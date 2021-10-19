ROOT=.
MK.pyver:=3

ifeq ($(wildcard $(ROOT)/deps/readies/mk),)
$(shell mkdir -p deps; cd deps; git clone https://github.com/RedisLabsModules/readies.git)
endif
include $(ROOT)/deps/readies/mk/main

#----------------------------------------------------------------------------------------------

ifneq ($(VG),)
VALGRIND=$(VG)
endif

ifeq ($(VALGRIND),1)
override DEBUG ?= 1
endif

ifneq ($(SAN),)
override DEBUG ?= 1
export ASAN_OPTIONS=detect_odr_violation=0

ifeq ($(SAN),mem)
CMAKE_SAN=-DUSE_MSAN=ON
export SAN=memory

else ifeq ($(SAN),memory)
CMAKE_SAN=-DUSE_MSAN=ON
export SAN=memory

else ifeq ($(SAN),addr)
CMAKE_SAN=-DUSE_ASAN=ON
export SAN=address

else ifeq ($(SAN),address)
CMAKE_SAN=-DUSE_ASAN=ON
export SAN=address

else ifeq ($(SAN),leak)
else ifeq ($(SAN),thread)
else
$(error SAN=mem|addr|leak|thread)
endif

endif # SAN

#----------------------------------------------------------------------------------------------

define HELP
make build
  DEBUG=1          # build debug variant
  VERBOSE=1        # print detailed build info
  VG|VALGRIND=1    # build for Valgrind
  SAN=type         # build with LLVM sanitizer (type=address|memory|leak|thread)
make clean         # remove binary files
  ALL=1            # remove binary directories

make all           # build all libraries and packages

make test          # run tests

make valgrind		   # run valgrind

make platform      # build for specific Linux distribution
  OSNICK=nick        # Linux distribution to build for
  REDIS_VER=ver      # use Redis version `ver`
  TEST=1             # test aftar build
  PACK=1             # create packages
  ARTIFACTS=1        # copy artifacts from docker image
  PUBLISH=1          # publish (i.e. docker push) after build


endef

#----------------------------------------------------------------------------------------------

MK_CUSTOM_CLEAN=1
BINDIR=$(BINROOT)

include $(MK)/defs
include $(MK)/rules

#----------------------------------------------------------------------------------------------

all: build

.PHONY: all

#----------------------------------------------------------------------------------------------

ifeq ($(DEBUG),1)
CMAKE_BUILD_TYPE=DEBUG
else
CMAKE_BUILD_TYPE=RelWithDebInfo
endif
CMAKE_FLAGS += -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)

ifeq ($(VERBOSE),1)
CMAKE_FLAGS += -DCMAKE_VERBOSE_MAKEFILE=on
endif

CMAKE_FLAGS += $(CMAKE_SAN)

build:
	@mkdir -p build
	@cd build; cmake $(CMAKE_FLAGS) ../src
	@-touch build/Makefile
	@make -C build

clean:
ifneq ($(ALL),1)
	@make -C build clean
	@make -C tests/unit/build clean
else
	@rm -rf build tests/unit/build
endif

.PHONY: build clean

#----------------------------------------------------------------------------------------------

test:
	@mkdir -p tests/unit/build
	@cd tests/unit/build && cmake $(CMAKE_FLAGS) .. && make
	@cd tests/unit/build && GTEST_COLOR=1 ctest --output-on-failure

.PHONY: test

#----------------------------------------------------------------------------------------------

valgrind:
	@./tests/valgrind.sh

.PHONY: valgrind

#----------------------------------------------------------------------------------------------

platform:
	@make -C build/platforms build
ifeq ($(PUBLISH),1)
	@make -C build/platforms publish
endif

format:
	@./sbin/clang-format-all.sh

lint:
	@./sbin/clang-check-all.sh
