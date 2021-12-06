
ifeq (n,$(findstring n,$(firstword -$(MAKEFLAGS))))
DRY_RUN:=1
else
DRY_RUN:=
endif

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
override SAN=memory
else ifeq ($(SAN),addr)
override SAN=address
endif

ifeq ($(SAN),memory)
CMAKE_SAN=-DUSE_MSAN=ON
override CTEST_ARGS += --exclude-regex BruteForceTest.sanity_rinsert_1280

else ifeq ($(SAN),address)
CMAKE_SAN=-DUSE_ASAN=ON
else ifeq ($(SAN),leak)
else ifeq ($(SAN),thread)
else
$(error SAN=mem|addr|leak|thread)
endif

export SAN
endif # SAN

#----------------------------------------------------------------------------------------------

ROOT=.
MK.pyver:=3

ifeq ($(wildcard $(ROOT)/deps/readies/mk),)
$(shell mkdir -p deps; cd deps; git clone https://github.com/RedisLabsModules/readies.git)
endif
include $(ROOT)/deps/readies/mk/main

#----------------------------------------------------------------------------------------------

define HELPTEXT
make build
  DEBUG=1          # build debug variant
  VERBOSE=1        # print detailed build info
  VG|VALGRIND=1    # build for Valgrind
  SAN=type         # build with LLVM sanitizer (type=address|memory|leak|thread)
make clean         # remove binary files
  ALL=1            # remove binary directories

make all           # build all libraries and packages

make test          # run tests
  CTEST_ARGS=args    # extra CTest arguments

make valgrind      # run valgrind

make benchmark	   # run benchmarks

make platform      # build for specific Linux distribution
  OSNICK=nick        # Linux distribution to build for
  REDIS_VER=ver      # use Redis version `ver`
  TEST=1             # test aftar build
  PUBLISH=1          # publish (i.e. docker push) after build

make format          # fix formatting of sources
make check-format    # check formatting of sources

endef

#----------------------------------------------------------------------------------------------

BINDIR=$(BINROOT)/src
TARGET=$(BINDIR)/libVectorSimilarity.so
SRCDIR=src

MK_CUSTOM_CLEAN=1

ifeq ($(SLOW),1)
MAKE_J=
else
MAKE_J:=-j$(shell nproc)
endif

#----------------------------------------------------------------------------------------------

CMAKE_DIR=$(ROOT)/src

CMAKE_FILES= \
	src/CMakeLists.txt \
	src/VecSim/spaces/CMakeLists.txt \
	cmake/common.cmake \
	cmake/gtest.cmake \
	cmake/clang-sanitizers.cmake

ifeq ($(DEBUG),1)
CMAKE_BUILD_TYPE=DEBUG
else
CMAKE_BUILD_TYPE=RelWithDebInfo
endif
CMAKE_FLAGS += -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)

ifeq ($(VERBOSE),1)
CMAKE_FLAGS += -DCMAKE_VERBOSE_MAKEFILE=on
endif

CMAKE_FLAGS += \
	-Wno-deprecated \
	-DCMAKE_WARN_DEPRECATED=OFF \
	-Wno-dev \
	-DOSNICK=$(OSNICK) \
	-DARCH=$(ARCH) \
	$(CMAKE_SAN)

#----------------------------------------------------------------------------------------------

include $(MK)/defs

#----------------------------------------------------------------------------------------------

all: bindirs $(TARGET)

.PHONY: all

include $(MK)/rules

#----------------------------------------------------------------------------------------------

.PHONY: __force

$(BINDIR)/Makefile: __force
	$(SHOW)cd $(BINDIR) && cmake $(CMAKE_FLAGS) $(CMAKE_DIR)

$(TARGET): $(BINDIR)/Makefile
	@echo Building $(TARGET) ...
ifneq ($(DRY_RUN),1)
	$(SHOW)$(MAKE) -C $(BINDIR) $(MAKE_J)
else
	@make -C $(BINDIR) $(MAKE_J)
endif

#build:
#	@mkdir -p build
#	@cd build; cmake $(CMAKE_FLAGS) ../src
#	@-touch build/Makefile
#	@make -C build

clean:
ifeq ($(ALL),1)
	$(SHOW)rm -rf $(BINROOT)
else
	$(SHOW)$(MAKE) -C $(BINDIR) clean
endif

.PHONY: clean

#----------------------------------------------------------------------------------------------

test:
#	$(SHOW)mkdir -p $(BINROOT)/test
#	$(SHOW)cd $(BINROOT)/tests && cmake $(CMAKE_FLAGS) $(ROOT)/tests/unit && $(MAKE)
#	@cd $(BINDIR)/test && cmake $(CMAKE_FLAGS) .. && make
#	@cd $(BINDIR)/test && GTEST_COLOR=1 ctest --output-on-failure $(CTEST_ARGS)
	$(SHOW)cd $(BINDIR)/unit_tests && ctest --output-on-failure $(CTEST_ARGS)

.PHONY: test

#----------------------------------------------------------------------------------------------

valgrind:
	@./tests/valgrind.sh

.PHONY: valgrind

#----------------------------------------------------------------------------------------------

check-format:
	./sbin/check-format.sh

format:
	FIX=1 ./sbin/check-format.sh

.PHONY: check-format format

#----------------------------------------------------------------------------------------------

platform:
	@make -C build/platforms build PACK=1 ARTIFACTS=1
ifeq ($(PUBLISH),1)
	@make -C build/platforms publish
endif

.PHONY: platform

#----------------------------------------------------------------------------------------------

benchmark:
	./tests/benchmark.sh

.PHONY: benchmark
