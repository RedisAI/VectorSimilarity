ROOT=.
MK.pyver:=3

ifeq ($(wildcard $(ROOT)/deps/readies/mk),)
$(shell mkdir -p deps; cd deps; git clone https://github.com/RedisLabsModules/readies.git)
endif
include $(ROOT)/deps/readies/mk/main

#----------------------------------------------------------------------------------------------

define HELP
make build
  DEBUG=1          # build debug variant
make clean         # remove binary files
  ALL=1            # remove binary directories

make all           # build all libraries and packages

make test          # run tests

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

build:
	mkdir -p build
	cd build; cmake ../src
	-touch build/Makefile
	make -C build

clean:
ifneq ($(ALL),1)
	make -C build clean
else
	rm -rf build
endif

.PHONY: build clean

#----------------------------------------------------------------------------------------------

test:
	cd test/module
	mkdir -p build
	cd test/module/build; cmake ..
	make -C test/module/build

.PHONY: test

#----------------------------------------------------------------------------------------------

platform:
	@make -C build/platforms build
ifeq ($(PUBLISH),1)
	@make -C build/platforms publish
endif
