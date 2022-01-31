
ifeq (n,$(findstring n,$(firstword -$(MAKEFLAGS))))
DRY_RUN:=1
else
DRY_RUN:=
endif

ifneq ($(filter coverage show-cov upload-cov,$(MAKECMDGOALS)),)
COV=1
endif

ifneq ($(VG),)
VALGRIND=$(VG)
endif

ifeq ($(VALGRIND),1)
override DEBUG ?= 1
endif

ifeq ($(COV),1)
override DEBUG ?= 1
CMAKE_COV += -DUSE_COVERAGE=ON
endif

ifeq ($(NO_TESTS),1)
CMAKE_TESTS += -DVECSIM_BUILD_TESTS=off
endif

#----------------------------------------------------------------------------------------------

define HELPTEXT
make build
  DEBUG=1          # build debug variant
  COV=1			   # build for code coverage
  VERBOSE=1        # print detailed build info
  VG|VALGRIND=1    # build for Valgrind
  SAN=type         # build with LLVM sanitizer (type=address|memory|leak|thread)
  SLOW=1           # don't run build in parallel (for diagnostics)
  PROFILE=1		   # enable profiling compile flags (and debug symbols) for release type.
make pybind        # build Python bindings
make clean         # remove binary files
  ALL=1            # remove binary directories

make all           # build all libraries and packages

make unit_test     # run unit tests
  CTEST_ARGS=args    # extra CTest arguments
  VG|VALGRIND=1      # run tests with valgrind
make valgrind      # build for Valgrind and run tests
make flow_test     # run flow tests (with pytest)
  TEST=file::name    # run specific test
make mod_test      # run Redis module intergration tests (with RLTest)
  TEST=file:name     # run specific test
  VERBOSE=1          # show more test detail
make benchmark	   # run benchmarks

make format          # fix formatting of sources
make check-format    # check formatting of sources

endef

#----------------------------------------------------------------------------------------------
ROOT=$(shell pwd)
ifeq ($(DEBUG),1)
FLAVOR=debug
else
FLAVOR=release
endif
FULL_VARIANT:=$(shell uname)-$(shell uname -m)-$(FLAVOR)
BINROOT=$(ROOT)/bin/$(FULL_VARIANT)
BINDIR=$(BINROOT)
TARGET=$(BINDIR)/libVectorSimilarity.so
SRCDIR=src


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

ifeq ($(PROFILE),1)
CMAKE_FLAGS += -DUSE_PROFILE=on
endif

ifeq ($(VERBOSE),1)
CMAKE_FLAGS += -DCMAKE_VERBOSE_MAKEFILE=on
endif

CMAKE_FLAGS += \
	-Wno-deprecated \
	-DCMAKE_WARN_DEPRECATED=OFF \
	-Wno-dev \
	--no-warn-unused-cli \
	$(CMAKE_SAN) \
	$(CMAKE_COV) \
	$(CMAKE_TESTS)

#----------------------------------------------------------------------------------------------


build:
	$(SHOW)mkdir -p $(BINDIR)
	$(SHOW)cd $(BINDIR) && cmake $(CMAKE_FLAGS) $(CMAKE_DIR)
	@make --no-print-directory -C $(BINDIR) $(MAKE_J)

.PHONY: build

clean:
ifeq ($(ALL),1)
	$(SHOW)rm -rf $(BINROOT) build dist .tox
else
	$(SHOW)$(MAKE) --no-print-directory -C $(BINDIR) clean
endif

.PHONY: clean

#----------------------------------------------------------------------------------------------

pybind:
	$(SHOW)python3 -m poetry build
.PHONY: pybind

#----------------------------------------------------------------------------------------------

_CTEST_ARGS=$(CTEST_ARGS)
_CTEST_ARGS += \
	--output-on-failure \
	$(MAKE_J)

ifeq ($(VERBOSE),1)
_CTEST_ARGS += -V
endif

ifeq ($(VALGRIND),1)
_CTEST_ARGS += \
	-T memcheck \
	--overwrite MemoryCheckCommandOptions="--leak-check=full --error-exitcode=255"
endif

unit_test:
	$(SHOW)cd $(BINDIR)/unit_tests && GTEST_COLOR=1 ctest $(_CTEST_ARGS)

valgrind:
	$(SHOW)$(MAKE) VG=1 build unit_test

.PHONY: unit_test valgrind

#----------------------------------------------------------------------------------------------

flow_test:
	$(SHOW)$(MAKE) pybind
	$(SHOW)tox -e flowenv

.PHONY: flow_test

#----------------------------------------------------------------------------------------------

ifneq ($(TEST),)
RLTEST_ARGS += --test $(TEST) -s
endif
ifeq ($(VERBOSE),1)
RLTEST_ARGS += -v
endif

mod_test:
	$(SHOW)cd $(ROOT)/tests/module/flow && \
		python3 -m RLTest \
		--module $(BINDIR)/module_tests//memory_test.so \
		--clear-logs \
		$(RLTEST_ARGS)

.PHONY: mod_test

#----------------------------------------------------------------------------------------------

benchmark:
	$(SHOW)$(BINDIR)/benchmark/bm_basics
	$(SHOW)$(BINDIR)/benchmark/bm_batch_iterator
	$(SHOW)python3 -m tox -e benchmark
#----------------------------------------------------------------------------------------------

check-format:
	$(SHOW)./sbin/check-format.sh

format:
	$(SHOW)FIX=1 ./sbin/check-format.sh

.PHONY: check-format format

#----------------------------------------------------------------------------------------------

COV_EXCLUDE_DIRS += bin tests
COV_EXCLUDE+=$(foreach D,$(COV_EXCLUDE_DIRS),'$(realpath $(ROOT))/$(D)/*')

coverage:
	$(SHOW)$(MAKE) build COV=1
	$(SHOW)$(COVERAGE_RESET)
	$(SHOW)$(MAKE) unit_test COV=1
	$(SHOW)$(COVERAGE_COLLECT_REPORT)

show-cov:
	$(SHOW)lcov -l $(COV_INFO)

upload-cov:
	$(SHOW)bash <(curl -s https://raw.githubusercontent.com/codecov/codecov-bash/master/codecov) -f bin/linux-x64-debug-cov/cov.info

.PHONY: coverage show-cov upload-cov

#----------------------------------------------------------------------------------------------
