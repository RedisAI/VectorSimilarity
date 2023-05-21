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

ifneq ($(SAN),)
override DEBUG ?= 1
export ASAN_OPTIONS=detect_odr_violation=0:allocator_may_return_null=1
export MSAN_OPTIONS=allocator_may_return_null=1

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

ROOT=.
export ROOT
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

make unit_test     # run unit tests
  CTEST_ARGS=args    # extra CTest arguments
  VG|VALGRIND=1      # run tests with valgrind
  FP_64=1			# run tests with 64-bit floating point
make valgrind      # build for Valgrind and run tests
make flow_test     # run flow tests (with pytest)
  TEST=file::name    # run specific test
make mod_test      # run Redis module intergration tests (with RLTest)
  TEST=file:name     # run specific test
  VERBOSE=1          # show more test detail
make benchmark	   # run benchmarks

make format          # fix formatting of sources
make check-format    # check formatting of sources

make sanbox        # create container with CLang Sanitizer

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
TESTDIR=$(BINDIR)/unit_tests
BENCHMARKDIR=$(BINDIR)/benchmark
SRCDIR=src

ifeq ($(SLOW),1)
MAKE_J=
else
MAKE_J:=-j$(shell nproc)
endif

CMAKE_DIR=$(ROOT)

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

# CMake flags for fp64 unit tests
ifeq ($(FP_64),1)
CMAKE_FLAGS += -DFP64_TESTS=on
endif

CMAKE_FLAGS += \
	-Wno-deprecated \
	-DCMAKE_WARN_DEPRECATED=OFF \
	-Wno-dev \
	--no-warn-unused-cli \
	$(CMAKE_SAN) \
	$(CMAKE_COV) \
	$(CMAKE_TESTS)

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
	$(SHOW)poetry build
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
	$(SHOW)mkdir -p $(BINDIR)
	$(SHOW)cd $(BINDIR) && cmake $(CMAKE_FLAGS) $(CMAKE_DIR)
	@make --no-print-directory -C $(BINDIR) $(MAKE_J)
	$(SHOW)cd $(TESTDIR) && GTEST_COLOR=1 ctest $(_CTEST_ARGS)

valgrind:
	$(SHOW)$(MAKE) VG=1 unit_test

.PHONY: unit_test valgrind

#----------------------------------------------------------------------------------------------

flow_test:
	$(SHOW)poetry install
	$(SHOW)poetry run pytest tests/flow -v -s

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
	$(SHOW)mkdir -p $(BINDIR)
	$(SHOW)cd $(BINDIR) && cmake $(CMAKE_FLAGS) $(CMAKE_DIR)
	@make --no-print-directory -C $(BINDIR) $(MAKE_J)
	$(ROOT)/tests/benchmark/benchmarks.sh $(BM_FILTER) | xargs -I {} bash -lc \
		"$(BENCHMARKDIR)/bm_{} --benchmark_out_format=json --benchmark_out={}_results.json || exit 255"

toxenv:
ifeq ($(wildcard .tox),)
	$(SHOW)tox -e flowenv
endif
	$(SHOW)bash -c ". ./.tox/flowenv/bin/activate; $$SHELL"

.PHONY: unit_test flow_test mem_test benchmark toxenv

#----------------------------------------------------------------------------------------------

check-format:
	$(SHOW)./check-format.sh

format:
	$(SHOW)FIX=1 ./check-format.sh

.PHONY: check-format format

COV_EXCLUDE_DIRS += bin tests
COV_EXCLUDE+=$(foreach D,$(COV_EXCLUDE_DIRS),'$(realpath $(ROOT))/$(D)/*')

COV_INFO=$(BINROOT)/cov.info
COV_DIR=$(BINROOT)/cov
COV_PROFDATA=$(COV_DIR)/cov.profdata

define COVERAGE_RESET
$(SHOW)set -e ;\
echo "Starting coverage analysis." ;\
mkdir -p $(COV_DIR) ;\
lcov --directory $(BINROOT) --base-directory $(SRCDIR) -z
endef


define COVERAGE_COLLECT
$(SHOW)set -e ;\
echo "Collecting coverage data ..." ;\
lcov --capture --directory $(BINROOT) --base-directory $(SRCDIR) --output-file $(COV_INFO);\
lcov -o $(COV_INFO).1 -r $(COV_INFO) $(COV_EXCLUDE);\
mv $(COV_INFO).1 $(COV_INFO)
endef

define COVERAGE_REPORT
$(SHOW)set -e ;\
lcov -l $(COV_INFO) ;\
genhtml --legend --ignore-errors source -o $(COV_DIR) $(COV_INFO) > /dev/null 2>&1 ;\
echo "Coverage info at $$(realpath $(COV_DIR))/index.html"
endef

define COVERAGE_COLLECT_REPORT
$(COVERAGE_COLLECT)
$(COVERAGE_REPORT)
endef

coverage:
	$(SHOW)$(MAKE) build COV=1
	$(SHOW)$(COVERAGE_RESET)
	$(SHOW)cd $(TESTDIR) && GTEST_COLOR=1 ctest $(_CTEST_ARGS)
	$(SHOW)$(COVERAGE_COLLECT_REPORT)

show-cov:
	$(SHOW)lcov -l $(COV_INFO)

upload-cov:
	$(SHOW)bash <(curl  https://raw.githubusercontent.com/codecov/codecov-bash/master/codecov) -f ${COV_INFO}

.PHONY: coverage show-cov upload-cov

#----------------------------------------------------------------------------------------------

ifneq ($(wildcard /w/*),)
SANBOX_ARGS += -v /w:/w
endif

sanbox:
	@docker run -it -v $(PWD):/vecsim -w /vecsim --cap-add=SYS_PTRACE --security-opt seccomp=unconfined $(SANBOX_ARGS) redisfab/clang:13-x64-bullseye bash

.PHONY: sanbox
