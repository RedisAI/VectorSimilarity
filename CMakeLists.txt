cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)

set(CMAKE_CXX_STANDARD 20)

# VECSIM_BUILD_TEST_HELPERS: Enables test helper methods (fitMemory, serialization, etc.)
# in VectorSimilarity classes without building test executables.
option(VECSIM_BUILD_TEST_HELPERS "Build test helper methods (fitMemory, serialization)" OFF)

# VECSIM_BUILD_TESTS: Builds VectorSimilarity test executables (unit tests, benchmarks).
# When ON, test helpers are also built even if VECSIM_BUILD_TEST_HELPERS is OFF.
option(VECSIM_BUILD_TESTS "Build VectorSimilarity test executables" ON)

get_filename_component(root ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)
message("# VectorSimilarity root: " ${root})
get_filename_component(binroot ${CMAKE_CURRENT_BINARY_DIR} ABSOLUTE)
message("# VectorSimilarity binroot: " ${binroot})

if(USE_COVERAGE)
	if(NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG")
		message(FATAL_ERROR "Build type must be DEBUG for coverage")
	endif()

	set(COV_CXX_FLAGS "-coverage")
endif()

include(cmake/san.cmake)

# ----------------------------------------------------------------------------------------------
project(VectorSimilarity)

# Official RaBitQ is an opt-in research dependency, never part of the public
# VectorSimilarity library or installed API.
option(VECSIM_RABITQ_RESEARCH "Build the RaBitQ research qualification tools" OFF)
if(VECSIM_RABITQ_RESEARCH)
	if(NOT VECSIM_BUILD_TESTS)
		message(FATAL_ERROR "VECSIM_RABITQ_RESEARCH requires VECSIM_BUILD_TESTS")
	endif()
	set(RABITQ_BUILD_INDEX OFF CACHE BOOL "" FORCE)
	set(RABITQ_BUILD_FASTSCAN OFF CACHE BOOL "" FORCE)
	set(RABITQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(RABITQ_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
	set(RABITQ_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
	set(RABITQ_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
	set(RABITQ_USE_OPENMP OFF CACHE BOOL "" FORCE)
	set(RABITQ_ENABLE_NATIVE_OPTIMIZATION OFF CACHE BOOL "" FORCE)
	include(FetchContent)
	FetchContent_Declare(
		rabitq
		GIT_REPOSITORY https://github.com/VectorDB-NTU/RaBitQ-Library.git
		GIT_TAG 4c91ef02bdfaa531b81fe679605acae31952d26a
	)
	FetchContent_MakeAvailable(rabitq)
endif()

# Only do these if this is the main project, and not if it is included through add_subdirectory
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS} ${COV_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

IF(USE_PROFILE)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
ENDIF()

include(cmake/svs.cmake)

include_directories(src)

# Define BUILD_TESTS macro if test helpers OR tests are requested.
# This enables test helper methods (fitMemory, serialization, etc.) in class definitions.
if(VECSIM_BUILD_TEST_HELPERS OR VECSIM_BUILD_TESTS)
	ADD_DEFINITIONS(-DBUILD_TESTS)
endif()

# Build test executables only if explicitly requested (VECSIM_BUILD_TESTS=ON).
# This fetches googletest/google_benchmark and builds test targets.
if(VECSIM_BUILD_TESTS)
	include(FetchContent)
	enable_testing()

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS} ${COV_CXX_FLAGS}")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

	FetchContent_Declare(
		googletest
		URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
	)

	# For Windows: Prevent overriding the parent project's compiler/linker settings
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(googletest)

	option(BENCHMARK_ENABLE_GTEST_TESTS "" OFF)
	option(BENCHMARK_ENABLE_TESTING "" OFF)
	FetchContent_Declare(
		google_benchmark
		URL https://github.com/google/benchmark/archive/refs/tags/v1.8.0.zip
	)
	FetchContent_MakeAvailable(google_benchmark)

	add_subdirectory(tests/unit unit_tests)
	add_subdirectory(tests/module module_tests)

	if(NOT(USE_ASAN OR USE_MSAN))
		add_subdirectory(tests/benchmark benchmark)
	endif()
endif()

add_subdirectory(src/VecSim)

# TurboQuant's inline runtime selector must have the same definition in the
# library and internal consumers. Individual SVE2 functions use target attributes;
# generic objects retain the baseline ISA and dispatch checks the running CPU.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$" AND
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "10")
	include(CheckIncludeFileCXX)
	check_include_file_cxx(arm_sve.h VECSIM_HAS_ARM_SVE_H)
	if(VECSIM_HAS_ARM_SVE_H)
		target_compile_definitions(VectorSimilarity PUBLIC OPT_SVE2)
		if(TARGET VectorSimilaritySerializer)
			target_compile_definitions(VectorSimilaritySerializer PRIVATE OPT_SVE2)
		endif()
	endif()
endif()

# Needed for build as ExternalProject (like RediSearch does)
install(TARGETS VectorSimilarity DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS VectorSimilaritySpaces DESTINATION ${CMAKE_INSTALL_PREFIX})

# Print compiler flags (similar to RediSearch)
get_directory_property(COMPILE_DEFS COMPILE_DEFINITIONS)
message(STATUS "CXXFLAGS:         ${CMAKE_CXX_FLAGS}")
message(STATUS "COMPILE_DEFS:     ${COMPILE_DEFS}")
message(STATUS "LDFLAGS:          ${CMAKE_SHARED_LINKER_FLAGS}")
