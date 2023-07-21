option(USE_CUDA "Build Cuda code" On)
if(USE_CUDA)
  cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
else()
  cmake_minimum_required(VERSION 3.10)
endif()
cmake_policy(SET CMP0077 NEW)

set(CMAKE_CXX_STANDARD 20)
option(VECSIM_BUILD_TESTS "Build tests" ON)

# option(VECSIM_STATIC "Build as static library" OFF)
get_filename_component(root ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)
message("# VectorSimilarity root: " ${root})
get_filename_component(binroot ${CMAKE_CURRENT_BINARY_DIR} ABSOLUTE)
message("# VectorSimilarity binroot: " ${binroot})
message("# VectorSimilarity static build: " ${VECSIM_STATIC})

if(USE_COVERAGE)
	if(NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG")
		message(FATAL_ERROR "Build type must be DEBUG for coverage")
	endif()

	set(COV_CXX_FLAGS "-coverage")
endif()

include(cmake/san.cmake)

# ----------------------------------------------------------------------------------------------
project(VectorSimilarity)

if (USE_CUDA)
    # Enable CUDA compilation for this project
    enable_language(CUDA)
 	# Add definition for conditional compilation of CUDA components
	add_definitions(-DUSE_CUDA)
    # Perform all RAFT-specific CMake setup
	include(cmake/raft.cmake)
    # Required flags for compiling RAFT
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -std=c++17")
endif()

# Only do these if this is the main project, and not if it is included through add_subdirectory
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS} ${COV_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

IF(USE_PROFILE)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
ENDIF()

include_directories(src)

if(VECSIM_BUILD_TESTS)
	ADD_DEFINITIONS(-DBUILD_TESTS)
	include(FetchContent)
	enable_testing()

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS} ${COV_CXX_FLAGS}")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

	FetchContent_Declare(
		googletest
		URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
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

# Needed for build as ExternalProject (like RediSearch does)
install(TARGETS VectorSimilarity DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS VectorSimilaritySpaces DESTINATION ${CMAKE_INSTALL_PREFIX})

# if(NOT VECSIM_STATIC)
# set_target_properties(VectorSimilarity PROPERTIES PREFIX "lib")
# set_target_properties(VectorSimilarity PROPERTIES SUFFIX ".so")
# endif()
