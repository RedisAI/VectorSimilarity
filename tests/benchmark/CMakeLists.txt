
message("# VectorSimilarity_Benchmark root: " ${root})
message("# VectorSimilarity_Benchmark binroot: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_testing()

ADD_DEFINITIONS(-DBUILD_TESTS)
include(FetchContent)
FetchContent_Declare(
googletest
URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

option(BENCHMARK_ENABLE_GTEST_TESTS "" OFF)
option(BENCHMARK_ENABLE_TESTING  "" OFF)
FetchContent_Declare(
	google_benchmark
	URL https://github.com/google/benchmark/archive/refs/tags/v1.7.0.zip
)
FetchContent_MakeAvailable(google_benchmark)

option(BUILD_TESTING "" ON)

include_directories(${PROJECT_SOURCE_DIR}/../../src)
add_subdirectory(${PROJECT_SOURCE_DIR}/../../ VectorSimilarity)

add_library(VectorSimilaritySerialization ../../src/VecSim/algorithms/hnsw/serialization.cpp)

add_executable(bm_batch_iterator bm_batch_iterator.cpp bm_utils.cpp)
target_link_libraries(bm_batch_iterator VectorSimilarity VectorSimilaritySerialization benchmark::benchmark)
add_executable(bm_basics bm_basics.cpp bm_utils.cpp)
target_link_libraries(bm_basics VectorSimilarity VectorSimilaritySerialization benchmark::benchmark)
add_executable(bm_updated_index bm_updated_index.cpp bm_utils.cpp)
target_link_libraries(bm_updated_index VectorSimilarity VectorSimilaritySerialization benchmark::benchmark)

#########################################################################################
# Passed this line we have optimization enabled by the compiler.                        #
# Adding files to complie after this line will have an effect on the resulted binaries. #
#########################################################################################

add_library(bm_spaces_class bm_spaces_class.cpp)
target_link_libraries(bm_spaces_class benchmark::benchmark)

include(CheckCXXCompilerFlag)
# TODO: Remove this once cpu_features get support for M1
if(NOT APPLE)
    include(${PROJECT_SOURCE_DIR}/../../cmake/cpu_features.cmake)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    include(${PROJECT_SOURCE_DIR}/../../cmake/cpu_features.cmake)
else()
    add_definitions(-DM1)
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	# build SSE/AVX* code only on x64 processors.
	# Check that the compiler supports instructions flag.
	# This will add the relevant flag both the the space selector and the optimization.
	CHECK_CXX_COMPILER_FLAG(-mavx512f CXX_AVX512F)
	CHECK_CXX_COMPILER_FLAG(-mavx512dq CXX_AVX512DQ)
	CHECK_CXX_COMPILER_FLAG(-mavx CXX_AVX)
	CHECK_CXX_COMPILER_FLAG(-msse CXX_SSE)
	if(CXX_AVX512F)
		add_compile_options(-mavx512f)
		message("Building with AVX512")
	endif()
	if(CXX_AVX512DQ)
		add_compile_options(-mavx512dq)
		message("Building with AVX512 DQ (Doubleword and Quadword) extension")
	endif()
	if(CXX_AVX)
		add_compile_options(-mavx)
		message("Building with AVX")
	endif()
	if(CXX_SSE)
		add_compile_options(-msse)
		message("Building with SSE")
	endif()
endif()

add_executable(bm_spaces bm_spaces.cpp)
target_link_libraries(bm_spaces VectorSimilarity VectorSimilaritySerialization benchmark::benchmark bm_spaces_class)
