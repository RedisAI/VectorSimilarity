cmake_minimum_required(VERSION 3.26)
project(SSD_Indexes_Benchmark)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

enable_testing()

# Get google benchmark
include(FetchContent)
option(BENCHMARK_ENABLE_GTEST_TESTS "" OFF)
option(BENCHMARK_ENABLE_TESTING "" OFF)
FetchContent_Declare(
    google_benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.8.4.zip
)
FetchContent_MakeAvailable(google_benchmark)

find_library(ROCKSDB rocksdb PATHS ./rocksdb NO_DEFAULT_PATH)

add_executable(bm_ssd bm_options.cpp data_block.cpp)
target_link_libraries(bm_ssd benchmark::benchmark ${ROCKSDB})
