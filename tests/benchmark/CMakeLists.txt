
cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if(NOT DEFINED root)
    get_filename_component(root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
    set(binroot ${root}/build)
endif()
message("# VectorSimilarity_Benchmark: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 14)

if (NOT DEFINED VECSIM_ARCH)
    set(VECSIM_ARCH "x86-64-v4")
endif()
message("# VecSim/Benchmark VECSIM_ARCH: " ${VECSIM_ARCH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -march=${VECSIM_ARCH}")

include_directories(${root}/src)

enable_testing()

include(${root}/cmake/gtest.cmake)

add_executable(bf_benchmark bf_google_benchmark.cpp)

target_link_libraries(bf_benchmark VectorSimilarity benchmark::benchmark)
