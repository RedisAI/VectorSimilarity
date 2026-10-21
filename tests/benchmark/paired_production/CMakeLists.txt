cmake_minimum_required(VERSION 3.14)
project(MOD19169PairedProduction LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCE_ROOT "" CACHE PATH "Unmodified production source tree (or dispatch-only baseline)")
set(PAIRED_NATIVE_MODE "" CACHE STRING "Driver mode: 0=mixedFP32, 1=finalnative")
if(NOT IS_DIRECTORY "${SOURCE_ROOT}/src/VecSim")
    message(FATAL_ERROR "SOURCE_ROOT must point to a complete VectorSimilarity source tree")
endif()
if(NOT PAIRED_NATIVE_MODE MATCHES "^[01]$")
    message(FATAL_ERROR "PAIRED_NATIVE_MODE must be 0 (mixedFP32) or 1 (finalnative)")
endif()

# Serialization belongs to the production build; experiment modes belong only to the driver.
set(VECSIM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(VECSIM_BUILD_TEST_HELPERS ON CACHE BOOL "" FORCE)
add_subdirectory("${SOURCE_ROOT}" production)

include(FetchContent)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    google_benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.8.0.zip
)
FetchContent_MakeAvailable(google_benchmark)

add_executable(bm_mod19169_paired ../mod19169_paired_production.cpp)
target_include_directories(bm_mod19169_paired PRIVATE "${SOURCE_ROOT}/src")
target_compile_definitions(bm_mod19169_paired PRIVATE BUILD_TESTS PAIRED_NATIVE_MODE=${PAIRED_NATIVE_MODE})
target_link_libraries(bm_mod19169_paired PRIVATE VectorSimilarity benchmark::benchmark)
