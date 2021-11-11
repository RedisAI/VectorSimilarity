project(VectorSimilarity_Benchmark)

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(CMAKE_CXX_STANDARD 14)

if (NOT DEFINED VECSIM_ARCH)
    set(VECSIM_ARCH "x86-64-v4")
endif()
message("VecSim/Benchmark VECSIM_ARCH: " ${VECSIM_ARCH})

if(NOT DEFINED root)
    get_filename_component(root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
    set(binroot ${root}/build)
endif()
message("VectorSimilarity_Benchmark: " ${binroot})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
        google_benchmark
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/benchmark/archive/refs/tags/v1.6.0.zip
)
FetchContent_MakeAvailable(google_benchmark)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${root}/src)

add_subdirectory(${root}/src ${binroot}/VectorSimilarity)

enable_testing()

find_package(benchmark REQUIRED)
add_executable(bf_benchmark bf_google_benchmark.cpp)
target_link_libraries(bf_benchmark VectorSimilarity benchmark::benchmark)
