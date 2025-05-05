cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
endif()

message("# VectorSimilarity/tests/unit root: " ${root})
message("# VectorSimilarity/tests/unit binroot: " ${binroot})

project(VectorSimilarity_UnitTest)

include_directories(../utils)
include(CTest)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

enable_testing()

option(FP64_TESTS "Build fp 64 tests" OFF)

if(FP64_TESTS)
	add_definitions(-DFP64_TESTS)
endif()

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)|(ARM64)|(armv8)|(armv9)")
	include(${root}/cmake/aarch64InstructionFlags.cmake)
else()
	include(${root}/cmake/x86_64InstructionFlags.cmake)
endif()

add_executable(test_hnsw ../utils/mock_thread_pool.cpp test_hnsw.cpp test_hnsw_multi.cpp test_hnsw_tiered.cpp unit_test_utils.cpp)
add_executable(test_hnsw_parallel test_hnsw_parallel.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_bruteforce test_bruteforce.cpp test_bruteforce_multi.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_allocator test_allocator.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_spaces test_spaces.cpp)
add_executable(test_types test_types.cpp)
add_executable(test_common ../utils/mock_thread_pool.cpp test_common.cpp unit_test_utils.cpp)
add_executable(test_components test_components.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_bf16 ../utils/mock_thread_pool.cpp test_bf16.cpp unit_test_utils.cpp)
add_executable(test_fp16 ../utils/mock_thread_pool.cpp test_fp16.cpp unit_test_utils.cpp)
add_executable(test_int8 ../utils/mock_thread_pool.cpp test_int8.cpp unit_test_utils.cpp)
add_executable(test_uint8 ../utils/mock_thread_pool.cpp test_uint8.cpp unit_test_utils.cpp)
add_executable(test_svs ../utils/mock_thread_pool.cpp test_svs.cpp unit_test_utils.cpp)

target_link_libraries(test_hnsw PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_hnsw_parallel PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_bruteforce PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_allocator PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_spaces PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_common PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_components PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_types PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_bf16 PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_fp16 PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_int8 PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_uint8 PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_svs PUBLIC gtest_main VectorSimilarity)

include(GoogleTest)

gtest_discover_tests(test_hnsw)
gtest_discover_tests(test_hnsw_parallel)
gtest_discover_tests(test_bruteforce)
gtest_discover_tests(test_allocator)
gtest_discover_tests(test_spaces)
gtest_discover_tests(test_common)
gtest_discover_tests(test_components)
gtest_discover_tests(test_types)
gtest_discover_tests(test_bf16 TEST_PREFIX BF16UNIT_)
gtest_discover_tests(test_fp16 TEST_PREFIX FP16UNIT_)
gtest_discover_tests(test_int8 TEST_PREFIX INT8UNIT_)
gtest_discover_tests(test_uint8 TEST_PREFIX UINT8UNIT_)
gtest_discover_tests(test_svs)
