cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
endif()
message("# VectorSimilarity/tests/unit root: " ${root})
message("# VectorSimilarity/tests/unit binroot: " ${binroot})

include(${root}/cmake/san.cmake)

project(VectorSimilarity_UnitTest)

include(CTest)
ADD_DEFINITIONS(-DBUILD_TESTS)
include(FetchContent)
	FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
	)
	# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/../../src)
add_subdirectory(${PROJECT_SOURCE_DIR}/../../ VectorSimilarity)

add_library(VectorSimilaritySerialization ../../src/VecSim/algorithms/hnsw/serialization.cpp)

add_executable(test_hnsw test_hnsw.cpp test_hnsw_multi.cpp test_utils.cpp)
add_executable(test_bruteforce test_bruteforce.cpp test_bruteforce_multi.cpp test_utils.cpp)
add_executable(test_allocator test_allocator.cpp test_utils.cpp)
add_executable(test_spaces test_spaces.cpp)
add_executable(test_common test_common.cpp test_utils.cpp)

target_link_libraries(test_hnsw PUBLIC gtest_main VectorSimilarity VectorSimilaritySerialization)
target_link_libraries(test_bruteforce PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_allocator PUBLIC gtest_main VectorSimilarity VectorSimilaritySerialization)
target_link_libraries(test_spaces PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_common PUBLIC gtest_main VectorSimilarity)

include(GoogleTest)

gtest_discover_tests(test_hnsw)
gtest_discover_tests(test_bruteforce)
gtest_discover_tests(test_allocator)
gtest_discover_tests(test_spaces)
gtest_discover_tests(test_common)
