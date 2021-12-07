cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
endif()
message("# VectorSimilarity/tests/unit root: " ${root})
message("# VectorSimilarity/tests/unit binroot: " ${binroot})

project(VectorSimilarity_UnitTest)

include(CTest)

set(CMAKE_CXX_STANDARD 14)

if (NOT DEFINED VECSIM_ARCH AND ARCH STREQUAL "x64")
	set(VECSIM_ARCH "x86-64-v4")
endif()
message("# VecSim/UnitTest VECSIM_ARCH: " ${VECSIM_ARCH})

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_testing()

add_executable(test_hnswlib test_hnswlib.cpp test_utils.cpp)
add_executable(test_bruteforce test_bruteforce.cpp test_utils.cpp)
add_executable(test_allocator test_allocator.cpp)

target_link_libraries(test_hnswlib PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_bruteforce PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_allocator PUBLIC gtest_main VectorSimilarity)

include(GoogleTest)

gtest_discover_tests(test_hnswlib)
gtest_discover_tests(test_bruteforce)
gtest_discover_tests(test_allocator)
