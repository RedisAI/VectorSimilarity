cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

option(USE_ASAN "Use AddressSanitizer (clang)" OFF)
option(USE_MSAN "Use MemorySanitizer (clang)" OFF)

if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
	set(binroot ${root}/build)
endif()
message("VectorSimilarity_UnitTest: " ${binroot})

if (USE_ASAN OR USE_MSAN)
	include(${root}/cmake/clang-sanitizers.cmake)
endif()

project(VectorSimilarity_UnitTest)

set(CMAKE_CXX_STANDARD 14)

if (NOT DEFINED VECSIM_ARCH)
set(VECSIM_ARCH "x86-64-v4")
endif()
message("VecSim/UnitTest VECSIM_ARCH: " ${VECSIM_ARCH})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

include_directories(${root}/src)

add_subdirectory(${root}/src ${binroot}/VectorSimilarity)

enable_testing()

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(test_hnswlib test_hnswlib.cpp test_utils.cpp)
add_executable(test_bruteforce test_bruteforce.cpp test_utils.cpp)
add_executable(test_allocator test_allocator.cpp)

target_link_libraries(test_hnswlib PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_bruteforce PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_allocator PUBLIC gtest_main VectorSimilarity)

include(${root}/cmake/gtest.cmake)

gtest_discover_tests(test_hnswlib)
gtest_discover_tests(test_bruteforce)
gtest_discover_tests(test_allocator)
