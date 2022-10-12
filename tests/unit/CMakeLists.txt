cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
endif()
message("# VectorSimilarity/tests/unit root: " ${root})
message("# VectorSimilarity/tests/unit binroot: " ${binroot})

project(VectorSimilarity_UnitTest)

include(CTest)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

enable_testing()

add_executable(test_hnsw test_hnsw.cpp test_hnsw_multi.cpp test_utils.cpp)
add_executable(test_bruteforce test_bruteforce.cpp test_bruteforce_multi.cpp test_utils.cpp)
add_executable(test_allocator test_allocator.cpp test_utils.cpp)
add_executable(test_common test_common.cpp)

target_link_libraries(test_hnsw PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_bruteforce PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_allocator PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_common PUBLIC gtest_main VectorSimilarity)

include(GoogleTest)

gtest_discover_tests(test_hnsw)
gtest_discover_tests(test_bruteforce)
gtest_discover_tests(test_allocator)
gtest_discover_tests(test_common)

include(CheckCXXCompilerFlag)
# TODO: Remove this once cpu_features get support for M1
if(NOT APPLE)
	include(${root}/cmake/cpu_features.cmake)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
	include(${root}/cmake/cpu_features.cmake)
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

add_executable(test_spaces test_spaces.cpp)
target_link_libraries(test_spaces PUBLIC gtest_main VectorSimilarity)
gtest_discover_tests(test_spaces)
