
message("# VectorSimilarity_Benchmark root: " ${root})
message("# VectorSimilarity_Benchmark binroot: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_testing()

set(BM_SET bm_basics bm_updated_index bm_batch_iterator)
set(INDEX_TYPE single multi)
set(DATA_TYPE fp32)

foreach(bm_set IN LISTS BM_SET)
	foreach(index_type IN LISTS INDEX_TYPE)
		foreach(data_type IN LISTS DATA_TYPE)
			set(bm_test ${bm_set}_${index_type}_${data_type})
			add_executable(${bm_test} ${bm_test}.cpp bm_utils.cpp)
			target_link_libraries(${bm_test} VectorSimilarity benchmark::benchmark)
		endforeach(data_type IN LISTS DATA_TYPE)
	endforeach(index_type IN LISTS INDEX_TYPE)
endforeach(bm_set IN LISTS BM_SET)

#########################################################################################
# Passed this line we have optimization enabled by the compiler.                        #
# Adding files to complie after this line will have an effect on the resulted binaries. #
#########################################################################################

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

add_executable(bm_spaces bm_spaces.cpp)
target_link_libraries(bm_spaces VectorSimilarity benchmark::benchmark)
