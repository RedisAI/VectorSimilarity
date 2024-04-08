
message("# VectorSimilarity_Benchmark root: " ${root})
message("# VectorSimilarity_Benchmark binroot: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories(../)

enable_testing()

exec_program(${root}/tests/benchmark/benchmarks.sh OUTPUT_VARIABLE BENCHMARKS)
string(REPLACE "\n" ";" BENCHMARKS ${BENCHMARKS})

foreach(benchmark IN ITEMS ${BENCHMARKS})
	if(${benchmark} MATCHES "spaces*")
		continue()
	endif()

	add_executable(bm_${benchmark} run_files/bm_${benchmark}.cpp bm_vecsim_general.cpp)
	target_link_libraries(bm_${benchmark} VectorSimilarity benchmark::benchmark)
endforeach()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Spaces benchmarks								                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

set(DATA_TYPE fp32 fp64)
include(CheckCXXCompilerFlag)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	# build SSE/AVX* code only on x64 processors.
	# Check that the compiler supports instructions flag.
	# This will add the relevant flag both the the space selector and the optimization.
	CHECK_CXX_COMPILER_FLAG(-mavx512f CXX_AVX512F)
	CHECK_CXX_COMPILER_FLAG(-mavx512dq CXX_AVX512DQ)
	CHECK_CXX_COMPILER_FLAG(-mavx CXX_AVX)
	CHECK_CXX_COMPILER_FLAG(-msse CXX_SSE)

	if(CXX_AVX512F)
		add_compile_definitions(OPT_AVX512F)
	endif()

	if(CXX_AVX512DQ)
		add_compile_definitions(OPT_AVX512DQ)
	endif()

	if(CXX_AVX)
		add_compile_definitions(OPT_AVX)
	endif()

	if(CXX_SSE)
		add_compile_definitions(OPT_SSE)
	endif()
endif()

foreach(data_type IN LISTS DATA_TYPE)
	add_executable(bm_spaces_${data_type} spaces_benchmarks/bm_spaces_${data_type}.cpp)
	target_link_libraries(bm_spaces_${data_type} VectorSimilarity benchmark::benchmark)
endforeach()
