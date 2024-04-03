
message("# VectorSimilarity_Benchmark root: " ${root})
message("# VectorSimilarity_Benchmark binroot: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 20)

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
	# NOTE: mock_thread_pool.cpp should appear *before* the benchmark files, so we can ensure that the thread pool
	# globals are initialized before we use them in the benchmark classes (as globals initialization is done by order).
	add_executable(bm_${benchmark} ../utils/mock_thread_pool.cpp bm_vecsim_general.cpp run_files/bm_${benchmark}.cpp)
	target_link_libraries(bm_${benchmark} VectorSimilarity benchmark::benchmark)
endforeach()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Spaces benchmarks								                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

include(CheckCXXCompilerFlag)
set(OPT_FLAGS "")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	# build SSE/AVX* code only on x64 processors.
	# Check that the compiler supports instructions flag.
	# This will add the relevant flag both the the space selector and the optimization.
	CHECK_CXX_COMPILER_FLAG(-mavx512f CXX_AVX512F)
	CHECK_CXX_COMPILER_FLAG(-mavx CXX_AVX)
	CHECK_CXX_COMPILER_FLAG(-msse CXX_SSE)

	if(CXX_AVX512F)
		list(APPEND OPT_FLAGS -mavx512f)
	endif()

	if(CXX_AVX)
		list(APPEND OPT_FLAGS -mavx)
	endif()

	if(CXX_SSE)
		list(APPEND OPT_FLAGS -msse)
	endif()
endif()

set(DATA_TYPE fp32 fp64)
foreach(data_type IN LISTS DATA_TYPE)
	add_library(bm_spaces_class_${data_type} spaces_benchmarks/bm_spaces_class_${data_type}.cpp)
	target_link_libraries(bm_spaces_class_${data_type} benchmark::benchmark)
	set_source_files_properties(spaces_benchmarks/bm_spaces_${data_type}.cpp PROPERTIES COMPILE_FLAGS ${OPT_FLAGS})
	add_executable(bm_spaces_${data_type} spaces_benchmarks/bm_spaces_${data_type}.cpp)
	target_link_libraries(bm_spaces_${data_type} VectorSimilarity benchmark::benchmark bm_spaces_class_${data_type})
endforeach()
