
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

# First we compile the spaces benchmarks class so it won't include any compiler flags
# that are not supported by the hardware.
set(DATA_TYPE fp32 fp64)

foreach(data_type IN LISTS DATA_TYPE)
	add_library(bm_spaces_class_${data_type} spaces_benchmarks/bm_spaces_class_${data_type}.cpp)
	target_link_libraries(bm_spaces_class_${data_type} benchmark::benchmark)
endforeach(data_type IN LISTS DATA_TYPE)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Passed this line we have optimization enabled by the compiler.                        #
# Adding files to complie after this line will have an effect on the resulted binaries. #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
	CHECK_CXX_COMPILER_FLAG(-mavx CXX_AVX)
	CHECK_CXX_COMPILER_FLAG(-msse CXX_SSE)

	if(CXX_AVX512F)
		add_compile_options(-mavx512f)
		message("Building with AVX512")
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

foreach(data_type IN LISTS DATA_TYPE)
	add_executable(bm_spaces_${data_type} spaces_benchmarks/bm_spaces_${data_type}.cpp)
	target_link_libraries(bm_spaces_${data_type} VectorSimilarity benchmark::benchmark bm_spaces_class_${data_type})
endforeach()
