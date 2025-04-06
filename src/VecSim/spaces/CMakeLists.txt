# Build non optimized code in a single project without architecture optimization flag.
project(VectorSimilaritySpaces_no_optimization)
add_library(VectorSimilaritySpaces_no_optimization
	L2/L2.cpp
	IP/IP.cpp
)

include(CheckCXXCompilerFlag)
include(${root}/cmake/cpu_features.cmake)

project(VectorSimilarity_Spaces)

set(OPTIMIZATIONS "")
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	# build SSE/AVX* code only on x64 processors.
	# Check that the compiler supports instructions flag.
	# This will add the relevant flag both the the space selector and the optimization.
	CHECK_CXX_COMPILER_FLAG(-mavx512f CXX_AVX512F)
	CHECK_CXX_COMPILER_FLAG(-mavx512dq CXX_AVX512DQ)
	CHECK_CXX_COMPILER_FLAG(-mavx CXX_AVX)
	CHECK_CXX_COMPILER_FLAG(-msse CXX_SSE)

	# Build and link the relevant optimization for X86.
	if(CXX_AVX512DQ)
		# project(VectorSimilaritySpaces_avx512dq)
		set_source_files_properties(L2/L2_AVX512DQ_FP64.cpp PROPERTIES COMPILE_FLAGS "-mavx512dq")
		set_source_files_properties(IP/IP_AVX512DQ_FP64.cpp PROPERTIES COMPILE_FLAGS "-mavx512dq")
		add_library(VectorSimilaritySpaces_avx512dq STATIC
			L2/L2_AVX512DQ_FP64.cpp
			IP/IP_AVX512DQ_FP64.cpp)
		list(APPEND OPTIMIZATIONS VectorSimilaritySpaces_avx512dq)
		add_compile_definitions(OPT_AVX512DQ)
		message("Building with AVX512 DQ (Doubleword and Quadword) extension")
	else()
		message("Building without AVX512DQ")
	endif()

	if(CXX_AVX512F)
		# project(VectorSimilaritySpaces_avx512)
		set_source_files_properties(L2/L2_AVX512_FP32.cpp PROPERTIES COMPILE_FLAGS "-mavx512f")
		set_source_files_properties(L2/L2_AVX512_FP64.cpp PROPERTIES COMPILE_FLAGS "-mavx512f")
		set_source_files_properties(IP/IP_AVX512_FP32.cpp PROPERTIES COMPILE_FLAGS "-mavx512f")
		set_source_files_properties(IP/IP_AVX512_FP64.cpp PROPERTIES COMPILE_FLAGS "-mavx512f")
		add_library(VectorSimilaritySpaces_avx512 STATIC
			L2/L2_AVX512_FP32.cpp
			L2/L2_AVX512_FP64.cpp
			IP/IP_AVX512_FP32.cpp
			IP/IP_AVX512_FP64.cpp)
		list(APPEND OPTIMIZATIONS VectorSimilaritySpaces_avx512)
		add_compile_definitions(OPT_AVX512F)
		message("Building with AVX512")
	else()
		message("Building without AVX512")
	endif()

	if(CXX_AVX)
		# project(VectorSimilaritySpaces_avx)
		set_source_files_properties(L2/L2_AVX_FP32.cpp PROPERTIES COMPILE_FLAGS "-mavx")
		set_source_files_properties(L2/L2_AVX_FP64.cpp PROPERTIES COMPILE_FLAGS "-mavx")
		set_source_files_properties(IP/IP_AVX_FP32.cpp PROPERTIES COMPILE_FLAGS "-mavx")
		set_source_files_properties(IP/IP_AVX_FP64.cpp PROPERTIES COMPILE_FLAGS "-mavx")
		add_library(VectorSimilaritySpaces_avx STATIC
			L2/L2_AVX_FP32.cpp
			L2/L2_AVX_FP64.cpp
			IP/IP_AVX_FP32.cpp
			IP/IP_AVX_FP64.cpp
		)
		list(APPEND OPTIMIZATIONS VectorSimilaritySpaces_avx)
		add_compile_definitions(OPT_AVX)
		message("Building with AVX")
	else()
		message("Building without AVX")
	endif()

	if(CXX_SSE)
		# project(VectorSimilaritySpaces_sse)
		set_source_files_properties(L2/L2_SSE_FP32.cpp PROPERTIES COMPILE_FLAGS "-msse")
		set_source_files_properties(L2/L2_SSE_FP64.cpp PROPERTIES COMPILE_FLAGS "-msse")
		set_source_files_properties(IP/IP_SSE_FP32.cpp PROPERTIES COMPILE_FLAGS "-msse")
		set_source_files_properties(IP/IP_SSE_FP64.cpp PROPERTIES COMPILE_FLAGS "-msse")
		add_library(VectorSimilaritySpaces_sse STATIC
			L2/L2_SSE_FP32.cpp
			L2/L2_SSE_FP64.cpp
			IP/IP_SSE_FP32.cpp
			IP/IP_SSE_FP64.cpp
		)
		list(APPEND OPTIMIZATIONS VectorSimilaritySpaces_sse)
		add_compile_definitions(OPT_SSE)
		message("Building with SSE")
	else()
		message("Building without SSE")
	endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

# Here we are compiling the space selectors with the relevant optimization flag.
add_library(VectorSimilaritySpaces
	space_aux.cpp
	L2_space.cpp
	IP_space.cpp
	spaces.cpp
)

target_link_libraries(VectorSimilaritySpaces cpu_features VectorSimilaritySpaces_no_optimization ${OPTIMIZATIONS})
