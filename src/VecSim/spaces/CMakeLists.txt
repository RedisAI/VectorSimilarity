# Build non optimized code in a single project without architecture optimization flag.
project(VectorSimilaritySpaces_no_optimization)
add_library(VectorSimilaritySpaces_no_optimization
	L2/L2.cpp
	IP/IP.cpp
)

include(CheckCXXCompilerFlag)

project(VectorSimilarity_Spaces)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

# Here we are compiling the space selectors with the relevant optimization flag.
add_library(VectorSimilaritySpaces
	space_aux.cpp
	L2_space.cpp
	IP_space.cpp
	spaces.cpp
)

target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_no_optimization)

if(NOT APPLE)
	target_link_libraries(VectorSimilaritySpaces cpu_features)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
	target_link_libraries(VectorSimilaritySpaces cpu_features)
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	# Build and link the relevant optimization for X86.
	if(CXX_AVX512DQ)
		# project(VectorSimilaritySpaces_avx512dq)
		add_library(VectorSimilaritySpaces_avx512dq STATIC
			L2/L2_AVX512DQ_FP64.cpp
			IP/IP_AVX512DQ_FP64.cpp)
		target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_avx512dq)
	else()
		message("Building without AVX512DQ")
	endif()

	if(CXX_AVX512F)
		# project(VectorSimilaritySpaces_avx512)
		add_library(VectorSimilaritySpaces_avx512 STATIC
			L2/L2_AVX512_FP32.cpp
			L2/L2_AVX512_FP64.cpp
			IP/IP_AVX512_FP32.cpp
			IP/IP_AVX512_FP64.cpp)
		target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_avx512)
	else()
		message("Building without AVX512")
	endif()

	if(CXX_AVX)
		# project(VectorSimilaritySpaces_avx)
		add_library(VectorSimilaritySpaces_avx STATIC
			L2/L2_AVX_FP32.cpp
			L2/L2_AVX_FP64.cpp
			IP/IP_AVX_FP32.cpp
			IP/IP_AVX_FP64.cpp
		)
		target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_avx)
	else()
		message("Building without AVX")
	endif()

	if(CXX_SSE)
		# project(VectorSimilaritySpaces_sse)
		add_library(VectorSimilaritySpaces_sse STATIC
			L2/L2_SSE_FP32.cpp
			L2/L2_SSE_FP64.cpp
			IP/IP_SSE_FP32.cpp
			IP/IP_SSE_FP64.cpp
		)
		target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_sse)
	else()
		message("Building without SSE")
	endif()
endif()
