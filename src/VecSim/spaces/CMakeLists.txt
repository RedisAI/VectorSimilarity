
project(VectorSimilaritySpaces_no_optimization)
	add_library(VectorSimilaritySpaces_no_optimization STATIC
	L2/L2.cpp
	IP/IP.cpp
	)

project(VectorSimilarity_Spaces)

include(${root}/cmake/cpu_features.cmake)
include(CheckCXXCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	# build SSE/AVX* code only on x64 processors
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
add_library(VectorSimilaritySpaces STATIC 
	space_aux.cpp
	L2_space.cpp
	IP_space.cpp
	)

target_link_libraries(VectorSimilaritySpaces cpu_features VectorSimilaritySpaces_no_optimization)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	# build SSE/AVX* code only on x64 processors
	CHECK_CXX_COMPILER_FLAG(-mavx512f CXX_AVX512F)
	CHECK_CXX_COMPILER_FLAG(-mavx CXX_AVX)
	CHECK_CXX_COMPILER_FLAG(-msse CXX_SSE)

	if(CXX_AVX512F)
			project(VectorSimilaritySpaces_avx512)
				add_library(VectorSimilaritySpaces_avx512 STATIC
					L2/L2_AVX512.cpp
					IP/IP_AVX512.cpp
					)
		target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_avx512)
	else()
		message("Building without AVX512")
	endif()
	if(CXX_AVX)
		project(VectorSimilaritySpaces_avx)
			add_library(VectorSimilaritySpaces_avx STATIC
				L2/L2_AVX.cpp
				IP/IP_AVX.cpp
				)
		target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_avx)
	else()
		message("Building without AVX")
	endif()
	if(CXX_SSE)
		project(VectorSimilaritySpaces_sse)
			add_library(VectorSimilaritySpaces_sse STATIC
				L2/L2_SSE.cpp
				IP/IP_SSE.cpp
				)
		target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_sse)
	else()
		message("Building without SSE")
	endif()
endif()
