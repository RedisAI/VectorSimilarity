
include(CheckCXXCompilerFlag)

# build SSE/AVX* code only on x64 processors
if (${ARCH} STREQUAL "x64")
	set(VectorSimilaritySpaces_libs)
	CHECK_CXX_COMPILER_FLAG(-mavx512f CXX_AVX512F)

	if (CXX_AVX512F)
		message("# VectorSimilarity/spaces: building with AVX512 (-mavx512f)")
		list(APPEND VectorSimilaritySpaces_libs VectorSimilaritySpaces_avx512)
		project(VectorSimilaritySpaces_avx512)
			add_compile_options(-mavx512f)
			add_library(VectorSimilaritySpaces_avx512 STATIC
				L2/L2_AVX512.cpp
				IP/IP_AVX512.cpp)
	else()
		message("# VectorSimilarity/spaces: building without AVX512")
	endif()

	CHECK_CXX_COMPILER_FLAG(-mavx CXX_AVX)
	if (CXX_AVX)
		message("# VectorSimilarity/spaces: building with AVX (-mavx)")
		list(APPEND VectorSimilaritySpaces_libs VectorSimilaritySpaces_avx)
		project(VectorSimilaritySpaces_avx)
			add_compile_options(-mavx)
			add_library(VectorSimilaritySpaces_avx STATIC
				L2/L2_AVX.cpp
				IP/IP_AVX.cpp)
	else()
		message("# VectorSimilarity/spaces: building without AVX")
	endif()

	CHECK_CXX_COMPILER_FLAG(-msse CXX_SSE)
	if (CXX_SSE)
		message("# VectorSimilarity/spaces: building with SSE (-msse)")
		list(APPEND VectorSimilaritySpaces_libs VectorSimilaritySpaces_sse)
		project(VectorSimilaritySpaces_sse)
			add_compile_options(-msse)
			add_library(VectorSimilaritySpaces_sse STATIC
				L2/L2_SSE.cpp
				IP/IP_SSE.cpp)
	else()
		message("# VectorSimilarity/spaces: building without SSE")
	endif()

endif() # ARCH == x64

project(VectorSimilaritySpaces)

include(${root}/cmake/cpu_features.cmake)

set(SPACES_SOURCES
	space_aux.cpp
	L2_space.cpp
	L2/L2.cpp
	IP_space.cpp
	IP/IP.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

add_library(VectorSimilaritySpaces STATIC ${SPACES_SOURCES})
target_link_libraries(VectorSimilaritySpaces cpu_features ${VectorSimilaritySpaces_libs})

