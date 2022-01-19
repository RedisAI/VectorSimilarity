
# build SSE/AVX* code only on x64 processors
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	project(VectorSimilaritySpaces_avx512)

	if (VECSIM_MARCH)
		add_compile_options(-march=${VECSIM_MARCH})
	else()
		add_compile_options(-march=x86-64-v4)
	endif()

	add_library(VectorSimilaritySpaces_avx512 STATIC
		L2/L2_AVX512.cpp
		IP/IP_AVX512.cpp
		)

	project(VectorSimilaritySpaces_avx)
	add_compile_options(-march=x86-64-v3)
	add_library(VectorSimilaritySpaces_avx STATIC
		L2/L2_AVX.cpp
		IP/IP_AVX.cpp
		)

	project(VectorSimilaritySpaces_sse)
	add_compile_options(-march=x86-64-v2)
	add_library(VectorSimilaritySpaces_sse STATIC
		L2/L2_SSE.cpp
		IP/IP_SSE.cpp
		)
endif()

project(VectorSimilarity_Spaces)

include(${root}/cmake/cpu_features.cmake)

set(SPACES_SOURCES
space_aux.cpp
L2_space.cpp
L2/L2.cpp
IP_space.cpp
IP/IP.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

add_library(VectorSimilaritySpaces STATIC ${SPACES_SOURCES})

target_link_libraries(VectorSimilaritySpaces cpu_features)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_avx512 VectorSimilaritySpaces_avx VectorSimilaritySpaces_sse)
endif()
