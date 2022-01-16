
# build SSE/AVX* code only on x64 processors
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	project(VectorSimilaritySpaces_avx512)
	enable_language(C)

	add_library(VectorSimilaritySpaces_avx512 STATIC
		L2/L2_AVX512.c
		IP/IP_AVX512.c
		)
	target_compile_options(VectorSimilaritySpaces_avx512 PRIVATE -march=x86-64-v4)

	project(VectorSimilaritySpaces_avx)
	enable_language(C)
	add_library(VectorSimilaritySpaces_avx STATIC
	L2/L2_AVX.c
	IP/IP_AVX.c
	)
	target_compile_options(VectorSimilaritySpaces_avx PRIVATE -march=x86-64-v3)

	project(VectorSimilaritySpaces_sse)
	enable_language(C)
	add_library(VectorSimilaritySpaces_sse STATIC
		L2/L2_SSE.c
		IP/IP_SSE.c
		)
	target_compile_options(VectorSimilaritySpaces_sse PRIVATE -march=x86-64-v2)
endif()

project(VectorSimilaritySpaces_aux)
enable_language(C)
include(${root}/cmake/cpu_features.cmake)
add_library(VectorSimilaritySpaces_aux STATIC
space_aux.c
)
target_link_libraries(VectorSimilaritySpaces_aux cpu_features)



project(VectorSimilarity_Spaces)

set(SPACES_SOURCES
L2_space.cpp
L2/L2.c
IP_space.cpp
IP/IP.c)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
add_library(VectorSimilaritySpaces STATIC ${SPACES_SOURCES})
target_link_libraries(VectorSimilaritySpaces PUBLIC VectorSimilaritySpaces_aux)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	target_link_libraries(VectorSimilaritySpaces PUBLIC VectorSimilaritySpaces_avx512 VectorSimilaritySpaces_avx VectorSimilaritySpaces_sse)
endif()
