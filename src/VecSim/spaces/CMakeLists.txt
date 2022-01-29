
# build SSE/AVX* code only on x64 processors
if (${ARCH} STREQUAL "x64")
	set(VectorSimilaritySpaces_libs "")
	
	if (HAVE_MARCH_X86_64_V4)
		message("# VectorSimilarity/spaces: building with AVX512 (-march=x86-64-v4)")
		set(VectorSimilaritySpaces_libs ${VectorSimilaritySpaces_libs} VectorSimilaritySpaces_avx512)
		project(VectorSimilaritySpaces_avx512)
			add_compile_options(-march=x86-64-v4)
			add_library(VectorSimilaritySpaces_avx512 STATIC
				L2/L2_AVX512.cpp
				IP/IP_AVX512.cpp)
	else()
		message("# VectorSimilarity/spaces: building without AVX512")
	endif()

	if (HAVE_MARCH_X86_64_V3)
		message("# VectorSimilarity/spaces: building with AVX (-march=x86-64-v3)")
		set(VectorSimilaritySpaces_libs ${VectorSimilaritySpaces_libs} VectorSimilaritySpaces_avx)
		project(VectorSimilaritySpaces_avx)
			add_compile_options(-march=x86-64-v3)
			add_library(VectorSimilaritySpaces_avx STATIC
				L2/L2_AVX.cpp
				IP/IP_AVX.cpp)
	else()
		message("# VectorSimilarity/spaces: building without AVX")
	endif()

	if (HAVE_MARCH_X86_64_V2)
		message("# VectorSimilarity/spaces: building with SSE (-march=x86-64-v2)")
		set(VectorSimilaritySpaces_libs ${VectorSimilaritySpaces_libs} VectorSimilaritySpaces_sse)
		project(VectorSimilaritySpaces_sse)
			add_compile_options(-march=x86-64-v2)
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

install(TARGETS VectorSimilaritySpaces DESTINATION ${CMAKE_INSTALL_PREFIX})
