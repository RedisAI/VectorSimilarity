
project(VectorSimilarity_Spaces)

if (NOT DEFINED VECSIM_ARCH AND ARCH STREQUAL "x64")
	set(VECSIM_ARCH "x86-64-v4")
endif()
message("VecSim/spaces VECSIM_ARCH: " ${VECSIM_ARCH})

if (DEFINED VECSIM_ARCH)
	add_compile_options(-march=${VECSIM_ARCH})
endif()

# build SSE/AVX* code only on x64 processors
if(ARCH STREQUAL "x64")
	set(SPACES_SOURCES
		space_aux.cpp
		L2_space.cpp
		L2/L2.cpp
		L2/L2_SSE.cpp
		L2/L2_AVX.cpp
		L2/L2_AVX512.cpp
		IP_space.cpp
		IP/IP.cpp
		IP/IP_SSE.cpp
		IP/IP_AVX.cpp
		IP/IP_AVX512.cpp)
else()
	set(SPACES_SOURCES
		space_aux.cpp
		L2_space.cpp
		L2/L2.cpp
		IP_space.cpp
		IP/IP.cpp)
endif()

add_library(VectorSimilaritySpaces STATIC ${SPACES_SOURCES})
