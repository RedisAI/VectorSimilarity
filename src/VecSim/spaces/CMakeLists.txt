project(VectorSimilarity_Spaces)

include(${root}/cmake/cpu_features.cmake)

if (VECSIM_MARCH)
	add_compile_options(-march=${VECSIM_MARCH})
endif()
message("VecSim/spaces VECSIM_MARCH: " ${VECSIM_MARCH})

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
add_library(VectorSimilaritySpaces STATIC ${SPACES_SOURCES})
target_link_libraries(VectorSimilaritySpaces cpu_features)
