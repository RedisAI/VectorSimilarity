cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
project(VectorSimilarity)

set(CMAKE_CXX_STANDARD 14)

option(VECSIM_STATIC "Build as static library" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../deps)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC")

if(VECSIM_STATIC)
	set(VECSIM_LIBTYPE STATIC)
else()
	set(VECSIM_LIBTYPE SHARED)
endif()

add_compile_options(-march=native)

add_library(VectorSimilarity ${VECSIM_LIBTYPE}
    VecSim/algorithms/brute_force.cpp
    VecSim/algorithms/hnsw/hnswlib_c.cpp
    VecSim/vecsim.cpp
    VecSim/spaces/space_aux.cpp
    VecSim/spaces/L2_space.cpp
    VecSim/spaces/L2/L2.cpp
    VecSim/spaces/L2/L2_SSE.cpp
    VecSim/spaces/L2/L2_AVX.cpp
    VecSim/spaces/L2/L2_AVX512.cpp
    VecSim/spaces/IP_space.cpp
    VecSim/spaces/IP/IP.cpp
    VecSim/spaces/IP/IP_SSE.cpp
    VecSim/spaces/IP/IP_AVX.cpp
    VecSim/spaces/IP/IP_AVX512.cpp
    VecSim/utils/vec_utils.cpp)

if(NOT VECSIM_STATIC)
    set_target_properties(VectorSimilarity PROPERTIES PREFIX "")
    set_target_properties(VectorSimilarity PROPERTIES SUFFIX ".so")
endif()
