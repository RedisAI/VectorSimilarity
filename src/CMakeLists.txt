cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
project(VectorSimilarity)

set(CMAKE_CXX_STANDARD 14)

option(VECSIM_STATIC "Build as static library" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../deps)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/OpenBLAS)

# openblas
set(USE_THREAD 0)
set(DYNAMIC_ARCH 1)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../deps/OpenBLAS OpenBLAS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC")

if(VECSIM_STATIC)
	set(VECSIM_LIBTYPE STATIC)
else()
	set(VECSIM_LIBTYPE SHARED)
endif()

add_library(VectorSimilarity ${VECSIM_LIBTYPE}
    VecSim/algorithms/brute_force.cpp
    VecSim/algorithms/hnsw/hnswlib_c.cpp
    VecSim/vecsim.cpp
    VecSim/spaces/internal_product.cpp
    VecSim/spaces/L2.cpp
    VecSim/utils/vec_utils.cpp)

target_link_libraries(VectorSimilarity openblas)

if(NOT VECSIM_STATIC)
    set_target_properties(VectorSimilarity PROPERTIES PREFIX "")
    set_target_properties(VectorSimilarity PROPERTIES SUFFIX ".so")
endif()
