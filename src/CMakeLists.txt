cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0015 NEW)

project(VectorSimilarity)

set(CMAKE_CXX_STANDARD 14)

option(VECSIM_STATIC "Build as static library" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../deps)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../deps_build/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

if(VECSIM_STATIC)
set(VECSIM_LIBTYPE STATIC)
else()
set(VECSIM_LIBTYPE SHARED)
endif()

add_library(VectorSimilarity ${VECSIM_LIBTYPE}
    VecSim/algorithms/brute_force.cpp
    VecSim/algorithms/hnsw_c.cpp
    VecSim/vecsim.cpp)


target_link_libraries(VectorSimilarity ${CMAKE_CURRENT_SOURCE_DIR}/../deps_build/lib/libopenblas.so) 

if(NOT VECSIM_STATIC)
    set_target_properties(VectorSimilarity PROPERTIES PREFIX "")
    set_target_properties(VectorSimilarity PROPERTIES SUFFIX ".so")
endif()

