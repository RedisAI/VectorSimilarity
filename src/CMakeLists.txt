cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0015 NEW)

project(VectorSimilarity)

set(CMAKE_CXX_STANDARD 14)

if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
	set(binroot ${root}/build)
endif()
message("VectorSimilarity: " ${binroot})

option(VECSIM_STATIC "Build as static library" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${root}/deps)
include_directories(${root}/deps/OpenBLAS)
include_directories(${binroot}/OpenBLAS)

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

if(NOT VECSIM_STATIC)
    set_target_properties(VectorSimilarity PROPERTIES PREFIX "")
    set_target_properties(VectorSimilarity PROPERTIES SUFFIX ".so")
endif()

add_subdirectory(${root}/deps/OpenBLAS ${binroot}/OpenBLAS)
target_link_libraries(VectorSimilarity ${binroot}/OpenBLAS/lib/libopenblas.a)
