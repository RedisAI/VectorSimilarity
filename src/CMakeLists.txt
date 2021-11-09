cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

project(VectorSimilarity)

set(CMAKE_CXX_STANDARD 14)

if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
endif()
if(NOT DEFINED binroot)
	set(binroot ${root}/build)
endif()
message("VectorSimilarity binroot: " ${binroot})

option(VECSIM_STATIC "Build as static library" OFF)

SET(CMAKE_CC_COMMON_FLAGS "${CMAKE_CC_COMMON_FLAGS} -fno-omit-frame-pointer")

message("VectorSimilarity VECSIM_ARCH: " ${VECSIM_ARCH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${root}/deps)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC")

if(VECSIM_STATIC)
	set(VECSIM_LIBTYPE STATIC)
else()
	set(VECSIM_LIBTYPE SHARED)
endif()

add_subdirectory(VecSim/spaces)

add_compile_options(-march=x86-64)

add_library(VectorSimilarity ${VECSIM_LIBTYPE}
    VecSim/algorithms/brute_force/brute_force.cpp
    VecSim/algorithms/brute_force/vector_block.cpp
    VecSim/algorithms/hnsw/hnswlib_c.cpp
	VecSim/vec_sim.cpp
	VecSim/query_results.cpp
	VecSim/query_result_struct.cpp
        VecSim/algorithms/brute_force/bf_batch_iterator.cpp
	VecSim/utils/vec_utils.cpp)

target_link_libraries(VectorSimilarity VectorSimilaritySpaces)

install(TARGETS VectorSimilarity
    DESTINATION ${CMAKE_INSTALL_PREFIX})

install(TARGETS VectorSimilaritySpaces
    DESTINATION ${CMAKE_INSTALL_PREFIX})

if(NOT VECSIM_STATIC)
    set_target_properties(VectorSimilarity PROPERTIES PREFIX "lib")
    set_target_properties(VectorSimilarity PROPERTIES SUFFIX ".so")
endif()
