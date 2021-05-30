cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0015 NEW)

project(VectorSimilarity)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../hnswlib)

add_library(vector_similarity SHARED
        algorithms/hnsw_c.cpp
        vecsim.c
        )

#target_link_libraries(vector_similarity ${PROJECT_SOURCE_DIR}/../nmslib/similarity_search/debug/libNonMetricSpaceLib.a)

SET_TARGET_PROPERTIES(vector_similarity PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(vector_similarity PROPERTIES SUFFIX ".so")