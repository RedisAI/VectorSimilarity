cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0015 NEW)

project(VectorSimilarity)

include_directories(${PROJECT_SOURCE_DIR}/../hnswlib/hnswlib)
include_directories($PROJECT_SOURCE_DIR)
#link_directories($PROJECT_SOURCE_DIR../nmslib/similarity_search/debug)

add_library(vector_similarity SHARED
        hnsw_c.cpp
        )

#target_link_libraries(vector_similarity ${PROJECT_SOURCE_DIR}/../nmslib/similarity_search/debug/libNonMetricSpaceLib.a)

SET_TARGET_PROPERTIES(vector_similarity PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(vector_similarity PROPERTIES SUFFIX ".so")