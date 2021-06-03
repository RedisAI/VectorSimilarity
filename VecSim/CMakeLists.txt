cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0015 NEW)
project(VectorSimilarity)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/hnswlib/hnswlib)

# # for prefixes like 'hnswlib/hnswlib'
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
# # for prefixes like 'VectorSimilarity/src'
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_library(VectorSimilarity SHARED
        algorithms/hnsw_c.cpp
        src/vecsim.c
        )

SET_TARGET_PROPERTIES(VectorSimilarity PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(VectorSimilarity PROPERTIES SUFFIX ".so")

