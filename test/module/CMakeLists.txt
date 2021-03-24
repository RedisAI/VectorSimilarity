cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0015 NEW)

project(Vector_similarity_test_module C)
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
link_directories(${PROJECT_SOURCE_DIR}/../../src/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/../../src/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/../../nmslib/similarity_search/include)
foreach(dir ${dirs})
    message(STATUS "dir='${dir}'")
endforeach()

ADD_LIBRARY(testmod SHARED test_module.c)

SET_TARGET_PROPERTIES(testmod PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(testmod PROPERTIES SUFFIX ".so")
target_link_libraries(testmod ${PROJECT_SOURCE_DIR}/../../src/vector_similarity.so)