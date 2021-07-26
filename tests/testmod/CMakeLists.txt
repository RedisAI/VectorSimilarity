cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0015 NEW)

project(Vector_similarity_test_module C)
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${PROJECT_SOURCE_DIR}/../../src)

foreach(dir ${dirs})
    message(STATUS "dir='${dir}'")
endforeach()

add_library(testmod SHARED test_module.c)

set_target_properties(testmod PROPERTIES PREFIX "")
set_target_properties(testmod PROPERTIES SUFFIX ".so")

add_subdirectory(${PROJECT_SOURCE_DIR}/../../src VectorSimilarity)
target_link_libraries(testmod PUBLIC VectorSimilarity)
