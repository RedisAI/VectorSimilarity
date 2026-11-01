# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

cmake_minimum_required(VERSION 3.20)
project(mod18264_measure LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(VECSIM_BUILD_TESTS ON CACHE BOOL "" FORCE)
set(VECSIM_BUILD_TEST_HELPERS ON CACHE BOOL "" FORCE)
set(USE_SVS ON CACHE BOOL "" FORCE)
set(SVS_SHARED_LIB OFF CACHE BOOL "" FORCE)
add_compile_definitions(BUILD_TESTS HAVE_SVS=1)
add_subdirectory("${VECSIM_SOURCE}" vecsim)

add_executable(auto_measure auto_measure.cpp)
target_include_directories(auto_measure PRIVATE "${VECSIM_SOURCE}/src")
target_link_libraries(auto_measure PRIVATE VectorSimilarity)
