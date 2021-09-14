cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if (NOT DEFINED VECSIM_ARCH)
set(VECSIM_ARCH "x86_64-v4")
endif()
message("VecSim/spaces VECSIM_ARCH: " ${VECSIM_ARCH})

add_compile_options(-march=${VECSIM_ARCH})

add_library(VectorSimilaritySpaces STATIC
    space_aux.cpp
    L2_space.cpp
    L2/L2.cpp
    L2/L2_SSE.cpp
    L2/L2_AVX.cpp
    L2/L2_AVX512.cpp
    IP_space.cpp
    IP/IP.cpp
    IP/IP_SSE.cpp
    IP/IP_AVX.cpp
    IP/IP_AVX512.cpp)
