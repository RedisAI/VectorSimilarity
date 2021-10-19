cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

option(VECSIM_STATIC "Build as static library" OFF)

option(USE_ASAN "Use AddressSanitizer (clang)" OFF)
option(USE_MSAN "Use MemorySanitizer (clang)" OFF)

if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
endif()
if(NOT DEFINED binroot)
	set(binroot ${root}/build)
endif()
message("VectorSimilarity binroot: " ${binroot})

if (USE_ASAN OR USE_MSAN)
	include(${root}/cmake/clang-sanitizers.cmake)
endif()

project(VectorSimilarity)

set(CMAKE_CXX_STANDARD 14)

message("VectorSimilarity VECSIM_ARCH: " ${VECSIM_ARCH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${root}/deps)

message("${CLANG_SAN_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

if(VECSIM_STATIC)
	set(VECSIM_LIBTYPE STATIC)
else()
	set(VECSIM_LIBTYPE SHARED)
endif()

add_subdirectory(VecSim/spaces)

add_compile_options(-march=x86-64)

add_library(VectorSimilarity ${VECSIM_LIBTYPE}
    VecSim/algorithms/brute_force.cpp
    VecSim/algorithms/hnsw/hnswlib_c.cpp
    VecSim/vecsim.cpp
    VecSim/utils/vec_utils.cpp)

target_link_libraries(VectorSimilarity VectorSimilaritySpaces)

install(TARGETS VectorSimilarity
    DESTINATION ${CMAKE_INSTALL_PREFIX})

install(TARGETS VectorSimilaritySpaces
    DESTINATION ${CMAKE_INSTALL_PREFIX})

if(NOT VECSIM_STATIC)
    set_target_properties(VectorSimilarity PROPERTIES PREFIX "")
    set_target_properties(VectorSimilarity PROPERTIES SUFFIX ".so")
endif()
