cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

option(VECSIM_STATIC "Build as static library" OFF)

get_filename_component(root ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)
message("# VectorSimilarity root: " ${root})
get_filename_component(binroot ${CMAKE_CURRENT_BINARY_DIR} ABSOLUTE)
message("# VectorSimilarity binroot: " ${binroot})
message("# VectorSimilarity static build: " ${VECSIM_STATIC})

if (USE_COVERAGE)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        message(FATAL_ERROR "Build type must be DEBUG for coverage")
    endif()
    set(COV_CXX_FLAGS "-coverage")
endif()

include(cmake/san.cmake)

#----------------------------------------------------------------------------------------------
project(VectorSimilarity)

set(CMAKE_CXX_STANDARD 20)

include_directories(${root}/src)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS} ${COV_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")
IF (USE_PROFILE)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
ENDIF()

if(VECSIM_STATIC)
	set(VECSIM_LIBTYPE STATIC)
else()
	set(VECSIM_LIBTYPE SHARED)
endif()

add_subdirectory(src/VecSim/spaces)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
add_library(VectorSimilarity ${VECSIM_LIBTYPE}
	src/VecSim/algorithms/brute_force/brute_force_factory.cpp
	src/VecSim/algorithms/hnsw/hnsw_factory.cpp
    src/VecSim/algorithms/brute_force/vector_block.cpp
	src/VecSim/algorithms/hnsw/visited_nodes_handler.cpp
    src/VecSim/vec_sim.cpp
    src/VecSim/vec_sim_interface.cpp
    src/VecSim/query_results.cpp
	src/VecSim/info_iterator.cpp
    src/VecSim/query_result_struct.cpp
    src/VecSim/utils/vec_utils.cpp
    src/VecSim/memory/vecsim_malloc.cpp
    src/VecSim/memory/vecsim_base.cpp
  )

target_link_libraries(VectorSimilarity VectorSimilaritySpaces)

# Needed for build as ExternalProject (like RediSearch does)
install(TARGETS VectorSimilarity DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS VectorSimilaritySpaces DESTINATION ${CMAKE_INSTALL_PREFIX})

if(NOT VECSIM_STATIC)
    set_target_properties(VectorSimilarity PROPERTIES PREFIX "lib")
    set_target_properties(VectorSimilarity PROPERTIES SUFFIX ".so")
endif()
