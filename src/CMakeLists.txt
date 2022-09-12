cmake_minimum_required(VERSION 3.10)

option(VECSIM_BUILD_TESTS "Build tests" ON)

option(VECSIM_STATIC "Build as static library" OFF)

option(USE_ASAN "Use AddressSanitizer (clang)" OFF)
option(USE_MSAN "Use MemorySanitizer (clang)" OFF)

get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
message("# VectorSimilarity root: " ${root})
get_filename_component(binroot ${CMAKE_CURRENT_BINARY_DIR} ABSOLUTE)
message("# VectorSimilarity binroot: " ${binroot})
message("# VectorSimilarity static build: " ${VECSIM_STATIC})
message("# VectorSimilarity build test: " ${VECSIM_BUILD_TESTS})

include(${root}/cmake/common.cmake)

if (USE_ASAN OR USE_MSAN)
	include(${root}/cmake/clang-sanitizers.cmake)
endif()

if (VECSIM_BUILD_TESTS)
	ADD_DEFINITIONS(-DBUILD_TESTS)
	include(${root}/cmake/gtest.cmake)
	option(BUILD_TESTING "" ON)
endif()
#----------------------------------------------------------------------------------------------
project(VectorSimilarity)

set(CMAKE_CXX_STANDARD 20)

include_directories(${root}/src)
include_directories(${root}/deps)

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

add_subdirectory(VecSim/spaces)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
add_library(VectorSimilarity ${VECSIM_LIBTYPE}
	VecSim/algorithms/brute_force/brute_force_factory.cpp
	VecSim/algorithms/hnsw/hnsw_factory.cpp
    VecSim/algorithms/brute_force/vector_block.cpp
	VecSim/algorithms/hnsw/visited_nodes_handler.cpp
    VecSim/vec_sim.cpp
    VecSim/vec_sim_interface.cpp
    VecSim/query_results.cpp
	VecSim/info_iterator.cpp
    VecSim/query_result_struct.cpp
    VecSim/utils/vec_utils.cpp
    VecSim/memory/vecsim_malloc.cpp
    VecSim/memory/vecsim_base.cpp
  )


target_link_libraries(VectorSimilarity VectorSimilaritySpaces)

# Needed for build as ExternalProject (like RediSearch does)
install(TARGETS VectorSimilarity DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS VectorSimilaritySpaces DESTINATION ${CMAKE_INSTALL_PREFIX})

if(NOT VECSIM_STATIC)
    set_target_properties(VectorSimilarity PROPERTIES PREFIX "lib")
    set_target_properties(VectorSimilarity PROPERTIES SUFFIX ".so")
endif()

if(VECSIM_BUILD_TESTS)
	add_subdirectory(${root}/tests/unit unit_tests)
	add_subdirectory(${root}/tests/module module_tests)

	if (NOT(USE_ASAN OR USE_MSAN))
		add_subdirectory(${root}/tests/benchmark benchmark)
	endif()
endif()
