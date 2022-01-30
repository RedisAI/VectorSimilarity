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

include(${root}/cmake/common.cmake)

if (USE_ASAN OR USE_MSAN)
	include(${root}/cmake/clang-sanitizers.cmake)
endif()

if (VECSIM_BUILD_TESTS)
	include(${root}/cmake/gtest.cmake)
	option(BUILD_TESTING "" ON)
endif()

#----------------------------------------------------------------------------------------------
project(VectorSimilarity)

set(CMAKE_CXX_STANDARD 20)

include_directories(${root}/src)
include_directories(${root}/deps)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS} ${COV_CXX_FLAGS} ${MARCH_CXX_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
IF (USE_PROFILE)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
ENDIF()

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

if(VECSIM_STATIC)
	set(VECSIM_LIBTYPE STATIC)
else()
	set(VECSIM_LIBTYPE SHARED)
endif()

add_library(VectorSimilarity ${VECSIM_LIBTYPE}
    VecSim/algorithms/brute_force/brute_force.cpp
    VecSim/algorithms/brute_force/vector_block.cpp
	VecSim/algorithms/hnsw/visited_nodes_handler.cpp
	VecSim/algorithms/hnsw/hnsw_wrapper.cpp
    VecSim/vec_sim.cpp
    VecSim/query_results.cpp
	VecSim/info_iterator.cpp
    VecSim/query_result_struct.cpp
    VecSim/utils/vec_utils.cpp
    VecSim/memory/vecsim_malloc.cpp
    VecSim/memory/vecsim_base.cpp
    VecSim/algorithms/brute_force/bf_batch_iterator.cpp
	VecSim/algorithms/hnsw/hnsw_batch_iterator.cpp
  )

if (${OSNICK} STREQUAL "xenial")
	include(ExternalProject)

	ExternalProject_Add(VectorSimilaritySpaces_project
		SOURCE_DIR      ${root}/src/VecSim/spaces
		PREFIX          ${binroot}/VectorSimilaritySpaces
		CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:path=${binroot}/VectorSimilaritySpaces
		CMAKE_ARGS      -DCMAKE_BUILD_TYPE:string=${CMAKE_BUILD_TYPE}
		CMAKE_ARGS      -DARCH:string=${ARCH}
		CMAKE_ARGS      -DCMAKE_C_COMPILER:string=/opt/miniforge/bin/gcc
		CMAKE_ARGS      -DCMAKE_CXX_COMPILER:string=/opt/miniforge/bin/g++
		CMAKE_ARGS      -DHAVE_MARCH_X86_64_V4:string=${HAVE_MARCH_X86_64_V4}
		CMAKE_ARGS      -DHAVE_MARCH_X86_64_V3:string=${HAVE_MARCH_X86_64_V3}
		CMAKE_ARGS      -DHAVE_MARCH_X86_64_V2:string=${HAVE_MARCH_X86_64_V2}
	)

	ExternalProject_Get_Property(VectorSimilaritySpaces_project install_dir)
	set(VectorSimilaritySpaces_bindir ${install_dir})

	set(spaces_libs)
	macro(add_vecsim_spaces_lib name)
		add_library("${name}" STATIC IMPORTED)
		set_target_properties("${name}" PROPERTIES IMPORTED_LOCATION "${VectorSimilaritySpaces_bindir}/lib${name}.a")
		if (NOT ${name} STREQUAL "VectorSimilaritySpaces")
			list(APPEND spaces_libs ${name})
		endif()
	endmacro()

	add_vecsim_spaces_lib(VectorSimilaritySpaces)
	add_vecsim_spaces_lib(VectorSimilaritySpaces_sse)
	add_vecsim_spaces_lib(VectorSimilaritySpaces_avx)
	add_vecsim_spaces_lib(VectorSimilaritySpaces_avx512)
	add_vecsim_spaces_lib(cpu_features)

	target_link_libraries(VectorSimilaritySpaces INTERFACE ${spaces_libs})
	target_link_libraries(VectorSimilarity VectorSimilaritySpaces)

	add_dependencies(VectorSimilaritySpaces VectorSimilaritySpaces_project)
	
	include_directories(${install_dir})
else()
	add_subdirectory(VecSim/spaces)
	target_link_libraries(VectorSimilarity VectorSimilaritySpaces)
endif()

if(NOT VECSIM_STATIC)
    set_target_properties(VectorSimilarity PROPERTIES PREFIX "lib")
    set_target_properties(VectorSimilarity PROPERTIES SUFFIX ".so")
endif()

if(VECSIM_BUILD_TESTS)
	add_library(VectorSimilaritySerialization VecSim/algorithms/hnsw/serialization.cpp)
	target_link_libraries(VectorSimilarity VectorSimilaritySerialization)
endif()

if(VECSIM_BUILD_TESTS)
	add_subdirectory(${root}/tests/unit unit_tests)
	add_subdirectory(${root}/tests/module module_tests)

	if (NOT(USE_ASAN OR USE_MSAN))
		add_subdirectory(${root}/tests/benchmark benchmark)
	endif()
endif()
