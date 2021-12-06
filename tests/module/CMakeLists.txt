
# cmake_minimum_required(VERSION 3.10)

# cmake_policy(SET CMP0077 NEW)
# set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
	set(binroot ${root}/build)
endif()
message("# VectorSimilarity_ModuleTest: " ${binroot})

project(VectorSimilarity_ModuleTest)

# set(CMAKE_CXX_STANDARD 14)

# if (NOT DEFINED VECSIM_ARCH)
# set(VECSIM_ARCH "x86-64-v4")
# endif()
# message("# VecSim/ModuleTest VECSIM_ARCH: " ${VECSIM_ARCH})

# if(NOT CMAKE_BUILD_TYPE)
#     set(CMAKE_BUILD_TYPE RelWithDebInfo)
# endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${VECSIM_ARCH}")

add_definitions(-DREDIS_MODULE_TARGET)

include_directories(${root}/src)

add_library(memory_test SHARED memory_test.c)

target_link_libraries(memory_test PUBLIC VectorSimilarity)

set_target_properties(memory_test PROPERTIES PREFIX "")
set_target_properties(memory_test PROPERTIES SUFFIX ".so")
