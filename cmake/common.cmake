
cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if(NOT DEFINED OS)
	set(OS $ENV{OS})
endif()

if(NOT DEFINED OSNICK)
	set(OSNICK $ENV{OSNICK})
endif()

if(NOT DEFINED ARCH)
	set(ARCH $ENV{ARCH})
endif()

message("# OS=${OS}")
message("# OSNICK=${OSNICK}")
message("# ARCH=${ARCH}")

# if (NOT DEFINED VECSIM_MARCH AND ARCH STREQUAL "x64")
# 	set(VECSIM_MARCH "x86-64-v4")
# endif()
# message("# VECSIM_MARCH: " ${VECSIM_MARCH})

if (USE_COVERAGE)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        message(FATAL_ERROR "Build type must be DEBUG for coverage")
    endif()
    set(COV_CXX_FLAGS "-coverage")
endif()
