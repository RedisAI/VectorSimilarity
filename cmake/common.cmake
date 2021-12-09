
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
