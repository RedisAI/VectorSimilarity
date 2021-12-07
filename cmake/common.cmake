
cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if(NOT DEFINED OSNICK)
	set(OSNICK $ENV{OSNICK})
endif()

if(NOT DEFINED ARCH)
	set(ARCH "x64")
#	set(ARCH $ENV{ARCH})
endif()
