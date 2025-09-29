option(USE_ASAN "Use AddressSanitizer (clang)" OFF)
option(USE_MSAN "Use MemorySanitizer (clang)" OFF)
message(STATUS "SAN: ${SAN}")

if (USE_ASAN)
	# define this before project()

	set(CMAKE_CXX_FLAGS "-fno-omit-frame-pointer -fsanitize=address -fsized-deallocation  -fsanitize-recover=all")

	set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
	message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_C_FLAGS}")
	message(STATUS "CMAKE_LINKER_FLAGS: ${CMAKE_LINKER_FLAGS}")
	message(STATUS "CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
endif()
