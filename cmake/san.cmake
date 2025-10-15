option(USE_ASAN "Use AddressSanitizer (clang)" OFF)
if (USE_ASAN)
	# define this before project()
	find_file(CMAKE_C_COMPILER "clang")
	find_file(CMAKE_CXX_COMPILER "clang++")
	set(CMAKE_LINKER "${CMAKE_C_COMPILER}")
	set(CLANG_SAN_FLAGS "-fno-omit-frame-pointer -fsanitize=address -fsized-deallocation")

endif()
