# Build non optimized code in a single project without architecture optimization flag.
project(VectorSimilaritySpaces_no_optimization)
add_library(VectorSimilaritySpaces_no_optimization
	L2/L2.cpp
	IP/IP.cpp
)

include(CheckCXXCompilerFlag)
include(${root}/cmake/cpu_features.cmake)

project(VectorSimilarity_Spaces)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

set(OPTIMIZATIONS "")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	# Check that the compiler supports instructions flag.
	CHECK_CXX_COMPILER_FLAG(-mavx512f CXX_AVX512F)
	CHECK_CXX_COMPILER_FLAG(-mavx CXX_AVX)
	CHECK_CXX_COMPILER_FLAG(-msse CXX_SSE)

	# build SSE/AVX* code only on x64 processors.
	# This will add the relevant flag both to the space selector and the optimization.
	if(CXX_AVX512F)
		message("Building with AVX512")
		set_source_files_properties(functions/AVX512.cpp PROPERTIES COMPILE_FLAGS -mavx512f)
		list(APPEND OPTIMIZATIONS functions/AVX512.cpp)
		add_compile_definitions(OPT_AVX512F)
	endif()

	if(CXX_AVX)
		message("Building with AVX")
		set_source_files_properties(functions/AVX.cpp PROPERTIES COMPILE_FLAGS -mavx)
		list(APPEND OPTIMIZATIONS functions/AVX.cpp)
		add_compile_definitions(OPT_AVX)
	endif()

	if(CXX_SSE)
		message("Building with SSE")
		set_source_files_properties(functions/SSE.cpp PROPERTIES COMPILE_FLAGS -msse)
		list(APPEND OPTIMIZATIONS functions/SSE.cpp)
		add_compile_definitions(OPT_SSE)
	endif()
endif()

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)|(ARM64)|(armv.*)")
	include(${root}/cmake/aarch64InstructionFlags.cmake)

	# Create different optimization implementations for ARM architecture
	if (CXX_ARMV8A)
		message("Building with ARMV8A")
		set_source_files_properties(functions/NEON.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
		list(APPEND OPTIMIZATIONS functions/NEON.cpp)
	endif()

	# SVE support
	if (CXX_SVE)
		message("Building with SVE")
		set_source_files_properties(functions/SVE.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+sve")
		list(APPEND OPTIMIZATIONS functions/SVE.cpp)
	endif()

	# SVE2 support
	if (CXX_SVE2)
		message("Building with ARMV9A and SVE2")
		set_source_files_properties(functions/SVE2.cpp PROPERTIES COMPILE_FLAGS "-march=armv9-a+sve2")
		list(APPEND OPTIMIZATIONS functions/SVE2.cpp)
	endif()
endif()

# Here we are compiling the space selectors with the relevant optimization flag.
add_library(VectorSimilaritySpaces
	space_aux.cpp
	L2_space.cpp
	IP_space.cpp
	spaces.cpp
	${OPTIMIZATIONS}
)

target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_no_optimization cpu_features)
