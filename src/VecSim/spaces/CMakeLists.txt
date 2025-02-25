# Build non optimized code in a single project without architecture optimization flag.
project(VectorSimilaritySpaces_no_optimization)



add_library(VectorSimilaritySpaces_no_optimization
	L2/L2.cpp
	IP/IP.cpp
)


if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
	message(STATUS "Including Arm Performance Libraries")
	# Set default install paths – adjust these if needed.
	set(ARMPL_INCLUDE_DIR "/opt/arm/armpl_24.10_gcc/include")
	set(ARMPL_LIBRARY_DIR "/opt/arm/armpl_24.10_gcc/lib")

	# Verify that the header exists.
	if(EXISTS "${ARMPL_INCLUDE_DIR}/armpl.h")
		message(STATUS "Found Arm Performance Libraries header in ${ARMPL_INCLUDE_DIR}")
		include_directories(${ARMPL_INCLUDE_DIR})
	else()
		message(FATAL_ERROR "Could not find Arm Performance Libraries header in ${ARMPL_INCLUDE_DIR}")
	endif()


	# Find the ArmPL library (for 32-bit integer mode, commonly named armpl_lp64)
	find_library(ARMPL_LIB NAMES armpl_lp64_mp HINTS ${ARMPL_LIBRARY_DIR})
	if(ARMPL_LIB)
		message(STATUS "Found Arm Performance Libraries: ${ARMPL_LIB}")
	else()
		message(FATAL_ERROR "Arm Performance Libraries not found in ${ARMPL_LIBRARY_DIR}")
	endif()

endif()


include(${root}/cmake/cpu_features.cmake)

project(VectorSimilarity_Spaces)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

set(OPTIMIZATIONS "")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
	# Check that the compiler supports instructions flag.
	# from gcc14+ -mavx512bw is implicitly enabled when -mavx512vbmi2 is requested
	include(${root}/cmake/x86_64InstructionFlags.cmake)

	# build SSE/AVX* code only on x64 processors.
	# This will add the relevant flag both to the space selector and the optimization.
	if(CXX_AVX512BF16 AND CXX_AVX512VL)
		message("Building with AVX512BF16 and AVX512VL")
		set_source_files_properties(functions/AVX512BF16_VL.cpp PROPERTIES COMPILE_FLAGS "-mavx512bf16  -mavx512vl")
		list(APPEND OPTIMIZATIONS functions/AVX512BF16_VL.cpp)
	endif()

	if(CXX_AVX512VL AND CXX_AVX512FP16)
		message("Building with AVX512FP16 and AVX512VL")
		set_source_files_properties(functions/AVX512FP16_VL.cpp PROPERTIES COMPILE_FLAGS "-mavx512fp16 -mavx512vl")
		list(APPEND OPTIMIZATIONS functions/AVX512FP16_VL.cpp)
	endif()

	if(CXX_AVX512BW AND CXX_AVX512VBMI2)
		message("Building with AVX512BW and AVX512VBMI2")
		set_source_files_properties(functions/AVX512BW_VBMI2.cpp PROPERTIES COMPILE_FLAGS "-mavx512bw  -mavx512vbmi2")
		list(APPEND OPTIMIZATIONS functions/AVX512BW_VBMI2.cpp)
	endif()

	if(CXX_AVX512F)
		message("Building with AVX512F")
		set_source_files_properties(functions/AVX512F.cpp PROPERTIES COMPILE_FLAGS "-mavx512f")
		list(APPEND OPTIMIZATIONS functions/AVX512F.cpp)
	endif()

	if(CXX_AVX512F AND CXX_AVX512BW AND CXX_AVX512VL AND CXX_AVX512VNNI)
		message("Building with AVX512F, AVX512BW, AVX512VL and AVX512VNNI")
		set_source_files_properties(functions/AVX512F_BW_VL_VNNI.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512vnni")
		list(APPEND OPTIMIZATIONS functions/AVX512F_BW_VL_VNNI.cpp)
	endif()

	if(CXX_AVX2)
		message("Building with AVX2")
		set_source_files_properties(functions/AVX2.cpp PROPERTIES COMPILE_FLAGS -mavx2)
		list(APPEND OPTIMIZATIONS functions/AVX2.cpp)
	endif()

	if(CXX_F16C AND CXX_FMA AND CXX_AVX)
		message("Building with CXX_F16C")
		set_source_files_properties(functions/F16C.cpp PROPERTIES COMPILE_FLAGS "-mf16c -mfma -mavx")
		list(APPEND OPTIMIZATIONS functions/F16C.cpp)
	endif()

	if(CXX_AVX)
		message("Building with AVX")
		set_source_files_properties(functions/AVX.cpp PROPERTIES COMPILE_FLAGS -mavx)
		list(APPEND OPTIMIZATIONS functions/AVX.cpp)
	endif()

	if(CXX_SSE3)
		message("Building with SSE3")
		set_source_files_properties(functions/SSE3.cpp PROPERTIES COMPILE_FLAGS -msse3)
		list(APPEND OPTIMIZATIONS functions/SSE3.cpp)
	endif()

	if(CXX_SSE)
		message("Building with SSE")
		set_source_files_properties(functions/SSE.cpp PROPERTIES COMPILE_FLAGS -msse)
		list(APPEND OPTIMIZATIONS functions/SSE.cpp)
	endif()
# elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
#     # aarch64-specific optimizations.
#     message("Building with NEON support for aarch64")
#     # Optionally include an ARM-specific cmake file for additional checks.
#     # include(${root}/cmake/aarch64NeonFlags.cmake)
#     # Add your NEON-specific source file and compile flags.
#     list(APPEND OPTIMIZATIONS functions/NEONF.cpp)
endif()

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
	message(STATUS "Enabling Arm Performance Libraries integration")
	include(${root}/cmake/aarch64InstructionFlags.cmake)

	if (CXX_ARMV8A)
		message("Building with ARMV8A")
		set_source_files_properties(functions/NEONF.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
	endif()

	if (CXX_SVE)
		message("Building with SVE")
		set_source_files_properties(functions/NEONF.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+sve")
	endif()

	if (CXX_SVE2)
		message("Building with SVE2")
		set_source_files_properties(functions/NEONF.cpp PROPERTIES COMPILE_FLAGS "-march=armv8.2-a+sve2")
	endif()

	if (CXX_ARMV9)
		message("Building with ARMV9")
		set_source_files_properties(functions/NEONF.cpp PROPERTIES COMPILE_FLAGS "-march=armv9.2-a+sve2")
	endif()

    list(APPEND OPTIMIZATIONS functions/NEONF.cpp) 

endif()

# Here we are compiling the space selectors with the relevant optimization flag.
add_library(VectorSimilaritySpaces
	L2_space.cpp
	IP_space.cpp
	spaces.cpp
	${OPTIMIZATIONS}
	computer/preprocessor_container.cpp
)


target_link_libraries(VectorSimilaritySpaces VectorSimilaritySpaces_no_optimization cpu_features ${ARMPL_LIB})