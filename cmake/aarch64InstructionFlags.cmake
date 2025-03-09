include(CheckCXXCompilerFlag)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Building for ARM aarch64")

  # Check if we can compile for various ARM architectures
  CHECK_CXX_COMPILER_FLAG("-march=armv8-a" CXX_ARMV8A)
  CHECK_CXX_COMPILER_FLAG("-march=armv8-a+sve" CXX_SVE)
  CHECK_CXX_COMPILER_FLAG("-march=armv8.2-a+sve2" CXX_SVE2)
  CHECK_CXX_COMPILER_FLAG("-march=armv9-a" CXX_ARMV9_ALT)
  CHECK_CXX_COMPILER_FLAG("-march=armv9.0-a" CXX_ARMV9)
  CHECK_CXX_COMPILER_FLAG("-msve2" CXX_SVE2_SUPPORTED)

  # Print which flags are supported by the compiler
  message(STATUS "ARM architecture flag support (compiler):")
  message(STATUS "  ARMv8-A (NEON): ${CXX_ARMV8A}")
  message(STATUS "  ARMv8-A+SVE: ${CXX_SVE}")
  message(STATUS "  ARMv8.2-A+SVE2: ${CXX_SVE2}")
  message(STATUS "  ARMv9-A (alt syntax): ${CXX_ARMV9_ALT}")
  message(STATUS "  ARMv9.0-A: ${CXX_ARMV9}")
  message(STATUS "  SVE2 support: ${CXX_SVE2_SUPPORTED}")

  # Determine architecture version (v8 or v9)
  if(CXX_ARMV9 OR CXX_ARMV9_ALT)
    message(STATUS "ARMv9 architecture detected")
    add_compile_definitions(OPT_ARMV9)
    set(ARM_ARCH "9")
  else()
    message(STATUS "ARMv8 architecture detected")
    add_compile_definitions(OPT_ARMV8)
    set(ARM_ARCH "8")
  endif()
  
  # Add feature detection based on compiler flag support
  if(CXX_ARMV8A)
    message(STATUS "NEON support detected (compiler)")
    add_compile_definitions(OPT_NEON)
  endif()
  
  if(CXX_SVE)
    message(STATUS "SVE support detected (compiler)")
    add_compile_definitions(OPT_SVE)
  endif()
  
  if(CXX_SVE2 OR CXX_SVE2_SUPPORTED)
    message(STATUS "SVE2 support detected (compiler)")
    add_compile_definitions(OPT_SVE2)
  endif()

endif()