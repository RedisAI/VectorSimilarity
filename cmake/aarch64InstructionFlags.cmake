include(CheckCXXCompilerFlag)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Building for ARM aarch64")

  # Check baseline: ARMv8-A (NEON is available by default on ARMv8-A)
  CHECK_CXX_COMPILER_FLAG("-march=armv8-a" CXX_ARMV8A)
  CHECK_CXX_COMPILER_FLAG("-march=armv8-a+sve" CXX_SVE)
  CHECK_CXX_COMPILER_FLAG("-march=armv8.2-a+sve2" CXX_SVE2)
  
  # Try different syntax variants for ARMv9
  CHECK_CXX_COMPILER_FLAG("-march=armv9-a" CXX_ARMV9_ALT)
  CHECK_CXX_COMPILER_FLAG("-march=armv9.0-a" CXX_ARMV9)
  
  # Specific check for Graviton4 (Neoverse V2)  
  CHECK_CXX_COMPILER_FLAG("-msve2" CXX_SVE2_SUPPORTED)

  # Print which flags are supported for better diagnostics
  message(STATUS "ARM architecture flag support:")
  message(STATUS "  ARMv8-A (NEON): ${CXX_ARMV8A}")
  message(STATUS "  ARMv8-A+SVE: ${CXX_SVE}")
  message(STATUS "  ARMv8.2-A+SVE2: ${CXX_SVE2}")
  message(STATUS "  ARMv9-A (alt syntax): ${CXX_ARMV9_ALT}")
  message(STATUS "  ARMv9.0-A: ${CXX_ARMV9}")
  message(STATUS "  SVE2 support: ${CXX_SVE2_SUPPORTED}")

  # Check baseline: ARMv8-A (NEON is available by default on ARMv8-A)
  if(CXX_ARMV8A)
    message(STATUS "Compiler supports -march=armv8-a (NEON baseline)")
    add_compile_definitions(OPT_NEON)
  endif()

  # Optionally check if the compiler supports SVE (ARMv8-A with SVE)
  if(CXX_SVE)
    message(STATUS "Compiler supports -march=armv8-a+sve")
    add_compile_definitions(OPT_SVE)
  endif()

  # Optionally check if the compiler supports SVE2 (ARMv8.2-A with SVE2)
  if(CXX_SVE2)
    message(STATUS "Compiler supports -march=armv8.2-a+sve2")
    add_compile_definitions(OPT_SVE2)
  endif()

  # Check if the compiler supports ARMv9 with either syntax
  if(CXX_ARMV9 OR CXX_ARMV9_ALT)
    message(STATUS "Compiler supports ARMv9-A")
    add_compile_definitions(OPT_ARMV9)
    if(CXX_ARMV9)
      set(ARMV9_FLAG "-march=armv9.0-a")
    else()
      set(ARMV9_FLAG "-march=armv9-a")
    endif()
  endif()
  
  # Check if the compiler supports SVE2 specifically
  if(CXX_SVE2_SUPPORTED)
    message(STATUS "Compiler supports -msve2 (SVE2)")
    add_compile_definitions(OPT_SVE2)
  endif()

  # If none of the ARMv9 or SVE/SVE2 features were detected, fall back to ARMv8.2-A
  if(NOT CXX_ARMV9 AND NOT CXX_ARMV9_ALT AND NOT CXX_NEOVERSE_V2 AND NOT CXX_SVE AND NOT CXX_SVE2)
    message(STATUS "ARMv9 or SVE/SVE2 not supported. Falling back to ARMv8.2-A")
    CHECK_CXX_COMPILER_FLAG("-march=armv8.2-a" CXX_ARMV82A)
    if(CXX_ARMV82A)
      message(STATUS "Compiler supports -march=armv8.2-a (SVE2 fallback)")
      add_compile_definitions(OPT_ARMV82A)
    else()
      message(FATAL_ERROR "Neither ARMv9 nor ARMv8.2 architecture supported!")
    endif()
  endif()

endif()