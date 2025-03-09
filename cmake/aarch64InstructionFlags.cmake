include(CheckCXXCompilerFlag)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Building for ARM aarch64")

  # Check baseline: ARMv8-A (NEON is available by default on ARMv8-A)
  CHECK_CXX_COMPILER_FLAG("-march=armv8-a" CXX_ARMV8A)
  CHECK_CXX_COMPILER_FLAG("-march=armv8-a+sve" CXX_SVE)
  CHECK_CXX_COMPILER_FLAG("-march=armv8.2-a+sve2" CXX_SVE2)
  CHECK_CXX_COMPILER_FLAG("-march=armv9.0-a" CXX_ARMV9)
  CHECK_CXX_COMPILER_FLAG("-msve2" CXX_SVE2_SUPPORTED)

  # Check if the compiler supports -march=armv8-a (NEON baseline)
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

  # Check if the compiler supports ARMv9.0-A
  if(CXX_ARMV9)
    message(STATUS "Compiler supports -march=armv9.0-a")
    add_compile_definitions(OPT_ARMV9)
  endif()

  # Check if the compiler supports SVE2 specifically for ARMv9.0-A
  if(CXX_SVE2_SUPPORTED)
    message(STATUS "Compiler supports -msve2 (SVE2)")
    add_compile_definitions(OPT_SVE2)
  endif()

  # If none of the ARMv9 or SVE/SVE2 features were detected, fall back to ARMv8.2-A
  if(NOT CXX_ARMV9 AND NOT CXX_SVE AND NOT CXX_SVE2)
    message(STATUS "ARMv9.0-A or SVE/SVE2 not supported. Falling back to ARMv8.2-A")
    CHECK_CXX_COMPILER_FLAG("-march=armv8.2-a" CXX_ARMV82A)
    if(CXX_ARMV82A)
      message(STATUS "Compiler supports -march=armv8.2-a (SVE2 fallback)")
      add_compile_definitions(OPT_ARMV82A)
    else()
      message(FATAL_ERROR "Neither ARMv9 nor ARMv8.2 architecture supported!")
    endif()
  endif()

endif()