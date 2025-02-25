include(CheckCXXCompilerFlag)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Building for ARM aarch64")

  # Check baseline: ARMv8-A (NEON is available by default on ARMv8-A)
  CHECK_CXX_COMPILER_FLAG("-march=armv8-a" CXX_ARMV8A)
  CHECK_CXX_COMPILER_FLAG("-march=armv8-a+sve" CXX_SVE)
  CHECK_CXX_COMPILER_FLAG("-march=armv8.2-a+sve2" CXX_SVE2)
  CHECK_CXX_COMPILER_FLAG("-march=armv9.2-a+sve2" CXX_ARMV9)
  
  if(CXX_ARMV8A)
    message(STATUS "Compiler supports -march=armv8-a (NEON baseline)")
    add_compile_definitions(OPT_NEON)
  endif()

  # Optionally check if the compiler supports SVE
  if(CXX_SVE)
    message(STATUS "Compiler supports -march=armv8-a+sve")
    add_compile_definitions(OPT_SVE)
  endif()

  # Optionally, check if the compiler supports SVE2
  if(CXX_SVE2)
    message(STATUS "Compiler supports -march=armv8.2-a+sve2")
    add_compile_definitions(OPT_SVE2)
  endif()

  if(CXX_ARMV9)
    message(STATUS "Compiler supports -march=armv9.2-a+sve2")
    add_compile_definitions(OPT_ARMV9)
  endif()

endif()