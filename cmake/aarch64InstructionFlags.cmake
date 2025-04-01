include(CheckCXXCompilerFlag)


message(STATUS "Building for ARM aarch64")

# Check what compiler flags are supported
CHECK_CXX_COMPILER_FLAG("-march=armv7-a+neon" CXX_ARMV7_NEON)
CHECK_CXX_COMPILER_FLAG("-march=armv8-a" CXX_ARMV8A)
CHECK_CXX_COMPILER_FLAG("-march=armv8.2-a+dotprod" CXX_NEON_DOTPROD)
CHECK_CXX_COMPILER_FLAG("-march=armv8-a+sve" CXX_SVE)
CHECK_CXX_COMPILER_FLAG("-march=armv9-a+sve2" CXX_SVE2)

# Only use ARMv9 if both compiler and CPU support it
if(CXX_SVE2)
  message(STATUS "Using ARMv9.0-a with SVE2 (supported by CPU)")
  add_compile_definitions(OPT_SVE2)
endif()
if (CXX_NEON_DOTPROD)
  add_compile_definitions(OPT_NEON_DOTPROD)
endif()
if (CXX_ARMV8A OR CXX_ARMV7_NEON)
  add_compile_definitions(OPT_NEON)
endif()
if (CXX_SVE)
  add_compile_definitions(OPT_SVE)
endif()
