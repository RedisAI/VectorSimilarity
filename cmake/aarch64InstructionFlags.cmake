include(CheckCXXCompilerFlag)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Building for ARM aarch64")

  # Create a test program to check CPU capabilities at configure time
  file(WRITE ${CMAKE_BINARY_DIR}/cpu_check.cpp [[
    #include <iostream>
    #include <sys/auxv.h>
    
    #ifndef AT_HWCAP
    #define AT_HWCAP 16
    #endif
    
    #ifndef AT_HWCAP2
    #define AT_HWCAP2 26
    #endif
    
    // ARM specific capabilities
    #ifndef HWCAP_SVE
    #define HWCAP_SVE (1 << 22)
    #endif
    
    #ifndef HWCAP2_SVE2
    #define HWCAP2_SVE2 (1 << 1)
    #endif
    
    int main() {
      unsigned long hwcaps = getauxval(AT_HWCAP);
      unsigned long hwcaps2 = getauxval(AT_HWCAP2);
      
      bool has_sve = (hwcaps & HWCAP_SVE) != 0;
      bool has_sve2 = (hwcaps2 & HWCAP2_SVE2) != 0;
      
      std::cout << "CPU_HAS_SVE=" << (has_sve ? "1" : "0") << std::endl;
      std::cout << "CPU_HAS_SVE2=" << (has_sve2 ? "1" : "0") << std::endl;
      
      return 0;
    }
  ]])

  # Try to compile and run the program
  try_run(
    RUN_RESULT COMPILE_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/cpu_check.cpp
    RUN_OUTPUT_VARIABLE CPU_INFO
  )

  # Set variables based on CPU detection
  if(RUN_RESULT EQUAL 0)
    if(CPU_INFO MATCHES "CPU_HAS_SVE=1")
      set(CPU_HAS_SVE TRUE)
      message(STATUS "CPU supports SVE instructions")
    else()
      set(CPU_HAS_SVE FALSE)
      message(STATUS "CPU does not support SVE instructions")
    endif()
    
    if(CPU_INFO MATCHES "CPU_HAS_SVE2=1")
      set(CPU_HAS_SVE2 TRUE)
      message(STATUS "CPU supports SVE2 instructions")
    else()
      set(CPU_HAS_SVE2 FALSE)
      message(STATUS "CPU does not support SVE2 instructions")
    endif()
  else()
    message(WARNING "Could not determine CPU capabilities, assuming baseline features only")
    set(CPU_HAS_SVE FALSE)
    set(CPU_HAS_SVE2 FALSE)
  endif()

  # Check what compiler flags are supported
  CHECK_CXX_COMPILER_FLAG("-march=armv8-a" CXX_ARMV8A)
  CHECK_CXX_COMPILER_FLAG("-march=armv8-a+sve" CXX_SVE)
  CHECK_CXX_COMPILER_FLAG("-march=armv8.2-a+sve2" CXX_SVE2)
  CHECK_CXX_COMPILER_FLAG("-march=armv9-a" CXX_ARMV9_ALT)
  CHECK_CXX_COMPILER_FLAG("-march=armv9.0-a" CXX_ARMV9)

  # Define ARM architecture option
  option(FORCE_BASELINE "Force baseline ARMv8-A without extensions" OFF)
  
  # Apply the best architecture that both the compiler and CPU support
  if(FORCE_BASELINE)
    message(STATUS "Forcing baseline ARMv8-A as requested")
    if(CXX_ARMV8A)
      add_compile_definitions(OPT_NEON)
      add_compile_options(-march=armv8-a)
    endif()
  
  # Only use ARMv9 if both compiler and CPU support it
  elseif(CPU_HAS_SVE2 AND CXX_ARMV9)
    message(STATUS "Using ARMv9.0-a with SVE2 (supported by CPU)")
    add_compile_definitions(OPT_ARMV9)
    add_compile_options(-march=armv9.0-a -msve-vector-bits=scalable)
  
  # Try alternative ARMv9 flag
  elseif(CPU_HAS_SVE2 AND CXX_ARMV9_ALT)
    message(STATUS "Using ARMv9-a with SVE2 (supported by CPU)")
    add_compile_definitions(OPT_ARMV9_ALT)
    add_compile_options(-march=armv9-a -msve-vector-bits=scalable)
  
  # Try ARMv8.5 with SVE2
  elseif(CPU_HAS_SVE2 AND CXX_ARMV85A)
    message(STATUS "Using ARMv8.5-a with SVE2 (supported by CPU)")
    add_compile_definitions(OPT_ARMV85A)
    add_compile_options(-march=armv8.5-a+sve2)
  
  # Try ARMv8.2 with SVE2
  elseif(CPU_HAS_SVE2 AND CXX_SVE2)
    message(STATUS "Using ARMv8.2-a with SVE2 (supported by CPU)")
    add_compile_definitions(OPT_SVE2)
    add_compile_options(-march=armv8.2-a+sve2)
  
  # Try basic SVE support
  elseif(CPU_HAS_SVE AND CXX_SVE)
    message(STATUS "Using ARMv8-a with SVE (supported by CPU)")
    add_compile_definitions(OPT_SVE)
    add_compile_options(-march=armv8-a+sve)
  
  # Fallback to baseline
  else()
    message(STATUS "Using baseline ARMv8-a (no SVE/SVE2 CPU support)")
    if(CXX_ARMV8A)
      add_compile_definitions(OPT_NEON)
      add_compile_options(-march=armv8-a)
    endif()
  endif()
endif()