cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0074 NEW)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Valgrind does not support AVX512 and Valgrind in running in Debug
# so disable it if we are in Debug mode
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message(STATUS "SVS: Disabling AVX512 support in Debug mode due to Valgrind")
    set(SVS_NO_AVX512 ON)
endif()

set(SVS_SUPPORTED 1)

# GCC < v11 does not support C++20 features required for SVS
# https://gcc.gnu.org/projects/cxx-status.html
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
        set(SVS_SUPPORTED 0)
        message(STATUS "Skipping SVS: requires GCC >= 11.")
    endif()
endif()

include(CMakeDependentOption)

# USE_SVS option forcibly OFF if CPU or compiler is not supported
# elsewhere let user disable SVS
cmake_dependent_option(USE_SVS "Build with SVS library support" ON "SVS_SUPPORTED" OFF)

if(USE_SVS)
    message(STATUS "SVS support enabled")
    # Configure SVS build
    add_compile_definitions("HAVE_SVS=1")

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86_64)|(AMD64|amd64)")
        set(SVS_LVQ_SUPPORTED 1)
    else()
        set(SVS_LVQ_SUPPORTED 0)
        message(STATUS "SVS LVQ is not supported on this architecture")
    endif()

    # detect if build environment is using glibc
    include(CheckSymbolExists)
    unset(GLIBC_FOUND CACHE)
    check_symbol_exists(__GLIBC__ "features.h" GLIBC_FOUND)
    if(NOT GLIBC_FOUND)
        message(STATUS "GLIBC is not detected - SVS shared library is not supported")
    endif()

    cmake_dependent_option(SVS_SHARED_LIB "Use SVS pre-compiled shared library" ON "USE_SVS AND GLIBC_FOUND AND SVS_LVQ_SUPPORTED" OFF)
    set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/v0.0.8-dev/svs-shared-library-0.0.8-NIGHTLY-254.tar.gz" CACHE STRING "SVS URL")

    if(SVS_SHARED_LIB)
        include(FetchContent)
        FetchContent_Declare(
            svs
            URL "${SVS_URL}"
        )
        FetchContent_MakeAvailable(svs)
        list(APPEND CMAKE_PREFIX_PATH "${svs_SOURCE_DIR}")
        find_package(svs REQUIRED)
        set(SVS_LVQ_HEADER "svs/extensions/vamana/lvq.h")
    else()
        # This file is included from both CMakeLists.txt and python_bindings/CMakeLists.txt
        # Set `root` relative to this file, regardless of where it is included from.
        get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
        add_subdirectory(
            ${root}/deps/ScalableVectorSearch
            deps/ScalableVectorSearch
        )
        set(SVS_LVQ_HEADER "svs/quantization/lvq/impl/lvq_impl.h")
    endif()

    if(SVS_LVQ_SUPPORTED AND EXISTS "${svs_SOURCE_DIR}/include/${SVS_LVQ_HEADER}")
        message("SVS LVQ implementation found")
        add_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS_LVQ=1" PUBLIC "SVS_LVQ_HEADER=\"${SVS_LVQ_HEADER}\"")
    else()
        message("SVS LVQ implementation not found")
        add_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS_LVQ=0")
    endif()
else()
    message(STATUS "SVS support disabled")
    add_compile_definitions("HAVE_SVS=0")
endif()
