cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0074 NEW)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
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
        # Valgrind does not support AVX512 and Valgrind in running in Debug
        # so disable it if we are in Debug mode
        string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
        if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
            message(STATUS "SVS: Disabling AVX512 support in Debug mode due to Valgrind")
            set(SVS_NO_AVX512 ON)
        endif()
    else()
        set(SVS_LVQ_SUPPORTED 0)
        message(STATUS "SVS LVQ is not supported on this architecture")
    endif()

    # detect if build environment is using glibc
    include(CheckSymbolExists)
    check_symbol_exists(__GLIBC__ "features.h" GLIBC_FOUND)
    if(GLIBC_FOUND)
        include(CheckCXXSourceRuns)
        check_cxx_source_runs("#include <features.h>
            int main(){ return __GLIBC__ == 2 && __GLIBC_MINOR__ >= 28 ?0:1; }"
            GLIBC_2_28_FOUND)
        check_cxx_source_runs("#include <features.h>
            int main(){ return __GLIBC__ == 2 && __GLIBC_MINOR__ >= 26 ?0:1; }"
            GLIBC_2_26_FOUND)
    endif()

    cmake_dependent_option(SVS_SHARED_LIB "Use SVS pre-compiled shared library" ON "USE_SVS AND GLIBC_FOUND AND SVS_LVQ_SUPPORTED" OFF)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if (GLIBC_2_28_FOUND)
            set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/v1.0.0-dev/svs-shared-library-1.0.0-NIGHTLY-20250910-627-reduced-clang.tar.gz" CACHE STRING "SVS URL")
        else()
            message(STATUS "GLIBC>=2.28 is required for Clang build - disabling SVS_SHARED_LIB")
            set(SVS_SHARED_LIB OFF)
        endif()
    else()
        if (GLIBC_2_28_FOUND)
            set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/v1.0.0-dev/svs-shared-library-1.0.0-NIGHTLY-20250910-627-reduced.tar.gz" CACHE STRING "SVS URL")
        elseif(GLIBC_2_26_FOUND)
            set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/v1.0.0-dev/svs-shared-library-1.0.0-NIGHTLY-20250910-627-reduced-glibc2_26.tar.gz" CACHE STRING "SVS URL")
        else()
            message(STATUS "GLIBC>=2.26 is required for SVS shared library - disabling SVS_SHARED_LIB")
            set(SVS_SHARED_LIB OFF)
        endif()
    endif()

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
        set(SVS_LEANVEC_HEADER "svs/extensions/vamana/leanvec.h")
    else()
        # This file is included from both CMakeLists.txt and python_bindings/CMakeLists.txt
        # Set `root` relative to this file, regardless of where it is included from.
        get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
        add_subdirectory(
            ${root}/deps/ScalableVectorSearch
            deps/ScalableVectorSearch
        )
        set(SVS_LVQ_HEADER "svs/quantization/lvq/impl/lvq_impl.h")
        set(SVS_LEANVEC_HEADER "svs/leanvec/impl/leanvec_impl.h")
    endif()

    if(SVS_LVQ_SUPPORTED AND EXISTS "${svs_SOURCE_DIR}/include/${SVS_LVQ_HEADER}")
        message("SVS LVQ implementation found")
        add_compile_definitions(VectorSimilarity
            PUBLIC "HAVE_SVS_LVQ=1"
            PUBLIC "SVS_LVQ_HEADER=\"${SVS_LVQ_HEADER}\""
            PUBLIC "SVS_LEANVEC_HEADER=\"${SVS_LEANVEC_HEADER}\""
        )
    else()
        message("SVS LVQ implementation not found")
        add_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS_LVQ=0")
    endif()
else()
    message(STATUS "SVS support disabled")
    add_compile_definitions("HAVE_SVS=0")
endif()
