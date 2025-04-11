cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0074 NEW)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# option(SVS_EXPERIMENTAL_LEANVEC "Enable experimental LeanVec in SVS" YES)
# if(SVS_EXPERIMENTAL_LEANVEC)
#     add_definitions(-DSVS_EXPERIMENTAL_LEANVEC)
# endif()

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86_64)|(AMD64|amd64)")
    set(SVS_SUPPORTED 1)
else()
    set(SVS_SUPPORTED 0)
    message(STATUS "SVS is not supported on this architecture")
endif()

# GCC < v11 does not support C++20 features required for SVS
# https://gcc.gnu.org/projects/cxx-status.html
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
        set(SVS_SUPPORTED 0)
        message(STATUS "Insufficient gcc version for SVS")
    endif()
endif()

include(CMakeDependentOption)

# USE_SVS option forsibly OFF if CPU or compiler is not supported
# elsewhere let user disable SVS
cmake_dependent_option(USE_SVS "Build with SVS library support" ON "SVS_SUPPORTED" OFF)
if(USE_SVS)
    message(STATUS "SVS support enabled")
else()
    message(STATUS "SVS support disabled")
endif()

# try to find MKL
set(MKL_HINTS ${MKLROOT} $ENV{MKL_DIR} $ENV{MKL_ROOT} $ENV{MKLROOT})
if(LINUX)
    list(APPEND MKL_HINTS "/opt/intel/oneapi/mkl/latest")
endif()
if(WIN32)
    list(APPEND MKL_HINTS "C:/Program Files (x86)/Intel/oneAPI/mkl/latest")
endif()
find_package(MKL HINTS ${MKL_HINTS})
if(NOT MKL_FOUND)
    message(WARNING "MKL not found - disabling SVS shared library")
endif()

# Configure SVS build
cmake_dependent_option(SVS_SHARED_LIB "Use SVS pre-compiled shared library" ON "SVS_SUPPORTED AND MKL_FOUND" OFF)
set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/v0.0.7/svs-shared-library-0.0.7-avx2.tar.gz" CACHE STRING "SVS URL")

if(USE_SVS)
    add_compile_definitions("HAVE_SVS=1")
    set(svs_factory_file "index_factories/svs_factory.cpp")
    set(SVS_TARGET_NAME "svs::svs")
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
        set(SVS_TARGET_NAME "svs::svs_shared_library")
    else()
        # this file is included from python_bindings/CMakeLists.txt
        # SVS sources path to be relative to this file
        get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
        add_subdirectory(
            ${root}/deps/ScalableVectorSearch
            deps/ScalableVectorSearch
        )
        set(SVS_LVQ_HEADER "svs/quantization/lvq/impl/lvq_impl.h")
        set(SVS_TARGET_NAME "svs::svs")
    endif()
else()
    add_compile_definitions("HAVE_SVS=0")
endif()

if(EXISTS "${svs_SOURCE_DIR}/include/${SVS_LVQ_HEADER}")
    message("SVS LVQ implementation found")
    add_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS_LVQ=1" PUBLIC "SVS_LVQ_HEADER=\"${SVS_LVQ_HEADER}\"")
else()
    message("SVS LVQ implementation not found")
    add_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS_LVQ=0")
endif()
