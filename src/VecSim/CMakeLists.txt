cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(CMAKE_CXX_STANDARD 20)

project(VecsimLib)

file(GLOB_RECURSE headers ./**.h)
set(HEADER_LIST "${headers}")

include_directories(../)

set(SVS_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++20 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

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

# USE_SVS option forsibly OFF if CPU is not supported
# elsewhere let user disable SVS
cmake_dependent_option(USE_SVS "Build with SVS library support" ON "SVS_SUPPORTED" OFF)
if(USE_SVS)
    message(STATUS "SVS support enabled")
else()
    message(STATUS "SVS support disabled")
endif()

# Configure SVS build
set(SVS_SHARED_LIB OFF CACHE BOOL "Use SVS pre-compiled shared library")
set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/v0.0.7/svs-shared-library-0.0.7-avx2.tar.gz" CACHE STRING "SVS URL")

set(MKL_HINTS
    ${MKLROOT}
    $ENV{MKL_DIR}
    $ENV{MKL_ROOT}
    $ENV{MKLROOT})

if(LINUX)
    list(APPEND MKL_HINTS "/opt/intel/oneapi/mkl/latest")
endif()
if(WIN32)
    list(APPEND MKL_HINTS "C:/Program Files (x86)/Intel/oneAPI/mkl/latest")
endif()

if(USE_SVS)
    set(svs_factory_file "index_factories/svs_factory.cpp")
    if(SVS_SHARED_LIB)
        include(FetchContent)
        FetchContent_Declare(
            svs
            URL "${SVS_URL}"
        )
        FetchContent_MakeAvailable(svs)
        list(APPEND CMAKE_PREFIX_PATH "${svs_SOURCE_DIR}")
        find_package(MKL REQUIRED HINTS ${MKL_HINTS})
        find_package(svs REQUIRED)
        set(SVS_LVQ_HEADER "svs/extensions/vamana/lvq.h")
    else()
        get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
        add_subdirectory(
            ${root}/deps/ScalableVectorSearch
            deps/ScalableVectorSearch
        )
        set(SVS_LVQ_HEADER "svs/quantization/lvq/impl/lvq_impl.h")
    endif()
endif()

add_subdirectory(spaces)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

add_library(VectorSimilarity ${VECSIM_LIBTYPE}
    index_factories/brute_force_factory.cpp
    index_factories/hnsw_factory.cpp
    index_factories/tiered_factory.cpp
    ${svs_factory_file}
    index_factories/index_factory.cpp
    index_factories/components/components_factory.cpp
    algorithms/hnsw/visited_nodes_handler.cpp
    vec_sim.cpp
    vec_sim_debug.cpp
    vec_sim_interface.cpp
    query_results.cpp
    info_iterator.cpp
    utils/vec_utils.cpp
    containers/data_block.cpp
    containers/data_blocks_container.cpp
    memory/vecsim_malloc.cpp
    memory/vecsim_base.cpp
    ${HEADER_LIST}
)

if (USE_SVS)
    target_link_libraries(VectorSimilarity VectorSimilaritySpaces svs::svs)
    if(TARGET svs::svs_shared_library)
        target_link_libraries(VectorSimilarity svs::svs_shared_library MKL::MKL)
    endif()
    target_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS=1")
else()
    target_link_libraries(VectorSimilarity VectorSimilaritySpaces)
    target_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS=0")
endif()

if(EXISTS "${svs_SOURCE_DIR}/include/${SVS_LVQ_HEADER}")
    message("SVS LVQ implementation found")
    target_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS_LVQ=1" PUBLIC "SVS_LVQ_HEADER=\"${SVS_LVQ_HEADER}\"")
else()
    message("SVS LVQ implementation not found")
    target_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS_LVQ=0")
endif()

if(VECSIM_BUILD_TESTS)
    add_library(VectorSimilaritySerializer utils/serializer.cpp)
    target_link_libraries(VectorSimilarity VectorSimilaritySerializer)
endif()
