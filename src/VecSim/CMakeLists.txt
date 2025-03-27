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

set(LVQ_IMPL_HEADER "${CMAKE_SOURCE_DIR}/deps/ScalableVectorSearch/include/svs/quantization/lvq/impl/lvq_impl.h")

add_subdirectory(spaces)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

add_library(VectorSimilarity ${VECSIM_LIBTYPE}
    index_factories/brute_force_factory.cpp
    index_factories/hnsw_factory.cpp
    index_factories/tiered_factory.cpp
    index_factories/svs_factory.cpp
    index_factories/index_factory.cpp
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

target_link_libraries(VectorSimilarity VectorSimilaritySpaces svs::svs)

if(EXISTS ${LVQ_IMPL_HEADER})
    message("SVS LVQ implementation found")
    target_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS_LVQ=1")
else()
    message("SVS LVQ implementation not found")
    target_compile_definitions(VectorSimilarity PUBLIC "HAVE_SVS_LVQ=0")
endif()

if(VECSIM_BUILD_TESTS)
    add_library(VectorSimilaritySerializer utils/serializer.cpp)
    target_link_libraries(VectorSimilarity VectorSimilaritySerializer)
endif()
