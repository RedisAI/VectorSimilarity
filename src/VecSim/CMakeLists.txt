cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(CMAKE_CXX_STANDARD 20)

project(VecsimLib)
file(GLOB_RECURSE headers ./**.h)
set(HEADER_LIST "${headers}")

include_directories(../)

add_subdirectory(spaces)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

add_library(VectorSimilarity ${VECSIM_LIBTYPE}
    algorithms/brute_force/brute_force_factory.cpp
    algorithms/hnsw/hnsw_factory.cpp
    algorithms/brute_force/vector_block.cpp
    algorithms/hnsw/visited_nodes_handler.cpp
    vec_sim.cpp
    vec_sim_interface.cpp
    query_results.cpp
    info_iterator.cpp
    query_result_struct.cpp
    utils/vec_utils.cpp
    memory/vecsim_malloc.cpp
    memory/vecsim_base.cpp
    ${HEADER_LIST}
)

target_link_libraries(VectorSimilarity VectorSimilaritySpaces)

if(VECSIM_BUILD_TESTS)
    add_library(VectorSimilaritySerializer utils/serializer.cpp)
    target_link_libraries(VectorSimilarity VectorSimilaritySerializer)
endif()
