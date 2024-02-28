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
    index_factories/brute_force_factory.cpp
    index_factories/hnsw_factory.cpp
    index_factories/tiered_factory.cpp
    index_factories/index_factory.cpp
    $<$<BOOL:${USE_CUDA}>:index_factories/raft_ivf_factory.cu>
    $<$<BOOL:${USE_CUDA}>:index_factories/raft_ivf_tiered_factory.cpp>
    algorithms/hnsw/visited_nodes_handler.cpp
    vec_sim.cpp
    vec_sim_debug.cpp
    vec_sim_interface.cpp
    query_results.cpp
    info_iterator.cpp
    utils/vec_utils.cpp
    utils/data_block.cpp
    memory/vecsim_malloc.cpp
    memory/vecsim_base.cpp
    ${HEADER_LIST}
)


if(VECSIM_BUILD_TESTS)
    add_library(VectorSimilaritySerializer utils/serializer.cpp)
endif()

target_link_libraries(VectorSimilarity
PUBLIC
	VectorSimilaritySpaces
	$<$<BOOL:${VECSIM_BUILD_TESTS}>:VectorSimilaritySerializer>
PRIVATE
	$<$<BOOL:${USE_CUDA}>:raft::raft>
    $<$<BOOL:${USE_CUDA}>:CUDA::cusolver>
    $<$<BOOL:${USE_CUDA}>:CUDA::cublas>
    $<$<BOOL:${USE_CUDA}>:CUDA::curand>
    $<$<BOOL:${USE_CUDA}>:CUDA::cusparse>
)
