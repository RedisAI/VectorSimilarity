cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(CMAKE_CXX_STANDARD 20)

project(VecsimLib)

file(GLOB_RECURSE headers ./**.h)
set(HEADER_LIST "${headers}")

include_directories(../)

set(SVS_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++20")

add_subdirectory(spaces)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

if(USE_SVS)
    set(svs_factory_file "index_factories/svs_factory.cpp")
endif()

add_library(VectorSimilarity ${VECSIM_LIBTYPE}
    index_factories/brute_force_factory.cpp
    index_factories/hnsw_factory.cpp
    index_factories/hnsw_disk_factory.cpp
    # index_factories/tiered_factory.cpp  # Temporarily disabled due to SVS header issues
    # index_factories/svs_factory.cpp  # Temporarily disabled due to SVS header issues
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

target_link_libraries(VectorSimilarity VectorSimilaritySpaces)

if (TARGET svs::svs)
    target_link_libraries(VectorSimilarity svs::svs)
    if(TARGET svs::svs_static_library)
        target_link_libraries(VectorSimilarity svs::svs_static_library)
    endif()
endif()

if(VECSIM_BUILD_TESTS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ROCKSDB REQUIRED IMPORTED_TARGET rocksdb)
    target_link_libraries(VectorSimilarity PkgConfig::ROCKSDB)

    add_library(VectorSimilaritySerializer utils/serializer.cpp)
    target_link_libraries(VectorSimilarity VectorSimilaritySerializer)
endif()
