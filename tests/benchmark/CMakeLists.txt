
message("# VectorSimilarity_Benchmark root: " ${root})
message("# VectorSimilarity_Benchmark binroot: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -fno-omit-frame-pointer")

enable_testing()

add_executable(bm_batch_iterator bm_batch_iterator.cpp)
target_link_libraries(bm_batch_iterator VectorSimilarity benchmark::benchmark)
add_executable(bm_hnsw searchKnnCloserFirst_test.cpp)
target_link_libraries(bm_hnsw VectorSimilarity)
add_executable(bm_basics bm_basics.cpp)
target_link_libraries(bm_basics VectorSimilarity benchmark::benchmark)
