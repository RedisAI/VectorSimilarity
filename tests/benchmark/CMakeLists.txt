
message("# VectorSimilarity_Benchmark root: " ${root})
message("# VectorSimilarity_Benchmark binroot: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

include_directories(${root}/src)

enable_testing()

add_executable(bm_batch_iterator bm_batch_iterator.cpp)

target_link_libraries(bm_batch_iterator VectorSimilarity benchmark::benchmark)
