
message("# VectorSimilarity_Benchmark root: " ${root})
message("# VectorSimilarity_Benchmark binroot: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

include_directories(${root}/src)

enable_testing()

include(${root}/cmake/gtest.cmake)

add_executable(bf_benchmark bf_google_benchmark.cpp)

target_link_libraries(bf_benchmark VectorSimilarity benchmark::benchmark)
