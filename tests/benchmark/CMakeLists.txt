
if(NOT DEFINED root)
    get_filename_component(root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
endif()
message("# VectorSimilarity_Benchmark root: " ${root})
message("# VectorSimilarity_Benchmark binroot: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 14)

if (NOT DEFINED VECSIM_ARCH AND ARCH STREQUAL "x64")
    set(VECSIM_ARCH "x86-64-v4")
endif()
message("# VecSim/Benchmark VECSIM_ARCH: " ${VECSIM_ARCH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -march=${VECSIM_ARCH}")

include_directories(${root}/src)

enable_testing()

include(${root}/cmake/gtest.cmake)

add_executable(bf_benchmark bf_google_benchmark.cpp)

target_link_libraries(bf_benchmark VectorSimilarity benchmark::benchmark)
