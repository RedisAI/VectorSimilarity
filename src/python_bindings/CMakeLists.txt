project(VecSim LANGUAGES CXX)

get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.1
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
add_subdirectory(${root} VectorSimilarity)

include_directories(${root}/src)

# build bindings with -DBUILD_TESTS flag
ADD_DEFINITIONS(-DBUILD_TESTS)
pybind11_add_module(VecSim bindings.cpp)
add_library(VectorSimilaritySerialization ${root}/src/VecSim/algorithms/hnsw/serialization.cpp)

target_link_libraries(VecSim VectorSimilarity VectorSimilaritySerialization)

add_dependencies(VecSim VectorSimilarity)
