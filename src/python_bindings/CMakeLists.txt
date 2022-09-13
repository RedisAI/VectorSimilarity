project(VecSim LANGUAGES CXX)

get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)

add_subdirectory(${root}/deps/pybind11 pybind11)
add_subdirectory(${root}/src VectorSimilarity)

include_directories(${root}/src)

# build bindings with -DBUILD_TESTS flag
ADD_DEFINITIONS(-DBUILD_TESTS)
pybind11_add_module(VecSim bindings.cpp)

target_link_libraries(VecSim VectorSimilarity)

add_dependencies(VecSim VectorSimilarity)
