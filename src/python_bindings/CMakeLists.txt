
get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
get_filename_component(binroot ${CMAKE_CURRENT_BINARY_DIR} ABSOLUTE)

set(VectorSimilarity_lib $ENV{VECSIM_LIB})

project(VecSim LANGUAGES CXX)

add_subdirectory(${root}/deps/pybind11 pybind11)

pybind11_add_module(VecSim bindings.cpp)

include_directories(${root}/src)

target_link_libraries(VecSim ${VectorSimilarity_lib})
