
if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
endif()

message("# VectorSimilarity_ModuleTest root:" ${root})
message("# VectorSimilarity_ModuleTest binroot:" ${binroot})

project(VectorSimilarity_ModuleTest)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC ${CLANG_SAN_FLAGS} ${LLVM_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${LLVM_LD_FLAGS}")

add_definitions(-DREDIS_MODULE_TARGET)

include_directories(${root}/src)

add_library(memory_test SHARED memory_test.c)

target_link_libraries(memory_test PUBLIC VectorSimilarity)

set_target_properties(memory_test PROPERTIES PREFIX "")
set_target_properties(memory_test PROPERTIES SUFFIX ".so")
