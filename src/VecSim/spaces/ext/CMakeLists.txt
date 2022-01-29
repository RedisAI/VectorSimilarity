
get_filename_component(root ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
message("# VectorSimilarity/spaces/ext root: " ${root})
get_filename_component(binroot ${CMAKE_CURRENT_BINARY_DIR} ABSOLUTE)
message("# VectorSimilarity/spaces/ext binroot: " ${binroot})

project(VectorSimilaritySpaces)

include (ExternalProject)

ExternalProject_Add (VectorSimilaritySpaces_project
	SOURCE_DIR      ${root}/src/VecSim/spaces
	PREFIX          ${binroot}/VectorSimilaritySpaces
	CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${binroot}/VectorSimilaritySpaces -Dbinroot:string=${binroot}
	CMAKE_ARGS      -DCMAKE_BUILD_TYPE:string=${CMAKE_BUILD_TYPE}
	CMAKE_ARGS      -DCMAKE_C_COMPILER:string=/opt/miniforge/bin/gcc
	CMAKE_ARGS      -DCMAKE_CXX_COMPILER:string=/opt/miniforge/bin/g++
	CMAKE_ARGS      -DHAVE_MARCH_X86_64_V4:string=on
	CMAKE_ARGS      -DHAVE_MARCH_X86_64_V3:string=on
	CMAKE_ARGS      -DHAVE_MARCH_X86_64_V2:string=on
)

ExternalProject_Get_Property(VectorSimilaritySpaces_project install_dir)
set(VectorSimilaritySpaces_bindir ${install_dir})

set(libs "")

function(add_vecsim_lib suffix)
	if (${sufffix})
		set(suffix "_${suffix}")
		set(lib "VectorSimilaritySpaces${suffix}")
		set(libs "${libs} ${lib}")
	else()
		set(lib "VectorSimilaritySpaces")
	endif()
	add_library(${lib} STATIC IMPORTED)
	set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${VectorSimilaritySpaces_bindir}/lib${lib}.a")
endfunction()

add_vecsim_lib("")
add_vecsim_lib("sse")
add_vecsim_lib("avx")
add_vecsim_lib("avx512")

target_link_libraries(VectorSimilaritySpaces INTERFACE ${libs})
