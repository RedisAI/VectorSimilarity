# CMakeLists.txt for vecsim-c Rust library
# This integrates the Rust vecsim-c library into the RediSearch build system

cmake_minimum_required(VERSION 3.14)
project(vecsim-c-rust LANGUAGES C CXX)

# Find Cargo
find_program(CARGO_EXECUTABLE cargo REQUIRED)

# Determine build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_BUILD_TYPE "debug")
    set(CARGO_BUILD_FLAGS "")
else()
    set(CARGO_BUILD_TYPE "release")
    set(CARGO_BUILD_FLAGS "--release")
endif()

# Set the target directory
set(CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/target")

# Determine the library name based on platform
if(APPLE)
    set(VECSIM_C_LIB_NAME "libvecsim_c.dylib")
    set(VECSIM_C_STATIC_LIB_NAME "libvecsim_c.a")
elseif(WIN32)
    set(VECSIM_C_LIB_NAME "vecsim_c.dll")
    set(VECSIM_C_STATIC_LIB_NAME "vecsim_c.lib")
else()
    set(VECSIM_C_LIB_NAME "libvecsim_c.so")
    set(VECSIM_C_STATIC_LIB_NAME "libvecsim_c.a")
endif()

# Full path to the built library
set(VECSIM_C_LIB_PATH "${CARGO_TARGET_DIR}/${CARGO_BUILD_TYPE}/${VECSIM_C_LIB_NAME}")
set(VECSIM_C_STATIC_LIB_PATH "${CARGO_TARGET_DIR}/${CARGO_BUILD_TYPE}/${VECSIM_C_STATIC_LIB_NAME}")

# Custom command to build the Rust library
add_custom_command(
    OUTPUT ${VECSIM_C_LIB_PATH} ${VECSIM_C_STATIC_LIB_PATH}
    COMMAND ${CMAKE_COMMAND} -E env 
        CARGO_TARGET_DIR=${CARGO_TARGET_DIR}
        ${CARGO_EXECUTABLE} build 
        ${CARGO_BUILD_FLAGS}
        -p vecsim-c
        --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/../Cargo.toml
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building vecsim-c Rust library (${CARGO_BUILD_TYPE})"
    VERBATIM
)

# Custom target for the Rust build
add_custom_target(vecsim_c_rust_build
    DEPENDS ${VECSIM_C_LIB_PATH} ${VECSIM_C_STATIC_LIB_PATH}
)

# Create imported library target for the shared library
add_library(vecsim_c_shared SHARED IMPORTED GLOBAL)
set_target_properties(vecsim_c_shared PROPERTIES
    IMPORTED_LOCATION ${VECSIM_C_LIB_PATH}
)
add_dependencies(vecsim_c_shared vecsim_c_rust_build)

# Create imported library target for the static library
add_library(vecsim_c_static STATIC IMPORTED GLOBAL)
set_target_properties(vecsim_c_static PROPERTIES
    IMPORTED_LOCATION ${VECSIM_C_STATIC_LIB_PATH}
)
add_dependencies(vecsim_c_static vecsim_c_rust_build)

# Default alias (prefer static linking)
add_library(vecsim_c ALIAS vecsim_c_static)

# Include directory for the C header
set(VECSIM_C_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "vecsim-c include directory")

# Interface library for easy consumption
add_library(vecsim_c_interface INTERFACE)
target_include_directories(vecsim_c_interface INTERFACE ${VECSIM_C_INCLUDE_DIR})

# Export variables for parent projects
set(VECSIM_C_LIBRARIES ${VECSIM_C_STATIC_LIB_PATH} CACHE FILEPATH "vecsim-c static library")
set(VECSIM_C_SHARED_LIBRARIES ${VECSIM_C_LIB_PATH} CACHE FILEPATH "vecsim-c shared library")

# Test target
add_custom_target(vecsim_c_test
    COMMAND ${CMAKE_COMMAND} -E env 
        CARGO_TARGET_DIR=${CARGO_TARGET_DIR}
        ${CARGO_EXECUTABLE} test 
        -p vecsim-c
        --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/../Cargo.toml
        -- --test-threads=1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running vecsim-c tests"
    VERBATIM
)

# Clean target
add_custom_target(vecsim_c_clean
    COMMAND ${CARGO_EXECUTABLE} clean
        --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/../Cargo.toml
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Cleaning vecsim-c Rust build"
    VERBATIM
)

# Print configuration info
message(STATUS "vecsim-c Rust library configuration:")
message(STATUS "  Build type: ${CARGO_BUILD_TYPE}")
message(STATUS "  Target dir: ${CARGO_TARGET_DIR}")
message(STATUS "  Static lib: ${VECSIM_C_STATIC_LIB_PATH}")
message(STATUS "  Shared lib: ${VECSIM_C_LIB_PATH}")
message(STATUS "  Include dir: ${VECSIM_C_INCLUDE_DIR}")

