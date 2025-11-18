cmake_minimum_required(VERSION 3.10)

# This CMakeLists.txt is designed to be used as a subdirectory of the main VectorSimilarity project
# It should NOT be built standalone - instead, add it to the main CMakeLists.txt

# If you're trying to build standalone, please build from the main project root instead:
#   cd /path/to/VectorSimilarity
#   mkdir -p build && cd build
#   cmake ..
#   make hnsw_disk_serializer
#   # Then the serializer will be available at: build/hnsw_disk_serializer/hnsw_disk_serializer

project(hnsw_disk_serializer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find RocksDB
find_package(PkgConfig REQUIRED)
pkg_check_modules(ROCKSDB REQUIRED IMPORTED_TARGET rocksdb)

# Create the executable
add_executable(hnsw_disk_serializer hnsw_disk_serializer.cpp)

# Link against VecSim and RocksDB (using CMake targets from parent project)
target_link_libraries(hnsw_disk_serializer
    VectorSimilarity
    PkgConfig::ROCKSDB
)

