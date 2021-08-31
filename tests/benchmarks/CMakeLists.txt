cmake_minimum_required(VERSION 3.10)
project(VectorSimilarity_UnitTest)


set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  googlebench
  GIT_REPOSITORY https://github.com/google/benchmark.git
)

FetchContent_MakeAvailable(googlebench)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../deps)

# open blas
set(USE_THREAD 0)
add_subdirectory(../../deps/OpenBLAS OpenBLAS)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/OpenBLAS)

add_executable(openblasbench openblas_bench.cpp)
target_link_libraries(openblasbench benchmark::benchmark)
target_link_libraries(openblasbench openblas)

include_directories($ENV{MKLROOT}/include)
add_executable(mklbench mkl_bench.cpp)
target_link_libraries(mklbench benchmark::benchmark)
target_link_libraries(mklbench $ENV{MKLROOT}/lib/intel64/libmkl_intel_ilp64.a $ENV{MKLROOT}/lib/intel64/libmkl_sequential.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a m dl)

add_compile_options(-march=native)
add_executable(native_avx512_bf_bench native_avx512_bf_bench.cpp)
target_link_libraries(native_avx512_bf_bench benchmark::benchmark)

add_executable(native_avx2_bf_bench native_avx2_bf_bench.cpp)
target_link_libraries(native_avx2_bf_bench benchmark::benchmark)

