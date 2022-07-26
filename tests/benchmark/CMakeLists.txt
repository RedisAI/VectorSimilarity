
message("# VectorSimilarity_Benchmark root: " ${root})
message("# VectorSimilarity_Benchmark binroot: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
    # build SSE/AVX* code only on x64 processors. 
    # Check that the compiler supports instructions flag. 
    # This will add the relevant flag both the the space selector and the optimization.
    CHECK_CXX_COMPILER_FLAG(-mavx512f CXX_AVX512F)
    CHECK_CXX_COMPILER_FLAG(-mavx CXX_AVX)
    CHECK_CXX_COMPILER_FLAG(-msse CXX_SSE)
    if(CXX_AVX512F)
        add_compile_options(-mavx512f)
        message("Building with AVX512")
    endif()
    if(CXX_AVX)
        add_compile_options(-mavx)
        message("Building with AVX")
    endif()
    if(CXX_SSE)
        add_compile_options(-msse)
        message("Building with SSE")
    endif()
endif()

enable_testing()

add_executable(bm_batch_iterator bm_batch_iterator.cpp)
target_link_libraries(bm_batch_iterator VectorSimilarity benchmark::benchmark)
add_executable(bm_basics bm_basics.cpp)
target_link_libraries(bm_basics VectorSimilarity benchmark::benchmark)
add_executable(bm_spaces bm_spaces.cpp)
target_link_libraries(bm_spaces VectorSimilarity benchmark::benchmark)
