
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories(${PROJECT_SOURCE_DIR}/../../src)
add_subdirectory(${PROJECT_SOURCE_DIR}/../deps/googletest gtest_main)
add_subdirectory(${PROJECT_SOURCE_DIR}/../../src VectorSimilarity)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(test_hnswlib test_hnswlib.cpp)
target_link_libraries(test_hnswlib gtest_main)
add_test(NAME test_hnswlib COMMAND test_hnswlib)

target_link_libraries(test_hnswlib PUBLIC VectorSimilarity)
