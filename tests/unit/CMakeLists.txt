cmake_minimum_required(VERSION 3.14)
include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories(${PROJECT_SOURCE_DIR}/../../src)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(test_hnswlib test_hnswlib.cpp)
target_link_libraries(test_hnswlib gtest_main)
add_test(NAME test_hnswlib COMMAND test_hnswlib)

add_subdirectory(${PROJECT_SOURCE_DIR}/../../src VectorSimilarity)
target_link_libraries(test_hnswlib PUBLIC VectorSimilarity)
