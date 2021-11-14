project(VectorSimilarity_UnitTest)

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(CMAKE_CXX_STANDARD 14)

if (NOT DEFINED VECSIM_ARCH)
set(VECSIM_ARCH "x86-64-v4")
endif()
message("VecSim/UnitTest VECSIM_ARCH: " ${VECSIM_ARCH})

if(NOT DEFINED root)
	get_filename_component(root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
	set(binroot ${root}/build)
endif()
message("VectorSimilarity_UnitTest: " ${binroot})

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

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${root}/src)

add_subdirectory(${root}/src ${binroot}/VectorSimilarity)

enable_testing()

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(test_hnswlib test_hnswlib.cpp test_utils.cpp)
add_executable(test_bruteforce test_bruteforce.cpp test_utils.cpp)

# add_test(NAME test_hnswlib COMMAND test_hnswlib)
# add_test(NAME test_bruteforce COMMAND test_bruteforce)

target_link_libraries(test_hnswlib PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_bruteforce PUBLIC gtest_main VectorSimilarity)

include(GoogleTest)
gtest_discover_tests(test_hnswlib)
gtest_discover_tests(test_bruteforce)
