add_executable(native_filtered_bench native_main.cpp native_candidates.cpp)
target_link_libraries(native_filtered_bench PRIVATE VectorSimilarity)

execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${root}"
                OUTPUT_VARIABLE FILTERED_SOURCE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
execute_process(COMMAND git status --porcelain --untracked-files=normal WORKING_DIRECTORY "${root}"
                OUTPUT_VARIABLE FILTERED_SOURCE_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
if(FILTERED_SOURCE_STATUS)
    set(FILTERED_SOURCE_DIRTY true)
else()
    set(FILTERED_SOURCE_DIRTY false)
endif()
target_compile_definitions(native_filtered_bench PRIVATE
    FILTERED_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    FILTERED_SOURCE_REVISION="${FILTERED_SOURCE_REVISION}"
    FILTERED_SOURCE_DIRTY="${FILTERED_SOURCE_DIRTY}")
