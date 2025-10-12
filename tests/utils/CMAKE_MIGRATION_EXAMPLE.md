# CMakeLists.txt Migration Example

This document shows the exact changes needed to enable global timeout protection for all tests.

## Overview

To enable automatic timeout protection for all tests, you need to:
1. Add `test_main_with_timeout.cpp` to each test executable
2. Change from `gtest_main` to `gtest` in `target_link_libraries`

That's it! No changes to test code needed.

---

## Complete Diff for tests/unit/CMakeLists.txt

### Before (Current)

```cmake
add_executable(test_hnsw ../utils/mock_thread_pool.cpp test_hnsw.cpp test_hnsw_multi.cpp test_hnsw_tiered.cpp unit_test_utils.cpp)
add_executable(test_hnsw_parallel test_hnsw_parallel.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_bruteforce test_bruteforce.cpp test_bruteforce_multi.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_allocator test_allocator.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_spaces test_spaces.cpp)
add_executable(test_types test_types.cpp)
add_executable(test_common ../utils/mock_thread_pool.cpp test_common.cpp unit_test_utils.cpp)
add_executable(test_components test_components.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_bf16 ../utils/mock_thread_pool.cpp test_bf16.cpp unit_test_utils.cpp)
add_executable(test_fp16 ../utils/mock_thread_pool.cpp test_fp16.cpp unit_test_utils.cpp)
add_executable(test_int8 ../utils/mock_thread_pool.cpp test_int8.cpp unit_test_utils.cpp)
add_executable(test_uint8 ../utils/mock_thread_pool.cpp test_uint8.cpp unit_test_utils.cpp)
add_executable(test_index_test_utils ../utils/mock_thread_pool.cpp test_index_test_utils.cpp unit_test_utils.cpp)
add_executable(test_svs ../utils/mock_thread_pool.cpp test_svs.cpp test_svs_tiered.cpp test_svs_multi.cpp unit_test_utils.cpp)

target_link_libraries(test_hnsw PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_hnsw_parallel PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_bruteforce PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_allocator PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_spaces PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_common PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_components PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_types PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_bf16 PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_fp16 PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_int8 PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_uint8 PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_index_test_utils PUBLIC gtest_main VectorSimilarity)
target_link_libraries(test_svs PUBLIC gtest_main VectorSimilarity)
```

### After (With Global Timeout)

```cmake
# Add test_main_with_timeout.cpp to each executable
add_executable(test_hnsw ../utils/test_main_with_timeout.cpp ../utils/mock_thread_pool.cpp test_hnsw.cpp test_hnsw_multi.cpp test_hnsw_tiered.cpp unit_test_utils.cpp)
add_executable(test_hnsw_parallel ../utils/test_main_with_timeout.cpp test_hnsw_parallel.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_bruteforce ../utils/test_main_with_timeout.cpp test_bruteforce.cpp test_bruteforce_multi.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_allocator ../utils/test_main_with_timeout.cpp test_allocator.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_spaces ../utils/test_main_with_timeout.cpp test_spaces.cpp)
add_executable(test_types ../utils/test_main_with_timeout.cpp test_types.cpp)
add_executable(test_common ../utils/test_main_with_timeout.cpp ../utils/mock_thread_pool.cpp test_common.cpp unit_test_utils.cpp)
add_executable(test_components ../utils/test_main_with_timeout.cpp test_components.cpp ../utils/mock_thread_pool.cpp unit_test_utils.cpp)
add_executable(test_bf16 ../utils/test_main_with_timeout.cpp ../utils/mock_thread_pool.cpp test_bf16.cpp unit_test_utils.cpp)
add_executable(test_fp16 ../utils/test_main_with_timeout.cpp ../utils/mock_thread_pool.cpp test_fp16.cpp unit_test_utils.cpp)
add_executable(test_int8 ../utils/test_main_with_timeout.cpp ../utils/mock_thread_pool.cpp test_int8.cpp unit_test_utils.cpp)
add_executable(test_uint8 ../utils/test_main_with_timeout.cpp ../utils/mock_thread_pool.cpp test_uint8.cpp unit_test_utils.cpp)
add_executable(test_index_test_utils ../utils/test_main_with_timeout.cpp ../utils/mock_thread_pool.cpp test_index_test_utils.cpp unit_test_utils.cpp)
add_executable(test_svs ../utils/test_main_with_timeout.cpp ../utils/mock_thread_pool.cpp test_svs.cpp test_svs_tiered.cpp test_svs_multi.cpp unit_test_utils.cpp)

# Change gtest_main to gtest
target_link_libraries(test_hnsw PUBLIC gtest VectorSimilarity)
target_link_libraries(test_hnsw_parallel PUBLIC gtest VectorSimilarity)
target_link_libraries(test_bruteforce PUBLIC gtest VectorSimilarity)
target_link_libraries(test_allocator PUBLIC gtest VectorSimilarity)
target_link_libraries(test_spaces PUBLIC gtest VectorSimilarity)
target_link_libraries(test_common PUBLIC gtest VectorSimilarity)
target_link_libraries(test_components PUBLIC gtest VectorSimilarity)
target_link_libraries(test_types PUBLIC gtest VectorSimilarity)
target_link_libraries(test_bf16 PUBLIC gtest VectorSimilarity)
target_link_libraries(test_fp16 PUBLIC gtest VectorSimilarity)
target_link_libraries(test_int8 PUBLIC gtest VectorSimilarity)
target_link_libraries(test_uint8 PUBLIC gtest VectorSimilarity)
target_link_libraries(test_index_test_utils PUBLIC gtest VectorSimilarity)
target_link_libraries(test_svs PUBLIC gtest VectorSimilarity)
```

---

## Line-by-Line Changes

### For Each Test Executable

**Pattern:**
```cmake
# BEFORE:
add_executable(test_name source1.cpp source2.cpp ...)

# AFTER:
add_executable(test_name ../utils/test_main_with_timeout.cpp source1.cpp source2.cpp ...)
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ADD THIS
```

### For Each target_link_libraries

**Pattern:**
```cmake
# BEFORE:
target_link_libraries(test_name PUBLIC gtest_main VectorSimilarity)

# AFTER:
target_link_libraries(test_name PUBLIC gtest VectorSimilarity)
#                                       ^^^^^ CHANGE FROM gtest_main
```

---

## Detailed Example: test_common

### Before
```cmake
add_executable(test_common 
    ../utils/mock_thread_pool.cpp 
    test_common.cpp 
    unit_test_utils.cpp)

target_link_libraries(test_common PUBLIC gtest_main VectorSimilarity)
```

### After
```cmake
add_executable(test_common 
    ../utils/test_main_with_timeout.cpp  # ← ADDED THIS LINE
    ../utils/mock_thread_pool.cpp 
    test_common.cpp 
    unit_test_utils.cpp)

target_link_libraries(test_common PUBLIC gtest VectorSimilarity)  # ← CHANGED gtest_main to gtest
```

---

## What This Does

### Before
- Uses Google Test's default `main()` from `gtest_main` library
- No automatic timeout protection
- Each test runs until completion or hangs forever

### After
- Uses custom `main()` from `test_main_with_timeout.cpp`
- Automatically registers `TimeoutTestListener`
- Every test gets timeout protection based on configurable rules
- Tests timeout after 30 seconds by default (customizable per test type)

---

## Testing the Migration

### Step 1: Migrate One Test Executable

Start with `test_common`:

```bash
# Edit tests/unit/CMakeLists.txt
# Make changes only for test_common

# Rebuild
make clean
make build

# Run just test_common
cd bin/debug/unit_tests
./test_common
```

### Step 2: Verify Output

You should see normal test output. If a test times out, you'll see:

```
TIMEOUT: Test UtilsTests.testMockThreadPool exceeded timeout!
```

### Step 3: Run Full Test Suite

```bash
make unit_test
```

All tests should pass as before.

### Step 4: Migrate Remaining Executables

Apply the same changes to all other test executables.

---

## Rollback Plan

If something goes wrong, simply revert the changes:

```cmake
# Remove test_main_with_timeout.cpp from add_executable
# Change gtest back to gtest_main in target_link_libraries
```

---

## Customizing Timeouts

After migration, you can customize timeouts by editing `tests/utils/timeout_test_environment.h`:

```cpp
std::chrono::seconds GetTimeoutForTest(const testing::TestInfo &test_info) {
    auto timeout = std::chrono::seconds(default_timeout_seconds_);
    
    std::string test_name = test_info.name();
    std::string suite_name = test_info.test_suite_name();
    
    // Example: Specific test needs longer timeout
    if (suite_name == "UtilsTests" && test_name == "testMockThreadPool") {
        timeout = std::chrono::seconds(100);
    }
    
    // Example: All HNSW tests get 60 seconds
    if (suite_name.find("HNSW") != std::string::npos) {
        timeout = std::chrono::seconds(60);
    }
    
    // Example: Tiered tests get 2 minutes
    if (test_name.find("tiered") != std::string::npos) {
        timeout = std::chrono::seconds(120);
    }
    
    return timeout;
}
```

---

## Verification Checklist

After migration, verify:

- [ ] All tests compile successfully
- [ ] All tests pass (same as before migration)
- [ ] Test output looks normal
- [ ] CI/CD pipeline passes
- [ ] Valgrind tests still work
- [ ] Sanitizer tests still work
- [ ] No new warnings or errors

---

## Common Issues

### Issue 1: Linker Error - Multiple main() Definitions

**Error:**
```
multiple definition of `main'
```

**Cause:** Both `gtest_main` and `test_main_with_timeout.cpp` define `main()`

**Solution:** Make sure you changed `gtest_main` to `gtest` in `target_link_libraries`

### Issue 2: Tests Timing Out Unexpectedly

**Error:**
```
TIMEOUT: Test MyTest.Example exceeded timeout!
```

**Cause:** Test legitimately takes longer than default timeout

**Solution:** Increase timeout for that test in `GetTimeoutForTest()`

### Issue 3: Compilation Error - timeout_guard.h Not Found

**Error:**
```
fatal error: timeout_guard.h: No such file or directory
```

**Cause:** Header files not in include path

**Solution:** Verify `include_directories(../utils)` is in CMakeLists.txt (should already be there)

---

## Performance Impact

### Build Time
- **Negligible** - one additional .cpp file per test executable
- Adds ~0.1-0.5 seconds to build time

### Test Execution Time
- **Negligible** - ~1-10ms overhead per test for thread creation
- For a test suite with 100 tests: ~1 second total overhead

### Memory
- **Minimal** - ~8KB per test for guard thread stack

---

## Alternative: Gradual Migration

If you want to migrate gradually:

### Option A: Migrate One Executable at a Time

Week 1: Migrate `test_common`
Week 2: Migrate `test_hnsw`
Week 3: Migrate remaining executables

### Option B: Create New Executables with Timeout

Keep existing executables, create new ones with timeout:

```cmake
# Old (no timeout)
add_executable(test_common ...)
target_link_libraries(test_common PUBLIC gtest_main VectorSimilarity)

# New (with timeout)
add_executable(test_common_timeout ../utils/test_main_with_timeout.cpp ...)
target_link_libraries(test_common_timeout PUBLIC gtest VectorSimilarity)
```

Run both in parallel, then remove old ones when confident.

---

## Summary

### Changes Required
1. Add `../utils/test_main_with_timeout.cpp` to each `add_executable()`
2. Change `gtest_main` to `gtest` in each `target_link_libraries()`

### Benefits
- ✅ All tests automatically get timeout protection
- ✅ Zero changes to test code
- ✅ Customizable timeouts per test
- ✅ Environment-aware (Valgrind, sanitizers)

### Effort
- **Low** - ~2 lines changed per test executable
- **Total**: ~28 lines changed in CMakeLists.txt

### Risk
- **Very Low** - Easy to rollback, no test code changes

