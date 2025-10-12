# Quick Start: Enable Timeout Protection for All Tests

## TL;DR

**Change 2 lines per test executable in `tests/unit/CMakeLists.txt` → All tests get automatic timeout protection!**

---

## The Changes

### Before
```cmake
add_executable(test_common ../utils/mock_thread_pool.cpp test_common.cpp unit_test_utils.cpp)
target_link_libraries(test_common PUBLIC gtest_main VectorSimilarity)
```

### After
```cmake
add_executable(test_common ../utils/test_main_with_timeout.cpp ../utils/mock_thread_pool.cpp test_common.cpp unit_test_utils.cpp)
target_link_libraries(test_common PUBLIC gtest VectorSimilarity)
```

**That's it! All tests in `test_common` now have automatic 30-second timeout (customizable).**

---

## Apply to All Test Executables

Edit `tests/unit/CMakeLists.txt` and apply the same pattern to all 14 executables:

1. test_hnsw
2. test_hnsw_parallel
3. test_bruteforce
4. test_allocator
5. test_spaces
6. test_types
7. test_common
8. test_components
9. test_bf16
10. test_fp16
11. test_int8
12. test_uint8
13. test_index_test_utils
14. test_svs

---

## Test It

```bash
make clean
make unit_test
```

All tests should pass as before, but now with timeout protection!

---

## Customize Timeouts

Edit `tests/utils/timeout_test_environment.h`:

```cpp
std::chrono::seconds GetTimeoutForTest(const testing::TestInfo &test_info) {
    auto timeout = std::chrono::seconds(30);  // Default
    
    // Custom timeout for specific test
    if (test_name == "testMockThreadPool") {
        timeout = std::chrono::seconds(100);
    }
    
    // Custom timeout for test suite
    if (suite_name == "HNSWTest") {
        timeout = std::chrono::seconds(60);
    }
    
    return timeout;
}
```

---

## What You Get

✅ All tests automatically timeout after 30 seconds (default)
✅ Thread pool tests get 100 seconds
✅ Tiered tests get 120 seconds
✅ SVS tests get 150 seconds
✅ Valgrind: 3x timeout
✅ Sanitizers: 2x timeout
✅ Zero changes to test code
✅ Customizable per test/suite

---

## Files Created

All files are ready to use in `tests/utils/`:

- ✅ `timeout_guard.h` - Core timeout mechanism
- ✅ `timeout_test_environment.h` - Test listener
- ✅ `test_main_with_timeout.cpp` - Custom main()
- ✅ Documentation (this file and others)

---

## Need More Info?

- **[ANSWER_TO_YOUR_QUESTION.md](ANSWER_TO_YOUR_QUESTION.md)** - Detailed answer to "do I need to add to every TEST?"
- **[CMAKE_MIGRATION_EXAMPLE.md](CMAKE_MIGRATION_EXAMPLE.md)** - Complete CMakeLists.txt diff
- **[GLOBAL_TIMEOUT_OPTIONS.md](GLOBAL_TIMEOUT_OPTIONS.md)** - All available options
- **[README_TIMEOUT.md](README_TIMEOUT.md)** - Full usage guide

---

## Summary

| Approach | Lines Changed | Test Code Changes | Effort |
|----------|---------------|-------------------|--------|
| **Automatic (Recommended)** | 28 | 0 | Low |
| Manual | 300+ | Every test | High |

**Use the automatic approach!**

