# Answer: Do I Need to Add TimeoutGuard to Every TEST?

## Short Answer

**NO!** You don't have to add `test_utils::TimeoutGuard guard(std::chrono::seconds(100));` to every TEST.

Instead, you can enable **automatic timeout protection for all tests** with just a small change to `CMakeLists.txt`.

---

## The Solution: Global Timeout Listener

### What You Need to Do

**Change 2 lines per test executable in `tests/unit/CMakeLists.txt`:**

```cmake
# BEFORE:
add_executable(test_common ../utils/mock_thread_pool.cpp test_common.cpp unit_test_utils.cpp)
target_link_libraries(test_common PUBLIC gtest_main VectorSimilarity)

# AFTER:
add_executable(test_common ../utils/test_main_with_timeout.cpp ../utils/mock_thread_pool.cpp test_common.cpp unit_test_utils.cpp)
target_link_libraries(test_common PUBLIC gtest VectorSimilarity)
#                                  1. Add this ↑                                                2. Change this ↑
```

### That's It!

**All tests in that executable now automatically have timeout protection. Zero changes to test code.**

---

## How It Works

### The Magic: Google Test Event Listener

I created a custom `main()` function that registers a **test event listener**:

```cpp
// tests/utils/test_main_with_timeout.cpp
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    
    // This one line adds timeout to ALL tests
    test_utils::RegisterGlobalTimeoutListener(std::chrono::seconds(30));
    
    return RUN_ALL_TESTS();
}
```

The listener automatically:
1. Creates a `TimeoutGuard` before each test starts
2. Destroys it after each test ends
3. Customizes timeout based on test name/type

---

## Example: Before and After

### Your Test Code (NO CHANGES!)

```cpp
// test_common.cpp - EXACTLY THE SAME BEFORE AND AFTER

TEST(MyTest, Example) {
    // Your test code
    // No TimeoutGuard needed!
}

TEST(MyTest, AnotherTest) {
    // Your test code
    // Still no TimeoutGuard needed!
}
```

### What Changed: Only CMakeLists.txt

```diff
  add_executable(test_common 
+     ../utils/test_main_with_timeout.cpp
      ../utils/mock_thread_pool.cpp 
      test_common.cpp 
      unit_test_utils.cpp)

- target_link_libraries(test_common PUBLIC gtest_main VectorSimilarity)
+ target_link_libraries(test_common PUBLIC gtest VectorSimilarity)
```

---

## Customizing Timeouts

### Default Behavior

All tests get **30 seconds** timeout by default.

### Automatic Customization

The listener automatically adjusts timeouts based on test characteristics:

```cpp
// In tests/utils/timeout_test_environment.h (already created for you)

std::chrono::seconds GetTimeoutForTest(const testing::TestInfo &test_info) {
    auto timeout = std::chrono::seconds(30);  // Default
    
    // Thread pool tests get 100 seconds
    if (test_name.find("thread") != std::string::npos) {
        timeout = std::chrono::seconds(100);
    }
    
    // Tiered tests get 120 seconds
    if (test_name.find("tiered") != std::string::npos) {
        timeout = std::chrono::seconds(120);
    }
    
    // SVS tests get 150 seconds
    if (suite_name.find("SVS") != std::string::npos) {
        timeout = std::chrono::seconds(150);
    }
    
    // Valgrind: 3x timeout
    #ifdef RUNNING_ON_VALGRIND
        timeout *= 3;
    #endif
    
    return timeout;
}
```

You can edit this function to customize timeouts for specific tests.

---

## Complete Migration Steps

### Step 1: Files Already Created

I've already created these files for you:
- ✅ `tests/utils/timeout_guard.h` - Core timeout mechanism
- ✅ `tests/utils/timeout_test_environment.h` - Test listener
- ✅ `tests/utils/test_main_with_timeout.cpp` - Custom main()

### Step 2: Update CMakeLists.txt

Edit `tests/unit/CMakeLists.txt` and make these changes:

```cmake
# For each test executable:

# 1. Add test_main_with_timeout.cpp to sources
add_executable(test_hnsw 
    ../utils/test_main_with_timeout.cpp  # ← ADD THIS
    ../utils/mock_thread_pool.cpp 
    test_hnsw.cpp 
    test_hnsw_multi.cpp 
    test_hnsw_tiered.cpp 
    unit_test_utils.cpp)

# 2. Change gtest_main to gtest
target_link_libraries(test_hnsw PUBLIC gtest VectorSimilarity)  # ← CHANGE THIS
```

Repeat for all 14 test executables:
- test_hnsw
- test_hnsw_parallel
- test_bruteforce
- test_allocator
- test_spaces
- test_types
- test_common
- test_components
- test_bf16
- test_fp16
- test_int8
- test_uint8
- test_index_test_utils
- test_svs

### Step 3: Test

```bash
make clean
make unit_test
```

All tests should pass as before, but now with timeout protection!

---

## What About Special Cases?

### Case 1: Test Needs Custom Timeout

**Option A:** Edit `GetTimeoutForTest()` in `timeout_test_environment.h`

```cpp
if (suite_name == "MyTest" && test_name == "VerySlowTest") {
    timeout = std::chrono::minutes(5);
}
```

**Option B:** Add manual `TimeoutGuard` to that specific test

```cpp
TEST(MyTest, VerySlowTest) {
    test_utils::TimeoutGuard guard(std::chrono::minutes(5));
    // Test code
}
```

### Case 2: Test Needs Custom Timeout Action

Add manual `TimeoutGuard`:

```cpp
TEST(MyTest, CustomAction) {
    test_utils::TimeoutGuard guard(
        std::chrono::seconds(30),
        []() {
            std::cerr << "Custom timeout handler!" << std::endl;
            dump_debug_info();
            std::exit(-1);
        }
    );
    // Test code
}
```

### Case 3: Test Should Never Timeout

Set very long timeout in `GetTimeoutForTest()`:

```cpp
if (test_name == "NeverTimeout") {
    timeout = std::chrono::hours(24);  // Effectively no timeout
}
```

---

## Comparison: Manual vs Automatic

### Manual Approach (What You Asked About)

```cpp
TEST(MyTest, Test1) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(30));  // ← Add to every test
    // Test code
}

TEST(MyTest, Test2) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(30));  // ← Add to every test
    // Test code
}

TEST(MyTest, Test3) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(30));  // ← Add to every test
    // Test code
}
```

**Effort:** High - must add to every test
**Lines changed:** 3 per test × 100s of tests = 300+ lines

### Automatic Approach (Recommended)

```cpp
// test_common.cpp - NO CHANGES!
TEST(MyTest, Test1) {
    // Test code - timeout automatic!
}

TEST(MyTest, Test2) {
    // Test code - timeout automatic!
}

TEST(MyTest, Test3) {
    // Test code - timeout automatic!
}
```

**Effort:** Low - change CMakeLists.txt only
**Lines changed:** 2 per executable × 14 executables = 28 lines

---

## Benefits of Automatic Approach

✅ **Zero test code changes** - all existing tests work as-is
✅ **Centralized configuration** - change timeout logic in one place
✅ **Automatic for new tests** - developers don't need to remember to add timeout
✅ **Customizable** - different timeouts for different test types
✅ **Environment-aware** - automatically adjusts for Valgrind, sanitizers
✅ **Maintainable** - easy to update timeout logic globally

---

## Documentation

I've created comprehensive documentation:

1. **[GLOBAL_TIMEOUT_OPTIONS.md](GLOBAL_TIMEOUT_OPTIONS.md)** - All options explained
2. **[CMAKE_MIGRATION_EXAMPLE.md](CMAKE_MIGRATION_EXAMPLE.md)** - Exact CMakeLists.txt changes
3. **[README_TIMEOUT.md](README_TIMEOUT.md)** - Quick start guide
4. **[TIMEOUT_GUARD_USAGE.md](TIMEOUT_GUARD_USAGE.md)** - Detailed usage
5. **[TIMEOUT_DESIGN.md](TIMEOUT_DESIGN.md)** - Architecture details

---

## Summary

### Your Question
> "Does that mean I will have to add `test_utils::TimeoutGuard guard(std::chrono::seconds(100));` in every TEST?"

### Answer
**No!** Just update `CMakeLists.txt` (2 lines per test executable) and all tests automatically get timeout protection.

### What to Do
1. Review [CMAKE_MIGRATION_EXAMPLE.md](CMAKE_MIGRATION_EXAMPLE.md)
2. Update `tests/unit/CMakeLists.txt` (28 lines total)
3. Run `make unit_test` to verify
4. Done! All tests now have timeout protection

### Effort
- **Manual approach**: 300+ lines (add to every test)
- **Automatic approach**: 28 lines (update CMakeLists.txt)

**Recommendation: Use the automatic approach!**

