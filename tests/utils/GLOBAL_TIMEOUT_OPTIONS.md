# Global Timeout Protection Options

This document explains different approaches to apply timeout protection to all tests without manually adding `TimeoutGuard` to every single `TEST()`.

## Overview of Options

| Option | Scope | Effort | Flexibility | Recommended For |
|--------|-------|--------|-------------|-----------------|
| **Option 1: Test Listener** | All tests globally | Low | High | **Most cases** |
| **Option 2: Test Fixture** | Tests using fixture | Medium | Medium | Specific test suites |
| **Option 3: Macro Wrapper** | All tests | Low | Low | Simple cases |
| **Option 4: CTest Timeout** | Per executable | Very Low | Low | Backup/fallback |
| **Option 5: Manual** | Per test | High | Very High | Special cases only |

---

## **Option 1: Test Event Listener (RECOMMENDED)**

### How It Works
Register a global listener that automatically creates a `TimeoutGuard` for every test.

### Implementation

**Step 1:** Use the custom main instead of `gtest_main`

In `tests/unit/CMakeLists.txt`, replace:
```cmake
# OLD:
target_link_libraries(test_hnsw PUBLIC gtest_main VectorSimilarity)

# NEW:
target_link_libraries(test_hnsw PUBLIC gtest VectorSimilarity)
add_executable(test_hnsw 
    ../utils/mock_thread_pool.cpp 
    ../utils/test_main_with_timeout.cpp  # Add this
    test_hnsw.cpp 
    test_hnsw_multi.cpp 
    test_hnsw_tiered.cpp 
    unit_test_utils.cpp)
```

**Step 2:** Done! All tests now have automatic timeout protection.

### Customization

Edit `tests/utils/timeout_test_environment.h` to customize timeouts per test:

```cpp
std::chrono::seconds GetTimeoutForTest(const testing::TestInfo &test_info) {
    auto timeout = std::chrono::seconds(default_timeout_seconds_);
    
    // Customize based on test name
    if (test_name.find("slow") != std::string::npos) {
        timeout = std::chrono::seconds(120);
    }
    
    // Customize based on test suite
    if (suite_name == "HNSWTest") {
        timeout = std::chrono::seconds(60);
    }
    
    return timeout;
}
```

### Pros
- ✅ **Zero code changes** to individual tests
- ✅ **Centralized configuration** - change timeout logic in one place
- ✅ **Flexible** - different timeouts for different test types
- ✅ **Environment-aware** - automatically adjusts for Valgrind, sanitizers
- ✅ **Clean** - no boilerplate in test code

### Cons
- ⚠️ Requires changing from `gtest_main` to custom main
- ⚠️ All tests get timeout protection (might be overkill for trivial tests)

---

## **Option 2: Test Fixture Base Class**

### How It Works
Create a base test fixture with timeout protection that other fixtures inherit from.

### Implementation

```cpp
// In tests/unit/unit_test_utils.h
namespace test_utils {

template <typename T = void>
class TimeoutTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Create timeout guard with default 30 seconds
        timeout_guard_ = std::make_unique<test_utils::TimeoutGuard>(
            GetTimeout(),
            [this]() {
                std::cerr << "Test timeout in " << typeid(*this).name() << std::endl;
                std::exit(-1);
            }
        );
    }

    void TearDown() override {
        if (timeout_guard_) {
            timeout_guard_->notify();
            timeout_guard_.reset();
        }
    }

    // Override this to customize timeout per fixture
    virtual std::chrono::seconds GetTimeout() {
        return std::chrono::seconds(30);
    }

private:
    std::unique_ptr<test_utils::TimeoutGuard> timeout_guard_;
};

} // namespace test_utils
```

### Usage

```cpp
// Instead of:
class MyTest : public ::testing::Test { };

// Use:
class MyTest : public test_utils::TimeoutTestFixture<> {
protected:
    // Override to customize timeout
    std::chrono::seconds GetTimeout() override {
        return std::chrono::seconds(60);
    }
};

TEST_F(MyTest, SomeTest) {
    // Automatically has timeout protection
}
```

### Pros
- ✅ **Opt-in** - only tests using the fixture get timeout
- ✅ **Customizable** per fixture
- ✅ **Type-safe** - compile-time checking

### Cons
- ⚠️ Requires changing test fixtures
- ⚠️ Doesn't work with plain `TEST()` (only `TEST_F()`)
- ⚠️ More boilerplate than Option 1

---

## **Option 3: Macro Wrapper**

### How It Works
Create a macro that wraps `TEST()` and adds timeout automatically.

### Implementation

```cpp
// In tests/unit/unit_test_utils.h
#define TEST_WITH_TIMEOUT(test_suite_name, test_name, timeout_seconds)                            \
    void test_suite_name##_##test_name##_Body();                                                  \
    TEST(test_suite_name, test_name) {                                                            \
        test_utils::TimeoutGuard guard(std::chrono::seconds(timeout_seconds));                    \
        test_suite_name##_##test_name##_Body();                                                   \
    }                                                                                             \
    void test_suite_name##_##test_name##_Body()

// Default 30 second timeout
#define TEST_TIMEOUT(test_suite_name, test_name)                                                  \
    TEST_WITH_TIMEOUT(test_suite_name, test_name, 30)
```

### Usage

```cpp
// Instead of:
TEST(MyTest, Example) {
    // test code
}

// Use:
TEST_TIMEOUT(MyTest, Example) {
    // test code - automatically has 30s timeout
}

// Or with custom timeout:
TEST_WITH_TIMEOUT(MyTest, SlowTest, 120) {
    // test code - has 120s timeout
}
```

### Pros
- ✅ **Simple** - just change `TEST` to `TEST_TIMEOUT`
- ✅ **Explicit** - clear that test has timeout
- ✅ **Flexible** - can specify timeout per test

### Cons
- ⚠️ Requires changing every test (from `TEST` to `TEST_TIMEOUT`)
- ⚠️ More verbose than Option 1
- ⚠️ Macro magic can be confusing

---

## **Option 4: CTest Timeout (Complementary)**

### How It Works
Use CMake's built-in test timeout feature.

### Implementation

In `tests/unit/CMakeLists.txt`:

```cmake
# Set default timeout for all tests
set_tests_properties(
    test_hnsw
    test_bruteforce
    test_common
    PROPERTIES TIMEOUT 60
)

# Or per-test:
gtest_discover_tests(test_svs PROPERTIES TIMEOUT 3000)
```

### Pros
- ✅ **Very simple** - just CMake configuration
- ✅ **No code changes** needed
- ✅ **Works with any test framework**

### Cons
- ⚠️ **Coarse-grained** - timeout per test executable, not per test
- ⚠️ **Kills process** - no graceful cleanup
- ⚠️ **Less informative** - doesn't show which specific test timed out

### Recommendation
Use this **in addition to** Option 1 as a safety net.

---

## **Option 5: Manual (Current Approach)**

### How It Works
Add `TimeoutGuard` manually to each test that needs it.

### Usage

```cpp
TEST(MyTest, Example) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(30));
    // test code
}
```

### Pros
- ✅ **Maximum control** - timeout only where needed
- ✅ **Explicit** - clear which tests have timeout

### Cons
- ⚠️ **High effort** - must add to every test
- ⚠️ **Easy to forget** - new tests might not have timeout
- ⚠️ **Repetitive** - lots of boilerplate

### Recommendation
Only use for **special cases** where you need custom timeout behavior.

---

## **Recommended Approach**

### For VectorSimilarity Repository

**Use Option 1 (Test Listener) + Option 4 (CTest Timeout)**

#### Step 1: Add Global Timeout Listener

Modify `tests/unit/CMakeLists.txt`:

```cmake
# For each test executable, replace gtest_main with custom main
add_executable(test_hnsw 
    ../utils/mock_thread_pool.cpp 
    ../utils/test_main_with_timeout.cpp  # Add this
    test_hnsw.cpp 
    test_hnsw_multi.cpp 
    test_hnsw_tiered.cpp 
    unit_test_utils.cpp)

# Change from gtest_main to gtest
target_link_libraries(test_hnsw PUBLIC gtest VectorSimilarity)  # Not gtest_main
```

Repeat for all test executables:
- `test_hnsw`
- `test_hnsw_parallel`
- `test_bruteforce`
- `test_allocator`
- `test_common`
- `test_components`
- `test_bf16`, `test_fp16`, `test_int8`, `test_uint8`
- `test_svs`
- `test_index_test_utils`

#### Step 2: Keep CTest Timeouts as Safety Net

```cmake
# Keep existing CTest timeouts
gtest_discover_tests(test_svs PROPERTIES TIMEOUT 3000)
```

#### Step 3: Customize Timeout Logic (Optional)

Edit `tests/utils/timeout_test_environment.h` to adjust timeouts based on:
- Test suite name
- Test name
- Environment (Valgrind, sanitizers)

### Result

- ✅ **All tests** automatically get timeout protection
- ✅ **Zero changes** to individual test code
- ✅ **Customizable** timeouts per test type
- ✅ **Environment-aware** (Valgrind, sanitizers)
- ✅ **Safety net** with CTest timeout
- ✅ **Informative** - shows which specific test timed out

---

## Migration Guide

### Phase 1: Add Infrastructure (No Impact)
1. ✅ Add `timeout_guard.h` (already done)
2. ✅ Add `timeout_test_environment.h` (already done)
3. ✅ Add `test_main_with_timeout.cpp` (already done)

### Phase 2: Migrate One Test Executable (Proof of Concept)
1. Pick one test executable (e.g., `test_common`)
2. Modify `CMakeLists.txt` to use custom main
3. Run tests and verify no regressions
4. Check CI/CD passes

### Phase 3: Migrate All Test Executables
1. Apply same changes to all test executables
2. Test each one individually
3. Run full test suite
4. Verify CI/CD passes

### Phase 4: Tune Timeout Logic
1. Monitor test execution times
2. Adjust timeout logic in `GetTimeoutForTest()`
3. Handle edge cases (very slow tests, very fast tests)

---

## Example: Migrating test_common

### Before

```cmake
add_executable(test_common ../utils/mock_thread_pool.cpp test_common.cpp unit_test_utils.cpp)
target_link_libraries(test_common PUBLIC gtest_main VectorSimilarity)
gtest_discover_tests(test_common)
```

### After

```cmake
add_executable(test_common 
    ../utils/mock_thread_pool.cpp 
    ../utils/test_main_with_timeout.cpp  # Added
    test_common.cpp 
    unit_test_utils.cpp)
target_link_libraries(test_common PUBLIC gtest VectorSimilarity)  # Changed gtest_main -> gtest
gtest_discover_tests(test_common)
```

### Test Code

**No changes needed!** All tests in `test_common.cpp` automatically get timeout protection.

---

## Comparison Summary

### Code Changes Required

| Option | CMakeLists.txt | Test Code | Custom Files |
|--------|----------------|-----------|--------------|
| **Option 1** | ✏️ Minor (change gtest_main) | ✅ None | ✅ Already created |
| **Option 2** | ✅ None | ✏️ Change fixtures | ✏️ Add base class |
| **Option 3** | ✅ None | ✏️ Change all TESTs | ✏️ Add macros |
| **Option 4** | ✏️ Add timeouts | ✅ None | ✅ None |
| **Option 5** | ✅ None | ✏️ Add to every test | ✅ None |

### Recommendation: **Option 1** (Test Listener)

**Why?**
- Minimal changes (just CMakeLists.txt)
- Zero changes to test code
- Centralized, maintainable configuration
- Flexible and customizable
- Works with all test types (TEST, TEST_F, TEST_P, etc.)

---

## FAQ

### Q: Will this slow down tests?
**A:** No. The timeout guard only creates one thread per test, which sleeps on a condition variable. Overhead is ~1-10ms per test.

### Q: What if a test legitimately takes longer than the timeout?
**A:** Customize the timeout in `GetTimeoutForTest()` based on test name or suite.

### Q: Can I disable timeout for specific tests?
**A:** Yes, return a very large timeout (e.g., `std::chrono::hours(24)`) for specific tests in `GetTimeoutForTest()`.

### Q: Does this work with parameterized tests (TEST_P)?
**A:** Yes, the listener works with all Google Test test types.

### Q: What about benchmarks?
**A:** Benchmarks should use `BenchmarkTimeoutGuard` manually (Option 5) since they don't use Google Test framework.

---

## Next Steps

1. **Review** this document and choose an approach
2. **Test** Option 1 with one test executable (e.g., `test_common`)
3. **Verify** no regressions in CI/CD
4. **Roll out** to all test executables
5. **Monitor** and tune timeout values based on actual test execution times

