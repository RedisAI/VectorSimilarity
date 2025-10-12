# Timeout Guard Usage Guide

This document describes how to use the `TimeoutGuard` utility to add timeout protection to tests and benchmarks.

## Overview

The `TimeoutGuard` is a RAII-style utility that creates a background thread to monitor execution time. If the timeout expires before the guard is destroyed or explicitly notified, it executes a configurable action (default: exit the process).

## Use Cases

### 1. **Unit Tests (Google Test)**

#### Simple Timeout Protection

```cpp
#include "timeout_guard.h"

TEST(MyTest, BasicTimeout) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(30));
    
    // Your test code here
    // If it takes longer than 30 seconds, the process will exit
    
    // Guard automatically notified when it goes out of scope
}
```

#### With EXPECT_EXIT Pattern (Recommended for Thread-Heavy Tests)

```cpp
TEST(MyTest, ThreadPoolTest) {
    auto test_body = []() {
        test_utils::TimeoutGuard guard(std::chrono::seconds(100));
        
        // Your test code here
        // ...
        
        guard.notify(); // Explicitly notify success
        std::cerr << "Success" << std::endl;
        std::exit(testing::Test::HasFailure() ? -1 : 0);
    };
    
    EXPECT_EXIT(test_body(), ::testing::ExitedWithCode(0), "Success");
}
```

#### Using the Helper Function

```cpp
TEST(MyTest, HelperExample) {
    test_utils::RunWithTimeout([]() {
        // Your test code here
    }, std::chrono::seconds(30), "Success");
}
```

#### Using the Macro

```cpp
TEST(MyTest, MacroExample) {
    auto test_body = []() {
        TIMEOUT_GUARD_SCOPE(std::chrono::seconds(30)) {
            // Your test code here
        }
        std::exit(0);
    };
    EXPECT_EXIT(test_body(), ::testing::ExitedWithCode(0), "");
}
```

### 2. **Benchmarks (Google Benchmark)**

```cpp
#include "timeout_guard.h"

BENCHMARK_F(MyBenchmark, BM_SlowOperation)(benchmark::State& state) {
    // Set a 5-minute timeout for the entire benchmark
    test_utils::BenchmarkTimeoutGuard guard(std::chrono::minutes(5));
    
    for (auto _ : state) {
        // Benchmark iteration code
    }
    
    // Guard automatically notified when it goes out of scope
}
```

### 3. **Custom Timeout Actions**

```cpp
TEST(MyTest, CustomAction) {
    test_utils::TimeoutGuard guard(
        std::chrono::seconds(30),
        []() {
            std::cerr << "Custom timeout handler!" << std::endl;
            // Log additional information
            // Cleanup resources
            std::abort();
        }
    );
    
    // Your test code
}
```

## Configuration

### Default Timeouts by Test Type

Recommended timeout values:

- **Fast unit tests**: 10-30 seconds
- **Thread pool / parallel tests**: 60-100 seconds
- **Integration tests**: 2-5 minutes
- **Benchmarks**: 5-10 minutes
- **CI/CD with sanitizers**: 2-3x normal timeout

### Environment-Specific Timeouts

```cpp
TEST(MyTest, AdaptiveTimeout) {
    auto timeout = std::chrono::seconds(30);
    
#ifdef RUNNING_ON_VALGRIND
    timeout *= 3; // Valgrind is slower
#endif
    
#ifdef USE_ASAN
    timeout *= 2; // Sanitizers add overhead
#endif
    
    test_utils::TimeoutGuard guard(timeout);
    // Test code
}
```

## Integration with Build System

### CMake Integration

The timeout guard is header-only and automatically available when you include it:

```cmake
# Already included in tests/utils/
include_directories(../utils)
```

### CTest Timeout (Complementary)

The `TimeoutGuard` works alongside CTest's built-in timeout:

```cmake
# In CMakeLists.txt
gtest_discover_tests(test_svs PROPERTIES TIMEOUT 3000)
```

**Difference:**
- **CTest TIMEOUT**: Kills the entire test process (less graceful)
- **TimeoutGuard**: Runs in-process, can log details, cleanup resources

## Best Practices

### 1. **Choose the Right Timeout**

```cpp
// ✅ Good: Reasonable timeout for the operation
test_utils::TimeoutGuard guard(std::chrono::seconds(30));

// ❌ Bad: Too short, may cause flaky tests
test_utils::TimeoutGuard guard(std::chrono::milliseconds(100));

// ❌ Bad: Too long, defeats the purpose
test_utils::TimeoutGuard guard(std::chrono::hours(1));
```

### 2. **Use EXPECT_EXIT for Thread-Heavy Tests**

```cpp
// ✅ Good: Isolates threading issues
TEST(ThreadTest, Parallel) {
    auto test_body = []() {
        test_utils::TimeoutGuard guard(std::chrono::seconds(60));
        // Thread pool operations
        guard.notify();
        std::exit(0);
    };
    EXPECT_EXIT(test_body(), ::testing::ExitedWithCode(0), "");
}

// ⚠️ Acceptable: Simple tests without complex threading
TEST(SimpleTest, Basic) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(10));
    // Simple operations
}
```

### 3. **Explicit Notification for Complex Control Flow**

```cpp
TEST(MyTest, ComplexFlow) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(30));
    
    if (condition) {
        // Early exit path
        guard.notify();
        return;
    }
    
    // Normal path
    // Guard auto-notified on scope exit
}
```

### 4. **Benchmark Timeouts**

```cpp
// ✅ Good: Timeout for entire benchmark fixture
BENCHMARK_F(MyBenchmark, BM_Operation)(benchmark::State& state) {
    test_utils::BenchmarkTimeoutGuard guard(std::chrono::minutes(5));
    
    for (auto _ : state) {
        // Each iteration should be fast
        // Timeout protects against infinite loops
    }
}
```

## Troubleshooting

### Test Times Out Unexpectedly

1. **Check if timeout is too short**: Increase timeout value
2. **Check for deadlocks**: Use debugger or add logging
3. **Check for infinite loops**: Review test logic
4. **Check CI environment**: May be slower than local machine

### Timeout Not Triggering

1. **Verify guard is in scope**: Guard must outlive the code being protected
2. **Check for early exits**: Ensure guard isn't destroyed prematurely
3. **Verify timeout duration**: Make sure it's actually exceeded

### Memory Leaks with Timeout

The guard thread is properly joined in the destructor, but if the timeout action calls `std::exit()`, destructors won't run. This is intentional for timeout scenarios.

## Migration Guide

### Migrating Existing Tests

**Before:**
```cpp
TEST(UtilsTests, testMockThreadPool) {
    const size_t num_repeats = 2;
    std::chrono::seconds test_timeout(100);
    
    auto TestBody = [=]() {
        std::mutex mtx;
        std::condition_variable cv;
        auto guard_thread = std::thread([&]() {
            std::unique_lock<std::mutex> lock(mtx);
            if (cv.wait_for(lock, test_timeout) == std::cv_status::timeout) {
                std::cerr << "Test timeout! Exiting..." << std::endl;
                std::exit(-1);
            }
        });
        
        // Test code...
        
        cv.notify_one();
        guard_thread.join();
        std::exit(0);
    };
    
    EXPECT_EXIT(TestBody(), ::testing::ExitedWithCode(0), "Success");
}
```

**After:**
```cpp
TEST(UtilsTests, testMockThreadPool) {
    const size_t num_repeats = 2;
    
    auto TestBody = [=]() {
        test_utils::TimeoutGuard guard(std::chrono::seconds(100));
        
        // Test code...
        
        guard.notify();
        std::cerr << "Success" << std::endl;
        std::exit(testing::Test::HasFailure() ? -1 : 0);
    };
    
    EXPECT_EXIT(TestBody(), ::testing::ExitedWithCode(0), "Success");
}
```

## Examples from Codebase

See `tests/unit/test_common.cpp::testMockThreadPool` for a real-world example of the timeout pattern.

