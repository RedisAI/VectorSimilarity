# Timeout Guard - Quick Start Guide

## What is TimeoutGuard?

`TimeoutGuard` is a reusable RAII-style utility that protects tests and benchmarks from hanging indefinitely. It's based on the timeout mechanism used in `testMockThreadPool` and provides a clean API for timeout protection.

## Quick Start

### 1. Include the Header

```cpp
#include "timeout_guard.h"
```

### 2. Add Timeout Protection

**Simple Test:**
```cpp
TEST(MyTest, Example) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(30));
    
    // Your test code here
    // If it takes longer than 30 seconds, the process will exit
}
```

**Thread-Heavy Test (Recommended Pattern):**
```cpp
TEST(MyTest, ThreadPool) {
    auto test_body = []() {
        test_utils::TimeoutGuard guard(std::chrono::seconds(100));
        
        // Your test code here
        
        guard.notify(); // Signal completion
        std::cerr << "Success" << std::endl;
        std::exit(testing::Test::HasFailure() ? -1 : 0);
    };
    
    EXPECT_EXIT(test_body(), ::testing::ExitedWithCode(0), "Success");
}
```

**Benchmark:**
```cpp
BENCHMARK_F(MyBenchmark, BM_Operation)(benchmark::State& state) {
    test_utils::BenchmarkTimeoutGuard guard(std::chrono::minutes(5));
    
    for (auto _ : state) {
        // Benchmark code
    }
}
```

## When to Use

### ✅ Use TimeoutGuard When:
- Test involves threading or async operations
- Test could potentially hang or deadlock
- Test runs in CI/CD and needs protection
- Test involves complex algorithms with unknown runtime
- Benchmark could run indefinitely

### ❌ Don't Use When:
- Test is trivial and fast (< 1 second)
- Test already has other timeout mechanisms
- Adding unnecessary complexity

## Recommended Timeouts

| Test Type | Timeout | Example |
|-----------|---------|---------|
| Fast unit test | 10-30s | `std::chrono::seconds(10)` |
| Thread pool test | 60-100s | `std::chrono::seconds(100)` |
| Integration test | 2-5min | `std::chrono::minutes(2)` |
| Benchmark | 5-10min | `std::chrono::minutes(5)` |
| CI with sanitizers | 2-3x normal | `std::chrono::seconds(60)` |

## Documentation

- **[TIMEOUT_GUARD_USAGE.md](TIMEOUT_GUARD_USAGE.md)** - Detailed usage guide with examples
- **[TIMEOUT_DESIGN.md](TIMEOUT_DESIGN.md)** - Architecture and design decisions
- **[MIGRATION_EXAMPLE.md](MIGRATION_EXAMPLE.md)** - How to migrate existing tests

## API Reference

### TimeoutGuard

```cpp
class TimeoutGuard {
public:
    // Constructor with timeout duration and optional custom action
    template <typename Rep, typename Period>
    explicit TimeoutGuard(
        std::chrono::duration<Rep, Period> timeout,
        std::function<void()> on_timeout = default_exit_action
    );
    
    // Destructor - automatically notifies and joins guard thread
    ~TimeoutGuard();
    
    // Explicitly notify that execution completed successfully
    void notify();
    
    // Check if timeout was triggered
    bool timed_out() const;
};
```

### BenchmarkTimeoutGuard

```cpp
class BenchmarkTimeoutGuard : public TimeoutGuard {
public:
    // Constructor with timeout duration (uses default benchmark timeout action)
    template <typename Rep, typename Period>
    explicit BenchmarkTimeoutGuard(std::chrono::duration<Rep, Period> timeout);
};
```

### Helper Functions

```cpp
// Run test with timeout and EXPECT_EXIT wrapper
template <typename TestFunc, typename Rep, typename Period>
void RunWithTimeout(
    TestFunc test_body,
    std::chrono::duration<Rep, Period> timeout,
    const char *success_message = "Success"
);
```

### Macros

```cpp
// Scope-based timeout guard
TIMEOUT_GUARD_SCOPE(duration) {
    // Code to protect
}
```

## Examples

### Example 1: Protecting a Thread Pool Test

```cpp
TEST(UtilsTests, testMockThreadPool) {
    auto test_body = []() {
        test_utils::TimeoutGuard guard(std::chrono::seconds(100));
        
        tieredIndexMock mock_thread_pool;
        mock_thread_pool.init_threads();
        
        // Submit jobs and wait
        for (size_t i = 0; i < 200; i++) {
            // ... submit jobs ...
            mock_thread_pool.thread_pool_wait();
        }
        
        mock_thread_pool.thread_pool_join();
        
        guard.notify();
        std::cerr << "Success" << std::endl;
        std::exit(testing::Test::HasFailure() ? -1 : 0);
    };
    
    EXPECT_EXIT(test_body(), ::testing::ExitedWithCode(0), "Success");
}
```

### Example 2: Simple RAII Pattern

```cpp
TEST(AlgorithmTest, ComplexOperation) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(30));
    
    // Complex algorithm that might hang
    auto result = complex_algorithm(input);
    
    ASSERT_EQ(result, expected);
    
    // Guard automatically cleaned up on scope exit
}
```

### Example 3: Custom Timeout Action

```cpp
TEST(MyTest, CustomAction) {
    test_utils::TimeoutGuard guard(
        std::chrono::seconds(30),
        []() {
            std::cerr << "Test timed out! Dumping state..." << std::endl;
            dump_debug_info();
            std::exit(-1);
        }
    );
    
    // Test code
}
```

### Example 4: Environment-Aware Timeout

```cpp
TEST(MyTest, AdaptiveTimeout) {
    auto timeout = std::chrono::seconds(30);
    
#ifdef RUNNING_ON_VALGRIND
    timeout *= 3; // Valgrind is slower
#endif
    
    test_utils::TimeoutGuard guard(timeout);
    
    // Test code
}
```

### Example 5: Benchmark Protection

```cpp
BENCHMARK_F(BM_VecSim, BM_ParallelSearch)(benchmark::State& state) {
    test_utils::BenchmarkTimeoutGuard guard(std::chrono::minutes(5));
    
    size_t k = state.range(0);
    size_t ef = state.range(1);
    
    for (auto _ : state) {
        auto results = VecSimIndex_TopKQuery(index, query, k, &params, BY_SCORE);
        VecSimQueryReply_Free(results);
    }
}
```

## Troubleshooting

### Test Times Out Unexpectedly

**Problem:** Test fails with timeout even though it should complete quickly.

**Solutions:**
1. Increase timeout value
2. Check for deadlocks using debugger
3. Add logging to identify slow sections
4. Check if CI environment is slower than local

### Timeout Not Triggering

**Problem:** Test hangs but timeout doesn't fire.

**Solutions:**
1. Verify guard is in scope during the hanging code
2. Check that timeout duration is actually exceeded
3. Ensure guard isn't destroyed prematurely

### Memory Leaks Reported

**Problem:** Valgrind or sanitizers report leaks related to timeout guard.

**Solutions:**
1. This is expected if timeout action calls `std::exit()`
2. Use EXPECT_EXIT pattern to isolate the test
3. Suppressions may be needed for intentional exits

## Best Practices

### ✅ DO:
- Use EXPECT_EXIT pattern for thread-heavy tests
- Choose reasonable timeout values (not too short, not too long)
- Call `notify()` explicitly in EXPECT_EXIT pattern
- Add timeouts to tests that run in CI/CD
- Document why a specific timeout value was chosen

### ❌ DON'T:
- Use extremely short timeouts (< 1 second) unless necessary
- Use extremely long timeouts (> 10 minutes) for unit tests
- Forget to call `notify()` in EXPECT_EXIT pattern
- Add timeouts to every trivial test
- Ignore timeout failures without investigation

## Integration with Build System

### CMake

No changes needed - header-only library:

```cmake
# Already included in tests/unit/CMakeLists.txt
include_directories(../utils)
```

### Makefile

No changes needed:

```bash
make unit_test              # Run all unit tests
make benchmark              # Run all benchmarks
```

### CTest

TimeoutGuard complements CTest's built-in timeout:

```cmake
# CTest timeout (last resort - kills process)
gtest_discover_tests(test_svs PROPERTIES TIMEOUT 3000)
```

**Recommendation:** Use both - TimeoutGuard for graceful handling, CTest TIMEOUT as safety net.

## Performance Impact

- **Thread Creation:** ~1-10ms per guard (one-time cost)
- **Memory:** ~8KB per guard (thread stack)
- **CPU:** Negligible (thread sleeps on condition variable)
- **Overhead:** Minimal and acceptable for tests/benchmarks

## Migration from Manual Implementation

See [MIGRATION_EXAMPLE.md](MIGRATION_EXAMPLE.md) for detailed migration guide.

**Quick comparison:**

**Before (11 lines):**
```cpp
std::mutex mtx;
std::condition_variable cv;
auto guard_thread = std::thread([&]() {
    std::unique_lock<std::mutex> lock(mtx);
    if (cv.wait_for(lock, timeout) == std::cv_status::timeout) {
        std::cerr << "Test timeout! Exiting..." << std::endl;
        std::exit(-1);
    }
});
// ... test code ...
cv.notify_one();
guard_thread.join();
```

**After (2 lines):**
```cpp
test_utils::TimeoutGuard guard(timeout);
// ... test code ...
guard.notify();
```

## Support

For questions or issues:
1. Check [TIMEOUT_GUARD_USAGE.md](TIMEOUT_GUARD_USAGE.md) for detailed examples
2. Check [TIMEOUT_DESIGN.md](TIMEOUT_DESIGN.md) for architecture details
3. Check [MIGRATION_EXAMPLE.md](MIGRATION_EXAMPLE.md) for migration help
4. Review existing tests using TimeoutGuard (e.g., `testMockThreadPool`)

## License

Same as VectorSimilarity repository (RSALv2 / SSPLv1 / AGPLv3).

