# Timeout Mechanism Design Document

## Executive Summary

This document describes the design of a reusable timeout mechanism for tests and benchmarks in the VectorSimilarity repository. The design is based on the timeout pattern used in `testMockThreadPool` and provides a clean, RAII-style API for timeout protection.

## Motivation

### Problems Addressed

1. **Test Hangs**: Tests with threading, async operations, or complex algorithms can hang indefinitely
2. **CI/CD Pipeline Failures**: Hung tests block CI/CD pipelines and waste resources
3. **Debugging Difficulty**: Without timeouts, it's hard to identify which test is stuck
4. **Code Duplication**: Each test that needs timeout protection reimplements the same pattern

### Goals

- ✅ Provide reusable timeout utilities for C++ tests and benchmarks
- ✅ Support both Google Test and Google Benchmark frameworks
- ✅ Maintain backward compatibility with existing tests
- ✅ Minimal boilerplate code
- ✅ Thread-safe and RAII-compliant
- ✅ Configurable timeout actions
- ✅ Easy to integrate into existing codebase

## Architecture

### Component Overview

```
tests/utils/
├── timeout_guard.h              # Header-only timeout guard utility
├── TIMEOUT_GUARD_USAGE.md       # Usage documentation
└── TIMEOUT_DESIGN.md            # This design document
```

### Class Hierarchy

```
test_utils::TimeoutGuard
    ├── Core timeout mechanism (RAII)
    └── test_utils::BenchmarkTimeoutGuard
            └── Specialized for Google Benchmark
```

### Design Patterns Used

1. **RAII (Resource Acquisition Is Initialization)**
   - Timeout guard automatically starts on construction
   - Automatically notifies and joins thread on destruction
   - Prevents resource leaks

2. **Strategy Pattern**
   - Configurable timeout action via `std::function`
   - Default action: exit process with error code
   - Custom actions: logging, cleanup, abort, etc.

3. **Guard Pattern**
   - Protects code sections from exceeding time limits
   - Automatic cleanup on scope exit

## Implementation Details

### Core Components

#### 1. TimeoutGuard Class

```cpp
class TimeoutGuard {
private:
    std::mutex mutex_;                    // Protects shared state
    std::condition_variable cv_;          // Signals completion
    std::thread guard_thread_;            // Background timeout monitor
    std::function<void()> timeout_action_; // Action on timeout
    bool timed_out_;                      // Timeout occurred flag
    bool notified_;                       // Completion notified flag
    
public:
    // Constructor: starts guard thread
    template <typename Rep, typename Period>
    explicit TimeoutGuard(
        std::chrono::duration<Rep, Period> timeout,
        std::function<void()> on_timeout = default_action
    );
    
    // Destructor: notifies and joins guard thread
    ~TimeoutGuard();
    
    // Explicit notification of completion
    void notify();
    
    // Check if timeout occurred
    bool timed_out() const;
};
```

#### 2. Thread Lifecycle

```
Construction
    ├── Create guard thread
    ├── Guard thread waits on condition variable with timeout
    └── Main thread continues with test/benchmark code

During Execution
    ├── If timeout expires: execute timeout_action_
    └── If notify() called: guard thread exits cleanly

Destruction
    ├── Call notify() if not already called
    ├── Join guard thread
    └── Clean up resources
```

#### 3. Timeout Action Flow

```
Timeout Expires
    ├── Set timed_out_ = true
    ├── Execute timeout_action_
    │   ├── Default: std::exit(-1)
    │   └── Custom: user-defined function
    └── Process terminates (if using default action)
```

### Thread Safety

- **Mutex Protection**: All shared state access is protected by `mutex_`
- **Condition Variable**: Used for thread synchronization
- **Atomic Operations**: `timed_out_` could be atomic, but mutex provides sufficient protection
- **No Data Races**: Guard thread only reads timeout duration, writes to `timed_out_`

### Memory Management

- **RAII Compliance**: All resources cleaned up in destructor
- **Thread Joining**: Guard thread always joined before destruction completes
- **No Leaks**: Even with custom timeout actions, guard thread is properly managed
- **Exception Safety**: Basic exception safety guarantee (resources cleaned up)

## Integration Points

### 1. Google Test Integration

```cpp
// Pattern 1: Simple RAII guard
TEST(MyTest, Simple) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(30));
    // Test code
}

// Pattern 2: EXPECT_EXIT (recommended for threading tests)
TEST(MyTest, WithExit) {
    auto test_body = []() {
        test_utils::TimeoutGuard guard(std::chrono::seconds(30));
        // Test code
        guard.notify();
        std::exit(0);
    };
    EXPECT_EXIT(test_body(), ::testing::ExitedWithCode(0), "");
}

// Pattern 3: Helper function
TEST(MyTest, Helper) {
    test_utils::RunWithTimeout([]() {
        // Test code
    }, std::chrono::seconds(30));
}
```

### 2. Google Benchmark Integration

```cpp
BENCHMARK_F(MyBenchmark, BM_Operation)(benchmark::State& state) {
    test_utils::BenchmarkTimeoutGuard guard(std::chrono::minutes(5));
    
    for (auto _ : state) {
        // Benchmark code
    }
}
```

### 3. CMake Integration

No changes required - header-only library automatically available:

```cmake
# Already in tests/unit/CMakeLists.txt
include_directories(../utils)
```

### 4. Build System Integration

```makefile
# Makefile - no changes needed
unit_test:
    $(SHOW)cd $(TESTDIR) && GTEST_COLOR=1 ctest $(_CTEST_ARGS)

benchmark:
    $(ROOT)/tests/benchmark/benchmarks.sh $(BM_FILTER) | xargs -I {} bash -lc \
        "$(BENCHMARKDIR)/bm_{} --benchmark_out_format=json ..."
```

## Usage Patterns

### Pattern 1: Simple Timeout (Most Common)

**Use Case**: Fast unit tests, simple operations

```cpp
TEST(MyTest, FastOperation) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(10));
    // Test code
}
```

**Pros**: Minimal boilerplate, automatic cleanup
**Cons**: Less control over exit behavior

### Pattern 2: EXPECT_EXIT (Recommended for Threading)

**Use Case**: Thread pools, parallel operations, complex async code

```cpp
TEST(MyTest, ThreadPool) {
    auto test_body = []() {
        test_utils::TimeoutGuard guard(std::chrono::seconds(100));
        // Thread pool test
        guard.notify();
        std::exit(0);
    };
    EXPECT_EXIT(test_body(), ::testing::ExitedWithCode(0), "Success");
}
```

**Pros**: Isolates threading issues, explicit exit control
**Cons**: Slightly more verbose

### Pattern 3: Custom Timeout Action

**Use Case**: Need custom logging, cleanup, or error handling

```cpp
TEST(MyTest, CustomAction) {
    test_utils::TimeoutGuard guard(
        std::chrono::seconds(30),
        []() {
            // Custom logging
            std::cerr << "Timeout in critical section!" << std::endl;
            // Cleanup
            cleanup_resources();
            std::abort();
        }
    );
    // Test code
}
```

**Pros**: Full control over timeout behavior
**Cons**: More complex

## Configuration Strategy

### Timeout Values by Test Type

| Test Type | Recommended Timeout | Rationale |
|-----------|-------------------|-----------|
| Fast unit tests | 10-30 seconds | Should complete quickly |
| Thread pool tests | 60-100 seconds | Thread synchronization overhead |
| Integration tests | 2-5 minutes | Multiple components involved |
| Benchmarks | 5-10 minutes | Many iterations |
| CI with sanitizers | 2-3x normal | Sanitizers add overhead |
| Valgrind | 3-5x normal | Valgrind is very slow |

### Environment-Aware Timeouts

```cpp
auto get_timeout() {
    auto base = std::chrono::seconds(30);
    
#ifdef RUNNING_ON_VALGRIND
    return base * 3;
#elif defined(USE_ASAN) || defined(USE_MSAN)
    return base * 2;
#else
    return base;
#endif
}

TEST(MyTest, Adaptive) {
    test_utils::TimeoutGuard guard(get_timeout());
    // Test code
}
```

## Comparison with Alternatives

### vs. CTest TIMEOUT

| Feature | TimeoutGuard | CTest TIMEOUT |
|---------|-------------|---------------|
| Granularity | Per-test or per-section | Per-test-executable |
| Cleanup | Can run cleanup code | Kills process (SIGTERM) |
| Logging | Custom logging possible | Limited |
| Portability | Cross-platform (C++) | CMake/CTest specific |
| Overhead | Minimal (one thread) | None |

**Recommendation**: Use both - CTest TIMEOUT as last resort, TimeoutGuard for graceful handling

### vs. Manual Thread + CV

| Feature | TimeoutGuard | Manual Implementation |
|---------|-------------|----------------------|
| Code reuse | ✅ Reusable | ❌ Copy-paste |
| RAII | ✅ Automatic | ❌ Manual cleanup |
| Error-prone | ✅ Low | ❌ High (easy to forget join) |
| Maintainability | ✅ High | ❌ Low |

**Recommendation**: Always use TimeoutGuard instead of manual implementation

### vs. std::async with timeout

| Feature | TimeoutGuard | std::async |
|---------|-------------|-----------|
| Purpose | Timeout protection | Async execution |
| Complexity | Simple | More complex |
| Control | Direct | Indirect (future) |
| Overhead | One thread | Thread pool |

**Recommendation**: TimeoutGuard is simpler for timeout-only use cases

## Migration Strategy

### Phase 1: Add Utility (Completed)
- ✅ Create `timeout_guard.h`
- ✅ Create documentation
- ✅ Create design document

### Phase 2: Migrate Existing Tests (Recommended)
1. Identify tests with manual timeout implementation
2. Replace with `TimeoutGuard`
3. Test in CI/CD pipeline
4. Update test documentation

### Phase 3: Adopt for New Tests (Ongoing)
1. Add to test template/guidelines
2. Code review checklist: "Does this test need timeout protection?"
3. Encourage use in complex tests

### Phase 4: Benchmark Integration (Optional)
1. Add to long-running benchmarks
2. Monitor for false positives
3. Adjust timeouts based on CI data

## Testing the Timeout Mechanism

### Unit Tests for TimeoutGuard

```cpp
TEST(TimeoutGuardTest, NoTimeout) {
    bool completed = false;
    {
        test_utils::TimeoutGuard guard(std::chrono::seconds(1));
        completed = true;
    }
    ASSERT_TRUE(completed);
}

TEST(TimeoutGuardTest, TimeoutTriggered) {
    auto test_body = []() {
        test_utils::TimeoutGuard guard(std::chrono::milliseconds(100));
        std::this_thread::sleep_for(std::chrono::seconds(10)); // Will timeout
        std::exit(0); // Should never reach here
    };
    EXPECT_EXIT(test_body(), ::testing::ExitedWithCode(-1), "timeout");
}

TEST(TimeoutGuardTest, ExplicitNotify) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(1));
    guard.notify();
    ASSERT_FALSE(guard.timed_out());
}
```

## Performance Considerations

### Overhead Analysis

- **Thread Creation**: ~1-10ms (one-time cost per guard)
- **Memory**: ~8KB per thread (stack size)
- **CPU**: Negligible (thread sleeps on condition variable)
- **Synchronization**: Minimal (mutex only on notify/destroy)

### Scalability

- **Per-Test Overhead**: Acceptable for unit tests
- **Benchmark Impact**: Minimal (guard outside iteration loop)
- **CI/CD Impact**: Positive (prevents hung tests)

### Optimization Opportunities

1. **Thread Pool**: Reuse threads across tests (complex, not recommended)
2. **Lazy Creation**: Only create thread if timeout > threshold (premature optimization)
3. **Static Timeout**: Compile-time timeout values (inflexible)

**Recommendation**: Current implementation is optimal for the use case

## Future Enhancements

### Potential Features

1. **Hierarchical Timeouts**: Parent timeout encompasses child timeouts
2. **Timeout Statistics**: Track which tests are close to timing out
3. **Dynamic Timeout Adjustment**: Based on historical test duration
4. **Timeout Warnings**: Warn at 80% of timeout before failing
5. **Python Bindings**: For pytest flow tests

### Python Integration (Future)

```python
# tests/utils/pytest_timeout.py
import pytest
import signal
from contextlib import contextmanager

@contextmanager
def timeout_guard(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Test timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Usage in pytest
def test_with_timeout():
    with timeout_guard(30):
        # Test code
        pass
```

## Conclusion

The `TimeoutGuard` utility provides a clean, reusable, and efficient mechanism for timeout protection in tests and benchmarks. It follows RAII principles, integrates seamlessly with existing test frameworks, and requires minimal changes to the build system.

### Key Benefits

- ✅ **Prevents hung tests** in CI/CD pipelines
- ✅ **Reusable** across all C++ tests and benchmarks
- ✅ **RAII-compliant** with automatic cleanup
- ✅ **Configurable** timeout actions
- ✅ **Thread-safe** implementation
- ✅ **Minimal overhead** (one thread per guard)
- ✅ **Easy to adopt** with clear migration path

### Recommended Next Steps

1. Review and approve this design
2. Migrate `testMockThreadPool` to use `TimeoutGuard` (demonstration)
3. Identify other tests that would benefit from timeout protection
4. Add timeout protection to new tests as needed
5. Monitor CI/CD for timeout-related improvements

