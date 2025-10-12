# Migration Example: testMockThreadPool

This document shows a side-by-side comparison of the original `testMockThreadPool` implementation and the refactored version using `TimeoutGuard`.

## Original Implementation

```cpp
TEST(UtilsTests, testMockThreadPool) {
    const size_t num_repeats = 2;
    const size_t num_submissions = 200;
    // 100 seconds timeout for the test should be enough for CI MemoryChecks
    std::chrono::seconds test_timeout(100);

    auto TestBody = [=]() {
        // Protection against test deadlock is implemented by a thread which exits process if
        // condition variable is not notified within a timeout.
        std::mutex mtx;
        std::condition_variable cv;
        auto guard_thread = std::thread([&]() {
            std::unique_lock<std::mutex> lock(mtx);
            if (cv.wait_for(lock, test_timeout) == std::cv_status::timeout) {
                std::cerr << "Test timeout! Exiting..." << std::endl;
                std::exit(-1);
            }
        });

        // Create and test a mock thread pool several times
        for (size_t i = 0; i < num_repeats; i++) {
            // Create a mock thread pool and verify its properties
            tieredIndexMock mock_thread_pool;
            ASSERT_EQ(mock_thread_pool.ctx->index_strong_ref, nullptr);
            ASSERT_TRUE(mock_thread_pool.jobQ.empty());

            // Create a new stub index to add to the mock thread pool
            BFParams params = {.dim = 4, .metric = VecSimMetric_L2};
            auto index = test_utils::CreateNewIndex(params, VecSimType_FLOAT32);
            mock_thread_pool.ctx->index_strong_ref.reset(index);
            auto allocator = index->getAllocator();

            // Very fast and simple job routine that increments a counter
            // This is just to simulate a job that does some work.
            std::atomic_int32_t job_counter = 0;
            auto job_mock = [&job_counter](AsyncJob * /*unused*/) { job_counter++; };

            // Define a mock job just to convert lambda with capture to a function pointer
            class LambdaJob : public AsyncJob {
            public:
                LambdaJob(std::shared_ptr<VecSimAllocator> allocator, JobType type,
                          std::function<void(AsyncJob *)> execute, VecSimIndex *index)
                    : AsyncJob(allocator, type, executeJob, index), impl_(execute) {}

                static void executeJob(AsyncJob *job) {
                    static_cast<LambdaJob *>(job)->impl_(job);
                    delete job; // Clean up the job after execution
                }
                std::function<void(AsyncJob *)> impl_;
            };

            mock_thread_pool.init_threads();
            // Verify the job queue is empty
            ASSERT_TRUE(mock_thread_pool.jobQ.empty());

            // Create a vector of jobs to submit to the mock thread pool
            // The number of jobs is equal to the thread pool size, so they will all be executed in
            // parallel
            std::vector<AsyncJob *> jobs(mock_thread_pool.thread_pool_size);

            // Submit jobs to the mock thread pool and wait several times
            for (size_t j = 0; j < num_submissions; j++) {
                job_counter.store(0); // Reset the counter for each iteration
                // Generate jobs and submit them to the mock thread pool
                std::generate(jobs.begin(), jobs.end(), [&]() {
                    return new (allocator) LambdaJob(allocator, HNSW_SEARCH_JOB, job_mock, index);
                });
                mock_thread_pool.submit_callback_internal(jobs.data(), nullptr /*unused*/,
                                                          jobs.size());
                mock_thread_pool.thread_pool_wait();
                // Verify the job queue is empty
                ASSERT_TRUE(mock_thread_pool.jobQ.empty());
                // Verify counter was incremented
                ASSERT_EQ(job_counter.load(), mock_thread_pool.thread_pool_size);
            }
            mock_thread_pool.thread_pool_join();
        }

        // Notify the guard thread that the test is done
        cv.notify_one();
        guard_thread.join();
        std::cerr << "Success" << std::endl;
        std::exit(testing::Test::HasFailure() ? -1 : 0); // Exit with failure if any test failed
    };

    EXPECT_EXIT(TestBody(), ::testing::ExitedWithCode(0), "Success");
}
```

## Refactored Implementation with TimeoutGuard

```cpp
#include "timeout_guard.h"

TEST(UtilsTests, testMockThreadPool) {
    const size_t num_repeats = 2;
    const size_t num_submissions = 200;

    auto TestBody = [=]() {
        // Protection against test deadlock using TimeoutGuard
        // 100 seconds timeout for the test should be enough for CI MemoryChecks
        test_utils::TimeoutGuard guard(std::chrono::seconds(100));

        // Create and test a mock thread pool several times
        for (size_t i = 0; i < num_repeats; i++) {
            // Create a mock thread pool and verify its properties
            tieredIndexMock mock_thread_pool;
            ASSERT_EQ(mock_thread_pool.ctx->index_strong_ref, nullptr);
            ASSERT_TRUE(mock_thread_pool.jobQ.empty());

            // Create a new stub index to add to the mock thread pool
            BFParams params = {.dim = 4, .metric = VecSimMetric_L2};
            auto index = test_utils::CreateNewIndex(params, VecSimType_FLOAT32);
            mock_thread_pool.ctx->index_strong_ref.reset(index);
            auto allocator = index->getAllocator();

            // Very fast and simple job routine that increments a counter
            // This is just to simulate a job that does some work.
            std::atomic_int32_t job_counter = 0;
            auto job_mock = [&job_counter](AsyncJob * /*unused*/) { job_counter++; };

            // Define a mock job just to convert lambda with capture to a function pointer
            class LambdaJob : public AsyncJob {
            public:
                LambdaJob(std::shared_ptr<VecSimAllocator> allocator, JobType type,
                          std::function<void(AsyncJob *)> execute, VecSimIndex *index)
                    : AsyncJob(allocator, type, executeJob, index), impl_(execute) {}

                static void executeJob(AsyncJob *job) {
                    static_cast<LambdaJob *>(job)->impl_(job);
                    delete job; // Clean up the job after execution
                }
                std::function<void(AsyncJob *)> impl_;
            };

            mock_thread_pool.init_threads();
            // Verify the job queue is empty
            ASSERT_TRUE(mock_thread_pool.jobQ.empty());

            // Create a vector of jobs to submit to the mock thread pool
            // The number of jobs is equal to the thread pool size, so they will all be executed in
            // parallel
            std::vector<AsyncJob *> jobs(mock_thread_pool.thread_pool_size);

            // Submit jobs to the mock thread pool and wait several times
            for (size_t j = 0; j < num_submissions; j++) {
                job_counter.store(0); // Reset the counter for each iteration
                // Generate jobs and submit them to the mock thread pool
                std::generate(jobs.begin(), jobs.end(), [&]() {
                    return new (allocator) LambdaJob(allocator, HNSW_SEARCH_JOB, job_mock, index);
                });
                mock_thread_pool.submit_callback_internal(jobs.data(), nullptr /*unused*/,
                                                          jobs.size());
                mock_thread_pool.thread_pool_wait();
                // Verify the job queue is empty
                ASSERT_TRUE(mock_thread_pool.jobQ.empty());
                // Verify counter was incremented
                ASSERT_EQ(job_counter.load(), mock_thread_pool.thread_pool_size);
            }
            mock_thread_pool.thread_pool_join();
        }

        // Notify the guard that the test completed successfully
        guard.notify();
        std::cerr << "Success" << std::endl;
        std::exit(testing::Test::HasFailure() ? -1 : 0);
    };

    EXPECT_EXIT(TestBody(), ::testing::ExitedWithCode(0), "Success");
}
```

## Key Changes

### Lines Removed (11 lines)
```cpp
// REMOVED: Manual timeout implementation
std::mutex mtx;
std::condition_variable cv;
auto guard_thread = std::thread([&]() {
    std::unique_lock<std::mutex> lock(mtx);
    if (cv.wait_for(lock, test_timeout) == std::cv_status::timeout) {
        std::cerr << "Test timeout! Exiting..." << std::endl;
        std::exit(-1);
    }
});

// REMOVED: Manual cleanup
cv.notify_one();
guard_thread.join();
```

### Lines Added (3 lines)
```cpp
// ADDED: Include header
#include "timeout_guard.h"

// ADDED: Create timeout guard (replaces 9 lines)
test_utils::TimeoutGuard guard(std::chrono::seconds(100));

// ADDED: Notify guard (replaces 2 lines)
guard.notify();
```

## Benefits of Refactoring

### Code Quality
- **-8 lines of code** (11 removed, 3 added)
- **Clearer intent**: `TimeoutGuard` name is self-documenting
- **Less error-prone**: No manual thread management
- **Reusable**: Same pattern can be used in other tests

### Maintainability
- **Single responsibility**: Timeout logic separated from test logic
- **Easier to modify**: Change timeout behavior in one place
- **Consistent**: All tests use the same timeout mechanism

### Safety
- **RAII compliance**: Automatic cleanup even if exceptions occur
- **Thread safety**: Mutex and CV properly managed
- **No resource leaks**: Guard thread always joined

## Performance Impact

**None** - The refactored version has identical runtime behavior:
- Same number of threads created (1 guard thread)
- Same synchronization primitives (mutex + condition variable)
- Same timeout mechanism (condition variable wait_for)

## Testing the Migration

### Before Migration
```bash
make unit_test CTEST_ARGS="--gtest_filter=UtilsTests.testMockThreadPool"
```

### After Migration
```bash
make unit_test CTEST_ARGS="--gtest_filter=UtilsTests.testMockThreadPool"
```

Expected output should be identical.

## Rollout Strategy

### Step 1: Add TimeoutGuard (No Changes to Existing Tests)
- Add `timeout_guard.h` to `tests/utils/`
- No impact on existing tests
- Can be tested independently

### Step 2: Migrate One Test (Proof of Concept)
- Migrate `testMockThreadPool` as shown above
- Run full test suite to ensure no regressions
- Verify CI/CD passes

### Step 3: Identify Other Candidates
Search for similar patterns:
```bash
grep -r "std::condition_variable" tests/unit/*.cpp
grep -r "wait_for.*timeout" tests/unit/*.cpp
grep -r "EXPECT_EXIT.*timeout" tests/unit/*.cpp
```

### Step 4: Gradual Migration
- Migrate tests one at a time
- Test each migration independently
- Update documentation

### Step 5: Adopt for New Tests
- Add to test guidelines
- Use in all new tests that need timeout protection

## Additional Examples

### Example 1: Simple Test (No EXPECT_EXIT)

**Before:**
```cpp
TEST(MyTest, Simple) {
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;
    
    auto guard = std::thread([&]() {
        std::unique_lock<std::mutex> lock(mtx);
        if (cv.wait_for(lock, std::chrono::seconds(30)) == std::cv_status::timeout) {
            std::exit(-1);
        }
    });
    
    // Test code
    
    {
        std::lock_guard<std::mutex> lock(mtx);
        done = true;
    }
    cv.notify_one();
    guard.join();
}
```

**After:**
```cpp
TEST(MyTest, Simple) {
    test_utils::TimeoutGuard guard(std::chrono::seconds(30));
    
    // Test code
    
    // Guard automatically notified on scope exit
}
```

### Example 2: Benchmark

**Before:**
```cpp
BENCHMARK_F(MyBenchmark, BM_Slow)(benchmark::State& state) {
    // No timeout protection - could hang forever
    for (auto _ : state) {
        slow_operation();
    }
}
```

**After:**
```cpp
BENCHMARK_F(MyBenchmark, BM_Slow)(benchmark::State& state) {
    test_utils::BenchmarkTimeoutGuard guard(std::chrono::minutes(5));
    
    for (auto _ : state) {
        slow_operation();
    }
}
```

## Conclusion

The migration to `TimeoutGuard` is:
- ✅ **Simple**: Minimal code changes
- ✅ **Safe**: No behavior changes
- ✅ **Beneficial**: Cleaner, more maintainable code
- ✅ **Low-risk**: Can be done incrementally
- ✅ **Testable**: Each migration can be verified independently

The refactored code is easier to read, maintain, and reuse across the codebase.

