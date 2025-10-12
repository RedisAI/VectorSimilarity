/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

namespace test_utils {

/**
 * @brief A RAII-style timeout guard that forcefully terminates execution if a timeout is reached.
 *
 * This class creates a background thread that monitors execution time. If the timeout expires
 * before the guard is destroyed (or explicitly notified), it will call the configured action
 * (default: exit the process with error code).
 *
 * Usage patterns:
 *
 * 1. Simple timeout with default exit behavior:
 *    {
 *        TimeoutGuard guard(std::chrono::seconds(30));
 *        // ... test code ...
 *    } // Guard destroyed, timeout cancelled
 *
 * 2. Explicit notification (for complex control flow):
 *    TimeoutGuard guard(std::chrono::seconds(30));
 *    // ... test code ...
 *    guard.notify(); // Cancel timeout early
 *
 * 3. Custom timeout action:
 *    TimeoutGuard guard(std::chrono::seconds(30), []() {
 *        std::cerr << "Custom timeout handler!" << std::endl;
 *        std::abort();
 *    });
 *
 * 4. With Google Test EXPECT_EXIT:
 *    auto test_body = []() {
 *        TimeoutGuard guard(std::chrono::seconds(30));
 *        // ... test code ...
 *        guard.notify();
 *        std::exit(0);
 *    };
 *    EXPECT_EXIT(test_body(), ::testing::ExitedWithCode(0), "");
 */
class TimeoutGuard {
public:
    /**
     * @brief Construct a timeout guard with specified duration and optional custom action.
     *
     * @param timeout Duration to wait before triggering timeout action
     * @param on_timeout Custom action to execute on timeout (default: exit with code -1)
     */
    template <typename Rep, typename Period>
    explicit TimeoutGuard(
        std::chrono::duration<Rep, Period> timeout,
        std::function<void()> on_timeout = []() {
            std::cerr << "TimeoutGuard: Test/Benchmark timeout! Exiting..." << std::endl;
            std::exit(-1);
        })
        : timeout_action_(std::move(on_timeout)), timed_out_(false), notified_(false) {

        guard_thread_ = std::thread([this, timeout]() {
            std::unique_lock<std::mutex> lock(mutex_);
            if (cv_.wait_for(lock, timeout) == std::cv_status::timeout) {
                timed_out_ = true;
                timeout_action_();
            }
        });
    }

    /**
     * @brief Destructor - notifies the guard thread and waits for it to finish.
     */
    ~TimeoutGuard() {
        if (!notified_) {
            notify();
        }
        if (guard_thread_.joinable()) {
            guard_thread_.join();
        }
    }

    // Delete copy and move operations to prevent misuse
    TimeoutGuard(const TimeoutGuard &) = delete;
    TimeoutGuard &operator=(const TimeoutGuard &) = delete;
    TimeoutGuard(TimeoutGuard &&) = delete;
    TimeoutGuard &operator=(TimeoutGuard &&) = delete;

    /**
     * @brief Explicitly notify the guard that execution completed successfully.
     *
     * This cancels the timeout and allows the guard thread to exit cleanly.
     * Safe to call multiple times.
     */
    void notify() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!notified_) {
            notified_ = true;
            cv_.notify_one();
        }
    }

    /**
     * @brief Check if the timeout was triggered.
     *
     * @return true if timeout occurred, false otherwise
     */
    bool timed_out() const { return timed_out_; }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread guard_thread_;
    std::function<void()> timeout_action_;
    bool timed_out_;
    bool notified_;
};

/**
 * @brief Helper macro for wrapping test bodies with timeout protection.
 *
 * This macro is useful for tests that need timeout protection with EXPECT_EXIT.
 *
 * Usage:
 *   TEST(MyTest, TestWithTimeout) {
 *       auto test_body = []() {
 *           TIMEOUT_GUARD_SCOPE(std::chrono::seconds(30)) {
 *               // ... test code ...
 *           }
 *       };
 *       EXPECT_EXIT(test_body(), ::testing::ExitedWithCode(0), "Success");
 *   }
 */
#define TIMEOUT_GUARD_SCOPE(duration)                                                              \
    test_utils::TimeoutGuard timeout_guard_##__LINE__(                                             \
        duration, []() {                                                                           \
            std::cerr << "Test timeout at " << __FILE__ << ":" << __LINE__ << std::endl;          \
            std::exit(-1);                                                                         \
        });                                                                                        \
    if (true)

/**
 * @brief Helper function to wrap a test body with timeout and EXPECT_EXIT.
 *
 * This is a convenience function that combines TimeoutGuard with EXPECT_EXIT pattern.
 *
 * @param test_body The test function to execute
 * @param timeout Duration to wait before timeout
 * @param success_message Expected message on success (for EXPECT_EXIT)
 *
 * Usage:
 *   TEST(MyTest, TestWithTimeout) {
 *       RUN_WITH_TIMEOUT([]() {
 *           // ... test code ...
 *       }, std::chrono::seconds(30), "Success");
 *   }
 */
template <typename TestFunc, typename Rep, typename Period>
inline void RunWithTimeout(TestFunc test_body, std::chrono::duration<Rep, Period> timeout,
                           const char *success_message = "Success") {
    auto wrapped_body = [test_body, timeout, success_message]() {
        TimeoutGuard guard(timeout);
        test_body();
        guard.notify();
        std::cerr << success_message << std::endl;
        std::exit(testing::Test::HasFailure() ? -1 : 0);
    };

#ifdef GTEST_API_
    EXPECT_EXIT(wrapped_body(), ::testing::ExitedWithCode(0), success_message);
#else
    // If not using GTest, just run the body directly
    wrapped_body();
#endif
}

/**
 * @brief Scoped timeout guard for benchmarks.
 *
 * Unlike the test version, this doesn't use EXPECT_EXIT and is suitable for
 * Google Benchmark fixtures.
 *
 * Usage in benchmark:
 *   BENCHMARK_F(MyBenchmark, BM_Operation)(benchmark::State& state) {
 *       BenchmarkTimeoutGuard guard(std::chrono::minutes(5));
 *       for (auto _ : state) {
 *           // ... benchmark code ...
 *       }
 *   }
 */
class BenchmarkTimeoutGuard : public TimeoutGuard {
public:
    template <typename Rep, typename Period>
    explicit BenchmarkTimeoutGuard(std::chrono::duration<Rep, Period> timeout)
        : TimeoutGuard(timeout, []() {
              std::cerr << "BenchmarkTimeoutGuard: Benchmark timeout! Exiting..." << std::endl;
              std::exit(-1);
          }) {}
};

} // namespace test_utils

