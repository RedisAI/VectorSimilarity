/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include "gtest/gtest.h"
#include "timeout_guard.h"
#include <chrono>

namespace test_utils {

/**
 * @brief Google Test Event Listener that adds timeout protection to every test.
 *
 * This listener automatically creates a TimeoutGuard for each test and destroys it
 * when the test completes. This provides global timeout protection without modifying
 * individual tests.
 */
class TimeoutTestListener : public testing::EmptyTestEventListener {
public:
    /**
     * @brief Construct a timeout listener with a default timeout for all tests.
     *
     * @param default_timeout Default timeout duration for all tests
     */
    template <typename Rep, typename Period>
    explicit TimeoutTestListener(std::chrono::duration<Rep, Period> default_timeout)
        : default_timeout_seconds_(
              std::chrono::duration_cast<std::chrono::seconds>(default_timeout).count()) {}

    /**
     * @brief Called before each test starts.
     */
    void OnTestStart(const testing::TestInfo &test_info) override {
        // Get timeout for this specific test (can be customized per test)
        auto timeout = GetTimeoutForTest(test_info);

        // Create timeout guard for this test
        current_guard_ = std::make_unique<TimeoutGuard>(timeout, [&test_info]() {
            std::cerr << "TIMEOUT: Test " << test_info.test_suite_name() << "."
                      << test_info.name() << " exceeded timeout!" << std::endl;
            std::exit(-1);
        });
    }

    /**
     * @brief Called after each test ends.
     */
    void OnTestEnd(const testing::TestInfo & /*test_info*/) override {
        // Notify and destroy the guard
        if (current_guard_) {
            current_guard_->notify();
            current_guard_.reset();
        }
    }

private:
    /**
     * @brief Get timeout duration for a specific test.
     *
     * This can be customized to return different timeouts based on test name,
     * test suite, or other criteria.
     */
    std::chrono::seconds GetTimeoutForTest(const testing::TestInfo &test_info) {
        // Default timeout
        auto timeout = std::chrono::seconds(default_timeout_seconds_);

        // Customize timeout based on test characteristics
        std::string test_name = test_info.name();
        std::string suite_name = test_info.test_suite_name();

        // Example: Longer timeout for thread pool tests
        if (suite_name.find("Thread") != std::string::npos ||
            test_name.find("thread") != std::string::npos ||
            test_name.find("Thread") != std::string::npos ||
            test_name.find("parallel") != std::string::npos ||
            test_name.find("Parallel") != std::string::npos) {
            timeout = std::chrono::seconds(100);
        }

        // Example: Longer timeout for tiered index tests
        if (suite_name.find("Tiered") != std::string::npos ||
            test_name.find("tiered") != std::string::npos) {
            timeout = std::chrono::seconds(120);
        }

        // Example: Longer timeout for SVS tests
        if (suite_name.find("SVS") != std::string::npos || suite_name.find("Svs") != std::string::npos) {
            timeout = std::chrono::seconds(150);
        }

#ifdef RUNNING_ON_VALGRIND
        // Triple timeout for Valgrind
        timeout *= 3;
#elif defined(USE_ASAN) || defined(USE_MSAN)
        // Double timeout for sanitizers
        timeout *= 2;
#endif

        return timeout;
    }

    int default_timeout_seconds_;
    std::unique_ptr<TimeoutGuard> current_guard_;
};

/**
 * @brief Helper function to register the timeout listener globally.
 *
 * Call this once in your test main() function.
 *
 * @param default_timeout Default timeout for all tests
 */
template <typename Rep, typename Period>
inline void RegisterGlobalTimeoutListener(std::chrono::duration<Rep, Period> default_timeout) {
    testing::TestEventListeners &listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new TimeoutTestListener(default_timeout));
}

} // namespace test_utils

