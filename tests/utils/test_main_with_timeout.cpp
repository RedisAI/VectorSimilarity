/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"
#include "timeout_test_environment.h"
#include <chrono>

/**
 * @brief Custom main function that registers global timeout protection for all tests.
 *
 * This replaces gtest_main and adds automatic timeout protection to every test.
 * Tests will automatically timeout after 30 seconds (or customized duration based on test type).
 */
int main(int argc, char **argv) {
    // Initialize Google Test
    testing::InitGoogleTest(&argc, argv);

    // Register global timeout listener with 30 second default timeout
    // Individual tests can have different timeouts based on their characteristics
    // (see TimeoutTestListener::GetTimeoutForTest for customization logic)
    test_utils::RegisterGlobalTimeoutListener(std::chrono::seconds(300));

    // Run all tests
    return RUN_ALL_TESTS();
}

