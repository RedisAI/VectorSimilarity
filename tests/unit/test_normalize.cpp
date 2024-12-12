/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <random> //TODO: remove once callinng populate_int8_vec

#include "gtest/gtest.h"
#include "VecSim/spaces/normalize/compute_norm.h"
class NormalizeTest : public ::testing::Test {};

TEST_F(NormalizeTest, TestINT8ComputeNorm) {
    size_t dim = 4;
    int8_t v[] = {-68, -100, 24, 127};
    float expected_norm = 177.0; // manually calculated

    float norm = spaces::IntegralType_ComputeNorm<int8_t>(v, dim);

    ASSERT_EQ(norm, expected_norm);
}
