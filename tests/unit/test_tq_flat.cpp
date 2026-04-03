/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"

#include "VecSim/vec_sim.h"

#include <utility>
#include <vector>

namespace {

VecSimParams CreateTQParams(size_t dim, VecSimMetric metric, size_t seed = 7,
                            bool use_rotation = true) {
    TQFlatParams tq_params = {.type = VecSimType_FLOAT32,
                              .dim = dim,
                              .metric = metric,
                              .multi = false,
                              .initialCapacity = 0,
                              .blockSize = 4,
                              .seed = seed,
                              .useRotation = use_rotation};
    return VecSimParams{.algo = VecSimAlgo_TQ, .algoParams = {.tqFlatParams = tq_params}};
}

std::vector<std::pair<size_t, double>> TopK(VecSimIndex *index, const float *query, size_t k) {
    auto *reply = VecSimIndex_TopKQuery(index, query, k, nullptr, BY_SCORE);
    auto *it = VecSimQueryReply_GetIterator(reply);
    std::vector<std::pair<size_t, double>> results;
    for (size_t i = 0; i < VecSimQueryReply_Len(reply); ++i) {
        auto *result = VecSimQueryReply_IteratorNext(it);
        results.emplace_back(VecSimQueryResult_GetId(result), VecSimQueryResult_GetScore(result));
    }
    VecSimQueryReply_IteratorFree(it);
    VecSimQueryReply_Free(reply);
    return results;
}

} // namespace

TEST(TQFlatTest, cosine_search_prefers_exact_match) {
    VecSimParams params = CreateTQParams(8, VecSimMetric_Cosine);
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_NE(index, nullptr);

    const float e1[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    const float e2[8] = {0, 1, 0, 0, 0, 0, 0, 0};
    const float mix[8] = {1, 1, 0, 0, 0, 0, 0, 0};

    ASSERT_EQ(VecSimIndex_AddVector(index, e1, 1), 1);
    ASSERT_EQ(VecSimIndex_AddVector(index, e2, 2), 1);
    ASSERT_EQ(VecSimIndex_AddVector(index, mix, 3), 1);

    auto results = TopK(index, e1, 3);
    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].first, 1U);
    EXPECT_EQ(results[1].first, 3U);
    EXPECT_EQ(results[2].first, 2U);

    auto info = VecSimIndex_BasicInfo(index);
    EXPECT_EQ(info.algo, VecSimAlgo_TQ);
    EXPECT_EQ(info.type, VecSimType_FLOAT32);
    EXPECT_EQ(info.dim, 8U);

    VecSimIndex_Free(index);
}

TEST(TQFlatTest, l2_search_update_delete_and_size_estimation) {
    VecSimParams params = CreateTQParams(8, VecSimMetric_L2);
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_NE(index, nullptr);

    const float near_left[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const float near_right[8] = {10, 0, 0, 0, 0, 0, 0, 0};
    const float query[8] = {9, 0, 0, 0, 0, 0, 0, 0};
    const float far_left[8] = {-10, 0, 0, 0, 0, 0, 0, 0};

    ASSERT_EQ(VecSimIndex_AddVector(index, near_left, 1), 1);
    ASSERT_EQ(VecSimIndex_AddVector(index, near_right, 2), 1);

    auto results = TopK(index, query, 2);
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].first, 2U);

    ASSERT_EQ(VecSimIndex_AddVector(index, far_left, 1), 0);
    results = TopK(index, query, 2);
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].first, 2U);
    EXPECT_EQ(results[1].first, 1U);

    ASSERT_EQ(VecSimIndex_DeleteVector(index, 2), 1);
    EXPECT_EQ(VecSimIndex_IndexSize(index), 1U);
    results = TopK(index, query, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].first, 1U);

    const size_t raw_element_size = 8 * sizeof(float) + sizeof(labelType) + sizeof(void *);
    EXPECT_LT(VecSimIndex_EstimateElementSize(&params), raw_element_size);

    VecSimIndex_Free(index);
}

TEST(TQFlatTest, rejects_non_power_of_two_rotation) {
    VecSimParams params = CreateTQParams(6, VecSimMetric_Cosine);
    VecSimIndex *index = VecSimIndex_New(&params);
    EXPECT_EQ(index, nullptr);
}
