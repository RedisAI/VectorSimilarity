/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"

#include "VecSim/algorithms/tq/tq_flat.h"
#include "VecSim/vec_sim.h"
#include "tq_golden_fixture.h"

#include <cmath>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace {

VecSimParams CreateTQParams(size_t dim, VecSimMetric metric, size_t seed = 7,
                            bool use_rotation = true, size_t bits = 8,
                            size_t projections = 0) {
    TQFlatParams tq_params = {.type = VecSimType_FLOAT32,
                              .dim = dim,
                              .metric = metric,
                              .multi = false,
                              .initialCapacity = 0,
                              .blockSize = 4,
                              .bits = bits,
                              .projections = projections ? projections : std::max<size_t>(1, dim / 2),
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

std::vector<std::pair<size_t, double>> Range(VecSimIndex *index, const float *query, double radius) {
    auto *reply = VecSimIndex_RangeQuery(index, query, radius, nullptr, BY_SCORE);
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

struct OracleComparison {
    float inner_product_estimate;
    float l2_distance_estimate;
    float code_norm_sq;
};

inline float AllowedError(float expected, float abs_tolerance, float rel_tolerance) {
    return std::max(abs_tolerance, std::abs(expected) * rel_tolerance);
}

template <VecSimMetric Metric>
OracleComparison CompareAgainstOracle(const tq_golden_fixture::OracleCase &oracle_case) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto state = std::make_shared<TQFlatDetails::TQModelState>(
        oracle_case.dim, oracle_case.bits, oracle_case.projections, oracle_case.seed, true);
    TQFlatDetails::TQPreprocessor<Metric> preprocessor(allocator, state);

    void *storage_blob = nullptr;
    size_t storage_blob_size = oracle_case.dim * sizeof(float);
    preprocessor.preprocessForStorage(oracle_case.vector.data(), storage_blob, storage_blob_size);

    void *query_blob = nullptr;
    size_t query_blob_size = oracle_case.dim * sizeof(float);
    preprocessor.preprocessQuery(oracle_case.query.data(), query_blob, query_blob_size, 0);

    const auto storage_view = state->storageView(storage_blob);
    const auto query_view = state->queryView(query_blob);
    const float ip_estimate = state->estimateInnerProduct(storage_view, query_view);
    const float l2_estimate =
        std::max(query_view.query_norm_sq + storage_view.code_norm_sq - 2.0f * ip_estimate, 0.0f);

    allocator->free_allocation(storage_blob);
    allocator->free_allocation(query_blob);

    return {
        .inner_product_estimate = ip_estimate,
        .l2_distance_estimate = l2_estimate,
        .code_norm_sq = storage_view.code_norm_sq,
    };
}

} // namespace

TEST(TQFlatTest, cosine_search_prefers_exact_match) {
    VecSimParams params = CreateTQParams(16, VecSimMetric_Cosine, 7, true, 16, 64);
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_NE(index, nullptr);

    const float e1[16] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float e2[16] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float mix[16] = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

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
    EXPECT_EQ(info.dim, 16U);

    VecSimIndex_Free(index);
}

TEST(TQFlatTest, l2_search_update_delete_and_size_estimation) {
    VecSimParams params = CreateTQParams(64, VecSimMetric_L2, 9, true, 8, 16);
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_NE(index, nullptr);

    float near_left[64] = {0};
    float near_right[64] = {0};
    float query[64] = {0};
    float far_left[64] = {0};
    near_right[0] = 10.0f;
    query[0] = 9.0f;
    far_left[0] = -10.0f;

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

    const size_t raw_element_size = 64 * sizeof(float) + sizeof(labelType) + sizeof(void *);
    EXPECT_LT(VecSimIndex_EstimateElementSize(&params), raw_element_size);

    VecSimIndex_Free(index);
}

TEST(TQFlatTest, range_query_returns_close_match) {
    VecSimParams params = CreateTQParams(16, VecSimMetric_Cosine, 11, true, 16, 64);
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_NE(index, nullptr);

    const float near_left[16] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float near_right[16] = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float query[16] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    ASSERT_EQ(VecSimIndex_AddVector(index, near_left, 1), 1);
    ASSERT_EQ(VecSimIndex_AddVector(index, near_right, 2), 1);

    auto results = Range(index, query, 0.8);
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].first, 1U);
    EXPECT_EQ(results[1].first, 2U);

    VecSimIndex_Free(index);
}

TEST(TQFlatTest, rejects_odd_dimensions) {
    VecSimParams params = CreateTQParams(15, VecSimMetric_Cosine, 7, true, 8, 4);
    VecSimIndex *index = VecSimIndex_New(&params);
    EXPECT_EQ(index, nullptr);
}

TEST(TQFlatTest, oracle_parity_matches_rust_scores_within_tolerance) {
    for (const auto &oracle_case : tq_golden_fixture::kCases) {
        OracleComparison comparison =
            std::string_view(oracle_case.metric) == "cosine"
                ? CompareAgainstOracle<VecSimMetric_Cosine>(oracle_case)
                : CompareAgainstOracle<VecSimMetric_IP>(oracle_case);

        SCOPED_TRACE(oracle_case.name);

        EXPECT_NEAR(
            comparison.code_norm_sq, oracle_case.code_norm_sq,
            AllowedError(oracle_case.code_norm_sq, 1e-4f, 1e-4f));

        const float oracle_ip_error =
            std::abs(oracle_case.inner_product_estimate - oracle_case.exact_inner_product);
        const float comparison_ip_error =
            std::abs(comparison.inner_product_estimate - oracle_case.exact_inner_product);
        EXPECT_LE(
            comparison_ip_error,
            std::max(1.25f, oracle_ip_error * 2.0f + 0.5f));

        const float oracle_l2_error =
            std::abs(oracle_case.l2_distance_estimate - oracle_case.exact_l2_distance);
        const float comparison_l2_error =
            std::abs(comparison.l2_distance_estimate - oracle_case.exact_l2_distance);
        EXPECT_LE(
            comparison_l2_error,
            std::max(2.5f, oracle_l2_error * 2.0f + 0.75f));

        EXPECT_NEAR(
            comparison.inner_product_estimate, oracle_case.inner_product_estimate,
            AllowedError(oracle_case.inner_product_estimate, 5.0f, 0.75f));
        EXPECT_NEAR(
            comparison.l2_distance_estimate, oracle_case.l2_distance_estimate,
            AllowedError(oracle_case.l2_distance_estimate, 10.0f, 0.35f));
    }
}
