/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"

#include "VecSim/algorithms/hnsw/hnsw_serializer.h"
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
                            bool use_rotation = true, size_t bits = 8, size_t projections = 0) {
    TQFlatParams tq_params = {.type = VecSimType_FLOAT32,
                              .dim = dim,
                              .metric = metric,
                              .multi = false,
                              .initialCapacity = 0,
                              .blockSize = 4,
                              .bits = bits,
                              .projections =
                                  projections ? projections : std::max<size_t>(1, dim / 2),
                              .seed = seed,
                              .useRotation = use_rotation};
    return VecSimParams{.algo = VecSimAlgo_TQ, .algoParams = {.tqFlatParams = tq_params}};
}

VecSimParams CreateTQHNSWParams(size_t dim, VecSimMetric metric, size_t seed = 7,
                                bool use_rotation = true, size_t bits = 8, size_t projections = 0,
                                size_t m = 16, size_t ef_construction = 200,
                                size_t ef_runtime = 50) {
    TQHNSWParams tq_params = {
        .type = VecSimType_FLOAT32,
        .dim = dim,
        .metric = metric,
        .multi = false,
        .initialCapacity = 0,
        .blockSize = 4,
        .bits = bits,
        .projections = projections ? projections : std::max<size_t>(1, dim / 2),
        .seed = seed,
        .useRotation = use_rotation,
        .M = m,
        .efConstruction = ef_construction,
        .efRuntime = ef_runtime,
        .epsilon = 0.01,
    };
    return VecSimParams{.algo = VecSimAlgo_TQ_HNSW, .algoParams = {.tqHnswParams = tq_params}};
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

std::vector<std::pair<size_t, double>> Range(VecSimIndex *index, const float *query,
                                             double radius) {
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
    preprocessor.preprocessForStorage(oracle_case.vector.data(), storage_blob, storage_blob_size,
                                      0);

    void *query_blob = nullptr;
    size_t query_blob_size = oracle_case.dim * sizeof(float);
    preprocessor.preprocessQuery(oracle_case.query.data(), query_blob, query_blob_size, 0);

    const auto storage_view = state->storageView(storage_blob);
    const auto query_view = state->queryView(query_blob);
    const float ip_estimate = state->estimateInnerProduct(storage_view, query_view);
    const float l2_estimate = std::max(
        query_view.query_norm_sq + storage_view.full_vector_norm_sq - 2.0f * ip_estimate, 0.0f);

    allocator->free_allocation(storage_blob);
    allocator->free_allocation(query_blob);

    return {
        .inner_product_estimate = ip_estimate,
        .l2_distance_estimate = l2_estimate,
        .code_norm_sq = storage_view.code_norm_sq,
    };
}

void SetStoredNorms(const std::shared_ptr<TQFlatDetails::TQModelState> &state, void *storage_blob,
                    float full_vector_norm_sq, float code_norm_sq) {
    auto *bytes = static_cast<uint8_t *>(storage_blob);
    bytes += state->pairs * sizeof(float);
    *reinterpret_cast<float *>(bytes) = full_vector_norm_sq;
    bytes += sizeof(float);
    *reinterpret_cast<float *>(bytes) = code_norm_sq;
}

std::vector<float> MakeSignal(size_t dim, float phase) {
    std::vector<float> values(dim);
    for (size_t i = 0; i < dim; ++i) {
        const float idx = static_cast<float>(i + 1);
        values[i] = std::sin(idx * 0.31f + phase) + 0.5f * std::cos(idx * 0.17f - phase * 0.5f) +
                    0.05f * static_cast<float>(static_cast<int>(i % 7) - 3);
    }
    return values;
}

template <VecSimMetric Metric>
std::vector<float> NormalizeForMetric(std::vector<float> values) {
    if constexpr (Metric == VecSimMetric_Cosine) {
        float norm_sq = 0.0f;
        for (float value : values) {
            norm_sq += value * value;
        }
        const float inv_norm = 1.0f / std::sqrt(norm_sq);
        for (float &value : values) {
            value *= inv_norm;
        }
    }
    return values;
}

inline bool PackedSignAt(const uint8_t *packed_signs, size_t projection_idx) {
    return (packed_signs[projection_idx / 8] & static_cast<uint8_t>(1u << (projection_idx % 8))) !=
           0;
}

inline float AngleFromCode(const TQFlatDetails::TQModelState &state, uint16_t angle_code) {
    return (static_cast<float>(angle_code) / static_cast<float>(state.levels)) *
               (2.0f * TQFlatDetails::kPi) -
           TQFlatDetails::kPi;
}

template <VecSimMetric Metric>
float ScalarEstimateInnerProduct(const TQFlatDetails::TQModelState &state,
                                 const TQFlatDetails::StorageView &storage,
                                 const std::vector<float> &query_vector) {
    auto normalized_query = NormalizeForMetric<Metric>(query_vector);
    std::vector<float> rotated_query(state.dim);
    std::vector<float> qjl_query_dots(state.projections);
    state.applyRotation(normalized_query.data(), rotated_query.data());
    state.projectQjl(normalized_query.data(), qjl_query_dots.data());

    float polar_estimate = 0.0f;
    for (size_t i = 0; i < state.pairs; ++i) {
        const uint16_t angle_code = state.angleCodeAt(storage, i);
        const float theta = AngleFromCode(state, angle_code);
        polar_estimate += storage.radii[i] * (rotated_query[2 * i] * std::cos(theta) +
                                              rotated_query[2 * i + 1] * std::sin(theta));
    }

    float qjl_estimate = 0.0f;
    for (size_t projection_idx = 0; projection_idx < state.projections; ++projection_idx) {
        qjl_estimate += PackedSignAt(storage.residual_signs, projection_idx)
                            ? qjl_query_dots[projection_idx]
                            : -qjl_query_dots[projection_idx];
    }

    return polar_estimate + state.qjl_scale * qjl_estimate;
}

float ScalarEstimateInnerProductSymmetric(const TQFlatDetails::TQModelState &state,
                                          const TQFlatDetails::StorageView &lhs,
                                          const TQFlatDetails::StorageView &rhs) {
    float polar_estimate = 0.0f;
    for (size_t i = 0; i < state.pairs; ++i) {
        const uint16_t lhs_angle = state.angleCodeAt(lhs, i);
        const uint16_t rhs_angle = state.angleCodeAt(rhs, i);
        const size_t delta = (static_cast<size_t>(lhs_angle) - static_cast<size_t>(rhs_angle)) &
                             state.angle_delta_mask;
        const float delta_theta = (static_cast<float>(delta) / static_cast<float>(state.levels)) *
                                  (2.0f * TQFlatDetails::kPi);
        polar_estimate += lhs.radii[i] * rhs.radii[i] * std::cos(delta_theta);
    }

    int sign_dot = 0;
    for (size_t projection_idx = 0; projection_idx < state.projections; ++projection_idx) {
        sign_dot += PackedSignAt(lhs.residual_signs, projection_idx) ==
                            PackedSignAt(rhs.residual_signs, projection_idx)
                        ? 1
                        : -1;
    }

    return polar_estimate + state.qjl_scale * static_cast<float>(sign_dot);
}

float ScalarDotProduct(const float *lhs, const float *rhs, size_t dim) {
    float sum = 0.0f;
    for (size_t idx = 0; idx < dim; ++idx) {
        sum += lhs[idx] * rhs[idx];
    }
    return sum;
}

float ScalarSumSquares(const float *values, size_t dim) {
    return ScalarDotProduct(values, values, dim);
}

std::vector<uint8_t> DecodeAngleCodes(const TQFlatDetails::TQModelState &state,
                                      const TQFlatDetails::StorageView &storage) {
    std::vector<uint8_t> decoded(state.pairs);
    for (size_t idx = 0; idx < state.pairs; ++idx) {
        decoded[idx] = static_cast<uint8_t>(state.angleCodeAt(storage, idx));
    }
    return decoded;
}

std::vector<float> BuildDeltaCosLut(size_t levels) {
    std::vector<float> delta_cos_lut(levels);
    for (size_t idx = 0; idx < levels; ++idx) {
        delta_cos_lut[idx] = std::cos((static_cast<float>(idx) / static_cast<float>(levels)) *
                                      (2.0f * TQFlatDetails::kPi));
    }
    return delta_cos_lut;
}

template <typename Features>
int RoutedPackedResidualSignDot(const TQFlatDetails::TQModelState &state,
                                const TQFlatDetails::StorageView &lhs,
                                const TQFlatDetails::StorageView &rhs,
                                const Features &optimization) {
    if (auto impl = spaces::Choose_TQ_PackedResidualSignDot_implementation(state.projections,
                                                                           &optimization)) {
        return impl(lhs.residual_signs, rhs.residual_signs, state.projections);
    }
    return spaces::TQ_PackedSignDot(lhs.residual_signs, rhs.residual_signs, state.projections);
}

template <typename Features>
float RoutedSymmetricPolarEstimate(const TQFlatDetails::TQModelState &state,
                                   const TQFlatDetails::StorageView &lhs,
                                   const TQFlatDetails::StorageView &rhs,
                                   const Features &optimization) {
    if (state.compactAngles()) {
        if (auto impl =
                spaces::Choose_TQ_SymmetricPolar_implementation(state.pairs, &optimization)) {
            const auto lhs_angles = DecodeAngleCodes(state, lhs);
            const auto rhs_angles = DecodeAngleCodes(state, rhs);
            const auto delta_cos_lut = BuildDeltaCosLut(state.levels);
            return impl(lhs.radii, lhs_angles.data(), rhs.radii, rhs_angles.data(),
                        delta_cos_lut.data(), static_cast<uint8_t>(state.angle_delta_mask),
                        state.pairs);
        }
    }

    float polar_estimate = 0.0f;
    for (size_t idx = 0; idx < state.pairs; ++idx) {
        const uint16_t lhs_angle = state.angleCodeAt(lhs, idx);
        const uint16_t rhs_angle = state.angleCodeAt(rhs, idx);
        const size_t delta = (static_cast<size_t>(lhs_angle) - static_cast<size_t>(rhs_angle)) &
                             state.angle_delta_mask;
        polar_estimate += lhs.radii[idx] * rhs.radii[idx] *
                          std::cos((static_cast<float>(delta) / static_cast<float>(state.levels)) *
                                   (2.0f * TQFlatDetails::kPi));
    }
    return polar_estimate;
}

template <typename Features>
float RoutedEstimateInnerProductSymmetric(const TQFlatDetails::TQModelState &state,
                                          const TQFlatDetails::StorageView &lhs,
                                          const TQFlatDetails::StorageView &rhs,
                                          const Features &optimization) {
    const float polar_estimate = RoutedSymmetricPolarEstimate(state, lhs, rhs, optimization);
    const int sign_dot = RoutedPackedResidualSignDot(state, lhs, rhs, optimization);
    return polar_estimate + state.qjl_scale * static_cast<float>(sign_dot);
}

template <typename Features>
bool HasOptimizedTQFP32Path(const Features &optimization) {
#ifdef OPT_AVX512F
    if (optimization.avx512f) {
        return true;
    }
#endif
#ifdef OPT_AVX
    if (optimization.avx) {
        return true;
    }
#endif
#ifdef OPT_SSE
    if (optimization.sse) {
        return true;
    }
#endif
#ifdef OPT_SVE2
    if (optimization.sve2) {
        return true;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        return true;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        return true;
    }
#endif
    return false;
}

template <typename Features>
size_t PackedResidualSelectorThreshold(const Features &optimization) {
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        return 64 * 8;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        return 32 * 8;
    }
#endif
#ifdef OPT_SSE4
    if (optimization.sse4_1) {
        return 16 * 8;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        return 16 * 8;
    }
#endif
    return 0;
}

template <typename Features>
size_t SymmetricPolarSelectorThreshold(const Features &optimization) {
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        return 16;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        return 8;
    }
#endif
#ifdef OPT_SSE4
    if (optimization.sse4_1) {
        return 4;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        return 8;
    }
#endif
    return 0;
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

TEST(TQFlatTest, cosine_scores_use_standard_one_minus_cosine_scale) {
    VecSimParams params = CreateTQParams(16, VecSimMetric_Cosine, 7, true, 8, 256);
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_NE(index, nullptr);

    const float e1[16] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float e2[16] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float negative_e1[16] = {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    ASSERT_EQ(VecSimIndex_AddVector(index, e1, 1), 1);
    ASSERT_EQ(VecSimIndex_AddVector(index, e2, 2), 1);
    ASSERT_EQ(VecSimIndex_AddVector(index, negative_e1, 3), 1);

    auto results = TopK(index, e1, 3);
    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].first, 1U);
    EXPECT_EQ(results[1].first, 2U);
    EXPECT_EQ(results[2].first, 3U);
    EXPECT_NEAR(results[0].second, 0.0, 0.1);
    EXPECT_NEAR(results[1].second, 1.0, 0.1);
    EXPECT_NEAR(results[2].second, 2.0, 0.1);

    auto close_results = Range(index, e1, 0.5);
    ASSERT_EQ(close_results.size(), 1);
    EXPECT_EQ(close_results[0].first, 1U);

    auto non_opposite_results = Range(index, e1, 1.5);
    ASSERT_EQ(non_opposite_results.size(), 2);
    EXPECT_EQ(non_opposite_results[0].first, 1U);
    EXPECT_EQ(non_opposite_results[1].first, 2U);

    VecSimIndex_Free(index);
}

TEST(TQFlatTest, tq_hnsw_cosine_search_prefers_exact_match) {
    VecSimParams params = CreateTQHNSWParams(16, VecSimMetric_Cosine, 7, true, 8, 64);
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

    auto info = VecSimIndex_BasicInfo(index);
    EXPECT_EQ(info.algo, VecSimAlgo_TQ_HNSW);
    EXPECT_EQ(info.type, VecSimType_FLOAT32);
    EXPECT_EQ(info.dim, 16U);
    EXPECT_FALSE(info.isTiered);

    VecSimIndex_Free(index);
}

TEST(TQFlatTest, tq_hnsw_rejects_unsupported_l2_and_serialization) {
    VecSimParams l2_params = CreateTQHNSWParams(16, VecSimMetric_L2, 7, true, 8, 64);
    EXPECT_EQ(VecSimIndex_New(&l2_params), nullptr);

    VecSimParams cosine_params = CreateTQHNSWParams(16, VecSimMetric_Cosine, 7, true, 8, 64);
    VecSimIndex *index = VecSimIndex_New(&cosine_params);
    ASSERT_NE(index, nullptr);
    auto *serializer = dynamic_cast<HNSWSerializer *>(index);
    ASSERT_NE(serializer, nullptr);
    EXPECT_THROW(serializer->saveIndex("unused-tq-hnsw-index"), std::runtime_error);
    VecSimIndex_Free(index);
}

TEST(TQFlatTest, low_bit_angle_codes_are_nibble_packed) {
    constexpr size_t dim = 18;
    constexpr size_t projections = 13;

    for (size_t bits : {size_t{2}, size_t{4}, size_t{5}}) {
        SCOPED_TRACE(testing::Message() << "bits=" << bits);
        auto state =
            std::make_shared<TQFlatDetails::TQModelState>(dim, bits, projections, 23, false);
        EXPECT_EQ(state->angleCodeBytes(), (state->pairs + 1) / 2);

        std::vector<uint16_t> angles(state->pairs);
        for (size_t i = 0; i < state->pairs; ++i) {
            angles[i] = static_cast<uint16_t>(i % state->levels);
        }
        std::vector<uint8_t> encoded(state->angleCodeBytes());
        state->writeAngleCodes(angles.data(), encoded.data());

        TQFlatDetails::StorageView storage = {
            .radii = nullptr,
            .full_vector_norm_sq = 0.0f,
            .code_norm_sq = 0.0f,
            .angle_indices = encoded.data(),
            .residual_signs = nullptr,
        };
        for (size_t i = 0; i < state->pairs; ++i) {
            EXPECT_EQ(state->angleCodeAt(storage, i), angles[i]);
        }
    }

    auto byte_state = std::make_shared<TQFlatDetails::TQModelState>(dim, 8, projections, 23, false);
    EXPECT_EQ(byte_state->angleCodeBytes(), byte_state->pairs);
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

TEST(TQFlatTest, l2_distance_uses_full_storage_norm_field) {
    constexpr size_t dim = 16;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto state = std::make_shared<TQFlatDetails::TQModelState>(dim, 8, 16, 17, true);
    TQFlatDetails::TQPreprocessor<VecSimMetric_L2> preprocessor(allocator, state);
    TQFlatDetails::TQDistanceCalculator<VecSimMetric_L2> calculator(allocator, state);

    const auto vector = MakeSignal(dim, 0.23f);
    const auto query = MakeSignal(dim, -0.41f);

    void *storage_blob = nullptr;
    size_t storage_blob_size = dim * sizeof(float);
    preprocessor.preprocessForStorage(vector.data(), storage_blob, storage_blob_size, 0);

    void *query_blob = nullptr;
    size_t query_blob_size = dim * sizeof(float);
    preprocessor.preprocessQuery(query.data(), query_blob, query_blob_size, 0);

    const auto original_storage = state->storageView(storage_blob);
    const auto query_view = state->queryView(query_blob);
    const float estimate = state->estimateInnerProduct(original_storage, query_view);
    const float forced_full_norm = original_storage.full_vector_norm_sq + 11.0f;
    const float forced_code_norm = original_storage.code_norm_sq * 0.25f + 0.5f;
    SetStoredNorms(state, storage_blob, forced_full_norm, forced_code_norm);

    const float actual = calculator.calcDistance(storage_blob, query_blob, dim);
    const float expected =
        std::max(query_view.query_norm_sq + forced_full_norm - 2.0f * estimate, 0.0f);
    const float wrong =
        std::max(query_view.query_norm_sq + forced_code_norm - 2.0f * estimate, 0.0f);

    ASSERT_GT(std::abs(expected - wrong), 1e-3f);
    EXPECT_NEAR(actual, expected, AllowedError(expected, 1e-5f, 1e-6f));

    allocator->free_allocation(storage_blob);
    allocator->free_allocation(query_blob);
}

TEST(TQFlatTest, l2_symmetric_distance_uses_full_storage_norm_fields) {
    constexpr size_t dim = 16;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto state = std::make_shared<TQFlatDetails::TQModelState>(dim, 8, 16, 19, true);
    TQFlatDetails::TQPreprocessor<VecSimMetric_L2> preprocessor(allocator, state);
    TQFlatDetails::TQSymmetricDistanceCalculator<VecSimMetric_L2> calculator(allocator, state);

    const auto lhs_vector = MakeSignal(dim, 0.11f);
    const auto rhs_vector = MakeSignal(dim, -0.37f);

    void *lhs_blob = nullptr;
    size_t lhs_blob_size = dim * sizeof(float);
    preprocessor.preprocessForStorage(lhs_vector.data(), lhs_blob, lhs_blob_size, 0);

    void *rhs_blob = nullptr;
    size_t rhs_blob_size = dim * sizeof(float);
    preprocessor.preprocessForStorage(rhs_vector.data(), rhs_blob, rhs_blob_size, 0);

    const auto lhs_view = state->storageView(lhs_blob);
    const auto rhs_view = state->storageView(rhs_blob);
    const float estimate = state->estimateInnerProductSymmetric(lhs_view, rhs_view);
    const float lhs_full_norm = lhs_view.full_vector_norm_sq + 7.0f;
    const float rhs_full_norm = rhs_view.full_vector_norm_sq + 9.0f;
    const float lhs_code_norm = lhs_view.code_norm_sq * 0.1f + 0.25f;
    const float rhs_code_norm = rhs_view.code_norm_sq * 0.2f + 0.5f;
    SetStoredNorms(state, lhs_blob, lhs_full_norm, lhs_code_norm);
    SetStoredNorms(state, rhs_blob, rhs_full_norm, rhs_code_norm);

    const float actual = calculator.calcDistance(lhs_blob, rhs_blob, dim);
    const float expected = std::max(lhs_full_norm + rhs_full_norm - 2.0f * estimate, 0.0f);
    const float wrong = std::max(lhs_code_norm + rhs_code_norm - 2.0f * estimate, 0.0f);

    ASSERT_GT(std::abs(expected - wrong), 1e-3f);
    EXPECT_NEAR(actual, expected, AllowedError(expected, 1e-5f, 1e-6f));

    allocator->free_allocation(lhs_blob);
    allocator->free_allocation(rhs_blob);
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

TEST(TQFlatTest, rejects_invalid_bit_budgets_and_projection_count) {
    for (size_t bits : {size_t{0}, size_t{1}, size_t{17}}) {
        SCOPED_TRACE(testing::Message() << "bits=" << bits);
        VecSimParams params = CreateTQParams(16, VecSimMetric_Cosine, 7, true, bits, 4);
        EXPECT_EQ(VecSimIndex_New(&params), nullptr);
        EXPECT_THROW(VecSimIndex_EstimateInitialSize(&params), std::invalid_argument);
    }

    VecSimParams params = CreateTQParams(16, VecSimMetric_Cosine, 7, true, 8, 0);
    params.algoParams.tqFlatParams.projections = 0;
    EXPECT_EQ(VecSimIndex_New(&params), nullptr);
    EXPECT_THROW(VecSimIndex_EstimateInitialSize(&params), std::invalid_argument);
}

TEST(TQFlatTest, fp32_helper_selector_thresholds_match_current_cpu_features) {
    auto optimization = spaces::getCpuOptimizationFeatures();

    EXPECT_EQ(spaces::Choose_FP32_InnerProduct_implementation_TQ(15, &optimization), nullptr);
    EXPECT_EQ(spaces::Choose_FP32_SumSquares_implementation_TQ(15, &optimization), nullptr);

    if (HasOptimizedTQFP32Path(optimization)) {
        EXPECT_NE(spaces::Choose_FP32_InnerProduct_implementation_TQ(16, &optimization), nullptr);
        EXPECT_NE(spaces::Choose_FP32_SumSquares_implementation_TQ(16, &optimization), nullptr);
    } else {
        EXPECT_EQ(spaces::Choose_FP32_InnerProduct_implementation_TQ(16, &optimization), nullptr);
        EXPECT_EQ(spaces::Choose_FP32_SumSquares_implementation_TQ(16, &optimization), nullptr);
    }
}

TEST(TQFlatTest, compact_helper_selector_thresholds_match_current_cpu_features) {
    auto optimization = spaces::getCpuOptimizationFeatures();

    EXPECT_EQ(spaces::Choose_TQ_PackedResidualSignDot_implementation(127, &optimization), nullptr);
    EXPECT_EQ(spaces::Choose_TQ_SymmetricPolar_implementation(3, &optimization), nullptr);

    const size_t packed_threshold = PackedResidualSelectorThreshold(optimization);
    if (packed_threshold != 0) {
        EXPECT_NE(
            spaces::Choose_TQ_PackedResidualSignDot_implementation(packed_threshold, &optimization),
            nullptr);
    } else {
        EXPECT_EQ(spaces::Choose_TQ_PackedResidualSignDot_implementation(128, &optimization),
                  nullptr);
    }

    const size_t polar_threshold = SymmetricPolarSelectorThreshold(optimization);
    if (polar_threshold != 0) {
        EXPECT_NE(spaces::Choose_TQ_SymmetricPolar_implementation(polar_threshold, &optimization),
                  nullptr);
    } else {
        EXPECT_EQ(spaces::Choose_TQ_SymmetricPolar_implementation(4, &optimization), nullptr);
    }
}

TEST(TQFlatTest, state_fp32_helpers_route_through_current_selectors) {
    auto optimization = spaces::getCpuOptimizationFeatures();

    struct RouteCase {
        size_t dim;
        size_t projections;
    };

    const std::vector<RouteCase> cases = {
        {.dim = 14, .projections = 13},
        {.dim = 16, .projections = 13},
        {.dim = 32, .projections = 13},
    };

    for (const auto &test_case : cases) {
        SCOPED_TRACE(testing::Message() << "dim=" << test_case.dim);

        auto state = std::make_shared<TQFlatDetails::TQModelState>(test_case.dim, 8,
                                                                   test_case.projections, 23, true);
        const auto lhs = MakeSignal(test_case.dim, 0.13f);
        const auto rhs = MakeSignal(test_case.dim, -0.41f);

        const auto dot_selector =
            spaces::Choose_FP32_InnerProduct_implementation_TQ(test_case.dim, &optimization);
        const float expected_dot = ScalarDotProduct(lhs.data(), rhs.data(), test_case.dim);
        if (dot_selector) {
            EXPECT_NEAR(dot_selector(lhs.data(), rhs.data(), test_case.dim), expected_dot,
                        AllowedError(expected_dot, 1e-5f, 1e-6f));
        }
        EXPECT_NEAR(state->dotProduct(lhs.data(), rhs.data(), test_case.dim), expected_dot,
                    AllowedError(expected_dot, 1e-5f, 1e-6f));

        const auto full_sum_selector =
            spaces::Choose_FP32_SumSquares_implementation_TQ(test_case.dim, &optimization);
        const float expected_full_sum = ScalarSumSquares(lhs.data(), test_case.dim);
        if (full_sum_selector) {
            EXPECT_NEAR(full_sum_selector(lhs.data(), test_case.dim), expected_full_sum,
                        AllowedError(expected_full_sum, 1e-5f, 1e-6f));
        }
        EXPECT_NEAR(state->sumSquares(lhs.data(), test_case.dim), expected_full_sum,
                    AllowedError(expected_full_sum, 1e-5f, 1e-6f));

        const auto pair_sum_selector =
            spaces::Choose_FP32_SumSquares_implementation_TQ(state->pairs, &optimization);
        const float expected_pair_sum = ScalarSumSquares(lhs.data(), state->pairs);
        if (pair_sum_selector) {
            EXPECT_NEAR(pair_sum_selector(lhs.data(), state->pairs), expected_pair_sum,
                        AllowedError(expected_pair_sum, 1e-5f, 1e-6f));
        }
        EXPECT_NEAR(state->sumSquares(lhs.data(), state->pairs), expected_pair_sum,
                    AllowedError(expected_pair_sum, 1e-5f, 1e-6f));
    }
}

TEST(TQFlatTest, oracle_parity_matches_rust_scores_within_tolerance) {
    for (const auto &oracle_case : tq_golden_fixture::kCases) {
        OracleComparison comparison = std::string_view(oracle_case.metric) == "cosine"
                                          ? CompareAgainstOracle<VecSimMetric_Cosine>(oracle_case)
                                          : CompareAgainstOracle<VecSimMetric_IP>(oracle_case);

        SCOPED_TRACE(oracle_case.name);

        EXPECT_NEAR(comparison.code_norm_sq, oracle_case.code_norm_sq,
                    AllowedError(oracle_case.code_norm_sq, 1e-4f, 1e-4f));

        const float oracle_ip_error =
            std::abs(oracle_case.inner_product_estimate - oracle_case.exact_inner_product);
        const float comparison_ip_error =
            std::abs(comparison.inner_product_estimate - oracle_case.exact_inner_product);
        EXPECT_LE(comparison_ip_error, std::max(1.25f, oracle_ip_error * 2.0f + 0.5f));

        const float oracle_l2_error =
            std::abs(oracle_case.l2_distance_estimate - oracle_case.exact_l2_distance);
        const float comparison_l2_error =
            std::abs(comparison.l2_distance_estimate - oracle_case.exact_l2_distance);
        EXPECT_LE(comparison_l2_error, std::max(2.5f, oracle_l2_error * 2.0f + 0.75f));

        EXPECT_NEAR(comparison.inner_product_estimate, oracle_case.inner_product_estimate,
                    AllowedError(oracle_case.inner_product_estimate, 5.0f, 0.75f));
        EXPECT_NEAR(comparison.l2_distance_estimate, oracle_case.l2_distance_estimate,
                    AllowedError(oracle_case.l2_distance_estimate, 10.0f, 0.35f));
    }
}

TEST(TQFlatTest, asymmetric_estimate_matches_scalar_reference_for_compact_and_fallback_paths) {
    constexpr size_t dim = 32;
    const auto storage_vector = MakeSignal(dim, 0.23f);
    const auto query_vector = MakeSignal(dim, -0.61f);

    struct EstimateCase {
        size_t bits;
        bool expect_compact_angles;
        bool expect_polar_lookup;
    };

    const std::vector<EstimateCase> cases = {
        {.bits = 2, .expect_compact_angles = true, .expect_polar_lookup = true},
        {.bits = 4, .expect_compact_angles = true, .expect_polar_lookup = true},
        {.bits = 7, .expect_compact_angles = true, .expect_polar_lookup = true},
        {.bits = 9, .expect_compact_angles = true, .expect_polar_lookup = true},
        {.bits = 10, .expect_compact_angles = false, .expect_polar_lookup = false},
    };

    for (size_t projections : {size_t{13}, size_t{257}}) {
        for (const auto &test_case : cases) {
            SCOPED_TRACE(testing::Message()
                         << "bits=" << test_case.bits << " projections=" << projections);

            auto allocator = VecSimAllocator::newVecsimAllocator();
            auto state = std::make_shared<TQFlatDetails::TQModelState>(dim, test_case.bits,
                                                                       projections, 31, true);
            TQFlatDetails::TQPreprocessor<VecSimMetric_Cosine> preprocessor(allocator, state);

            void *storage_blob = nullptr;
            size_t storage_blob_size = dim * sizeof(float);
            preprocessor.preprocessForStorage(storage_vector.data(), storage_blob,
                                              storage_blob_size, 0);

            void *query_blob = nullptr;
            size_t query_blob_size = dim * sizeof(float);
            preprocessor.preprocessQuery(query_vector.data(), query_blob, query_blob_size, 0);

            const auto storage_view = state->storageView(storage_blob);
            const auto query_view = state->queryView(query_blob);
            const float expected =
                ScalarEstimateInnerProduct<VecSimMetric_Cosine>(*state, storage_view, query_vector);
            const float actual = state->estimateInnerProduct(storage_view, query_view);

            EXPECT_EQ(state->compactAngles(), test_case.expect_compact_angles);
            EXPECT_EQ(state->usePolarLut(), test_case.expect_polar_lookup);
            EXPECT_NEAR(actual, expected, AllowedError(expected, 1e-5f, 1e-5f));

            allocator->free_allocation(storage_blob);
            allocator->free_allocation(query_blob);
        }
    }
}

TEST(TQFlatTest, symmetric_estimate_matches_scalar_reference_for_compact_and_fallback_paths) {
    constexpr size_t dim = 32;
    const auto lhs_vector = MakeSignal(dim, 0.11f);
    const auto rhs_vector = MakeSignal(dim, -0.47f);

    for (size_t projections : {size_t{13}, size_t{257}}) {
        for (size_t bits : {size_t{2}, size_t{4}, size_t{5}, size_t{7}, size_t{9}, size_t{10}}) {
            SCOPED_TRACE(testing::Message() << "bits=" << bits << " projections=" << projections);

            auto allocator = VecSimAllocator::newVecsimAllocator();
            auto state =
                std::make_shared<TQFlatDetails::TQModelState>(dim, bits, projections, 43, true);
            TQFlatDetails::TQPreprocessor<VecSimMetric_Cosine> preprocessor(allocator, state);
            TQFlatDetails::TQSymmetricDistanceCalculator<VecSimMetric_Cosine> calculator(allocator,
                                                                                         state);

            void *lhs_blob = nullptr;
            size_t lhs_blob_size = dim * sizeof(float);
            preprocessor.preprocessForStorage(lhs_vector.data(), lhs_blob, lhs_blob_size, 0);

            void *rhs_blob = nullptr;
            size_t rhs_blob_size = dim * sizeof(float);
            preprocessor.preprocessForStorage(rhs_vector.data(), rhs_blob, rhs_blob_size, 0);

            const auto lhs_view = state->storageView(lhs_blob);
            const auto rhs_view = state->storageView(rhs_blob);
            const float expected = ScalarEstimateInnerProductSymmetric(*state, lhs_view, rhs_view);
            const float actual = state->estimateInnerProductSymmetric(lhs_view, rhs_view);

            EXPECT_EQ(bits <= 9, state->compactAngles());
            EXPECT_EQ(bits <= 5, state->angleCodeBytes() == (state->pairs + 1) / 2);
            EXPECT_NEAR(actual, expected, AllowedError(expected, 1e-5f, 1e-5f));
            EXPECT_NEAR(calculator.calcDistance(lhs_blob, rhs_blob, dim), 1.0f - expected,
                        AllowedError(1.0f - expected, 1e-5f, 1e-5f));

            allocator->free_allocation(lhs_blob);
            allocator->free_allocation(rhs_blob);
        }
    }
}

TEST(TQFlatTest, symmetric_state_routing_matches_current_helper_selection) {
    auto optimization = spaces::getCpuOptimizationFeatures();
    const size_t packed_threshold = PackedResidualSelectorThreshold(optimization);
    const size_t polar_threshold = SymmetricPolarSelectorThreshold(optimization);
    const size_t fallback_pairs = polar_threshold > 0 ? polar_threshold - 1 : 3;
    const size_t selected_pairs = polar_threshold > 0 ? polar_threshold : 8;
    const size_t fallback_projections = packed_threshold > 0 ? packed_threshold - 1 : 13;
    const size_t selected_projections = packed_threshold > 0 ? packed_threshold : 13;

    struct RouteCase {
        size_t bits;
        size_t pairs;
        size_t projections;
    };

    const std::vector<RouteCase> cases = {
        {.bits = 5, .pairs = fallback_pairs, .projections = fallback_projections},
        {.bits = 5, .pairs = selected_pairs, .projections = selected_projections},
        {.bits = 9, .pairs = selected_pairs, .projections = selected_projections},
        {.bits = 10, .pairs = selected_pairs, .projections = selected_projections},
    };

    for (const auto &test_case : cases) {
        const size_t dim = test_case.pairs * 2;
        SCOPED_TRACE(testing::Message() << "bits=" << test_case.bits << " pairs=" << test_case.pairs
                                        << " projections=" << test_case.projections);

        auto allocator = VecSimAllocator::newVecsimAllocator();
        auto state = std::make_shared<TQFlatDetails::TQModelState>(dim, test_case.bits,
                                                                   test_case.projections, 43, true);
        TQFlatDetails::TQPreprocessor<VecSimMetric_Cosine> preprocessor(allocator, state);
        TQFlatDetails::TQSymmetricDistanceCalculator<VecSimMetric_Cosine> calculator(allocator,
                                                                                     state);

        const auto lhs_vector = MakeSignal(dim, 0.11f + static_cast<float>(test_case.bits));
        const auto rhs_vector = MakeSignal(dim, -0.47f - static_cast<float>(test_case.bits));

        void *lhs_blob = nullptr;
        size_t lhs_blob_size = dim * sizeof(float);
        preprocessor.preprocessForStorage(lhs_vector.data(), lhs_blob, lhs_blob_size, 0);

        void *rhs_blob = nullptr;
        size_t rhs_blob_size = dim * sizeof(float);
        preprocessor.preprocessForStorage(rhs_vector.data(), rhs_blob, rhs_blob_size, 0);

        const auto lhs_view = state->storageView(lhs_blob);
        const auto rhs_view = state->storageView(rhs_blob);
        const float expected =
            RoutedEstimateInnerProductSymmetric(*state, lhs_view, rhs_view, optimization);
        const float actual = state->estimateInnerProductSymmetric(lhs_view, rhs_view);

        EXPECT_EQ(test_case.bits < 10, state->compactAngles());
        EXPECT_NEAR(actual, expected, AllowedError(expected, 1e-5f, 1e-6f));
        EXPECT_NEAR(calculator.calcDistance(lhs_blob, rhs_blob, dim), 1.0f - expected,
                    AllowedError(1.0f - expected, 1e-5f, 1e-6f));

        allocator->free_allocation(lhs_blob);
        allocator->free_allocation(rhs_blob);
    }
}
