/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/spaces/computer/preprocessors.h"
#include "VecSim/utils/vec_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace TQFlatDetails {

inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr uint64_t kQjlSeedOffset = 0xCAFEBABE00000001ULL;

inline bool IsEven(size_t value) { return value != 0 && value % 2 == 0; }
inline size_t PairCount(size_t dim) { return dim / 2; }

inline size_t PolarBits(size_t total_bits) {
    if (total_bits < 2 || total_bits > 16) {
        throw std::invalid_argument("TQ-FLAT bits must be between 2 and 16");
    }
    return total_bits - 1;
}

inline float QjlScale(size_t projections) {
    if (projections == 0) {
        throw std::invalid_argument("TQ-FLAT requires at least one projection");
    }
    return kPi / (2.0f * static_cast<float>(projections));
}

struct QueryView {
    const float *polar_lookup;
    const float *rotated_query;
    const float *qjl_byte_lut;
    float query_norm_sq;
};

struct StorageView {
    const float *radii;
    float full_vector_norm_sq;
    float code_norm_sq;
    const void *angle_indices;
    const uint8_t *residual_signs;
};

class TQModelState {
public:
    TQModelState(size_t dim, size_t total_bits, size_t projections, size_t seed, bool use_rotation)
        : dim(dim), pairs(PairCount(dim)), total_bits(total_bits),
          polar_bits(PolarBits(total_bits)), projections(projections), seed(seed),
          use_rotation(use_rotation), levels(size_t{1} << polar_bits),
          packed_qjl_bytes((projections + 7) / 8), nibble_angle_codes(levels <= 16),
          compact_angle_codes(levels <= 256), use_polar_lookup(levels <= 256),
          qjl_scale(QjlScale(projections)), rotation_columns(dim * dim, 0.0f),
          qjl_projection_rows(projections * dim, 0.0f), cos_lut(levels), sin_lut(levels) {
        if (!IsEven(dim)) {
            throw std::invalid_argument("TQ-FLAT requires even dimensions");
        }

        initializeRotation();
        initializeQjlProjectionRows();
        initializeTrigLut();
    }

    size_t storageBlobSize() const {
        return pairs * sizeof(float) + 2 * sizeof(float) + angleCodeBytes() + packedQjlBytes();
    }

    size_t queryBlobSize() const {
        return (polarQueryWordCount() + packedQjlBytes() * 256 + 1) * sizeof(float);
    }

    StorageView storageView(const void *blob) const {
        const auto *bytes = static_cast<const uint8_t *>(blob);
        const auto *radii = reinterpret_cast<const float *>(bytes);
        bytes += pairs * sizeof(float);
        const auto *full_vector_norm_sq = reinterpret_cast<const float *>(bytes);
        bytes += sizeof(float);
        const auto *code_norm_sq = reinterpret_cast<const float *>(bytes);
        bytes += sizeof(float);
        const void *angles = bytes;
        bytes += angleCodeBytes();
        const auto *signs = reinterpret_cast<const uint8_t *>(bytes);
        return {.radii = radii,
                .full_vector_norm_sq = *full_vector_norm_sq,
                .code_norm_sq = *code_norm_sq,
                .angle_indices = angles,
                .residual_signs = signs};
    }

    QueryView queryView(const void *blob) const {
        const auto *bytes = static_cast<const float *>(blob);
        const auto *polar_lookup = usePolarLut() ? bytes : nullptr;
        const auto *rotated_query = usePolarLut() ? nullptr : bytes;
        bytes += polarQueryWordCount();
        const auto *qjl_byte_lut = bytes;
        bytes += packedQjlBytes() * 256;
        const float query_norm_sq = *bytes;
        return {.polar_lookup = polar_lookup,
                .rotated_query = rotated_query,
                .qjl_byte_lut = qjl_byte_lut,
                .query_norm_sq = query_norm_sq};
    }

    size_t packedQjlBytes() const { return packed_qjl_bytes; }
    bool usePolarLut() const { return use_polar_lookup; }
    bool compactAngles() const { return compact_angle_codes; }
    size_t angleCodeBytes() const {
        if (nibble_angle_codes) {
            return (pairs + 1) / 2;
        }
        return pairs * (compactAngles() ? sizeof(uint8_t) : sizeof(uint16_t));
    }
    size_t polarQueryWordCount() const { return usePolarLut() ? pairs * levels : dim; }

    void applyRotation(const float *input, float *output) const {
        if (!use_rotation) {
            std::memcpy(output, input, dim * sizeof(float));
            return;
        }
        for (size_t row = 0; row < dim; ++row) {
            float sum = 0.0f;
            for (size_t col = 0; col < dim; ++col) {
                sum += rotation_columns[col * dim + row] * input[col];
            }
            output[row] = sum;
        }
    }

    void applyInverseRotation(const float *input, float *output) const {
        if (!use_rotation) {
            std::memcpy(output, input, dim * sizeof(float));
            return;
        }
        for (size_t col = 0; col < dim; ++col) {
            float sum = 0.0f;
            for (size_t row = 0; row < dim; ++row) {
                sum += rotation_columns[col * dim + row] * input[row];
            }
            output[col] = sum;
        }
    }

    void encodePolar(const float *rotated, float *radii, uint16_t *angles) const {
        for (size_t i = 0; i < pairs; ++i) {
            const float a = rotated[2 * i];
            const float b = rotated[2 * i + 1];
            const float radius = std::sqrt(a * a + b * b);
            const float theta = std::atan2(b, a);
            const float normalized = (theta + kPi) / (2.0f * kPi);
            const auto index = static_cast<uint16_t>(
                static_cast<uint32_t>(std::floor(normalized * levels)) % levels);
            radii[i] = radius;
            angles[i] = index;
        }
    }

    void reconstructRotated(const float *radii, const uint16_t *angles, float *rotated) const {
        for (size_t i = 0; i < pairs; ++i) {
            const uint16_t angle_index = angles[i];
            const float radius = radii[i];
            rotated[2 * i] = radius * cos_lut[angle_index];
            rotated[2 * i + 1] = radius * sin_lut[angle_index];
        }
    }

    void projectQjl(const float *input, float *output) const {
        for (size_t row = 0; row < projections; ++row) {
            float sum = 0.0f;
            const float *projection_row = qjl_projection_rows.data() + row * dim;
            for (size_t col = 0; col < dim; ++col) {
                sum += projection_row[col] * input[col];
            }
            output[row] = sum;
        }
    }

    void sketchResidual(const float *residual, int8_t *signs) const {
        std::vector<float> dots(projections);
        projectQjl(residual, dots.data());
        for (size_t i = 0; i < projections; ++i) {
            signs[i] = dots[i] >= 0.0f ? int8_t{1} : int8_t{-1};
        }
    }

    void packResidualSigns(const float *residual, uint8_t *packed_signs) const {
        std::memset(packed_signs, 0, packedQjlBytes());
        std::vector<float> dots(projections);
        projectQjl(residual, dots.data());
        for (size_t i = 0; i < projections; ++i) {
            if (dots[i] >= 0.0f) {
                packed_signs[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            }
        }
    }

    void writeAngleCodes(const uint16_t *source_angles, void *destination) const {
        if (nibble_angle_codes) {
            auto *encoded = static_cast<uint8_t *>(destination);
            std::memset(encoded, 0, angleCodeBytes());
            for (size_t i = 0; i < pairs; ++i) {
                const uint8_t angle = static_cast<uint8_t>(source_angles[i] & 0x0F);
                const size_t byte_idx = i / 2;
                if ((i % 2) == 0) {
                    encoded[byte_idx] = angle;
                } else {
                    encoded[byte_idx] |= static_cast<uint8_t>(angle << 4);
                }
            }
            return;
        }
        if (compactAngles()) {
            auto *encoded = static_cast<uint8_t *>(destination);
            for (size_t i = 0; i < pairs; ++i) {
                encoded[i] = static_cast<uint8_t>(source_angles[i]);
            }
            return;
        }
        std::memcpy(destination, source_angles, pairs * sizeof(uint16_t));
    }

    uint16_t angleCodeAt(const StorageView &storage, size_t idx) const {
        if (nibble_angle_codes) {
            const auto *encoded = static_cast<const uint8_t *>(storage.angle_indices);
            const uint8_t packed = encoded[idx / 2];
            return (idx % 2 == 0) ? static_cast<uint16_t>(packed & 0x0F)
                                  : static_cast<uint16_t>((packed >> 4) & 0x0F);
        }
        if (compactAngles()) {
            return static_cast<const uint8_t *>(storage.angle_indices)[idx];
        }
        return static_cast<const uint16_t *>(storage.angle_indices)[idx];
    }

    void buildPolarLookup(const float *rotated_query, float *polar_lookup) const {
        for (size_t pair_idx = 0; pair_idx < pairs; ++pair_idx) {
            const float q_a = rotated_query[2 * pair_idx];
            const float q_b = rotated_query[2 * pair_idx + 1];
            float *pair_lookup = polar_lookup + pair_idx * levels;
            for (size_t angle_idx = 0; angle_idx < levels; ++angle_idx) {
                pair_lookup[angle_idx] = q_a * cos_lut[angle_idx] + q_b * sin_lut[angle_idx];
            }
        }
    }

    void buildQjlByteLookup(const float *qjl_query_dots, float *qjl_byte_lut) const {
        for (size_t byte_idx = 0; byte_idx < packedQjlBytes(); ++byte_idx) {
            const size_t base_projection = byte_idx * 8;
            const size_t valid_bits = std::min<size_t>(8, projections - base_projection);
            float *byte_lookup = qjl_byte_lut + byte_idx * 256;
            for (size_t pattern = 0; pattern < 256; ++pattern) {
                float sum = 0.0f;
                for (size_t bit_idx = 0; bit_idx < valid_bits; ++bit_idx) {
                    const bool positive = (pattern & (size_t{1} << bit_idx)) != 0;
                    const float sign = positive ? 1.0f : -1.0f;
                    sum += sign * qjl_query_dots[base_projection + bit_idx];
                }
                byte_lookup[pattern] = sum;
            }
        }
    }

    float estimateInnerProduct(const StorageView &storage, const QueryView &query) const {
        float polar_estimate = 0.0f;
        if (usePolarLut()) {
            for (size_t i = 0; i < pairs; ++i) {
                const uint16_t angle_index = angleCodeAt(storage, i);
                polar_estimate += storage.radii[i] * query.polar_lookup[i * levels + angle_index];
            }
        } else {
            for (size_t i = 0; i < pairs; ++i) {
                const uint16_t angle_index = angleCodeAt(storage, i);
                const float q_a = query.rotated_query[2 * i];
                const float q_b = query.rotated_query[2 * i + 1];
                polar_estimate +=
                    storage.radii[i] * (q_a * cos_lut[angle_index] + q_b * sin_lut[angle_index]);
            }
        }

        float qjl_estimate = 0.0f;
        for (size_t i = 0; i < packedQjlBytes(); ++i) {
            qjl_estimate += query.qjl_byte_lut[i * 256 + storage.residual_signs[i]];
        }

        return polar_estimate + qjl_scale * qjl_estimate;
    }

    float estimateInnerProductSymmetric(const StorageView &lhs, const StorageView &rhs) const {
        float polar_estimate = 0.0f;
        for (size_t i = 0; i < pairs; ++i) {
            const uint16_t lhs_angle = angleCodeAt(lhs, i);
            const uint16_t rhs_angle = angleCodeAt(rhs, i);
            polar_estimate +=
                lhs.radii[i] * rhs.radii[i] *
                (cos_lut[lhs_angle] * cos_lut[rhs_angle] + sin_lut[lhs_angle] * sin_lut[rhs_angle]);
        }

        int sign_dot = 0;
        for (size_t byte_idx = 0; byte_idx < packedQjlBytes(); ++byte_idx) {
            const size_t base_projection = byte_idx * 8;
            const size_t valid_bits = std::min<size_t>(8, projections - base_projection);
            const uint8_t valid_mask = valid_bits == 8
                                           ? static_cast<uint8_t>(0xFFu)
                                           : static_cast<uint8_t>((uint16_t{1} << valid_bits) - 1u);
            const uint8_t diff_bits = static_cast<uint8_t>(
                (lhs.residual_signs[byte_idx] ^ rhs.residual_signs[byte_idx]) & valid_mask);
            const int diff_count = __builtin_popcount(static_cast<unsigned int>(diff_bits));
            sign_dot += static_cast<int>(valid_bits) - (2 * diff_count);
        }

        return polar_estimate + qjl_scale * static_cast<float>(sign_dot);
    }

    size_t dim;
    size_t pairs;
    size_t total_bits;
    size_t polar_bits;
    size_t projections;
    size_t seed;
    bool use_rotation;
    size_t levels;
    size_t packed_qjl_bytes;
    bool nibble_angle_codes;
    bool compact_angle_codes;
    bool use_polar_lookup;
    float qjl_scale;

private:
    void initializeRotation() {
        if (!use_rotation) {
            return;
        }

        std::mt19937_64 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        std::vector<float> candidate(dim);
        for (size_t col = 0; col < dim; ++col) {
            bool accepted = false;
            for (size_t attempt = 0; attempt < 16 && !accepted; ++attempt) {
                for (size_t row = 0; row < dim; ++row) {
                    candidate[row] = normal(rng);
                }

                for (size_t prev = 0; prev < col; ++prev) {
                    const float *prev_col = rotation_columns.data() + prev * dim;
                    float dot = 0.0f;
                    for (size_t row = 0; row < dim; ++row) {
                        dot += candidate[row] * prev_col[row];
                    }
                    for (size_t row = 0; row < dim; ++row) {
                        candidate[row] -= dot * prev_col[row];
                    }
                }

                float norm_sq = 0.0f;
                for (float value : candidate) {
                    norm_sq += value * value;
                }

                if (norm_sq > 1e-12f) {
                    float inv_norm = 1.0f / std::sqrt(norm_sq);
                    if (candidate[col] < 0.0f) {
                        inv_norm = -inv_norm;
                    }
                    float *dst_col = rotation_columns.data() + col * dim;
                    for (size_t row = 0; row < dim; ++row) {
                        dst_col[row] = candidate[row] * inv_norm;
                    }
                    accepted = true;
                }
            }

            if (!accepted) {
                throw std::runtime_error("Failed to construct TQ rotation");
            }
        }
    }

    void initializeQjlProjectionRows() {
        std::mt19937_64 rng(seed + kQjlSeedOffset);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float &value : qjl_projection_rows) {
            value = normal(rng);
        }
    }

    void initializeTrigLut() {
        for (size_t i = 0; i < levels; ++i) {
            const float theta =
                (static_cast<float>(i) / static_cast<float>(levels)) * (2.0f * kPi) - kPi;
            cos_lut[i] = std::cos(theta);
            sin_lut[i] = std::sin(theta);
        }
    }

    std::vector<float> rotation_columns;
    std::vector<float> qjl_projection_rows;
    std::vector<float> cos_lut;
    std::vector<float> sin_lut;
};

template <VecSimMetric Metric>
class TQDistanceCalculator : public IndexCalculatorInterface<float> {
private:
    static float calcWithContext(const void *opaque_state, const void *storage_blob,
                                 const void *query_blob, size_t dim) {
        UNUSED(dim);
        const auto *state = static_cast<const TQModelState *>(opaque_state);
        const auto storage = state->storageView(storage_blob);
        const auto query = state->queryView(query_blob);
        const float estimate = state->estimateInnerProduct(storage, query);

        if constexpr (Metric == VecSimMetric_L2) {
            return std::max(query.query_norm_sq + storage.full_vector_norm_sq - 2.0f * estimate,
                            0.0f);
        }

        return 1.0f - estimate;
    }

public:
    TQDistanceCalculator(std::shared_ptr<VecSimAllocator> allocator,
                         std::shared_ptr<TQModelState> state)
        : IndexCalculatorInterface<float>(allocator), state(std::move(state)) {}

    float calcDistance(const void *v1, const void *v2, size_t dim) const override {
        return calcWithContext(state.get(), v1, v2, dim);
    }

    float calcDistanceForQuery(const void *candidate_vector, const void *query_vector,
                               size_t dim) const override {
        return calcWithContext(state.get(), candidate_vector, query_vector, dim);
    }

    DistanceDispatch<float> getDistanceDispatch(DistanceMode mode) const override {
        // BruteForceIndex currently routes preprocessed queries through calcDistance(), so its
        // stored dispatch is intentionally asymmetric as well. TQ-HNSW supplies a separate
        // symmetric calculator for graph construction.
        UNUSED(mode);
        return DistanceDispatch<float>::stateful(state.get(), calcWithContext);
    }

private:
    std::shared_ptr<TQModelState> state;
};

template <VecSimMetric Metric>
class TQSymmetricDistanceCalculator : public IndexCalculatorInterface<float> {
private:
    static float calcWithContext(const void *opaque_state, const void *lhs_blob,
                                 const void *rhs_blob, size_t dim) {
        UNUSED(dim);
        const auto *state = static_cast<const TQModelState *>(opaque_state);
        const auto lhs = state->storageView(lhs_blob);
        const auto rhs = state->storageView(rhs_blob);
        const float estimate = state->estimateInnerProductSymmetric(lhs, rhs);

        if constexpr (Metric == VecSimMetric_L2) {
            return std::max(lhs.full_vector_norm_sq + rhs.full_vector_norm_sq - 2.0f * estimate,
                            0.0f);
        }

        return 1.0f - estimate;
    }

public:
    TQSymmetricDistanceCalculator(std::shared_ptr<VecSimAllocator> allocator,
                                  std::shared_ptr<TQModelState> state)
        : IndexCalculatorInterface<float>(allocator), state(std::move(state)) {}

    float calcDistance(const void *v1, const void *v2, size_t dim) const override {
        return calcWithContext(state.get(), v1, v2, dim);
    }

    float calcDistanceForQuery(const void *candidate_vector, const void *query_vector,
                               size_t dim) const override {
        return calcWithContext(state.get(), candidate_vector, query_vector, dim);
    }

    DistanceDispatch<float> getDistanceDispatch(DistanceMode mode) const override {
        UNUSED(mode);
        return DistanceDispatch<float>::stateful(state.get(), calcWithContext);
    }

private:
    std::shared_ptr<TQModelState> state;
};

template <VecSimMetric Metric>
class TQPreprocessor : public PreprocessorInterface {
public:
    TQPreprocessor(std::shared_ptr<VecSimAllocator> allocator, std::shared_ptr<TQModelState> state)
        : PreprocessorInterface(allocator), normalize_func(spaces::GetNormalizeFunc<float>()),
          state(std::move(state)), working_dim(this->state->dim) {}

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char storage_alignment, unsigned char query_alignment) const override {
        preprocessForStorage(original_blob, storage_blob, storage_blob_size, storage_alignment);
        preprocessQuery(original_blob, query_blob, query_blob_size, query_alignment);
    }

    void preprocessForStorage(const void *original_blob, void *&storage_blob,
                              size_t &input_blob_size,
                              unsigned char storage_alignment) const override {
        if (!storage_blob) {
            storage_blob =
                this->allocator->allocate_aligned(state->storageBlobSize(), storage_alignment);
        }

        const auto *typed_blob = static_cast<const float *>(original_blob);
        std::vector<float> normalized(typed_blob, typed_blob + working_dim);
        normalizeIfNeeded(normalized.data());

        std::vector<float> rotated(working_dim);
        std::vector<uint16_t> angles(state->pairs);
        std::vector<float> reconstructed_rotated(working_dim);
        std::vector<float> reconstructed(working_dim);
        std::vector<float> residual(working_dim);

        state->applyRotation(normalized.data(), rotated.data());

        auto *bytes = static_cast<uint8_t *>(storage_blob);
        auto *radii = reinterpret_cast<float *>(bytes);
        bytes += state->pairs * sizeof(float);
        auto *full_vector_norm_sq = reinterpret_cast<float *>(bytes);
        bytes += sizeof(float);
        auto *code_norm_sq = reinterpret_cast<float *>(bytes);
        bytes += sizeof(float);
        void *encoded_angles = bytes;
        bytes += state->angleCodeBytes();
        auto *signs = reinterpret_cast<uint8_t *>(bytes);

        state->encodePolar(rotated.data(), radii, angles.data());
        state->writeAngleCodes(angles.data(), encoded_angles);
        *full_vector_norm_sq = 0.0f;
        for (float value : normalized) {
            *full_vector_norm_sq += value * value;
        }
        *code_norm_sq = 0.0f;
        for (size_t i = 0; i < state->pairs; ++i) {
            *code_norm_sq += radii[i] * radii[i];
        }

        state->reconstructRotated(radii, angles.data(), reconstructed_rotated.data());
        state->applyInverseRotation(reconstructed_rotated.data(), reconstructed.data());
        for (size_t i = 0; i < working_dim; ++i) {
            residual[i] = normalized[i] - reconstructed[i];
        }
        state->packResidualSigns(residual.data(), signs);

        input_blob_size = state->storageBlobSize();
    }

    void preprocessQuery(const void *original_blob, void *&query_blob, size_t &input_blob_size,
                         unsigned char alignment) const override {
        if (!query_blob) {
            query_blob = this->allocator->allocate_aligned(state->queryBlobSize(), alignment);
        }

        const auto *typed_blob = static_cast<const float *>(original_blob);
        std::vector<float> normalized(typed_blob, typed_blob + working_dim);
        normalizeIfNeeded(normalized.data());

        auto *query_words = static_cast<float *>(query_blob);
        auto *polar_query_data = query_words;
        query_words += state->polarQueryWordCount();
        auto *qjl_byte_lut = query_words;
        auto *query_norm_sq = qjl_byte_lut + state->packedQjlBytes() * 256;

        std::vector<float> rotated_query(working_dim);
        std::vector<float> qjl_query_dots(state->projections);

        state->applyRotation(normalized.data(), rotated_query.data());
        if (state->usePolarLut()) {
            state->buildPolarLookup(rotated_query.data(), polar_query_data);
        } else {
            std::memcpy(polar_query_data, rotated_query.data(), working_dim * sizeof(float));
        }
        state->projectQjl(normalized.data(), qjl_query_dots.data());
        state->buildQjlByteLookup(qjl_query_dots.data(), qjl_byte_lut);
        *query_norm_sq = 0.0f;
        for (float value : normalized) {
            *query_norm_sq += value * value;
        }

        input_blob_size = state->queryBlobSize();
    }

    void preprocessStorageInPlace(void *original_blob, size_t input_blob_size) const override {
        assert(original_blob);
        assert(input_blob_size >= state->storageBlobSize());
        const size_t encoded_words = (state->storageBlobSize() + sizeof(float) - 1) / sizeof(float);
        std::vector<float> encoded(encoded_words);
        void *encoded_blob = encoded.data();
        size_t storage_blob_size = input_blob_size;
        // encoded_blob is non-null, so no aligned allocation happens; alignment is unused.
        preprocessForStorage(original_blob, encoded_blob, storage_blob_size, 0);
        std::memcpy(original_blob, encoded.data(), state->storageBlobSize());
    }

private:
    void normalizeIfNeeded(float *values) const {
        if constexpr (Metric == VecSimMetric_Cosine) {
            normalize_func(values, working_dim);
        }
    }

    spaces::normalizeVector_f<float> normalize_func;
    std::shared_ptr<TQModelState> state;
    size_t working_dim;
};

template <VecSimMetric Metric>
class TQSymmetricPreprocessor : public PreprocessorInterface {
public:
    TQSymmetricPreprocessor(std::shared_ptr<VecSimAllocator> allocator,
                            std::shared_ptr<TQModelState> state)
        : PreprocessorInterface(allocator), delegate(allocator, std::move(state)) {}

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char storage_alignment, unsigned char query_alignment) const override {
        delegate.preprocessForStorage(original_blob, storage_blob, storage_blob_size,
                                      storage_alignment);
        delegate.preprocessForStorage(original_blob, query_blob, query_blob_size, query_alignment);
    }

    void preprocessForStorage(const void *original_blob, void *&storage_blob,
                              size_t &input_blob_size,
                              unsigned char storage_alignment) const override {
        delegate.preprocessForStorage(original_blob, storage_blob, input_blob_size,
                                      storage_alignment);
    }

    void preprocessQuery(const void *original_blob, void *&query_blob, size_t &input_blob_size,
                         unsigned char query_alignment) const override {
        delegate.preprocessForStorage(original_blob, query_blob, input_blob_size, query_alignment);
    }

    void preprocessStorageInPlace(void *original_blob, size_t input_blob_size) const override {
        delegate.preprocessStorageInPlace(original_blob, input_blob_size);
    }

private:
    TQPreprocessor<Metric> delegate;
};

template <VecSimMetric Metric>
inline size_t GetStorageDataSize(const TQFlatParams *params) {
    return TQModelState(params->dim, params->bits, params->projections, params->seed,
                        params->useRotation)
        .storageBlobSize();
}

template <VecSimMetric Metric>
inline IndexComponents<float, float> CreateTQComponents(std::shared_ptr<VecSimAllocator> allocator,
                                                        const TQFlatParams *params) {
    auto state = std::make_shared<TQModelState>(params->dim, params->bits, params->projections,
                                                params->seed, params->useRotation);
    auto *index_calculator = new (allocator) TQDistanceCalculator<Metric>(allocator, state);
    auto *preprocessors =
        new (allocator) MultiPreprocessorsContainer<float, 1>(allocator, alignof(float));
    auto *tq_preprocessor = new (allocator) TQPreprocessor<Metric>(allocator, state);
    int rc = preprocessors->addPreprocessor(tq_preprocessor);
    UNUSED(rc);
    assert(rc != -1 && "TQ preprocessor was not added correctly");
    return {index_calculator, preprocessors};
}

template <VecSimMetric Metric>
inline IndexComponents<float, float>
CreateTQHNSWComponents(std::shared_ptr<VecSimAllocator> allocator, const TQFlatParams *params) {
    auto state = std::make_shared<TQModelState>(params->dim, params->bits, params->projections,
                                                params->seed, params->useRotation);
    auto *index_calculator =
        new (allocator) TQSymmetricDistanceCalculator<Metric>(allocator, state);
    auto *preprocessors =
        new (allocator) MultiPreprocessorsContainer<float, 1>(allocator, alignof(float));
    auto *tq_preprocessor = new (allocator) TQSymmetricPreprocessor<Metric>(allocator, state);
    int rc = preprocessors->addPreprocessor(tq_preprocessor);
    UNUSED(rc);
    assert(rc != -1 && "TQ symmetric preprocessor was not added correctly");
    return {index_calculator, preprocessors};
}

class TQFlatIndex : public BruteForceIndex_Single<float, float> {
public:
    TQFlatIndex(const BFParams *params, const AbstractIndexInitParams &abstract_init_params,
                const IndexComponents<float, float> &components)
        : BruteForceIndex_Single<float, float>(params, abstract_init_params, components) {}

    int addVector(const void *vector_data, labelType label) override {
        auto existing_id = this->labelToIdLookup.find(label);
        if (existing_id != this->labelToIdLookup.end()) {
            auto processed_blob = this->preprocessForStorage(vector_data);
            this->vectors->updateElement(existing_id->second, processed_blob.get());
            return 0;
        }

        this->appendVector(vector_data, label);
        return 1;
    }

    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        auto optional_id = this->labelToIdLookup.find(label);
        if (optional_id == this->labelToIdLookup.end()) {
            return INVALID_SCORE;
        }

        auto processed_query = this->preprocessQuery(vector_data);
        return this->calcDistance(this->getDataByInternalId(optional_id->second),
                                  processed_query.get());
    }

    VecSimIndexDebugInfo debugInfo() const override {
        VecSimIndexDebugInfo info = BruteForceIndex_Single<float, float>::debugInfo();
        info.commonInfo.basicInfo.algo = VecSimAlgo_TQ;
        return info;
    }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_TQ;
        info.isTiered = false;
        return info;
    }
};

} // namespace TQFlatDetails
