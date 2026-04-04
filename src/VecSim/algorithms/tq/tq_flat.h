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

struct QueryView {
    const float *rotated_query;
    const float *qjl_query_dots;
    float query_norm_sq;
};

struct StorageView {
    const float *radii;
    float code_norm_sq;
    const uint16_t *angle_indices;
    const int8_t *residual_signs;
};

class TQModelState {
public:
    TQModelState(size_t dim, size_t total_bits, size_t projections, size_t seed, bool use_rotation)
        : dim(dim), pairs(PairCount(dim)), total_bits(total_bits), polar_bits(total_bits - 1),
          projections(projections), seed(seed), use_rotation(use_rotation),
          levels(size_t{1} << polar_bits), rotation_columns(dim * dim, 0.0f),
          qjl_projection_rows(projections * dim, 0.0f), cos_lut(levels), sin_lut(levels) {
        if (!IsEven(dim)) {
            throw std::invalid_argument("TQ-FLAT requires even dimensions");
        }
        if (total_bits < 2 || total_bits > 16) {
            throw std::invalid_argument("TQ-FLAT bits must be between 2 and 16");
        }
        if (projections == 0) {
            throw std::invalid_argument("TQ-FLAT requires at least one projection");
        }

        initializeRotation();
        initializeQjlProjectionRows();
        initializeTrigLut();
    }

    size_t storageBlobSize() const {
        return pairs * sizeof(float) + sizeof(float) + pairs * sizeof(uint16_t) +
               projections * sizeof(int8_t);
    }

    size_t queryBlobSize() const {
        return (dim + projections + 1) * sizeof(float);
    }

    StorageView storageView(const void *blob) const {
        const auto *bytes = static_cast<const uint8_t *>(blob);
        const auto *radii = reinterpret_cast<const float *>(bytes);
        bytes += pairs * sizeof(float);
        const auto *code_norm_sq = reinterpret_cast<const float *>(bytes);
        bytes += sizeof(float);
        const auto *angles = reinterpret_cast<const uint16_t *>(bytes);
        bytes += pairs * sizeof(uint16_t);
        const auto *signs = reinterpret_cast<const int8_t *>(bytes);
        return {.radii = radii,
                .code_norm_sq = *code_norm_sq,
                .angle_indices = angles,
                .residual_signs = signs};
    }

    QueryView queryView(const void *blob) const {
        const auto *bytes = static_cast<const float *>(blob);
        const auto *rotated_query = bytes;
        const auto *qjl_query_dots = bytes + dim;
        const float query_norm_sq = *(bytes + dim + projections);
        return {.rotated_query = rotated_query,
                .qjl_query_dots = qjl_query_dots,
                .query_norm_sq = query_norm_sq};
    }

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
            const auto index =
                static_cast<uint16_t>(static_cast<uint32_t>(std::floor(normalized * levels)) %
                                      levels);
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

    float estimateInnerProduct(const StorageView &storage, const QueryView &query) const {
        float polar_estimate = 0.0f;
        for (size_t i = 0; i < pairs; ++i) {
            const uint16_t angle_index = storage.angle_indices[i];
            const float q_a = query.rotated_query[2 * i];
            const float q_b = query.rotated_query[2 * i + 1];
            polar_estimate +=
                storage.radii[i] * (q_a * cos_lut[angle_index] + q_b * sin_lut[angle_index]);
        }

        const float qjl_scale = kPi / (2.0f * static_cast<float>(projections));
        float qjl_estimate = 0.0f;
        for (size_t i = 0; i < projections; ++i) {
            qjl_estimate +=
                static_cast<float>(storage.residual_signs[i]) * query.qjl_query_dots[i];
        }

        return polar_estimate + qjl_scale * qjl_estimate;
    }

    size_t dim;
    size_t pairs;
    size_t total_bits;
    size_t polar_bits;
    size_t projections;
    size_t seed;
    bool use_rotation;
    size_t levels;

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
            const float theta = (static_cast<float>(i) / static_cast<float>(levels)) *
                                    (2.0f * kPi) -
                                kPi;
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
public:
    TQDistanceCalculator(std::shared_ptr<VecSimAllocator> allocator,
                         std::shared_ptr<TQModelState> state)
        : IndexCalculatorInterface<float>(allocator), state(std::move(state)) {}

    float calcDistance(const void *v1, const void *v2, size_t dim) const override {
        UNUSED(dim);
        const auto storage = state->storageView(v1);
        const auto query = state->queryView(v2);
        const float estimate = state->estimateInnerProduct(storage, query);

        if constexpr (Metric == VecSimMetric_L2) {
            return std::max(query.query_norm_sq + storage.code_norm_sq - 2.0f * estimate, 0.0f);
        }

        return 1.0f - estimate;
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
                    size_t &input_blob_size, unsigned char alignment) const override {
        size_t storage_blob_size = input_blob_size;
        size_t query_blob_size = input_blob_size;
        preprocess(original_blob, storage_blob, query_blob, storage_blob_size, query_blob_size,
                   alignment);
        input_blob_size = storage_blob_size;
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char alignment) const override {
        preprocessForStorage(original_blob, storage_blob, storage_blob_size);
        preprocessQuery(original_blob, query_blob, query_blob_size, alignment);
    }

    void preprocessForStorage(const void *original_blob, void *&storage_blob,
                              size_t &input_blob_size) const override {
        if (!storage_blob) {
            storage_blob = this->allocator->allocate(state->storageBlobSize());
        }

        const auto *typed_blob = static_cast<const float *>(original_blob);
        std::vector<float> normalized(typed_blob, typed_blob + working_dim);
        normalizeIfNeeded(normalized.data());

        std::vector<float> rotated(working_dim);
        std::vector<float> reconstructed_rotated(working_dim);
        std::vector<float> reconstructed(working_dim);
        std::vector<float> residual(working_dim);

        state->applyRotation(normalized.data(), rotated.data());

        auto *bytes = static_cast<uint8_t *>(storage_blob);
        auto *radii = reinterpret_cast<float *>(bytes);
        bytes += state->pairs * sizeof(float);
        auto *code_norm_sq = reinterpret_cast<float *>(bytes);
        bytes += sizeof(float);
        auto *angles = reinterpret_cast<uint16_t *>(bytes);
        bytes += state->pairs * sizeof(uint16_t);
        auto *signs = reinterpret_cast<int8_t *>(bytes);

        state->encodePolar(rotated.data(), radii, angles);
        *code_norm_sq = 0.0f;
        for (size_t i = 0; i < state->pairs; ++i) {
            *code_norm_sq += radii[i] * radii[i];
        }

        state->reconstructRotated(radii, angles, reconstructed_rotated.data());
        state->applyInverseRotation(reconstructed_rotated.data(), reconstructed.data());
        for (size_t i = 0; i < working_dim; ++i) {
            residual[i] = normalized[i] - reconstructed[i];
        }
        state->sketchResidual(residual.data(), signs);

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
        auto *rotated_query = query_words;
        auto *qjl_query_dots = query_words + working_dim;
        auto *query_norm_sq = query_words + working_dim + state->projections;

        state->applyRotation(normalized.data(), rotated_query);
        state->projectQjl(normalized.data(), qjl_query_dots);
        *query_norm_sq = 0.0f;
        for (float value : normalized) {
            *query_norm_sq += value * value;
        }

        input_blob_size = state->queryBlobSize();
    }

    void preprocessStorageInPlace(void *original_blob, size_t input_blob_size) const override {
        assert(original_blob);
        assert(input_blob_size >= state->storageBlobSize());
        std::vector<uint8_t> encoded(state->storageBlobSize());
        void *encoded_blob = encoded.data();
        size_t storage_blob_size = input_blob_size;
        preprocessForStorage(original_blob, encoded_blob, storage_blob_size);
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
inline size_t GetStorageDataSize(const TQFlatParams *params) {
    return TQModelState(params->dim, params->bits, params->projections, params->seed,
                        params->useRotation)
        .storageBlobSize();
}

template <VecSimMetric Metric>
inline IndexComponents<float, float>
CreateTQComponents(std::shared_ptr<VecSimAllocator> allocator, const TQFlatParams *params) {
    auto state = std::make_shared<TQModelState>(params->dim, params->bits, params->projections,
                                                params->seed, params->useRotation);
    auto *index_calculator =
        new (allocator) TQDistanceCalculator<Metric>(allocator, state);
    auto *preprocessors =
        new (allocator) MultiPreprocessorsContainer<float, 1>(allocator, alignof(float));
    auto *tq_preprocessor = new (allocator) TQPreprocessor<Metric>(allocator, state);
    int rc = preprocessors->addPreprocessor(tq_preprocessor);
    UNUSED(rc);
    assert(rc != -1 && "TQ preprocessor was not added correctly");
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
