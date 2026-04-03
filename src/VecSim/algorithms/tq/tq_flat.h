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
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/types/sq8.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/utils/vecsim_stl.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace TQFlatDetails {

inline bool IsPowerOfTwo(size_t value) { return value != 0 && (value & (value - 1)) == 0; }

template <VecSimMetric Metric>
class TQPreprocessor : public PreprocessorInterface {
public:
    TQPreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim, size_t seed,
                   bool use_rotation)
        : PreprocessorInterface(allocator), normalize_func(spaces::GetNormalizeFunc<float>()),
          dim(dim), seed(seed), use_rotation(use_rotation),
          quantizer(allocator, dim), signs(allocator) {
        if (this->use_rotation && !IsPowerOfTwo(dim)) {
            throw std::invalid_argument("TQ-FLAT rotation requires power-of-two dimensions");
        }
        initializeSigns();
    }

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
        auto transformed = transformInput(original_blob);
        size_t transformed_size = dim * sizeof(float);
        quantizer.preprocessForStorage(transformed.data(), storage_blob, transformed_size);
        input_blob_size = transformed_size;
    }

    void preprocessQuery(const void *original_blob, void *&query_blob, size_t &input_blob_size,
                         unsigned char alignment) const override {
        auto transformed = transformInput(original_blob);
        size_t transformed_size = dim * sizeof(float);
        quantizer.preprocessQuery(transformed.data(), query_blob, transformed_size, alignment);
        input_blob_size = transformed_size;
    }

    void preprocessStorageInPlace(void *original_blob, size_t input_blob_size) const override {
        auto *typed_blob = static_cast<float *>(original_blob);
        if constexpr (Metric == VecSimMetric_Cosine) {
            normalize_func(typed_blob, dim);
        }
        applyRotation(typed_blob);
        quantizer.preprocessStorageInPlace(original_blob, input_blob_size);
    }

private:
    std::vector<float> transformInput(const void *original_blob) const {
        const auto *typed_blob = static_cast<const float *>(original_blob);
        std::vector<float> transformed(typed_blob, typed_blob + dim);
        if constexpr (Metric == VecSimMetric_Cosine) {
            normalize_func(transformed.data(), dim);
        }
        applyRotation(transformed.data());
        return transformed;
    }

    void applyRotation(float *values) const {
        if (!use_rotation) {
            return;
        }
        for (size_t i = 0; i < dim; ++i) {
            values[i] *= signs[i];
        }

        for (size_t block = 1; block < dim; block <<= 1) {
            const size_t step = block << 1;
            for (size_t start = 0; start < dim; start += step) {
                for (size_t offset = 0; offset < block; ++offset) {
                    const size_t left = start + offset;
                    const size_t right = left + block;
                    const float a = values[left];
                    const float b = values[right];
                    values[left] = a + b;
                    values[right] = a - b;
                }
            }
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
        for (size_t i = 0; i < dim; ++i) {
            values[i] *= scale;
        }
    }

    void initializeSigns() {
        signs.resize(dim);
        std::mt19937_64 rng(seed);
        for (size_t i = 0; i < dim; ++i) {
            signs[i] = (rng() & 1ULL) ? 1.0f : -1.0f;
        }
    }

    spaces::normalizeVector_f<float> normalize_func;
    const size_t dim;
    const size_t seed;
    const bool use_rotation;
    QuantPreprocessor<float, Metric> quantizer;
    vecsim_stl::vector<float> signs;
};

template <VecSimMetric Metric>
inline spaces::dist_func_t<float> GetTQDistFunc(size_t dim, unsigned char *alignment) {
    if constexpr (Metric == VecSimMetric_L2) {
        return spaces::L2_SQ8_FP32_GetDistFunc(dim, alignment);
    }
    if constexpr (Metric == VecSimMetric_IP) {
        return spaces::IP_SQ8_FP32_GetDistFunc(dim, alignment);
    }
    return spaces::Cosine_SQ8_FP32_GetDistFunc(dim, alignment);
}

template <VecSimMetric Metric>
inline size_t GetStorageDataSize(size_t dim) {
    return dim * sizeof(vecsim_types::sq8::value_type) +
           vecsim_types::sq8::storage_metadata_count<Metric>() * sizeof(float);
}

template <VecSimMetric Metric>
inline size_t GetQueryDataSize(size_t dim) {
    return (dim + vecsim_types::sq8::query_metadata_count<Metric>()) * sizeof(float);
}

template <VecSimMetric Metric>
inline IndexComponents<float, float>
CreateTQComponents(std::shared_ptr<VecSimAllocator> allocator, size_t dim, size_t seed,
                   bool use_rotation) {
    unsigned char alignment = 0;
    auto dist_func = GetTQDistFunc<Metric>(dim, &alignment);
    auto *index_calculator = new (allocator) DistanceCalculatorCommon<float>(allocator, dist_func);
    auto *preprocessors =
        new (allocator) MultiPreprocessorsContainer<float, 1>(allocator, alignment);
    auto *tq_preprocessor =
        new (allocator) TQPreprocessor<Metric>(allocator, dim, seed, use_rotation);
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
