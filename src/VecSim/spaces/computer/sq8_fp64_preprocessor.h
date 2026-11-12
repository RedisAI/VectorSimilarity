/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/spaces/computer/preprocessors.h"
#include "VecSim/types/sq8_fp64.h"

// Cosine inputs must already be normalized by the caller.
template <VecSimMetric Metric>
class SQ8FP64Preprocessor : public PreprocessorInterface {
    static_assert(Metric == VecSimMetric_L2 || Metric == VecSimMetric_IP ||
                  Metric == VecSimMetric_Cosine);

public:
    SQ8FP64Preprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim);
    void preprocess(const void *original, void *&storage, void *&query, size_t &storage_size,
                    size_t &query_size, unsigned char storage_alignment,
                    unsigned char query_alignment) const override;
    void preprocessForStorage(const void *original, void *&storage, size_t &storage_size,
                              unsigned char alignment) const override;
    void preprocessQuery(const void *original, void *&query, size_t &query_size,
                         unsigned char alignment) const override;
    void preprocessStorageInPlace(void *original, size_t input_size) const override;

private:
    static uint8_t to_code(double scaled);
    void quantize(const double *input, uint8_t *output) const;
    const size_t dim;
};

template <VecSimMetric Metric>
SQ8FP64Preprocessor<Metric>::SQ8FP64Preprocessor(std::shared_ptr<VecSimAllocator> allocator,
                                                 size_t dim)
    : PreprocessorInterface(allocator), dim(dim) {
    assert(dim > 0);
}

template <VecSimMetric Metric>
uint8_t SQ8FP64Preprocessor<Metric>::to_code(double scaled) {
    if (!(scaled > 0.0)) {
        return 0;
    }

    if (scaled >= vecsim_types::sq8_fp64::max_code) {
        return vecsim_types::sq8_fp64::max_code;
    }

    return static_cast<uint8_t>(scaled + 0.5);
}

template <VecSimMetric Metric>
void SQ8FP64Preprocessor<Metric>::quantize(const double *input, uint8_t *output) const {
    const auto [min_it, max_it] = std::minmax_element(input, input + dim);
    const double minimum = *min_it;
    const double range = *max_it - minimum;
    const double delta = range == 0.0 ? 1.0 : range / vecsim_types::sq8_fp64::max_code;
    double sum = 0.0;
    double squares = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const uint8_t code = to_code((input[i] - minimum) / delta);
        output[i] = code;
        const double value = minimum + delta * code;
        sum += value;
        if constexpr (Metric == VecSimMetric_L2) {
            squares += value * value;
        }
    }
    const double metadata[] = {minimum, delta, sum, squares};
    std::memcpy(output + dim, metadata,
                vecsim_types::sq8_fp64::storage_metadata_count<Metric>() * sizeof(double));
}

template <VecSimMetric Metric>
void SQ8FP64Preprocessor<Metric>::preprocess(const void *original, void *&storage, void *&query,
                                             size_t &storage_size, size_t &query_size,
                                             unsigned char storage_alignment,
                                             unsigned char query_alignment) const {
    assert(!storage && !query);
    assert(&storage_size != &query_size);
    preprocessForStorage(original, storage, storage_size, storage_alignment);
    preprocessQuery(original, query, query_size, query_alignment);
}

template <VecSimMetric Metric>
void SQ8FP64Preprocessor<Metric>::preprocessForStorage(const void *original, void *&storage,
                                                       size_t &storage_size,
                                                       unsigned char alignment) const {
    assert(!storage);
    storage_size = vecsim_types::sq8_fp64::storage_bytes_count<Metric>(dim);
    storage = this->allocator->allocate_aligned(storage_size, alignment);
    quantize(static_cast<const double *>(original), static_cast<uint8_t *>(storage));
}

template <VecSimMetric Metric>
void SQ8FP64Preprocessor<Metric>::preprocessQuery(const void *original, void *&query,
                                                  size_t &query_size,
                                                  unsigned char alignment) const {
    assert(!query);
    const auto *const input = static_cast<const double *>(original);
    double sum = 0.0;
    double squares = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += input[i];
        if constexpr (Metric == VecSimMetric_L2) {
            squares += input[i] * input[i];
        }
    }
    query_size = vecsim_types::sq8_fp64::query_bytes_count<Metric>(dim);
    query = this->allocator->allocate_aligned(query_size, alignment);
    std::memcpy(query, original, dim * sizeof(double));
    const double metadata[] = {sum, squares};
    std::memcpy(static_cast<uint8_t *>(query) + dim * sizeof(double), metadata,
                vecsim_types::sq8_fp64::query_metadata_count<Metric>() * sizeof(double));
}

template <VecSimMetric Metric>
void SQ8FP64Preprocessor<Metric>::preprocessStorageInPlace(void *original,
                                                           size_t input_size) const {
    assert(input_size >= vecsim_types::sq8_fp64::storage_bytes_count<Metric>(dim));
    quantize(static_cast<const double *>(original), static_cast<uint8_t *>(original));
}
