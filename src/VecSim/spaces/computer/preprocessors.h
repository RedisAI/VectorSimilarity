/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/memory/memory_utils.h"

class PreprocessorInterface : public VecsimBaseObject {
public:
    PreprocessorInterface(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}
    // Note: input_blob_size is relevant for both storage blob and query blob, as we assume results
    // are the same size.
    // Use the overload below for different sizes.
    virtual void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                            size_t &input_blob_size, unsigned char alignment) const = 0;
    virtual void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                            size_t &storage_blob_size, size_t &query_blob_size,
                            unsigned char alignment) const = 0;
    virtual void preprocessForStorage(const void *original_blob, void *&storage_blob,
                                      size_t &input_blob_size) const = 0;
    virtual void preprocessQuery(const void *original_blob, void *&query_blob,
                                 size_t &input_blob_size, unsigned char alignment) const = 0;
    virtual void preprocessStorageInPlace(void *original_blob, size_t input_blob_size) const = 0;
};

template <typename DataType>
class CosinePreprocessor : public PreprocessorInterface {
public:
    // This preprocessor requires that storage_blob and query_blob have identical memory sizes
    // both before processing (as input) and after preprocessing completes.
    CosinePreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim,
                       size_t processed_bytes_count)
        : PreprocessorInterface(allocator), normalize_func(spaces::GetNormalizeFunc<DataType>()),
          dim(dim), processed_bytes_count(processed_bytes_count) {}

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char alignment) const override {
        // This assert verifies that the current use of this function is for blobs of the same
        // size, which is the case for the Cosine preprocessor. If we ever need to support different
        // sizes for storage and query blobs, we can remove the assert and implement the logic to
        // handle different sizes.
        assert(storage_blob_size == query_blob_size);

        preprocess(original_blob, storage_blob, query_blob, storage_blob_size, alignment);
        // Ensure both blobs have the same size after processing.
        query_blob_size = storage_blob_size;
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {
        // This assert verifies that if a blob was allocated by a previous preprocessor, its
        // size matches our expected processed size. Therefore, it is safe to skip re-allocation and
        // process it inplace. Supporting dynamic resizing would require additional size checks (if
        // statements) and memory management logic, which could impact performance. Currently, no
        // code path requires this capability. If resizing becomes necessary in the future, remove
        // the assertions and implement appropriate allocation handling with performance
        // considerations.
        assert(storage_blob == nullptr || input_blob_size == processed_bytes_count);
        assert(query_blob == nullptr || input_blob_size == processed_bytes_count);

        // Case 1: Blobs are different (one might be null, or both are allocated and processed
        // separately).
        if (storage_blob != query_blob) {
            // If one of them is null, allocate memory for it and copy the original_blob to it.
            if (storage_blob == nullptr) {
                storage_blob = this->allocator->allocate(processed_bytes_count);
                memcpy(storage_blob, original_blob, input_blob_size);
            } else if (query_blob == nullptr) {
                query_blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                memcpy(query_blob, original_blob, input_blob_size);
            }

            // Normalize both blobs.
            normalize_func(storage_blob, this->dim);
            normalize_func(query_blob, this->dim);
        } else { // Case 2: Blobs are the same (either both are null or processed in the same way).
            if (query_blob == nullptr) { // If both blobs are null, allocate query_blob and set
                                         // storage_blob to point to it.
                query_blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                memcpy(query_blob, original_blob, input_blob_size);
                storage_blob = query_blob;
            }
            // normalize one of them (since they point to the same memory).
            normalize_func(query_blob, this->dim);
        }

        input_blob_size = processed_bytes_count;
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t &input_blob_size) const override {
        // see assert docs in preprocess
        assert(blob == nullptr || input_blob_size == processed_bytes_count);

        if (blob == nullptr) {
            blob = this->allocator->allocate(processed_bytes_count);
            memcpy(blob, original_blob, input_blob_size);
        }
        normalize_func(blob, this->dim);
        input_blob_size = processed_bytes_count;
    }

    void preprocessQuery(const void *original_blob, void *&blob, size_t &input_blob_size,
                         unsigned char alignment) const override {
        // see assert docs in preprocess
        assert(blob == nullptr || input_blob_size == processed_bytes_count);
        if (blob == nullptr) {
            blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
            memcpy(blob, original_blob, input_blob_size);
        }
        normalize_func(blob, this->dim);
        input_blob_size = processed_bytes_count;
    }

    void preprocessStorageInPlace(void *blob, size_t input_blob_size) const override {
        assert(blob);
        assert(input_blob_size == this->processed_bytes_count);
        normalize_func(blob, this->dim);
    }

private:
    spaces::normalizeVector_f<DataType> normalize_func;
    const size_t dim;
    const size_t processed_bytes_count;
};

/*
 * QuantPreprocessor is a preprocessor that quantizes storage vectors from DataType to a
 * lower precision representation using OUTPUT_TYPE (uint8_t).
 * Query vectors remain as DataType for asymmetric distance computation.
 *
 * The quantized storage blob contains the quantized values along with metadata (min value,
 * scaling factor, and precomputed sums for reconstruction) in a single contiguous blob.
 * The quantization is done by finding the minimum and maximum values of the input vector,
 * and then scaling the values to fit in the range of [0, 255].
 *
 * Storage layout:
 * | quantized_values[dim] | min_val | delta | x_sum | (x_sum_squares for L2 only) |
 * where:
 * x_sum = Σx_i: sum of the original values,
 * x_sum_squares = Σx_i²: sum of squares of the original values.
 *
 * The quantized blob size is:
 * - For L2:        dim * sizeof(OUTPUT_TYPE) + 4 * sizeof(DataType)
 * - For IP/Cosine: dim * sizeof(OUTPUT_TYPE) + 3 * sizeof(DataType)
 *
 * Reconstruction formulas:
 * Given quantized value q_i, the original value is reconstructed as:
 *   x_i ≈ min + delta * q_i
 *
 * === Asymmetric distance (storage x quantized, query y remains float) ===
 *
 * For IP/Cosine:
 *   IP(x, y) = Σ(x_i * y_i)
 *            ≈ Σ((min + delta * q_i) * y_i)
 *            = min * Σy_i + delta * Σ(q_i * y_i)
 *            = min * sum_query + delta * quantized_dot_product
 *   where sum_query = Σy_i is computed at query time.
 *
 * For L2:
 *   ||x - y||² = Σx_i² - 2*Σ(x_i * y_i) + Σy_i²
 *              = sum_squares - 2 * IP(x, y) + sum_sq_query
 *   where:
 *     - sum_squares = Σx_i² is precomputed and stored
 *     - IP(x, y) is computed using the formula above
 *     - sum_sq_query = Σy_i² is computed at query time
 *
 * === Symmetric distance (both x and y are quantized) ===
 *
 * For IP/Cosine:
 *   IP(x, y) = Σ((min_x + delta_x * qx_i) * (min_y + delta_y * qy_i))
 *            = dim * min_x * min_y
 *              + min_x * delta_y * Σqy_i + min_y * delta_x * Σqx_i
 *              + delta_x * delta_y * Σ(qx_i * qy_i)
 *            = dim * min_x * min_y
 *              + min_x * (sum_y - dim * min_y) + min_y * (sum_x - dim * min_x)
 *              + delta_x * delta_y * Σ(qx_i * qy_i)
 *            = min_x * sum_y + min_y * sum_x - dim * min_x * min_y
 *              + delta_x * delta_y * Σ(qx_i * qy_i)
 *   where:
 *     - sum_x, sum_y are precomputed sums of original values
 *     - Σqx_i = (sum_x - dim * min_x) / delta_x  (sum of quantized values, derived from stored sum)
 *     - Σqy_i = (sum_y - dim * min_y) / delta_y
 *
 * For L2:
 *   ||x - y||² = sum_sq_x + sum_sq_y - 2 * IP(x, y)
 *   where sum_sq_x, sum_sq_y are precomputed sums of squared original values.
 */
template <typename DataType, VecSimMetric Metric>
class QuantPreprocessor : public PreprocessorInterface {
    using OUTPUT_TYPE = uint8_t;

    // For L2:   store sum + sum_of_squares (2 extra values)
    // For IP/Cosine: store only sum (1 extra value)
    static constexpr size_t extra_values_count = (Metric == VecSimMetric_L2) ? 2 : 1;
    static_assert(Metric == VecSimMetric_L2 || Metric == VecSimMetric_IP ||
                      Metric == VecSimMetric_Cosine,
                  "QuantPreprocessor only supports L2, IP and Cosine metrics");

    // Helper function to perform quantization. This function is used by the storage preprocessing
    // methods.
    void quantize(const DataType *input, OUTPUT_TYPE *quantized) const {
        assert(input && quantized);
        // Find min and max values
        auto [min_val, max_val] = find_min_max(input);

        // Calculate scaling factor
        const DataType diff = (max_val - min_val);
        // Delta = diff / 255.0f
        const DataType delta = (diff == DataType{0}) ? DataType{1} : diff / DataType{255};
        const DataType inv_delta = DataType{1} / delta;

        // Compute sum (and sum of squares for L2) while quantizing
        DataType sum = DataType{0};
        DataType sum_squares = DataType{0};

        // Quantize the values
        for (size_t i = 0; i < this->dim; i++) {
            // We know (input - min) => 0
            // If min == max, all values are the same and should be quantized to 0.
            // reconstruction will yield the same original value for all vectors.
            quantized[i] = static_cast<OUTPUT_TYPE>(std::round((input[i] - min_val) * inv_delta));

            // Accumulate sum for all metrics
            sum += input[i];
            // Accumulate sum of squares only for L2 metric
            if constexpr (Metric == VecSimMetric_L2) {
                sum_squares += input[i] * input[i];
            }
        }

        DataType *metadata = reinterpret_cast<DataType *>(quantized + this->dim);

        // Store min_val, delta, in the metadata
        metadata[0] = min_val;
        metadata[1] = delta;

        // Store sum (for all metrics) and sum_squares (for L2 only)
        metadata[2] = sum;
        if constexpr (Metric == VecSimMetric_L2) {
            metadata[3] = sum_squares;
        }
    }

public:
    QuantPreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : PreprocessorInterface(allocator), dim(dim),
          storage_bytes_count(dim * sizeof(OUTPUT_TYPE) +
                              (2 + extra_values_count) * sizeof(DataType)) {
        static_assert(std::is_floating_point_v<DataType>,
                      "QuantPreprocessor only supports floating-point types");
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {
        assert(false &&
               "QuantPreprocessor does not support identical size for storage and query blobs");
    }

    /**
     * Quantizes the storage blob (DataType → OUTPUT_TYPE) while leaving the query blob unchanged.
     *
     * Storage vectors are quantized to uint8_t values, with metadata (min, delta, sum, and
     * sum_squares for L2) appended for distance reconstruction. Query vectors remain as DataType
     * for asymmetric distance computation.
     *
     * Note: query_blob and query_blob_size are not modified, nor allocated by this function.
     *
     * Possible scenarios (currently only CASE 1 is implemented):
     * - CASE 1: STORAGE BLOB NEEDS ALLOCATION (storage_blob == nullptr)
     * - CASE 2: STORAGE BLOB EXISTS (storage_blob != nullptr)
     *   - CASE 2A: STORAGE BLOB EXISTS and its size is insufficient
     * (storage_blob_size < required_size) - reallocate storage
     *   - CASE 2B: STORAGE AND QUERY SHARE MEMORY (storage_blob == query_blob) - reallocate storage
     *   - CASE 2C: SEPARATE STORAGE AND QUERY BLOBS (storage_blob != query_blob) - quantize storage
     * in-place
     */
    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char alignment) const override {
        // CASE 1: STORAGE BLOB NEEDS ALLOCATION - the only implemented case
        assert(!storage_blob && "CASE 1: storage_blob must be nullptr");

        // CASE 2A: STORAGE BLOB EXISTS and its size is insufficient - not implemented
        // storage_blob && storage_blob_size < required_size
        // CASE 2B: STORAGE EXISTS AND EQUALS QUERY BLOB - not implemented
        // storage_blob && storage_blob == query_blob
        // (if we want to handle this, we need to separate the blobs)
        // CASE 2C: SEPARATE STORAGE AND QUERY BLOBS - not implemented
        // storage_blob && storage_blob != query_blob
        // We can quantize the storage blob in-place (if we already checked storage_blob_size is
        // sufficient)

        // Allocate aligned memory for the quantized storage blob
        storage_blob = static_cast<OUTPUT_TYPE *>(
            this->allocator->allocate_aligned(this->storage_bytes_count, alignment));

        // Quantize directly from original data
        const DataType *input = static_cast<const DataType *>(original_blob);
        quantize(input, static_cast<OUTPUT_TYPE *>(storage_blob));

        storage_blob_size = this->storage_bytes_count;
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t &input_blob_size) const override {
        assert(!blob && "storage_blob must be nullptr");

        blob = this->allocator->allocate(storage_bytes_count);
        // Cast to appropriate types
        const DataType *input = static_cast<const DataType *>(original_blob);
        OUTPUT_TYPE *quantized = static_cast<OUTPUT_TYPE *>(blob);
        quantize(input, quantized);

        input_blob_size = storage_bytes_count;
    }

    void preprocessQuery(const void *original_blob, void *&blob, size_t &query_blob_size,
                         unsigned char alignment) const override {
        // No-op: queries remain as original DataType
    }

    void preprocessStorageInPlace(void *original_blob, size_t input_blob_size) const override {
        assert(original_blob);
        assert(input_blob_size >= storage_bytes_count &&
               "Input buffer too small for in-place quantization");

        quantize(static_cast<const DataType *>(original_blob),
                 static_cast<OUTPUT_TYPE *>(original_blob));
    }

private:
    std::pair<DataType, DataType> find_min_max(const DataType *input) const {
        auto [min_it, max_it] = std::minmax_element(input, input + dim);
        return {*min_it, *max_it};
    }

    const size_t dim;
    const size_t storage_bytes_count;
};
