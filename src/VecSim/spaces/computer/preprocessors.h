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
#include "VecSim/types/sq8.h"

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
 * Query processing:
 * The query vector is not quantized. It remains as DataType, but we precompute
 * and store metric-specific values to accelerate asymmetric distance computation:
 * - For IP/Cosine: y_sum = Σy_i (sum of query values)
 * - For L2: y_sum_squares = Σy_i² (sum of squared query values)
 *
 * Query blob layout:
 * | query_values[dim] | y_sum (IP/Cosine) OR y_sum_squares (L2) |
 *
 * Query blob size: (dim + 1) * sizeof(DataType)
 *
 * === Asymmetric distance (storage x quantized, query y remains float) ===
 *
 * For IP/Cosine:
 *   IP(x, y) = Σ(x_i * y_i)
 *            ≈ Σ((min + delta * q_i) * y_i)
 *            = min * Σy_i + delta * Σ(q_i * y_i)
 *            = min * y_sum + delta * quantized_dot_product
 *   where y_sum = Σy_i is precomputed and stored in the query blob.
 *
 * For L2:
 *   ||x - y||² = Σx_i² - 2*Σ(x_i * y_i) + Σy_i²
 *              = x_sum_squares - 2 * IP(x, y) + y_sum_squares
 *   where:
 *     - x_sum_squares = Σx_i² is precomputed and stored in the storage blob
 *     - IP(x, y) is computed using the formula above
 *     - y_sum_squares = Σy_i² is precomputed and stored in the query blob
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
    using sq8 = vecsim_types::sq8;

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
        // 4 independent accumulators (sum)
        DataType s0{}, s1{}, s2{}, s3{};

        // 4 independent accumulators (sum of squares), only used for L2
        DataType q0{}, q1{}, q2{}, q3{};

        size_t i = 0;
        // round dim down to the nearest multiple of 4
        size_t dim_round_down = this->dim & ~size_t(3);

        // Quantize the values
        for (; i < dim_round_down; i += 4) {
            // Load once
            const DataType x0 = input[i + 0];
            const DataType x1 = input[i + 1];
            const DataType x2 = input[i + 2];
            const DataType x3 = input[i + 3];
            // We know (input - min) => 0
            // If min == max, all values are the same and should be quantized to 0.
            // reconstruction will yield the same original value for all vectors.
            quantized[i + 0] = static_cast<OUTPUT_TYPE>(std::round((x0 - min_val) * inv_delta));
            quantized[i + 1] = static_cast<OUTPUT_TYPE>(std::round((x1 - min_val) * inv_delta));
            quantized[i + 2] = static_cast<OUTPUT_TYPE>(std::round((x2 - min_val) * inv_delta));
            quantized[i + 3] = static_cast<OUTPUT_TYPE>(std::round((x3 - min_val) * inv_delta));

            // Accumulate sum for all metrics
            s0 += x0;
            s1 += x1;
            s2 += x2;
            s3 += x3;

            // Accumulate sum of squares only for L2 metric
            if constexpr (Metric == VecSimMetric_L2) {
                q0 += x0 * x0;
                q1 += x1 * x1;
                q2 += x2 * x2;
                q3 += x3 * x3;
            }
        }

        // Tail: 0..3 remaining elements (still the same pass, just finishing work)
        DataType sum = (s0 + s1) + (s2 + s3);
        DataType sum_squares = (q0 + q1) + (q2 + q3);

        for (; i < this->dim; ++i) {
            const DataType x = input[i];
            quantized[i] = static_cast<OUTPUT_TYPE>(std::round((x - min_val) * inv_delta));
            sum += x;
            if constexpr (Metric == VecSimMetric_L2) {
                sum_squares += x * x;
            }
        }

        DataType *metadata = reinterpret_cast<DataType *>(quantized + this->dim);

        // Store min_val, delta, in the metadata
        metadata[sq8::MIN_VAL] = min_val;
        metadata[sq8::DELTA] = delta;

        // Store sum (for all metrics) and sum_squares (for L2 only)
        metadata[sq8::SUM] = sum;
        if constexpr (Metric == VecSimMetric_L2) {
            metadata[sq8::SUM_SQUARES] = sum_squares;
        }
    }

    DataType sum_fast(const DataType *p) const {
        DataType s0{}, s1{}, s2{}, s3{};

        size_t i = 0;
        // round dim down to the nearest multiple of 4
        size_t dim_round_down = this->dim & ~size_t(3);

        for (; i < dim_round_down; i += 4) {
            s0 += p[i + 0];
            s1 += p[i + 1];
            s2 += p[i + 2];
            s3 += p[i + 3];
        }

        DataType sum = (s0 + s1) + (s2 + s3);

        for (; i < dim; ++i) {
            sum += p[i];
        }
        return sum;
    }

    DataType sum_squares_fast(const DataType *p) const {
        DataType s0{}, s1{}, s2{}, s3{};

        size_t i = 0;
        // round dim down to the nearest multiple of 4
        size_t dim_round_down = this->dim & ~size_t(3);

        for (; i < dim_round_down; i += 4) {
            s0 += p[i + 0] * p[i + 0];
            s1 += p[i + 1] * p[i + 1];
            s2 += p[i + 2] * p[i + 2];
            s3 += p[i + 3] * p[i + 3];
        }

        DataType sum = (s0 + s1) + (s2 + s3);

        for (; i < dim; ++i) {
            sum += p[i] * p[i];
        }
        return sum;
    }

public:
    QuantPreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : PreprocessorInterface(allocator), dim(dim),
          storage_bytes_count(dim * sizeof(OUTPUT_TYPE) +
                              (vecsim_types::sq8::metadata_count<Metric>()) * sizeof(DataType)),
          query_bytes_count((dim + 1) * sizeof(DataType)) {
        static_assert(std::is_floating_point_v<DataType>,
                      "QuantPreprocessor only supports floating-point types");
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {
        assert(false &&
               "QuantPreprocessor does not support identical size for storage and query blobs");
    }

    /**
     * Preprocesses the original blob into separate storage and query blobs.
     *
     * Storage vectors are quantized to uint8_t values, with metadata (min, delta, sum, and
     * sum_squares for L2) appended for distance reconstruction.
     *
     * Query vectors remain as DataType for asymmetric distance computation, with a precomputed
     * sum (for IP/Cosine) or sum of squares (for L2) appended for efficient distance calculation.
     *
     * Possible scenarios (currently only CASE 1 is implemented):
     * - CASE 1: STORAGE BLOB AND QUERY BLOB NEED ALLOCATION (storage_blob == query_blob == nullptr)
     * - CASE 2: STORAGE BLOB EXISTS (storage_blob != nullptr)
     *   - CASE 2A: STORAGE BLOB EXISTS and its size is insufficient
     * (storage_blob_size < required_size) - reallocate storage
     *   - CASE 2B: STORAGE AND QUERY SHARE MEMORY (storage_blob == query_blob != nullptr) -
     * reallocate storage
     *   - CASE 2C: SEPARATE STORAGE AND QUERY BLOBS (storage_blob != query_blob) - quantize storage
     * in-place
     */
    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char alignment) const override {
        // CASE 1: STORAGE BLOB NEEDS ALLOCATION - the only implemented case
        assert(!storage_blob && "CASE 1: storage_blob must be nullptr");
        assert(!query_blob && "CASE 1: query_blob must be nullptr");

        // storage_blob_size and query_blob_size must point to different memory slots.
        assert(&storage_blob_size != &query_blob_size);

        // CASE 2A: STORAGE BLOB EXISTS and its size is insufficient - not implemented
        // storage_blob && storage_blob_size < required_size
        // CASE 2B: STORAGE EXISTS AND EQUALS QUERY BLOB - not implemented
        // storage_blob && storage_blob == query_blob
        // (if we want to handle this, we need to separate the blobs)
        // CASE 2C: SEPARATE STORAGE AND QUERY BLOBS - not implemented
        // storage_blob && storage_blob != query_blob
        // We can quantize the storage blob in-place (if we already checked storage_blob_size is
        // sufficient)

        preprocessForStorage(original_blob, storage_blob, storage_blob_size);
        preprocessQuery(original_blob, query_blob, query_blob_size, alignment);
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

    /**
     * Preprocesses the query vector for asymmetric distance computation.
     *
     * The query blob contains the original float values followed by a precomputed value:
     * - For IP/Cosine: y_sum = Σy_i (sum of query values)
     * - For L2: y_sum_squares = Σy_i² (sum of squared query values)
     *
     * Query blob layout: | query_values[dim] | y_sum OR y_sum_squares |
     * Query blob size: (dim + 1) * sizeof(DataType)
     */
    void preprocessQuery(const void *original_blob, void *&blob, size_t &query_blob_size,
                         unsigned char alignment) const override {
        assert(!blob && "query_blob must be nullptr");

        // Allocate aligned memory for the query blob
        blob = this->allocator->allocate_aligned(this->query_bytes_count, alignment);
        memcpy(blob, original_blob, this->dim * sizeof(DataType));
        const DataType *input = static_cast<const DataType *>(original_blob);
        // For IP/Cosine, we need to store the sum of the query vector.
        if constexpr (Metric == VecSimMetric_IP || Metric == VecSimMetric_Cosine) {
            static_cast<DataType *>(blob)[this->dim] = sum_fast(input);
        } // For L2, compute the sum of squares.
        else if constexpr (Metric == VecSimMetric_L2) {
            static_cast<DataType *>(blob)[this->dim] = sum_squares_fast(input);
        }

        query_blob_size = this->query_bytes_count;
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
    const size_t query_bytes_count;
};
