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
#include <concepts>
#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/memory/memory_utils.h"
#include "VecSim/types/float16.h"
#include "VecSim/types/sq8.h"

class PreprocessorInterface : public VecsimBaseObject {
public:
    PreprocessorInterface(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}
    // Combined preprocessing into both storage and query blobs. storage_alignment applies to any
    // newly allocated storage blob; query_alignment applies to any newly allocated query blob.
    // Implementations that allocate a single shared buffer for both must align it to satisfy both
    // requirements (use combineAlignments).
    virtual void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                            size_t &storage_blob_size, size_t &query_blob_size,
                            unsigned char storage_alignment,
                            unsigned char query_alignment) const = 0;
    virtual void preprocessForStorage(const void *original_blob, void *&storage_blob,
                                      size_t &input_blob_size,
                                      unsigned char storage_alignment) const = 0;
    virtual void preprocessQuery(const void *original_blob, void *&query_blob,
                                 size_t &input_blob_size, unsigned char query_alignment) const = 0;
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
                    unsigned char storage_alignment, unsigned char query_alignment) const override {
        // CosinePreprocessor produces equally-sized storage and query blobs.
        assert(storage_blob_size == query_blob_size);
        // see assert docs below
        assert(storage_blob == nullptr || storage_blob_size == processed_bytes_count);
        assert(query_blob == nullptr || query_blob_size == processed_bytes_count);

        // Case 1: Blobs are different (one might be null, or both are allocated and processed
        // separately).
        if (storage_blob != query_blob) {
            // If one of them is null, allocate memory for it and copy the original_blob to it.
            if (storage_blob == nullptr) {
                storage_blob =
                    this->allocator->allocate_aligned(processed_bytes_count, storage_alignment);
                memcpy(storage_blob, original_blob, storage_blob_size);
            } else if (query_blob == nullptr) {
                query_blob =
                    this->allocator->allocate_aligned(processed_bytes_count, query_alignment);
                memcpy(query_blob, original_blob, query_blob_size);
            }

            // Normalize both blobs.
            normalize_func(storage_blob, this->dim);
            normalize_func(query_blob, this->dim);
        } else { // Case 2: Blobs are the same (either both are null or processed in the same way).
            if (query_blob == nullptr) {
                // Single buffer must satisfy both the storage and the query alignment hint.
                const unsigned char shared_alignment =
                    spaces::combineAlignments(storage_alignment, query_alignment);
                query_blob =
                    this->allocator->allocate_aligned(processed_bytes_count, shared_alignment);
                memcpy(query_blob, original_blob, storage_blob_size);
                storage_blob = query_blob;
            }
            // normalize one of them (since they point to the same memory).
            normalize_func(query_blob, this->dim);
        }

        storage_blob_size = processed_bytes_count;
        query_blob_size = processed_bytes_count;
    }

    void preprocessForStorage(const void *original_blob, void *&blob, size_t &input_blob_size,
                              unsigned char storage_alignment) const override {
        // The assert here verifies that if a blob was allocated by a previous preprocessor, its
        // size matches our expected processed size, allowing in-place normalization. Dynamic
        // resizing is intentionally not supported: handling it would require runtime size checks
        // and reallocation logic in a hot path, and no current caller needs it.
        assert(blob == nullptr || input_blob_size == processed_bytes_count);

        if (blob == nullptr) {
            blob = this->allocator->allocate_aligned(processed_bytes_count, storage_alignment);
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
 * Storage metadata is always FP32 (independent of DataType) to match the asymmetric distance
 * kernels. The quantized blob size is:
 * - For L2:        dim * sizeof(OUTPUT_TYPE) + 4 * sizeof(float)
 * - For IP/Cosine: dim * sizeof(OUTPUT_TYPE) + 3 * sizeof(float)
 *
 * Reconstruction formulas:
 * Given quantized value q_i, the original value is reconstructed as:
 *   x_i ≈ min + delta * q_i
 *
 * Query processing:
 * The query vector is not quantized. It remains in DataType width (FP32 stays FP32, FP16 stays
 * FP16), but we precompute and store metric-specific FP32 values to accelerate asymmetric
 * distance computation:
 * - For IP/Cosine: y_sum = Σy_i (sum of query values)
 * - For L2: y_sum = Σy_i (sum of query values), y_sum_squares = Σy_i² (sum of squared query values)
 *
 * Query blob layout:
 * - For IP/Cosine: | query_values[dim] | y_sum |
 * - For L2:        | query_values[dim] | y_sum | y_sum_squares |
 *
 * Query metadata is always FP32. The query blob size is:
 * - For IP/Cosine: dim * sizeof(DataType) + 1 * sizeof(float)
 * - For L2:        dim * sizeof(DataType) + 2 * sizeof(float)
 *
 * Note: when DataType is float16 the metadata region may not be 4-byte aligned; both writes
 * and reads of metadata must therefore go through memcpy.
 *
 * === Asymmetric distance (storage x quantized, query y in DataType) ===
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
// Input types accepted by QuantPreprocessor. Opt-in via std::same_as so unrelated types
// (e.g. integers, double, bfloat16) are rejected at the template head with a named constraint.
template <typename T>
concept QuantInput = std::same_as<T, float> || std::same_as<T, vecsim_types::float16>;

// Convert a single input element to FP32 for accumulation/comparison. Identity for float,
// FP16 -> FP32 widening for vecsim_types::float16.
template <QuantInput T>
static inline float to_fp32(T x) {
    if constexpr (std::is_same_v<T, vecsim_types::float16>) {
        return vecsim_types::FP16_to_FP32(x);
    } else {
        return x;
    }
}

template <QuantInput DataType, VecSimMetric Metric>
class QuantPreprocessor : public PreprocessorInterface {
    using OUTPUT_TYPE = uint8_t;
    using MetadataType = float; // SQ8 metadata is always FP32 (see class doc).
    using sq8 = vecsim_types::sq8;

    static_assert(Metric == VecSimMetric_L2 || Metric == VecSimMetric_IP ||
                      Metric == VecSimMetric_Cosine ||
                      Metric == VecSimMetric_CosineSimilarity,
                  "QuantPreprocessor only supports L2, IP and cosine-based metrics");

    // Helper function to perform quantization. This function is used by the storage preprocessing
    // methods.
    void quantize(const DataType *input, OUTPUT_TYPE *quantized) const {
        assert(input && quantized);
        // Find min and max values (computed in MetadataType regardless of DataType).
        auto [min_val, max_val] = find_min_max(input);

        // Calculate scaling factor (typed as MetadataType because they end up as metadata).
        const MetadataType diff = (max_val - min_val);
        const MetadataType delta = (diff == 0.0f) ? MetadataType{1} : diff / MetadataType{255};
        const MetadataType inv_delta = MetadataType{1} / delta;

        // Compute sum (and sum of squares for L2) while quantizing.
        // Accumulators are FP32 to preserve metadata precision for FP16 inputs.
        // 4 independent accumulators (sum)
        float s0{}, s1{}, s2{}, s3{};

        // 4 independent accumulators (sum of squares), only used for L2
        float q0{}, q1{}, q2{}, q3{};

        size_t i = 0;
        // round dim down to the nearest multiple of 4
        size_t dim_round_down = this->dim & ~size_t(3);

        // Quantize the values
        for (; i < dim_round_down; i += 4) {
            // Load once (widened to FP32 if DataType is FP16).
            const float x0 = to_fp32<DataType>(input[i + 0]);
            const float x1 = to_fp32<DataType>(input[i + 1]);
            const float x2 = to_fp32<DataType>(input[i + 2]);
            const float x3 = to_fp32<DataType>(input[i + 3]);
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

        // Tail: 0..3 remaining elements (still the same pass, just finishing work).
        // Sum/sum_squares become metadata, so they are MetadataType.
        MetadataType sum = (s0 + s1) + (s2 + s3);
        MetadataType sum_squares = (q0 + q1) + (q2 + q3);

        for (; i < this->dim; ++i) {
            const float x = to_fp32<DataType>(input[i]);
            quantized[i] = static_cast<OUTPUT_TYPE>(std::round((x - min_val) * inv_delta));
            sum += x;
            if constexpr (Metric == VecSimMetric_L2) {
                sum_squares += x * x;
            }
        }

        // Metadata uses MetadataType. Use memcpy because the metadata offset
        // (dim * sizeof(uint8_t)) is not guaranteed to be sizeof(MetadataType)-aligned.
        void *meta_dst = quantized + this->dim;
        if constexpr (Metric == VecSimMetric_L2) {
            const MetadataType buf[4] = {min_val, delta, sum, sum_squares};
            memcpy(meta_dst, buf, sizeof(buf));
        } else {
            const MetadataType buf[3] = {min_val, delta, sum};
            memcpy(meta_dst, buf, sizeof(buf));
        }
    }

    // Computes and writes query metadata (FP32) in a single pass over the input vector.
    // For IP/Cosine: writes y_sum = Σy_i
    // For L2: writes y_sum = Σy_i and y_sum_squares = Σy_i²
    // The output pointer addresses the metadata region after the query body and may not be
    // 4-byte aligned (e.g. FP16 query body with odd dim), so writes go through memcpy.
    void assign_query_metadata(const DataType *input, void *output_metadata) const {
        // Accumulators are FP32 to preserve precision for FP16 inputs.
        // 4 independent accumulators for sum
        float s0{}, s1{}, s2{}, s3{};
        // 4 independent accumulators for sum of squares (only used for L2)
        float q0{}, q1{}, q2{}, q3{};

        size_t i = 0;
        // round dim down to the nearest multiple of 4
        size_t dim_round_down = this->dim & ~size_t(3);

        for (; i < dim_round_down; i += 4) {
            const float y0 = to_fp32<DataType>(input[i + 0]);
            const float y1 = to_fp32<DataType>(input[i + 1]);
            const float y2 = to_fp32<DataType>(input[i + 2]);
            const float y3 = to_fp32<DataType>(input[i + 3]);

            s0 += y0;
            s1 += y1;
            s2 += y2;
            s3 += y3;

            if constexpr (Metric == VecSimMetric_L2) {
                q0 += y0 * y0;
                q1 += y1 * y1;
                q2 += y2 * y2;
                q3 += y3 * y3;
            }
        }

        // Sum/sum_squares become metadata, so they are MetadataType.
        MetadataType sum = (s0 + s1) + (s2 + s3);
        MetadataType sum_squares = (q0 + q1) + (q2 + q3);

        // Tail: handle remaining elements
        for (; i < this->dim; ++i) {
            const float y = to_fp32<DataType>(input[i]);
            sum += y;
            if constexpr (Metric == VecSimMetric_L2) {
                sum_squares += y * y;
            }
        }

        // Metadata uses MetadataType. Use memcpy because the metadata offset (after the query
        // body of dim * sizeof(DataType)) is not guaranteed to be sizeof(MetadataType)-aligned
        // when DataType is float16 and dim is odd.
        if constexpr (Metric == VecSimMetric_L2) {
            const MetadataType buf[2] = {sum, sum_squares};
            memcpy(output_metadata, buf, sizeof(buf));
        } else {
            const MetadataType buf[1] = {sum};
            memcpy(output_metadata, buf, sizeof(buf));
        }
    }

public:
    QuantPreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : PreprocessorInterface(allocator), dim(dim),
          storage_bytes_count(dim * sizeof(OUTPUT_TYPE) +
                              (vecsim_types::sq8::storage_metadata_count<Metric>()) *
                                  sizeof(MetadataType)),
          query_bytes_count(dim * sizeof(DataType) +
                            (vecsim_types::sq8::query_metadata_count<Metric>()) *
                                sizeof(MetadataType)) {}

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
                    unsigned char storage_alignment, unsigned char query_alignment) const override {
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

        preprocessForStorage(original_blob, storage_blob, storage_blob_size, storage_alignment);
        preprocessQuery(original_blob, query_blob, query_blob_size, query_alignment);
    }

    void preprocessForStorage(const void *original_blob, void *&blob, size_t &input_blob_size,
                              unsigned char storage_alignment) const override {
        assert(!blob && "storage_blob must be nullptr");

        blob = this->allocator->allocate_aligned(storage_bytes_count, storage_alignment);
        // Cast to appropriate types
        const DataType *input = static_cast<const DataType *>(original_blob);
        OUTPUT_TYPE *quantized = static_cast<OUTPUT_TYPE *>(blob);
        quantize(input, quantized);

        input_blob_size = storage_bytes_count;
    }

    /**
     * Preprocesses the query vector for asymmetric distance computation.
     *
     * The query blob contains the original DataType values followed by FP32 precomputed values:
     * - For IP/Cosine: y_sum = Σy_i (sum of query values)
     * - For L2: y_sum = Σy_i (sum of query values), y_sum_squares = Σy_i² (sum of squared query
     *                                                                      values)
     *
     * Query blob layout:
     * - For IP/Cosine: | query_values[dim] | y_sum |
     * - For L2:        | query_values[dim] | y_sum | y_sum_squares |
     *
     * Query blob size:
     * - For IP/Cosine: dim * sizeof(DataType) + 1 * sizeof(float)
     * - For L2:        dim * sizeof(DataType) + 2 * sizeof(float)
     */
    void preprocessQuery(const void *original_blob, void *&blob, size_t &query_blob_size,
                         unsigned char alignment) const override {
        assert(!blob && "query_blob must be nullptr");

        // Allocate aligned memory for the query blob
        blob = this->allocator->allocate_aligned(this->query_bytes_count, alignment);
        const size_t body_bytes = this->dim * sizeof(DataType);
        memcpy(blob, original_blob, body_bytes);
        const DataType *input = static_cast<const DataType *>(original_blob);

        // Compute and write FP32 query metadata after the query body. The metadata offset is
        // body_bytes, which is not guaranteed to be 4-byte aligned for FP16 query bodies.
        void *metadata_dst = static_cast<uint8_t *>(blob) + body_bytes;
        assign_query_metadata(input, metadata_dst);

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
    // Returns (min, max) of the input vector evaluated in MetadataType. Both float and float16
    // expose a usable operator< (float16's overload delegates to FP32 semantics), so a single
    // std::minmax_element call covers both DataTypes. The returned values are written verbatim
    // into the metadata region.
    std::pair<MetadataType, MetadataType> find_min_max(const DataType *input) const {
        auto [min_it, max_it] = std::minmax_element(input, input + dim);
        return {to_fp32<DataType>(*min_it), to_fp32<DataType>(*max_it)};
    }

    const size_t dim;
    const size_t storage_bytes_count;
    const size_t query_bytes_count;
};
