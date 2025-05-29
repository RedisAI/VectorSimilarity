/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include <cstddef>
#include <memory>
#include <cassert>
#include <cmath>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/memory/memory_utils.h"

class PreprocessorInterface : public VecsimBaseObject {
public:
    PreprocessorInterface(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}
    // Note: input_blob_size is relevant for both storage blob and query blob, as we assume results
    // are the same size.
    // TODO: Add query_blob_size as a parameter to the preprocess functions, to allow
    // different sizes for storage and query blobs in the future, if needed.
    virtual void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                            size_t &input_blob_size, unsigned char alignment) const = 0;
    virtual void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                            size_t &storage_blob_size, size_t &query_blob_size,
                            unsigned char alignment) const = 0;
    virtual void preprocessForStorage(const void *original_blob, void *&storage_blob,
                                      size_t &input_blob_size) const = 0;
    virtual void preprocessQuery(const void *original_blob, void *&query_blob,
                                 size_t &input_blob_size, unsigned char alignment) const = 0;
    virtual void preprocessQueryInPlace(void *original_blob, size_t input_blob_size,
                                        unsigned char alignment) const = 0;
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
        // This assert verifies that that the current use of this function is for blobs of the same
        // size, which is the case for the Cosine preprocessor. If we ever need to support different
        // sizes for storage and query blobs, we can remove the assert and implement the logic to
        // handle different sizes.
        assert(storage_blob_size == query_blob_size);

        preprocess(original_blob, storage_blob, query_blob, storage_blob_size, alignment);
        query_blob_size =
            storage_blob_size; // Ensure both blobs have the same size after processing.
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

    void preprocessQueryInPlace(void *blob, size_t input_blob_size,
                                unsigned char alignment) const override {
        assert(blob);
        assert(input_blob_size == this->processed_bytes_count);
        normalize_func(blob, this->dim);
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

// QuantPreprocessor is a preprocessor that quantizes the input vector of fp32 to a lower precision
// of uint8_t. The quantization is done by finding the minimum and maximum values of the input
// vector, and then scaling the values to fit in the range of [0, 255]. The quantized values are
// then stored in a uint8_t array. [Quantized values, min, delta] Quantized Blob size  =
// dim_elements * sizeof(int8)  +  2 * sizeof(float) delta = (max_val - min_val) / 255.0f
// quantized_v[i] = (v[i] - min_val) / delta
// preprocessForStorage:
// if null:
//      - We are not reallocing because it will be released after the query.
//      Allocate quantized blob size
// 3. Compute (min, delta) and quantize to the quantized blob or in place.
// preprocessQuery: No-op – queries arrive as float32 and remain uncompressed

class QuantPreprocessor : public PreprocessorInterface {
public:
    // Constructor for backward compatibility (single blob size)
    QuantPreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : PreprocessorInterface(allocator), dim(dim),
          storage_bytes_count(dim * sizeof(uint8_t) + 2 * sizeof(float)) {
    } // quantized + min + delta + inverted_norm {}

    // Helper function to perform quantization
    void quantize(const float *input, uint8_t *quantized) const {
        assert(dim > 0);
        // Find min and max values
        float min_val = input[0];
        float max_val = input[0];
        for (size_t i = 1; i < this->dim; i++) {
            min_val = std::min(min_val, input[i]);
            max_val = std::max(max_val, input[i]);
        }

        // Calculate scaling factor
        const float diff = (max_val - min_val);
        const float delta = diff == 0.0f ? 1.0f : diff / 255.0f;
        const float inv_delta = 1.0f / delta;

        // Quantize the values
        for (size_t i = 0; i < this->dim; i++) {
            quantized[i] = static_cast<uint8_t>(std::round((input[i] - min_val) * inv_delta));
        }

        // Reserve space for metadata at the end of the blob
        // [quantized values, min_val, delta, inverted_norm]
        float *metadata = reinterpret_cast<float *>(quantized + this->dim);

        // Store min_val, delta, and inverted_norm in the metadata
        metadata[0] = min_val;
        metadata[1] = delta;
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {
        // For backward compatibility - both blobs have the same size
        preprocess(original_blob, storage_blob, query_blob, input_blob_size, input_blob_size,
                   alignment);
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char alignment) const override {

        if (storage_blob) {
            if (storage_blob == query_blob) {
                storage_blob =
                    this->allocator->allocate_aligned(this->storage_bytes_count, alignment);
            } else if (storage_blob_size < storage_bytes_count) {
                // Check if the blob allocated by a previous preprocessor is big enough, otherwise,
                // realloc it. Can happen when the dim is smaller than the quantization metadata.
                // For example, din size is 2 so the storage_blob_size is 2 * sizeof(float) = 8
                // bytes. But the quantized blob size is 2 * sizeof(uint8_t) + 2 * sizeof(float) =
                // 10 bytes.
                storage_blob = this->allocator->reallocate(storage_blob, this->storage_bytes_count);
            }

        } else {
            // storage_blob is nullptr, so we need to allocate new memory
            storage_blob = this->allocator->allocate_aligned(this->storage_bytes_count, alignment);
        }
        storage_blob_size = this->storage_bytes_count;

        // Cast to appropriate types
        const float *input = static_cast<const float *>(original_blob);
        uint8_t *quantized = static_cast<uint8_t *>(storage_blob);
        quantize(input, quantized);
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t &input_blob_size) const override {
        // Allocate quantized blob if needed
        if (blob == nullptr) {
            blob = this->allocator->allocate(storage_bytes_count);
        }

        // Cast to appropriate types
        const float *input = static_cast<const float *>(original_blob);
        uint8_t *quantized = static_cast<uint8_t *>(blob);
        quantize(input, quantized);

        input_blob_size = storage_bytes_count;
    }

    void preprocessQuery(const void *original_blob, void *&blob, size_t &query_blob_size,
                         unsigned char alignment) const override {
        // No-op: queries remain as float32
        if (blob == nullptr) {
            blob = this->allocator->allocate_aligned(query_blob_size, alignment);
            memcpy(blob, original_blob, query_blob_size);
        }
    }

    void preprocessQueryInPlace(void *blob, size_t input_blob_size,
                                unsigned char alignment) const override {
        // No-op: queries remain as float32
        assert(blob);
    }

    void preprocessStorageInPlace(void *original_blob, size_t input_blob_size) const override {
        // This function is unused for this preprocessor.
        assert(original_blob);
        assert(input_blob_size >= storage_bytes_count);
    }

private:
    const size_t dim;
    const size_t storage_bytes_count;
};
