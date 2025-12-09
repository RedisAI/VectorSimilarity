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
#include <iostream>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/memory/memory_utils.h"

class PreprocessorInterface : public VecsimBaseObject {
public:
    PreprocessorInterface(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}
    // Note: input_blob_size is relevant for both storage blob and query blob, as we assume results
    // are the same size.
    virtual void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                            size_t &input_blob_size, unsigned char alignment) const = 0;
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

// Scalar Quantization Preprocessor: Quantizes float vectors to int8
// Assumes input vectors are in the range [-1, 1] (e.g., normalized embeddings)
// For Cosine metric, this should be applied AFTER CosinePreprocessor normalization
template <typename DataType = float>
class ScalarQuantizationPreprocessor : public PreprocessorInterface {
public:
    ScalarQuantizationPreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : PreprocessorInterface(allocator), dim(dim), quantized_bytes_count(dim * sizeof(int8_t)),
          input_bytes_count(dim * sizeof(DataType)) {}

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {
        // Verify input size matches expected float vector size
        assert(input_blob_size == input_bytes_count);

        // Case 1: Blobs are different (separate storage and query)
        if (storage_blob != query_blob) {
            // Quantize for storage
            if (storage_blob == nullptr) {
                storage_blob = this->allocator->allocate(quantized_bytes_count);
            }
            quantizeVector(original_blob, storage_blob);

            // Quantize for query
            if (query_blob == nullptr) {
                query_blob = this->allocator->allocate_aligned(quantized_bytes_count, alignment);
            }
            quantizeVector(original_blob, query_blob);
        } else { // Case 2: Blobs are the same
            if (query_blob == nullptr) {
                query_blob = this->allocator->allocate_aligned(quantized_bytes_count, alignment);
                storage_blob = query_blob;
            }
            quantizeVector(original_blob, query_blob);
        }

        input_blob_size = quantized_bytes_count;
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t &input_blob_size) const override {
        assert(input_blob_size == input_bytes_count);

        if (blob == nullptr) {
            blob = this->allocator->allocate(quantized_bytes_count);
        }
        quantizeVector(original_blob, blob);
        input_blob_size = quantized_bytes_count;
    }

    void preprocessQuery(const void *original_blob, void *&blob, size_t &input_blob_size,
                         unsigned char alignment) const override {
        assert(input_blob_size == input_bytes_count);

        if (blob == nullptr) {
            blob = this->allocator->allocate_aligned(quantized_bytes_count, alignment);
        }
        quantizeVector(original_blob, blob);
        input_blob_size = quantized_bytes_count;
    }

    void preprocessStorageInPlace(void *blob, size_t input_blob_size) const override {
        // In-place quantization not supported (type conversion from float to int8)
        // This would require the blob to already be allocated with the correct size
        assert(false && "In-place quantization not supported for type conversion");
    }

private:
    // Quantization parameters for normalized vectors in [-1, 1] range
    // Maps [-1, 1] to [-127, 127] (symmetric quantization)
    static constexpr float SCALE = 127.0f;

    const size_t dim;
    const size_t quantized_bytes_count; // dim * sizeof(int8_t)
    const size_t input_bytes_count;     // dim * sizeof(DataType)

    // Quantize a float vector to int8
    // For normalized vectors in [-1, 1], maps to [-127, 127]
    void quantizeVector(const void *input, void *output) const {
        const DataType *input_vec = static_cast<const DataType *>(input);
        int8_t *output_vec = static_cast<int8_t *>(output);

        for (size_t i = 0; i < dim; i++) {
            // Quantize: int8 = clamp(float * 127, -128, 127)
            float scaled = static_cast<float>(input_vec[i]) * SCALE;

            // Clamp to int8 range [-128, 127]
            if (scaled < -128.0f) {
                output_vec[i] = -128;
            } else if (scaled > 127.0f) {
                output_vec[i] = 127;
            } else {
                output_vec[i] = static_cast<int8_t>(std::round(scaled));
            }
        }

        // std::cout << "quantized_0: " << static_cast<int>(output_vec[0]) << std::endl;
        // std::cout << "original_0: " << input_vec[0] << std::endl;
        // std::cout << "quantized_n: " << static_cast<int>(output_vec[dim - 1]) << std::endl;
        // std::cout << "original_n: " << input_vec[dim - 1] << std::endl;
    }
};
