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

    // TODO: add storage_blob_size and query_blob_size parameters
    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size,  unsigned char alignment) const override {
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


// QuantPreprocessor is a preprocessor that quantizes the input vector of fp32 to a lower precision of uint8_t.
// The quantization is done by finding the minimum and maximum values of the input vector, and then
// scaling the values to fit in the range of [0, 255]. The quantized values are then stored in a uint8_t array.
// [Quantized values, min, delta]
// Quantized Blob size  =  dim_elements * sizeof(int8)  +  2 * sizeof(float)
// delta = (max_val - min_val) / 255.0f
// quantized_v[i] = (v[i] - min_val) / delta
// preprocessForStorage:
// if null:
//      - We are not reallocing because it will be released after the query.
//      Allocate quantized blob size
// 3. Compute (min, delta) and quantize to the quantized blob or in place.
// preprocessQuery: No-op – queries arrive as float32 and remain uncompressed


// preprocess -> 
// if storage_blob == null || storage_blob == query_blob:
//      allocate storage blob with storage_blob_size
// edge case:
// Check if the input size is not big enough and if not, reallocate it, and update the size. -add this scenario to tests
// quantize it the storage_blob


// add class member const size_t storage_bytes_count, and calculate it as follows:
// storage_bytes_count = dim * sizeof(uint8_t) + 2 * sizeof(float);


template <typename DataType>
class QuantPreprocessor : public PreprocessorInterface {
public:
    QuantPreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim,
                      size_t processed_bytes_count)
        : PreprocessorInterface(allocator), dim(dim), processed_bytes_count(processed_bytes_count) {}

    // Helper function to perform quantization
    void quantize(const float* input, uint8_t* output, float& min_val, float& delta) const {
        // Find min and max values
        min_val = input[0];
        float max_val = input[0];
        for (size_t i = 1; i < dim; i++) {
            min_val = std::min(min_val, input[i]);
            max_val = std::max(max_val, input[i]);
        }
        
        // Calculate scaling factor
        delta = (max_val - min_val) / 255.0f;
        
        // Quantize values
        if (delta > 0) {
            for (size_t i = 0; i < dim; i++) {
                output[i] = static_cast<uint8_t>((input[i] - min_val) / delta);
            }
        } else {
            // Handle case where all values are the same
            memset(output, 0, dim);
        }
    }

    // Calculate the norm of the dequantized vector
    float calculateDequantizedNorm(const uint8_t* quantized, const float min_val, const float delta) const {
        float norm = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float dequantized = min_val + quantized[i] * delta;
            norm += dequantized * dequantized;
        }
        return norm > 0 ? 1.0f / std::sqrt(norm) : 0.0f;
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
        } else { // Case 2: Blobs are the same (either both are null or processed in the same way).
            if (query_blob == nullptr) { // If both blobs are null, allocate query_blob and set
                                         // storage_blob to point to it.
                query_blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                memcpy(query_blob, original_blob, input_blob_size);
                storage_blob = query_blob;
            }
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
        
        // Only quantize if input is float32 and we're quantizing to uint8_t
        if constexpr (std::is_same_v<DataType, float>) {
            // Cast to appropriate types
            const float* input = static_cast<const float*>(original_blob);
            uint8_t* quantized = static_cast<uint8_t*>(blob);
            
            // Reserve space for metadata at the end of the blob
            // [quantized values, min_val, delta, inverted_norm]
            float* metadata = reinterpret_cast<float*>(quantized + dim);
            
            // Perform quantization
            float min_val, delta;
            quantize(input, quantized, min_val, delta);
            
            // Store min_val and delta in the metadata
            metadata[0] = min_val;
            metadata[1] = delta;
            
            // Calculate and store the inverted norm for cosine similarity
            metadata[2] = calculateDequantizedNorm(quantized, min_val, delta);
        }
        
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
        input_blob_size = processed_bytes_count;
    }

    void preprocessQueryInPlace(void *blob, size_t input_blob_size,
                                unsigned char alignment) const override {
        assert(blob);
        assert(input_blob_size == this->processed_bytes_count);
    }

    void preprocessStorageInPlace(void *blob, size_t input_blob_size) const override {
        assert(blob);
        assert(input_blob_size == this->processed_bytes_count);
        
        if constexpr (std::is_same_v<DataType, float>) {
            // Cast to appropriate types
            float* input = static_cast<float*>(blob);
            uint8_t* quantized = static_cast<uint8_t*>(blob);
            
            // Create temporary copy of input data
            std::vector<float> temp_input(input, input + dim);
            
            // Reserve space for metadata at the end of the blob
            float* metadata = reinterpret_cast<float*>(quantized + dim);
            
            // Perform quantization
            float min_val, delta;
            quantize(temp_input.data(), quantized, min_val, delta);
            
            // Store min_val and delta in the metadata
            metadata[0] = min_val;
            metadata[1] = delta;
            
            // Calculate and store the inverted norm for cosine similarity
            metadata[2] = calculateDequantizedNorm(quantized, min_val, delta);
        }
    }

private:
    const size_t dim;
    const size_t processed_bytes_count;
};
