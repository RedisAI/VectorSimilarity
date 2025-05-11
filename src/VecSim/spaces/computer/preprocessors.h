/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/#pragma once

#include <cstddef>
#include <memory>
#include <cassert>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/memory/memory_utils.h"

// TODO: Handle processed_bytes_count that might change down the preprocessors pipeline.
// The preprocess function calls a pipeline of preprocessors, one of which can be a quantization
// preprocessor. In such cases, the quantization preprocessor compresses the vector, resulting in a
// change in the allocation size.
class PreprocessorInterface : public VecsimBaseObject {
public:
    PreprocessorInterface(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}
    // TODO: handle a dynamic processed_bytes_count, as the allocation size of the blob might change
    // down the preprocessors pipeline (such as in quantization preprocessor that compresses the
    // vector).
    virtual void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                            size_t processed_bytes_count, unsigned char alignment) const = 0;
    virtual void preprocessForStorage(const void *original_blob, void *&storage_blob,
                                      size_t processed_bytes_count) const = 0;
    virtual void preprocessQuery(const void *original_blob, void *&query_blob,
                                 size_t processed_bytes_count, unsigned char alignment) const = 0;
    virtual void preprocessQueryInPlace(void *original_blob, size_t processed_bytes_count,
                                        unsigned char alignment) const = 0;
    virtual void preprocessStorageInPlace(void *original_blob,
                                          size_t processed_bytes_count) const = 0;
};

template <typename DataType>
class CosinePreprocessor : public PreprocessorInterface {
public:
    CosinePreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : PreprocessorInterface(allocator), normalize_func(spaces::GetNormalizeFunc<DataType>()),
          dim(dim) {}

    // If a blob (storage_blob or query_blob) is not nullptr, it means a previous preprocessor
    // already allocated and processed it. So, we process it inplace. If it's null, we need to
    // allocate memory for it and copy the original_blob to it.
    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t processed_bytes_count, unsigned char alignment) const override {

        // Case 1: Blobs are different (one might be null, or both are allocated and processed
        // separately).
        if (storage_blob != query_blob) {
            // If one of them is null, allocate memory for it and copy the original_blob to it.
            if (storage_blob == nullptr) {
                storage_blob = this->allocator->allocate(processed_bytes_count);
                memcpy(storage_blob, original_blob, processed_bytes_count);
            } else if (query_blob == nullptr) {
                query_blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                memcpy(query_blob, original_blob, processed_bytes_count);
            }

            // Normalize both blobs.
            normalize_func(storage_blob, this->dim);
            normalize_func(query_blob, this->dim);
        } else { // Case 2: Blobs are the same (either both are null or processed in the same way).
            if (query_blob == nullptr) { // If both blobs are null, allocate query_blob and set
                                         // storage_blob to point to it.
                query_blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                memcpy(query_blob, original_blob, processed_bytes_count);
                storage_blob = query_blob;
            }
            // normalize one of them (since they point to the same memory).
            normalize_func(query_blob, this->dim);
        }
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t processed_bytes_count) const override {
        if (blob == nullptr) {
            blob = this->allocator->allocate(processed_bytes_count);
            memcpy(blob, original_blob, processed_bytes_count);
        }
        normalize_func(blob, this->dim);
    }

    void preprocessQuery(const void *original_blob, void *&blob, size_t processed_bytes_count,
                         unsigned char alignment) const override {
        if (blob == nullptr) {
            blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
            memcpy(blob, original_blob, processed_bytes_count);
        }
        normalize_func(blob, this->dim);
    }

    void preprocessQueryInPlace(void *blob, size_t processed_bytes_count,
                                unsigned char alignment) const override {
        assert(blob);
        normalize_func(blob, this->dim);
    }

    void preprocessStorageInPlace(void *blob, size_t processed_bytes_count) const override {
        assert(blob);
        normalize_func(blob, this->dim);
    }

private:
    spaces::normalizeVector_f<DataType> normalize_func;
    const size_t dim;
};

template <typename DataType>
class QuantPreprocessor : public PreprocessorInterface {
public:
    QuantPreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim, size_t bits_per_dim = 8)
        : PreprocessorInterface(allocator), dim(dim), bits_per_dim(bits_per_dim),
          compressed_bytes_count(calculateCompressedSize(dim)) {}

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t processed_bytes_count, unsigned char alignment) const override {
        // Case 1: Blobs are different (one might be null, or both are allocated and processed separately)
        if (storage_blob != query_blob) {
            // Process storage blob (compress)
            if (storage_blob == nullptr) {
                storage_blob = this->allocator->allocate(compressed_bytes_count);
                quantize(original_blob, storage_blob);
            }
            
            // Query blob remains uncompressed
            if (query_blob == nullptr) {
                query_blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                memcpy(query_blob, original_blob, processed_bytes_count);
            }
        } else { // Case 2: Blobs are the same or both null
            if (query_blob == nullptr) {
                // For query, we keep the original format
                query_blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                memcpy(query_blob, original_blob, processed_bytes_count);
                
                // For storage, we compress
                storage_blob = this->allocator->allocate(compressed_bytes_count);
                quantize(original_blob, storage_blob);
            } else {
                // If both point to the same memory, we need to separate them
                void* new_storage = this->allocator->allocate(compressed_bytes_count);
                quantize(query_blob, new_storage);
                storage_blob = new_storage;
            }
        }
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t processed_bytes_count) const override {
        if (blob == nullptr) {
            blob = this->allocator->allocate(compressed_bytes_count);
            quantize(original_blob, blob);
        } else {
            // If blob is already allocated, we need to compress in-place
            void* temp = this->allocator->allocate(compressed_bytes_count);
            quantize(blob, temp);
            this->allocator->free_allocation(blob);
            blob = temp;
        }
    }

    void preprocessQuery(const void *original_blob, void *&blob, size_t processed_bytes_count,
                         unsigned char alignment) const override {
        // For query, we keep the original format
        if (blob == nullptr) {
            blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
            memcpy(blob, original_blob, processed_bytes_count);
        }
    }

    void preprocessQueryInPlace(void *blob, size_t processed_bytes_count,
                                unsigned char alignment) const override {
        // No compression for query vectors
        assert(blob);
    }

    void preprocessStorageInPlace(void *blob, size_t processed_bytes_count) const override {
        assert(blob);
        // Create temporary storage for compressed data
        void* temp = this->allocator->allocate(compressed_bytes_count);
        quantize(blob, temp);
        
        // Copy compressed data back to original location
        // Note: This assumes blob has enough space for the compressed data
        memcpy(blob, temp, compressed_bytes_count);
        this->allocator->free_allocation(temp);
    }

private:
    const size_t dim;
    const size_t bits_per_dim;
    const size_t compressed_bytes_count;

    // Calculate the size needed for the compressed vector
    static size_t calculateCompressedSize(size_t dim) {
        // Quantized values (int8 per dimension) + min (float32) + delta (float32)
        return dim * sizeof(int8_t) + 2 * sizeof(float);
    }

    // Quantize the vector from original format to compressed format
    void quantize(const void *src, void *dst) const {
        const DataType* src_data = static_cast<const DataType*>(src);
        
        // Find min and max values in the vector
        DataType min_val = src_data[0];
        DataType max_val = src_data[0];
        
        for (size_t i = 0; i < dim; i++) {
            DataType val = src_data[i];
            min_val = val < min_val ? val : min_val;
            max_val = val > max_val ? val : max_val;
        }
        
        // Calculate delta (quantization step)
        float delta = (max_val - min_val) / 255.0f;
        if (delta == 0){
            delta = 1.0f; // Avoid division by zero if all values are the same
        }
        
        // Structure of compressed data:
        // [quantized values (int8_t * dim)][min_val (float)][delta (float)]
        int8_t* quant_values = static_cast<int8_t*>(dst); // convert to int8_t pointer
        float* params = reinterpret_cast<float*>(quant_values + dim); // convert to float pointer starting after quantized values
        
        // Store min and delta values for dequantization
        params[0] = static_cast<float>(min_val);
        params[1] = delta;
        
        // Quantize each value
        for (size_t i = 0; i < dim; i++) {
            float normalized = (src_data[i] - min_val) / delta;
            if (normalized < 0) normalized = 0;
            if (normalized > 255) normalized = 255;
            quant_values[i] = static_cast<int8_t>(normalized);
        }
    }
};
