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
    // Use the the overload below for different sizes.
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
        // This assert verifies that that the current use of this function is for blobs of the same
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
 * The quantized storage blob contains the quantized values along with metadata (min value and
 * scaling factor) in a single contiguous blob. The quantization is done by finding the minimum and
 * maximum values of the input vector, and then scaling the values to fit in the range of [0, 255].
 *
 * The quantized blob size is: dim_elements * sizeof(OUTPUT_TYPE) + 2 * sizeof(DataType)
 */
template <typename DataType>
class QuantPreprocessor : public PreprocessorInterface {
    using OUTPUT_TYPE = uint8_t;

public:
    // Constructor for backward compatibility (single blob size)
    QuantPreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : PreprocessorInterface(allocator), dim(dim),
          storage_bytes_count(dim * sizeof(OUTPUT_TYPE) + 2 * sizeof(DataType)) {
    } // quantized + min + delta

    // Helper function to perform quantization. This function is used by both preprocess and
    // preprocessQuery and supports in-place quantization of the storage blob.
    void quantize(const DataType *input, OUTPUT_TYPE *quantized) const {
        assert(input && quantized);
        // Find min and max values
        auto [min_val, max_val] = find_min_max(input);

        // Calculate scaling factor
        const DataType diff = (max_val - min_val);
        const DataType delta = (diff == DataType{0}) ? DataType{1} : diff / DataType{255};
        const DataType inv_delta = DataType{1} / delta;

        // Quantize the values
        for (size_t i = 0; i < this->dim; i++) {
            quantized[i] = static_cast<OUTPUT_TYPE>(std::round((input[i] - min_val) * inv_delta));
        }

        DataType *metadata = reinterpret_cast<DataType *>(quantized + this->dim);

        // Store min_val, delta, in the metadata
        metadata[0] = min_val;
        metadata[1] = delta;
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {
        // For backward compatibility - delegate to the two-size version with identical sizes
        preprocess(original_blob, storage_blob, query_blob, input_blob_size, input_blob_size,
                   alignment);
    }

    /**
     * Quantizes the storage blob (DataType → OUTPUT_TYPE) while leaving the query blob unchanged.
     *
     * Storage vectors are quantized, while query vectors remain as DataType for asymmetric distance
     * computation.
     *
     * Note: query_blob and query_blob_size are not modified, nor allocated by this function.
     */
    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char alignment) const override {
        // CASE 1: STORAGE BLOB NEEDS ALLOCATION
        if (!storage_blob) {
            // Allocate aligned memory for the quantized storage blob
            storage_blob = static_cast<OUTPUT_TYPE *>(
                this->allocator->allocate_aligned(this->storage_bytes_count, alignment));

            // Quantize directly from original data
            const DataType *input = static_cast<const DataType *>(original_blob);
            quantize(input, static_cast<OUTPUT_TYPE *>(storage_blob));
        }
        // CASE 2: STORAGE BLOB EXISTS
        else {
            // CASE 2A: STORAGE AND QUERY SHARE MEMORY
            if (storage_blob == query_blob) {
                // Need to allocate a separate storage blob since query remains DataType
                // while storage needs to be quantized
                void *new_storage =
                    this->allocator->allocate_aligned(this->storage_bytes_count, alignment);

                // Quantize from the shared blob (query_blob) to the new storage blob
                quantize(static_cast<const DataType *>(query_blob),
                         static_cast<OUTPUT_TYPE *>(new_storage));

                // Update storage_blob to point to the new memory
                storage_blob = new_storage;
            }
            // CASE 2B: SEPARATE STORAGE AND QUERY BLOBS
            else {
                // Check if storage blob needs resizing
                if (storage_blob_size < this->storage_bytes_count) {
                    // Allocate new storage with correct size
                    OUTPUT_TYPE *new_storage = static_cast<OUTPUT_TYPE *>(
                        this->allocator->allocate_aligned(this->storage_bytes_count, alignment));

                    // Quantize from old storage to new storage
                    quantize(static_cast<const DataType *>(storage_blob),
                             static_cast<OUTPUT_TYPE *>(new_storage));

                    // Free old storage and update pointer
                    this->allocator->free_allocation(storage_blob);
                    storage_blob = new_storage;
                } else {
                    // Storage blob is large enough, quantize in-place
                    quantize(static_cast<const DataType *>(storage_blob),
                             static_cast<OUTPUT_TYPE *>(storage_blob));
                }
            }
        }

        storage_blob_size = this->storage_bytes_count;
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t &input_blob_size) const override {
        // Allocate quantized blob if needed
        if (!blob) {
            blob = this->allocator->allocate(storage_bytes_count);
        }

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
