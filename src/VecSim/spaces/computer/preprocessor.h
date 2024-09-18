/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#include <cstddef>
#include <memory>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/memory/memory_utils.h"

class PreprocessorAbstract : public VecsimBaseObject {
public:
    PreprocessorAbstract(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}
    virtual void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                            size_t processed_bytes_count, unsigned char alignment) const = 0;
    virtual void preprocessForStorage(const void *original_blob, void *&storage_blob,
                                      size_t processed_bytes_count) const = 0;
    virtual void preprocessQuery(const void *original_blob, void *&query_blob,
                                 size_t processed_bytes_count, unsigned char alignment) const = 0;
};

template <typename DataType>
class CosinePreprocessor : public PreprocessorAbstract {
public:
    CosinePreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : PreprocessorAbstract(allocator), normalize_func(spaces::GetNormalizeFunc<DataType>()),
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

private:
    spaces::normalizeVector_f<DataType> normalize_func;
    const size_t dim;
};
