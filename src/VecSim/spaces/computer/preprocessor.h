/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#include <cstddef>
#include <memory>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"

class PreprocessorAbstract : public VecsimBaseObject {
public:
    PreprocessorAbstract(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}
    struct PreprocessorParams;
    virtual void preprocess(const void *original_blob,
                            std::unique_ptr<void, alloc_deleter_t> &storage_blob,
                            std::unique_ptr<void, alloc_deleter_t> &query_blob,
                            PreprocessorParams &params) = 0;
    virtual void preprocessForStorage(std::unique_ptr<void, alloc_deleter_t> &blob) = 0;
    virtual void preprocessQuery(std::unique_ptr<void, alloc_deleter_t> &blob) = 0;

    struct PreprocessorParams {
        const size_t processed_bytes_count;
        bool is_populated_storage;
        bool is_populated_query;
    };
};

template <typename DataType>
class CosinePreprocessor : public PreprocessorAbstract {
public:
    CosinePreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : PreprocessorAbstract(allocator), normalize_func(spaces::GetNormalizeFunc<DataType>()),
          dim(dim) {}

    virtual void preprocess(const void *original_blob,
                            std::unique_ptr<void, alloc_deleter_t> &storage_blob,
                            std::unique_ptr<void, alloc_deleter_t> &query_blob,
                            PreprocessorParams &params) override {
        if (!params.is_populated_storage) {
            memcpy(storage_blob.get(), original_blob, params.processed_bytes_count);
            params.is_populated_storage = true;
        }
        normalize_func(storage_blob.get(), this->dim);

        if (!params.is_populated_query) {
            // No need to normalize again, just copy the normalized vector
            memcpy(query_blob.get(), storage_blob.get(), params.processed_bytes_count);
            params.is_populated_query = true;
        } else {
            // Normalize the query vector
            normalize_func(query_blob.get(), this->dim);
        }
    }

    virtual void preprocessForStorage(std::unique_ptr<void, alloc_deleter_t> &blob) override {
        normalize_func(blob.get(), this->dim);
    }

    virtual void preprocessQuery(std::unique_ptr<void, alloc_deleter_t> &blob) override {
        normalize_func(blob.get(), this->dim);
    }

private:
    spaces::normalizeVector_f<DataType> normalize_func;
    const size_t dim;
};
