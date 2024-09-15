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
    struct PreprocessParams;
    virtual void preprocess(const void *original_blob, MemoryUtils::unique_blob &storage_blob,
                            MemoryUtils::unique_blob &query_blob,
                            PreprocessParams &params) const = 0;
    virtual void preprocessForStorage(MemoryUtils::unique_blob &blob) const = 0;
    virtual void preprocessQuery(MemoryUtils::unique_blob &blob) const = 0;

    virtual bool hasQueryPreprocessor() const = 0;
    virtual bool hasStoragePreprocessor() const = 0;

    struct PreprocessParams {
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

    void preprocess(const void *original_blob, MemoryUtils::unique_blob &storage_blob,
                    MemoryUtils::unique_blob &query_blob, PreprocessParams &params) const override {
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

    void preprocessForStorage(MemoryUtils::unique_blob &blob) const override {
        normalize_func(blob.get(), this->dim);
    }

    void preprocessQuery(MemoryUtils::unique_blob &blob) const override {
        normalize_func(blob.get(), this->dim);
    }

    bool hasQueryPreprocessor() const override { return true; };

    bool hasStoragePreprocessor() const override { return true; };

private:
    spaces::normalizeVector_f<DataType> normalize_func;
    const size_t dim;
};
