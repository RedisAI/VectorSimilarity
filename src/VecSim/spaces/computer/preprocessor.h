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
    // virtual void preprocessForStorage() = 0;
    virtual void preprocessQuery(std::unique_ptr<void, alloc_deleter_t> &blob) = 0;
    // pre process for const memory, need to copy.
};

template <typename DataType>
class CosinePreprocessor : public PreprocessorAbstract {
public:
    CosinePreprocessor(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : PreprocessorAbstract(allocator), normalize_func(spaces::GetNormalizeFunc<DataType>()),
          dim(dim) {}

    // virtual void preprocessForStorage(std::unique_ptr<void, alloc_deleter_t> blob, size_t dim) {
    //     normalize_func(blob, this->dim);
    // }

    virtual void preprocessQuery(std::unique_ptr<void, alloc_deleter_t> &blob) override {
        normalize_func(blob.get(), this->dim);
    }

private:
    spaces::normalizeVector_f<DataType> normalize_func;
    const size_t dim;
};
