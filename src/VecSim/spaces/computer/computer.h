/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "VecSim/vec_sim_common.h"
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/spaces/computer/preprocessor.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/memory/memory_utils.h"

struct ProcessedBlobs {
    explicit ProcessedBlobs() = default;

    explicit ProcessedBlobs(MemoryUtils::unique_blob &&storage_blob,
                            MemoryUtils::unique_blob &&query_blob)
        : storage_blob(std::move(storage_blob)), query_blob(std::move(query_blob)) {}

    ProcessedBlobs(ProcessedBlobs &&other) noexcept
        : storage_blob(std::move(other.storage_blob)), query_blob(std::move(other.query_blob)) {}

    // Move assignment operator
    ProcessedBlobs &operator=(ProcessedBlobs &&other) noexcept {
        if (this != &other) {
            storage_blob = std::move(other.storage_blob);
            query_blob = std::move(other.query_blob);
        }
        return *this;
    }

    // Delete copy constructor and assignment operator to avoid copying unique_ptr
    ProcessedBlobs(const ProcessedBlobs &) = delete;
    ProcessedBlobs &operator=(const ProcessedBlobs &) = delete;

    const void *getStorageBlob() const { return storage_blob.get(); }
    const void *getQueryBlob() const { return query_blob.get(); }

private:
    MemoryUtils::unique_blob storage_blob;
    MemoryUtils::unique_blob query_blob;
};

template <typename DistType>
class IndexComputerAbstract : public VecsimBaseObject {
public:
    IndexComputerAbstract(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}

    virtual ProcessedBlobs preprocess(const void *blob, size_t processed_bytes_count) const = 0;

    virtual MemoryUtils::unique_blob preprocessForStorage(const void *blob,
                                                          size_t processed_bytes_count) const = 0;

    virtual MemoryUtils::unique_blob preprocessQuery(const void *blob,
                                                     size_t processed_bytes_count) const = 0;

    virtual void preprocessQueryInPlace(void *blob, size_t processed_bytes_count) const = 0;

    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const = 0;
    // TODO: remove alignment once datablock is implemented in HNSW
    virtual unsigned char getAlignment() const = 0;
};

template <typename DistType, typename DistFuncType>
class IndexComputerBasic : public IndexComputerAbstract<DistType> {
public:
    IndexComputerBasic(
        std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment,
        DistanceCalculatorAbstract<DistType, DistFuncType> *distance_calculator = nullptr)
        : IndexComputerAbstract<DistType>(allocator), alignment(alignment),
          distance_calculator(distance_calculator) {}

    ~IndexComputerBasic() override {
        if (distance_calculator) {
            delete distance_calculator;
        }
    }

    ProcessedBlobs preprocess(const void *original_blob,
                              size_t processed_bytes_count) const override;

    MemoryUtils::unique_blob preprocessForStorage(const void *original_blob,
                                                  size_t processed_bytes_count) const override;

    unsigned char getAlignment() const override { return alignment; }

    MemoryUtils::unique_blob preprocessQuery(const void *original_blob,
                                             size_t processed_bytes_count) const override;

    void preprocessQueryInPlace(void *blob, size_t processed_bytes_count) const override;

    DistType calcDistance(const void *v1, const void *v2, size_t dim) const override {
        assert(this->distance_calculator);
        return this->distance_calculator->calcDistance(v1, v2, dim);
    }

protected:
    const unsigned char alignment;
    DistanceCalculatorAbstract<DistType, DistFuncType> *distance_calculator;

    // Allocate and copy the blob only if the original blob is not aligned.
    MemoryUtils::unique_blob maybeCopyToAlignedMem(const void *original_blob,
                                                   size_t blob_bytes_count) const;

    MemoryUtils::unique_blob wrapAllocated(void *blob) const {
        return MemoryUtils::unique_blob(
            blob, [this](void *ptr) { this->allocator->free_allocation(ptr); });
    }

    static MemoryUtils::unique_blob wrapWithDummyDeleter(void *ptr) {
        return MemoryUtils::unique_blob(ptr, [](void *) {});
    }
};

template <typename DistType, typename DistFuncType, size_t n_preprocessors>
class IndexComputerExtended : public IndexComputerBasic<DistType, DistFuncType> {
protected:
    std::array<PreprocessorAbstract *, n_preprocessors> preprocessors;

public:
    IndexComputerExtended(
        std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment,
        DistanceCalculatorAbstract<DistType, DistFuncType> *distance_calculator = nullptr)
        : IndexComputerBasic<DistType, DistFuncType>(allocator, alignment, distance_calculator) {
        assert(n_preprocessors);
        std::fill_n(preprocessors.begin(), n_preprocessors, nullptr);
    }

    ~IndexComputerExtended() override {
        for (auto pp : preprocessors) {
            if (!pp)
                break;

            delete pp;
        }
    }

    /** @returns On success, next uninitialized index, or 0 in case capacity is reached (after
     * inserting the preprocessor). -1 if capacity is full and we failed to add the preprocessor.
     */
    int addPreprocessor(PreprocessorAbstract *preprocessor);

    ProcessedBlobs preprocess(const void *original_blob,
                              size_t processed_bytes_count) const override;

    MemoryUtils::unique_blob preprocessForStorage(const void *original_blob,
                                                  size_t processed_bytes_count) const override;

    MemoryUtils::unique_blob preprocessQuery(const void *original_blob,
                                             size_t processed_bytes_count) const override;

    void preprocessQueryInPlace(void *blob, size_t processed_bytes_count) const override;

private:
    using Base = IndexComputerBasic<DistType, DistFuncType>;
};

/* ====================================== Implementation ======================================*/

/* ======================= IndexComputerBasic ======================= */

template <typename DistType, typename DistFuncType>
ProcessedBlobs
IndexComputerBasic<DistType, DistFuncType>::preprocess(const void *original_blob,
                                                       size_t processed_bytes_count) const {
    auto storage_blob = preprocessForStorage(original_blob, processed_bytes_count);
    auto query_blob = preprocessQuery(original_blob, processed_bytes_count);
    return ProcessedBlobs(std::move(storage_blob), std::move(query_blob));
}

template <typename DistType, typename DistFuncType>
MemoryUtils::unique_blob IndexComputerBasic<DistType, DistFuncType>::preprocessForStorage(
    const void *original_blob, size_t processed_bytes_count) const {
    return std::move(wrapWithDummyDeleter(const_cast<void *>(original_blob)));
}

template <typename DistType, typename DistFuncType>
MemoryUtils::unique_blob
IndexComputerBasic<DistType, DistFuncType>::preprocessQuery(const void *original_blob,
                                                            size_t processed_bytes_count) const {
    return maybeCopyToAlignedMem(original_blob, processed_bytes_count);
}

template <typename DistType, typename DistFuncType>
void IndexComputerBasic<DistType, DistFuncType>::preprocessQueryInPlace(
    void *blob, size_t processed_bytes_count) const {}

template <typename DistType, typename DistFuncType>
MemoryUtils::unique_blob
IndexComputerBasic<DistType, DistFuncType>::maybeCopyToAlignedMem(const void *original_blob,
                                                                  size_t blob_bytes_count) const {
    if (this->alignment) {
        if ((uintptr_t)original_blob % this->alignment) {
            auto aligned_mem = this->allocator->allocate_aligned(blob_bytes_count, this->alignment);
            memcpy(aligned_mem, original_blob, blob_bytes_count);
            return std::move(this->wrapAllocated(aligned_mem));
        }
    }

    // Returning a unique_ptr with a no-op deleter
    return std::move(wrapWithDummyDeleter(const_cast<void *>(original_blob)));
}

/* ======================= IndexComputerExtended ======================= */

template <typename DistType, typename DistFuncType, size_t n_preprocessors>
int IndexComputerExtended<DistType, DistFuncType, n_preprocessors>::addPreprocessor(
    PreprocessorAbstract *preprocessor) {
    for (size_t i = 0; i < n_preprocessors; i++) {
        if (preprocessors[i] == nullptr) {
            preprocessors[i] = preprocessor;
            return i + 1 >= n_preprocessors ? 0 : i + 1;
        }
    }
    return -1;
}

template <typename DistType, typename DistFuncType, size_t n_preprocessors>
ProcessedBlobs IndexComputerExtended<DistType, DistFuncType, n_preprocessors>::preprocess(
    const void *original_blob, size_t processed_bytes_count) const {
    // No preprocessors were added yet.
    if (preprocessors[0] == nullptr) {
        // query might need to be aligned
        auto query_ptr = this->maybeCopyToAlignedMem(original_blob, processed_bytes_count);
        return ProcessedBlobs(
            std::move(Base::wrapWithDummyDeleter(const_cast<void *>(original_blob))),
            std::move(query_ptr));
    }

    void *storage_blob = nullptr;
    void *query_blob = nullptr;
    for (auto pp : preprocessors) {
        if (!pp)
            break;
        pp->preprocess(original_blob, storage_blob, query_blob, processed_bytes_count,
                       this->alignment);
    }

    // If they point to the same memory, we need to free only one of them.
    if (storage_blob == query_blob) {
        return ProcessedBlobs(std::move(this->wrapAllocated(storage_blob)),
                              std::move(Base::wrapWithDummyDeleter(storage_blob)));
    }

    if (storage_blob == nullptr) { // we processed only the query
        return ProcessedBlobs(
            std::move(Base::wrapWithDummyDeleter(const_cast<void *>(original_blob))),
            std::move(this->wrapAllocated(query_blob)));
    }

    if (query_blob == nullptr) { // we processed only the storage
        // query might need to be aligned
        auto query_ptr = this->maybeCopyToAlignedMem(original_blob, processed_bytes_count);
        return ProcessedBlobs(std::move(this->wrapAllocated(storage_blob)), std::move(query_ptr));
    }

    // Else, both were allocated separately, we need to release both.
    return ProcessedBlobs(std::move(this->wrapAllocated(storage_blob)),
                          std::move(this->wrapAllocated(query_blob)));
}

template <typename DistType, typename DistFuncType, size_t n_preprocessors>
MemoryUtils::unique_blob
IndexComputerExtended<DistType, DistFuncType, n_preprocessors>::preprocessForStorage(
    const void *original_blob, size_t processed_bytes_count) const {

    void *storage_blob = nullptr;
    for (auto pp : preprocessors) {
        if (!pp)
            break;
        pp->preprocessForStorage(original_blob, storage_blob, processed_bytes_count);
    }

    return storage_blob ? std::move(this->wrapAllocated(storage_blob))
                        : std::move(Base::wrapWithDummyDeleter(const_cast<void *>(original_blob)));
}

template <typename DistType, typename DistFuncType, size_t n_preprocessors>
MemoryUtils::unique_blob
IndexComputerExtended<DistType, DistFuncType, n_preprocessors>::preprocessQuery(
    const void *original_blob, size_t processed_bytes_count) const {

    void *query_blob = nullptr;
    for (auto pp : preprocessors) {
        if (!pp)
            break;
        // modifies the memory in place
        pp->preprocessQuery(original_blob, query_blob, processed_bytes_count, this->alignment);
    }
    return query_blob
               ? std::move(this->wrapAllocated(query_blob))
               : std::move(this->maybeCopyToAlignedMem(original_blob, processed_bytes_count));
}

template <typename DistType, typename DistFuncType, size_t n_preprocessors>
void IndexComputerExtended<DistType, DistFuncType, n_preprocessors>::preprocessQueryInPlace(
    void *blob, size_t processed_bytes_count) const {

    for (auto pp : preprocessors) {
        if (!pp)
            break;
        // modifies the memory in place
        pp->preprocessQueryInPlace(blob, processed_bytes_count, this->alignment);
    }
}
