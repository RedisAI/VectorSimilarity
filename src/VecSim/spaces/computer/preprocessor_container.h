/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <cstddef>
#include <memory>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/memory/memory_utils.h"
#include "VecSim/spaces/computer/preprocessors.h"

struct ProcessedBlobs;

class PreprocessorsContainerAbstract : public VecsimBaseObject {
public:
    PreprocessorsContainerAbstract(std::shared_ptr<VecSimAllocator> allocator,
                                   unsigned char alignment)
        : VecsimBaseObject(allocator), alignment(alignment) {}
    virtual ProcessedBlobs preprocess(const void *original_blob,
                                      size_t processed_bytes_count) const;

    virtual MemoryUtils::unique_blob preprocessForStorage(const void *original_blob,
                                                          size_t processed_bytes_count) const;

    virtual MemoryUtils::unique_blob preprocessQuery(const void *original_blob,
                                                     size_t processed_bytes_count,
                                                     bool force_copy = false) const;

    virtual void preprocessQueryInPlace(void *blob, size_t processed_bytes_count) const;

    virtual void preprocessStorageInPlace(void *blob, size_t processed_bytes_count) const;

    unsigned char getAlignment() const { return alignment; }

protected:
    const unsigned char alignment;

    // Allocate and copy the blob only if the original blob is not aligned.
    MemoryUtils::unique_blob maybeCopyToAlignedMem(const void *original_blob,
                                                   size_t blob_bytes_count,
                                                   bool force_copy = false) const;

    MemoryUtils::unique_blob wrapAllocated(void *blob) const {
        return MemoryUtils::unique_blob(
            blob, [this](void *ptr) { this->allocator->free_allocation(ptr); });
    }

    static MemoryUtils::unique_blob wrapWithDummyDeleter(void *ptr) {
        return MemoryUtils::unique_blob(ptr, [](void *) {});
    }
};

template <typename DataType, size_t n_preprocessors>
class MultiPreprocessorsContainer : public PreprocessorsContainerAbstract {
protected:
    std::array<PreprocessorInterface *, n_preprocessors> preprocessors;

public:
    MultiPreprocessorsContainer(std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment)
        : PreprocessorsContainerAbstract(allocator, alignment) {
        assert(n_preprocessors);
        std::fill_n(preprocessors.begin(), n_preprocessors, nullptr);
    }

    ~MultiPreprocessorsContainer() override {
        for (auto pp : preprocessors) {
            if (!pp)
                break;

            delete pp;
        }
    }

    /** @returns On success, next uninitialized index, or 0 in case capacity is reached (after
     * inserting the preprocessor). -1 if capacity is full and we failed to add the preprocessor.
     */
    int addPreprocessor(PreprocessorInterface *preprocessor);

    ProcessedBlobs preprocess(const void *original_blob,
                              size_t processed_bytes_count) const override;

    MemoryUtils::unique_blob preprocessForStorage(const void *original_blob,
                                                  size_t processed_bytes_count) const override;

    MemoryUtils::unique_blob preprocessQuery(const void *original_blob,
                                             size_t processed_bytes_count,
                                             bool force_copy = false) const override;

    void preprocessQueryInPlace(void *blob, size_t processed_bytes_count) const override;

    void preprocessStorageInPlace(void *blob, size_t processed_bytes_count) const override;

#ifdef BUILD_TESTS
    std::array<PreprocessorInterface *, n_preprocessors> getPreprocessors() const {
        return preprocessors;
    }
#endif

private:
    using Base = PreprocessorsContainerAbstract;
};

/* ======================= ProcessedBlobs Definition ======================= */

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

/* ====================================== Implementation ======================================*/

/* ======================= MultiPreprocessorsContainer ======================= */

// On success, returns the array size after adding the preprocessor, or 0 when we add the last
// preprocessor. Returns -1 if the array is full and we failed to add the preprocessor.
template <typename DataType, size_t n_preprocessors>
int MultiPreprocessorsContainer<DataType, n_preprocessors>::addPreprocessor(
    PreprocessorInterface *preprocessor) {
    for (size_t curr_pp_idx = 0; curr_pp_idx < n_preprocessors; curr_pp_idx++) {
        if (preprocessors[curr_pp_idx] == nullptr) {
            preprocessors[curr_pp_idx] = preprocessor;
            const size_t pp_arr_size = curr_pp_idx + 1;
            return pp_arr_size >= n_preprocessors ? 0 : pp_arr_size;
        }
    }
    return -1;
}

template <typename DataType, size_t n_preprocessors>
ProcessedBlobs MultiPreprocessorsContainer<DataType, n_preprocessors>::preprocess(
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
    // At least one blob was allocated.

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

template <typename DataType, size_t n_preprocessors>
MemoryUtils::unique_blob
MultiPreprocessorsContainer<DataType, n_preprocessors>::preprocessForStorage(
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

template <typename DataType, size_t n_preprocessors>
MemoryUtils::unique_blob MultiPreprocessorsContainer<DataType, n_preprocessors>::preprocessQuery(
    const void *original_blob, size_t processed_bytes_count, bool force_copy) const {

    void *query_blob = nullptr;
    for (auto pp : preprocessors) {
        if (!pp)
            break;
        // modifies the memory in place
        pp->preprocessQuery(original_blob, query_blob, processed_bytes_count, this->alignment);
    }
    return query_blob ? std::move(this->wrapAllocated(query_blob))
                      : std::move(this->maybeCopyToAlignedMem(original_blob, processed_bytes_count,
                                                              force_copy));
}

template <typename DataType, size_t n_preprocessors>
void MultiPreprocessorsContainer<DataType, n_preprocessors>::preprocessQueryInPlace(
    void *blob, size_t processed_bytes_count) const {

    for (auto pp : preprocessors) {
        if (!pp)
            break;
        // modifies the memory in place
        pp->preprocessQueryInPlace(blob, processed_bytes_count, this->alignment);
    }
}

template <typename DataType, size_t n_preprocessors>
void MultiPreprocessorsContainer<DataType, n_preprocessors>::preprocessStorageInPlace(
    void *blob, size_t processed_bytes_count) const {

    for (auto pp : preprocessors) {
        if (!pp)
            break;
        // modifies the memory in place
        pp->preprocessStorageInPlace(blob, processed_bytes_count);
    }
}
