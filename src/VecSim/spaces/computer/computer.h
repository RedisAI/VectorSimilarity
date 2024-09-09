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

#include <iostream> // TODO: REMOVE!!!!!
// TODO: think where to place

static void dummyFreeAllocation(void *ptr) {}

// TODO : move oto a general lcation and use in preprocess.h
using unique_blob = std::unique_ptr<void, alloc_deleter_t>;

struct ProcessedBlobs {
    explicit ProcessedBlobs() = default;

    explicit ProcessedBlobs(const void *storage_blob, const void *query_blob)
        : storage_blob(const_cast<void *>(storage_blob), dummyFreeAllocation),
          query_blob(const_cast<void *>(query_blob), dummyFreeAllocation) {}

    explicit ProcessedBlobs(unique_blob &&storage_blob, unique_blob &&query_blob)
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

    static unique_blob uniqueWithDummyDeleter(void *ptr) {
        return unique_blob(ptr, dummyFreeAllocation);
    }

private:
    unique_blob storage_blob;
    unique_blob query_blob;
};

template <typename DistType>
class IndexComputerAbstract : public VecsimBaseObject {
public:
    IndexComputerAbstract(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}

    virtual ProcessedBlobs preprocess(const void *blob, size_t processed_bytes_count) const = 0;

    virtual unique_blob preprocessForStorage(const void *blob,
                                             size_t processed_bytes_count) const = 0;

    virtual unique_blob preprocessQuery(const void *blob, size_t processed_bytes_count) const = 0;

    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const = 0;
    virtual unsigned char getAlignment() const = 0; // TODO:remove!!!!!!!
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

    virtual ProcessedBlobs preprocess(const void *original_blob,
                                      size_t processed_bytes_count) const override;

    virtual unique_blob preprocessForStorage(const void *original_blob,
                                             size_t processed_bytes_count) const override;

    // TODO:remove!!!!!!!
    virtual unsigned char getAlignment() const override { return alignment; }

    virtual unique_blob preprocessQuery(const void *original_blob,
                                        size_t processed_bytes_count) const override;

    DistType calcDistance(const void *v1, const void *v2, size_t dim) const override {
        assert(this->distance_calculator);
        return this->distance_calculator->calcDistance(v1, v2, dim);
    }

protected:
    const unsigned char alignment;
    DistanceCalculatorAbstract<DistType, DistFuncType> *distance_calculator;

    // Allocate and copy the blob only if the original blob is not aligned.
    unique_blob maybeCopyToAlignedMem(const void *original_blob, size_t blob_bytes_count) const;
};

template <typename DistType, typename DistFuncType>
class IndexComputerExtended : public IndexComputerBasic<DistType, DistFuncType> {
protected:
    PreprocessorAbstract **preprocessors;
    const size_t n_preprocessors;

public:
    IndexComputerExtended(
        std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment, size_t n_preprocessors,
        DistanceCalculatorAbstract<DistType, DistFuncType> *distance_calculator = nullptr)
        : IndexComputerBasic<DistType, DistFuncType>(allocator, alignment, distance_calculator),
          n_preprocessors(n_preprocessors) {
        assert(n_preprocessors);
        preprocessors = static_cast<PreprocessorAbstract **>(
            this->allocator->allocate(sizeof(PreprocessorAbstract *) * n_preprocessors));
        for (size_t i = 0; i < n_preprocessors; i++) {
            preprocessors[i] = nullptr;
        }
    }

    ~IndexComputerExtended() override {
        for (size_t i = 0; i < n_preprocessors; i++) {
            if (preprocessors[i] == nullptr)
                break;
            delete preprocessors[i];
        }

        this->allocator->free_allocation(preprocessors);
    }

    // returns next uninitialized index. returns 0 in case capacity is reached.
    size_t addPreprocessor(PreprocessorAbstract *preprocessor);

    virtual ProcessedBlobs preprocess(const void *original_blob,
                                      size_t processed_bytes_count) const override;

    virtual unique_blob preprocessForStorage(const void *original_blob,
                                             size_t processed_bytes_count) const override;

    virtual unique_blob preprocessQuery(const void *original_blob,
                                        size_t processed_bytes_count) const override;

private:
    unsigned char flags = 0;

    // Allocate a memory slot for the blob. If the aligned = true, the blob will be allocated in an
    // aligned memory address.
    template <bool aligned>
    unique_blob allocBlob(const void *original_blob, size_t blob_bytes_count) const;

    // Allocate and copy the blob. If the aligned = true, the blob will be allocated in an aligned
    // memory address.
    template <bool aligned>
    unique_blob allocBlobCopy(const void *original_blob, size_t blob_bytes_count) const;

    struct Flags {
        static constexpr unsigned char STORAGE = 0x1;
        static constexpr unsigned char QUERY = 0x2;
    };
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
unique_blob IndexComputerBasic<DistType, DistFuncType>::preprocessForStorage(
    const void *original_blob, size_t processed_bytes_count) const {
    return unique_blob(const_cast<void *>(original_blob), dummyFreeAllocation);
}

template <typename DistType, typename DistFuncType>
unique_blob
IndexComputerBasic<DistType, DistFuncType>::preprocessQuery(const void *original_blob,
                                                            size_t processed_bytes_count) const {
    return maybeCopyToAlignedMem(original_blob, processed_bytes_count);
}

template <typename DistType, typename DistFuncType>
unique_blob
IndexComputerBasic<DistType, DistFuncType>::maybeCopyToAlignedMem(const void *original_blob,
                                                                  size_t blob_bytes_count) const {
    if (this->alignment) {
        if ((this->alignment && (uintptr_t)original_blob % this->alignment)) {
            auto aligned_mem =
                this->allocator->allocate_force_aligned(blob_bytes_count, this->alignment);
            memcpy(aligned_mem, original_blob, blob_bytes_count);
            return unique_blob(aligned_mem,
                               [this](void *ptr) { this->allocator->free_allocation(ptr); });
        }
    }

    // Returning a unique_ptr with a no-op deleter
    return unique_blob(const_cast<void *>(original_blob), dummyFreeAllocation);
}

/* ======================= IndexComputerExtended ======================= */

template <typename DistType, typename DistFuncType>
size_t
IndexComputerExtended<DistType, DistFuncType>::addPreprocessor(PreprocessorAbstract *preprocessor) {
    for (size_t i = 0; i < n_preprocessors; i++) {
        if (preprocessors[i] == nullptr) {
            if (preprocessor->hasQueryPreprocessor()) {
                flags |= Flags::QUERY;
            }

            if (preprocessor->hasStoragePreprocessor()) {
                flags |= Flags::STORAGE;
            }

            preprocessors[i] = preprocessor;
            return i + 1;
        }
    }
    return 0;
}

template <typename DistType, typename DistFuncType>
ProcessedBlobs
IndexComputerExtended<DistType, DistFuncType>::preprocess(const void *original_blob,
                                                          size_t processed_bytes_count) const {
    assert(preprocessors[0]);

    if (this->flags & Flags::STORAGE) {
        auto storage_blob = allocBlobCopy<false>(original_blob, processed_bytes_count);

        PreprocessorAbstract::PreprocessParams params = {.processed_bytes_count =
                                                             processed_bytes_count,
                                                         .is_populated_storage = true,
                                                         .is_populated_query = false};

        if (this->flags & Flags::QUERY) {
            auto query_blob = allocBlob<true>(original_blob, processed_bytes_count);
            for (size_t i = 0; i < n_preprocessors; i++) {
                if (preprocessors[i] == nullptr)
                    break;
                preprocessors[i]->preprocess(original_blob, storage_blob, query_blob, params);
            }

            return ProcessedBlobs(std::move(storage_blob), std::move(query_blob));
            // only storage preprocessing, maybe alignment for query
        } else { // if (this->flags & Flags::QUERY)
            for (size_t i = 0; i < n_preprocessors; i++) {
                if (preprocessors[i] == nullptr)
                    break;
                preprocessors[i]->preprocessForStorage(storage_blob);
            }
            auto query_blob = this->maybeCopyToAlignedMem(original_blob, processed_bytes_count);
            return ProcessedBlobs(std::move(storage_blob), std::move(query_blob));
        }
    }

    // We always have at least one preprocessor, so it must be a query preprocessor
    auto query_blob = allocBlobCopy<true>(original_blob, processed_bytes_count);
    for (size_t i = 0; i < n_preprocessors; i++) {
        if (preprocessors[i] == nullptr)
            break;
        preprocessors[i]->preprocessQuery(query_blob);
    }

    return ProcessedBlobs(
        std::move(ProcessedBlobs::uniqueWithDummyDeleter(const_cast<void *>(original_blob))),
        std::move(query_blob));
}

template <typename DistType, typename DistFuncType>
unique_blob IndexComputerExtended<DistType, DistFuncType>::preprocessForStorage(
    const void *original_blob, size_t processed_bytes_count) const {
    assert(preprocessors[0]);
    auto storage_blob = allocBlobCopy<false>(original_blob, processed_bytes_count);

    for (size_t i = 0; i < n_preprocessors; i++) {
        if (preprocessors[i] == nullptr)
            break;
        preprocessors[i]->preprocessForStorage(storage_blob);
    }
    return storage_blob;
}

template <typename DistType, typename DistFuncType>
unique_blob
IndexComputerExtended<DistType, DistFuncType>::preprocessQuery(const void *original_blob,
                                                               size_t processed_bytes_count) const {
    assert(preprocessors[0]);
    auto query_blob = allocBlobCopy<true>(original_blob, processed_bytes_count);
    for (size_t i = 0; i < n_preprocessors; i++) {
        if (preprocessors[i] == nullptr)
            break;
        // modifies the aligned memory in place
        preprocessors[i]->preprocessQuery(query_blob);
    }
    return query_blob;
}

template <typename DistType, typename DistFuncType>
template <bool aligned>
unique_blob
IndexComputerExtended<DistType, DistFuncType>::allocBlob(const void *original_blob,
                                                         size_t blob_bytes_count) const {
    void *allocated_mem;
    if constexpr (aligned) {
        allocated_mem = this->allocator->allocate_aligned(blob_bytes_count, this->alignment);
    } else {
        allocated_mem = this->allocator->allocate(blob_bytes_count);
    }

    return unique_blob(allocated_mem, [this](void *ptr) { this->allocator->free_allocation(ptr); });
}

template <typename DistType, typename DistFuncType>
template <bool aligned>
unique_blob
IndexComputerExtended<DistType, DistFuncType>::allocBlobCopy(const void *original_blob,
                                                             size_t blob_bytes_count) const {
    auto allocated_mem_ptr = allocBlob<aligned>(original_blob, blob_bytes_count);

    memcpy(allocated_mem_ptr.get(), original_blob, blob_bytes_count);
    return allocated_mem_ptr;
}
