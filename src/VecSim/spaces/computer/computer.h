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

struct ProcessedBlobs {
    using unique_blob = std::unique_ptr<void, alloc_deleter_t>;
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

private:
    std::unique_ptr<void, alloc_deleter_t> storage_blob;
    std::unique_ptr<void, alloc_deleter_t> query_blob;
};

template <typename DistType>
class IndexComputerAbstract : public VecsimBaseObject {
public:
    IndexComputerAbstract(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}

    virtual ProcessedBlobs preprocess(const void *blob, size_t processed_bytes_count) = 0;

    virtual std::unique_ptr<void, alloc_deleter_t>
    preprocessForStorage(const void *blob, size_t processed_bytes_count) = 0;

    virtual std::unique_ptr<void, alloc_deleter_t>
    preprocessQuery(const void *blob, size_t processed_bytes_count) = 0;

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

    virtual ProcessedBlobs preprocess(const void *blob, size_t processed_bytes_count) {
        auto storage_blob = preprocessForStorage(blob, processed_bytes_count);
        auto query_blob = preprocessQuery(blob, processed_bytes_count);
        return ProcessedBlobs(std::move(storage_blob), std::move(query_blob));
    }

    virtual std::unique_ptr<void, alloc_deleter_t>
    preprocessForStorage(const void *original_blob, size_t processed_bytes_count) {
        return std::unique_ptr<void, alloc_deleter_t>(const_cast<void *>(original_blob),
                                                      dummyFreeAllocation);
    }

    // TODO:remove!!!!!!!
    virtual unsigned char getAlignment() const override { return alignment; }

    // static void FreeAllocation(void *ptr) { this->allocator->free_allocation(ptr); }

    virtual std::unique_ptr<void, alloc_deleter_t> preprocessQuery(const void *original_blob,
                                                                   size_t processed_bytes_count) {
        if (this->alignment) {
            // if the blob is not aligned, or we need to normalize, we copy it
            if ((this->alignment && (uintptr_t)original_blob % this->alignment)) {
                auto aligned_mem =
                    this->allocator->allocate_aligned(processed_bytes_count, this->alignment);
                memcpy(aligned_mem, original_blob, processed_bytes_count);
                return std::unique_ptr<void, alloc_deleter_t>(
                    aligned_mem, [this](void *ptr) { this->allocator->free_allocation(ptr); });
            }
        }

        // Returning a unique_ptr with a no-op deleter
        return std::unique_ptr<void, alloc_deleter_t>(const_cast<void *>(original_blob),
                                                      dummyFreeAllocation);
    }

    DistType calcDistance(const void *v1, const void *v2, size_t dim) const override {
        assert(this->distance_calculator);
        return this->distance_calculator->calcDistance(v1, v2, dim);
    }

protected:
    const unsigned char alignment;
    DistanceCalculatorAbstract<DistType, DistFuncType> *distance_calculator;
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

    // returns next uninitialized index. returns 0 in case capacity is reached.
    size_t addPreprocessor(PreprocessorAbstract *preprocessor) {
        for (size_t i = 0; i < n_preprocessors; i++) {
            if (preprocessors[i] == nullptr) {
                preprocessors[i] = preprocessor;
                return i + 1;
            }
        }
        return 0;
    }

    ~IndexComputerExtended() override {
        for (size_t i = 0; i < n_preprocessors; i++) {
            if (preprocessors[i] == nullptr)
                break;
            delete preprocessors[i];
        }

        this->allocator->free_allocation(preprocessors);
    }

    virtual ProcessedBlobs preprocess(const void *original_blob, size_t processed_bytes_count) {
        assert(preprocessors[0]);

        // Create modifiable memory slot for the processed blobs.
        auto storage_blob = allocateBlob<false>(processed_bytes_count);
        auto query_blob = allocateBlob<true>(processed_bytes_count);

        // Flags indicating rather the memory was populated, to be updated by the preprocessor
        PreprocessorAbstract::PreprocessorParams params = {.processed_bytes_count =
                                                               processed_bytes_count,
                                                           .is_populated_storage = false,
                                                           .is_populated_query = false};
        for (size_t i = 0; i < n_preprocessors; i++) {
            if (preprocessors[i] == nullptr)
                break;
            // modifies the blob for storage or for query or both, and updates is_populated* flags
            // accordingly.
            preprocessors[i]->preprocess(original_blob, storage_blob, query_blob, params);
        }
        // This class is always used with some preprocessor
        assert(params.is_populated_storage || params.is_populated_query);

        if (!params.is_populated_storage) {
            memcpy(storage_blob.get(), original_blob, processed_bytes_count);
        }
        if (!params.is_populated_query) {
            memcpy(query_blob.get(), original_blob, processed_bytes_count);
        }
        return ProcessedBlobs(std::move(storage_blob), std::move(query_blob));
    }

    virtual std::unique_ptr<void, alloc_deleter_t>
    preprocessForStorage(const void *original_blob, size_t processed_bytes_count) {

        assert(preprocessors[0]);
        // when Using this class, we always need to modify the blob, so we always copy it
        auto processed_mem = copyBlobToNewAllocation<false>(original_blob, processed_bytes_count);

        for (size_t i = 0; i < n_preprocessors; i++) {
            if (preprocessors[i] == nullptr)
                break;
            // modifies the aligned memory in place
            preprocessors[i]->preprocessForStorage(processed_mem);
        }
        return processed_mem;
    }

    virtual std::unique_ptr<void, alloc_deleter_t> preprocessQuery(const void *original_blob,
                                                                   size_t processed_bytes_count) {
        assert(preprocessors[0]);
        // when Using this class, we always need to modify the blob, so we always copy it
        // if aligment = 0, allocate aligned will call simple allocate
        auto aligned_mem = copyBlobToNewAllocation<true>(original_blob, processed_bytes_count);
        for (size_t i = 0; i < n_preprocessors; i++) {
            if (preprocessors[i] == nullptr)
                break;
            // modifies the aligned memory in place
            preprocessors[i]->preprocessQuery(aligned_mem);
        }
        return aligned_mem;
    }

protected:
    template <bool aligned>
    std::unique_ptr<void, alloc_deleter_t> copyBlobToNewAllocation(const void *original_blob,
                                                                   size_t blob_bytes_count) {
        auto blob_copy = allocateBlob<aligned>(blob_bytes_count);
        memcpy(blob_copy.get(), original_blob, blob_bytes_count);

        return blob_copy;
    }

    template <bool aligned>
    std::unique_ptr<void, alloc_deleter_t> allocateBlob(size_t blob_bytes_count) {
        void *allocated_mem;
        if constexpr (aligned) {
            allocated_mem = this->allocator->allocate_aligned(blob_bytes_count, this->alignment);
        } else {
            allocated_mem = this->allocator->allocate(blob_bytes_count);
        }

        std::unique_ptr<void, alloc_deleter_t> ret_mem(
            allocated_mem, [this](void *ptr) { this->allocator->free_allocation(ptr); });
        return ret_mem;
    }
};
