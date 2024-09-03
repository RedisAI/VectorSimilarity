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

template <typename DistType>
class IndexComputerAbstract : public VecsimBaseObject {
public:
    IndexComputerAbstract(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}
    // virtual std::unique_ptr<void, alloc_deleter_t>
    // preprocessForStorage(const void *blob, size_t processed_bytes_count) = 0;
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
    // virtual void preprocessForStorage() { /*nothing to do*/
    // }

    // TODO:remove!!!!!!!
    virtual unsigned char getAlignment() const override { return alignment; }

    // TODO: think where to place

    static void dummyFreeAllocation(void *ptr) {}
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
    virtual std::unique_ptr<void, alloc_deleter_t> preprocessQuery(const void *original_blob,
                                                                   size_t processed_bytes_count) {
        assert(preprocessors[0]);
        // when Using this class, we always need to modify the blob, so we always copy it
        // if aligment = 0, allocate aligned will call simple allocate
        std::unique_ptr<void, alloc_deleter_t> aligned_mem(
            this->allocator->allocate_aligned(processed_bytes_count, this->alignment),
            [this](void *ptr) { this->allocator->free_allocation(ptr); });
        memcpy(aligned_mem.get(), original_blob, processed_bytes_count);
        for (size_t i = 0; i < n_preprocessors; i++) {
            if (preprocessors[i] == nullptr)
                break;
            // modifies the aligned memory in place
            preprocessors[i]->preprocessQuery(aligned_mem);
        }
        return aligned_mem;
    }

protected:
    PreprocessorAbstract **preprocessors;
    const size_t n_preprocessors;
};
