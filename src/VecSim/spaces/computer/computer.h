/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "VecSim/vec_sim_common.h"
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/memory/memory_utils.h"

// We need this "wrapper" class to hold the indexComputer in the index, that is not templated
// according to the distance function signature.
template <typename DistType>
class IndexComputerAbstract : public VecsimBaseObject {
public:
    // Creates a new idexComputer with a given preprocessors container.
    // The indexComputer is responsible to manage the preprocessor memory and will release it when
    // the indexComputer is released.
    explicit IndexComputerAbstract(std::shared_ptr<VecSimAllocator> allocator,
                                   PreprocessorsContainerAbstract *preprocessors)
        : VecsimBaseObject(allocator), preprocessors(preprocessors) {
        assert(this->preprocessors);
    }

    // Creates a new idexComputer with the default preprocessors container,
    // currntly only aligns queries allocations.
    explicit IndexComputerAbstract(std::shared_ptr<VecSimAllocator> allocator,
                                   unsigned char alignment)
        : VecsimBaseObject(allocator) {
        this->preprocessors = new (allocator) PreprocessorsContainerAbstract(allocator, alignment);
    }

    virtual ~IndexComputerAbstract() {
        if (preprocessors) {
            delete preprocessors;
        }
    }

    virtual ProcessedBlobs preprocess(const void *original_blob,
                                      size_t processed_bytes_count) const {
        return preprocessors->preprocess(original_blob, processed_bytes_count);
    }

    virtual MemoryUtils::unique_blob preprocessForStorage(const void *original_blob,
                                                          size_t processed_bytes_count) const {
        return preprocessors->preprocessForStorage(original_blob, processed_bytes_count);
    }

    virtual MemoryUtils::unique_blob preprocessQuery(const void *original_blob,
                                                     size_t processed_bytes_count) const {
        return preprocessors->preprocessQuery(original_blob, processed_bytes_count);
    }
    virtual void preprocessQueryInPlace(void *blob, size_t processed_bytes_count) const {
        preprocessors->preprocessQueryInPlace(blob, processed_bytes_count);
    };

    // TODO: remove alignment once datablock is implemented in HNSW
    virtual unsigned char getAlignment() const { return preprocessors->getAlignment(); }

    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const = 0;
#ifdef BUILD_TESTS
    void replacePPContainer(PreprocessorsContainerAbstract *newPPComputer) {
        delete preprocessors;
        preprocessors = newPPComputer;
    }
#endif

protected:
    PreprocessorsContainerAbstract *preprocessors;
};

template <typename DistType, typename DistFuncType>
class IndexComputerBasic : public IndexComputerAbstract<DistType> {
public:
    explicit IndexComputerBasic(
        std::shared_ptr<VecSimAllocator> allocator, PreprocessorsContainerAbstract *preprocessors,
        DistanceCalculatorInterface<DistType, DistFuncType> *distance_calculator)
        : IndexComputerAbstract<DistType>(allocator, preprocessors),
          distance_calculator(distance_calculator) {}

    explicit IndexComputerBasic(
        std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment,
        DistanceCalculatorInterface<DistType, DistFuncType> *distance_calculator)
        : IndexComputerAbstract<DistType>(allocator, alignment),
          distance_calculator(distance_calculator) {}

    ~IndexComputerBasic() override {
        if (distance_calculator) {
            delete distance_calculator;
        }
    }

    DistType calcDistance(const void *v1, const void *v2, size_t dim) const override {
        assert(this->distance_calculator);
        return this->distance_calculator->calcDistance(v1, v2, dim);
    }

protected:
    DistanceCalculatorInterface<DistType, DistFuncType> *distance_calculator;
};
