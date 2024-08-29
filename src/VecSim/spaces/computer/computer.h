/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "VecSim/vec_sim_common.h"
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/computer/calculator.h"
#include <iostream> // TODO: REMOVE!!!!!

template <typename DistType>
class IndexComputerAbstract : public VecsimBaseObject {
public:
    IndexComputerAbstract(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}
    // virtual const void *preprocessForStorage(const void *blob) = 0;
    // virtual const void *preprocessQuery(const void *blob) = 0;
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
    virtual void preprocessForStorage() {
        std::cout << "Computer::preprocessForStorage nothing to do" << std::endl;
    }

    // TODO:remove!!!!!!!
    virtual unsigned char getAlignment() const override { return alignment; }

    virtual void preprocessQuery() {
        std::cout << "IndexComputerBasic::preprocessQuery" << std::endl;
        if (alignment) {
            std::cout << "IndexComputerBasic::preprocessQuery alignment: " << alignment
                      << std::endl;
        } else {
            std::cout << "IndexComputer::preprocessQuery no alignment" << std::endl;
        }
    }

    DistType calcDistance(const void *v1, const void *v2, size_t dim) const override {
        assert(this->distance_calculator);
        return this->distance_calculator->calcDistance(v1, v2, dim);
    }

protected:
    const unsigned char alignment;
    DistanceCalculatorAbstract<DistType, DistFuncType> *distance_calculator;
};
