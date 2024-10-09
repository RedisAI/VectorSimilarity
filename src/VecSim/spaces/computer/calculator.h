/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include <cstddef>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"

/**
 * This object purpose is to calculate the distance between two vectors.
 * It holds the distance function of the abstract index and the parameters required for the
 * calculation. The distance calculation API of all DistanceCalculator classes is:
 * calc_dist(v1,v2,dim). Internally it calls the distance function according the template signature.
 */

template <typename DistType, typename DistFuncType>
class DistanceCalculatorInterface : public VecsimBaseObject {
public:
    DistanceCalculatorInterface(std::shared_ptr<VecSimAllocator> allocator, DistFuncType dist_func)
        : VecsimBaseObject(allocator), dist_func(dist_func) {}
    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const = 0;

protected:
    DistFuncType dist_func;
};

template <typename DistType>
class DistanceCalculatorCommon
    : public DistanceCalculatorInterface<DistType, spaces::dist_func_t<DistType>> {
public:
    DistanceCalculatorCommon(std::shared_ptr<VecSimAllocator> allocator,
                             spaces::dist_func_t<DistType> dist_func)
        : DistanceCalculatorInterface<DistType, spaces::dist_func_t<DistType>>(allocator,
                                                                               dist_func) {}

    DistType calcDistance(const void *v1, const void *v2, size_t dim) const override {
        return this->dist_func(v1, v2, dim);
    }
};
