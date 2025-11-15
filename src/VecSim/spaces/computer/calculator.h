/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include <cstddef>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"

// We need this "wrapper" class to hold the DistanceCalculatorInterface in the index, that is not
// templated according to the distance function signature.
template <typename DistType>
class IndexCalculatorInterface : public VecsimBaseObject {
public:
    explicit IndexCalculatorInterface(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}

    virtual ~IndexCalculatorInterface() = default;

    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const = 0;
    virtual DistType calcDistanceRaw(const void *v1, const void *v2, size_t dim) const = 0;
};

/**
 * This object purpose is to calculate the distance between two vectors.
 * It extends the IndexCalculatorInterface class' type to hold the distance function.
 * Every specific implementation of the distance calculator should hold by reference or by value the
 * parameters required for the calculation. The distance calculation API of all DistanceCalculator
 * classes is: calc_dist(v1,v2,dim). Internally it calls the distance function according the
 * template signature, allowing flexibility in the distance function arguments.
 */
template <typename DistType, typename DistFuncType>
class DistanceCalculatorInterface : public IndexCalculatorInterface<DistType> {
public:
    DistanceCalculatorInterface(std::shared_ptr<VecSimAllocator> allocator, DistFuncType dist_func)
        : IndexCalculatorInterface<DistType>(allocator), dist_func(dist_func) {}
    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const = 0;
    virtual DistType calcDistanceRaw(const void *v1, const void *v2, size_t dim) const = 0;
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
    DistType calcDistanceRaw(const void *v1, const void *v2, size_t dim) const override {
        return this->dist_func(v1, v2, dim);
    }
};

template <typename DistType>
class DistanceCalculatorQuantized
    : public DistanceCalculatorInterface<DistType, spaces::dist_func_t<DistType>> {
protected:
    spaces::dist_func_t<DistType> raw_dist_func;

public:
    DistanceCalculatorQuantized(std::shared_ptr<VecSimAllocator> allocator,
                                spaces::dist_func_t<DistType> quant_dist_func, spaces::dist_func_t<DistType> raw_dist_func)
        : DistanceCalculatorInterface<DistType, spaces::dist_func_t<DistType>>(allocator,
                                                                               quant_dist_func),
          raw_dist_func(raw_dist_func) {}

    DistType calcDistance(const void *v1, const void *v2, size_t dim) const override {
        return this->dist_func(v1, v2, dim);
    }
    DistType calcDistanceRaw(const void *v1, const void *v2, size_t dim) const {
        return this->raw_dist_func(v1, v2, dim);
    }
};
