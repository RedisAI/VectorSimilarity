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

    virtual DistType calcDistanceForQuery(const void *candidate_vector, const void *query_vector,
                                          size_t dim) const = 0;

    // Raw distance function; cached by the index to skip the vtable on the hot path.
    virtual spaces::dist_func_t<DistType> getDistFunc() const = 0;

    virtual spaces::dist_func_t<DistType> getDistFuncForQuery() const = 0;
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
    DistanceCalculatorInterface(std::shared_ptr<VecSimAllocator> allocator, DistFuncType dist_func,
                                DistFuncType query_dist_func = nullptr)
        : IndexCalculatorInterface<DistType>(allocator), dist_func(dist_func),
          query_dist_func(query_dist_func ? query_dist_func : dist_func) {}
    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const = 0;
    virtual DistType calcDistanceForQuery(const void *candidate_vector, const void *query_vector,
                                          size_t dim) const = 0;

protected:
    DistFuncType dist_func;
    DistFuncType query_dist_func;
};

template <typename DistType>
class DistanceCalculatorCommon
    : public DistanceCalculatorInterface<DistType, spaces::dist_func_t<DistType>> {
public:
    DistanceCalculatorCommon(std::shared_ptr<VecSimAllocator> allocator,
                             spaces::dist_func_t<DistType> dist_func,
                             spaces::dist_func_t<DistType> query_dist_func = nullptr)
        : DistanceCalculatorInterface<DistType, spaces::dist_func_t<DistType>>(allocator, dist_func,
                                                                               query_dist_func) {}

    DistType calcDistance(const void *v1, const void *v2, size_t dim) const override {
        return this->dist_func(v1, v2, dim);
    }

    DistType calcDistanceForQuery(const void *candidate_vector, const void *query_vector,
                                  size_t dim) const override {
        return this->query_dist_func(candidate_vector, query_vector, dim);
    }

    spaces::dist_func_t<DistType> getDistFunc() const override { return this->dist_func; }

    spaces::dist_func_t<DistType> getDistFuncForQuery() const override {
        return this->query_dist_func;
    }
};
