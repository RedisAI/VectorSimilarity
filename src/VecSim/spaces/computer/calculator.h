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

enum class DistanceMode {
    StoredToStored,
    StoredToQuery,
};

/**
 * A distance callback selected once when an index is constructed.
 *
 * Stateless calculators populate `stateless_func`, preserving the existing hot path of one
 * indirect call to the selected distance kernel. Stateful calculators populate `stateful_func`
 * and `context`; this avoids a vtable lookup in the hot path without forcing stateless
 * calculators through an adapter.
 */
template <typename DistType>
struct DistanceDispatch {
    using stateful_dist_func_t = DistType (*)(const void *context, const void *lhs, const void *rhs,
                                              size_t dim);

    spaces::dist_func_t<DistType> stateless_func = nullptr;
    stateful_dist_func_t stateful_func = nullptr;
    const void *context = nullptr;

    static DistanceDispatch stateless(spaces::dist_func_t<DistType> func) {
        return {.stateless_func = func};
    }

    static DistanceDispatch stateful(const void *context, stateful_dist_func_t func) {
        return {.stateful_func = func, .context = context};
    }

    bool isValid() const { return (stateless_func != nullptr) != (stateful_func != nullptr); }

    DistType operator()(const void *lhs, const void *rhs, size_t dim) const {
        if (stateless_func) {
            return stateless_func(lhs, rhs, dim);
        }
        return stateful_func(context, lhs, rhs, dim);
    }
};

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

    // Called once per mode when constructing the index. The returned dispatch is cached so
    // distance calculations in the hot path do not perform virtual calls.
    virtual DistanceDispatch<DistType> getDistanceDispatch(DistanceMode mode) const = 0;
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

    DistanceDispatch<DistType> getDistanceDispatch(DistanceMode mode) const override {
        auto func = mode == DistanceMode::StoredToStored ? this->dist_func : this->query_dist_func;
        return DistanceDispatch<DistType>::stateless(func);
    }
};
