/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/algorithms/brute_force/brute_force_factory.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/brute_force/brute_force_multi.h"

namespace BruteForceFactory {
template <typename DataType, typename DistType = DataType>
inline VecSimIndex *NewIndex_ChooseMultiOrSingle(const BFParams *params,
                                                 std::shared_ptr<VecSimAllocator> allocator) {
    // check if single and return new bf_index
    if (params->multi)
        return new (allocator) BruteForceIndex_Multi<DataType, DistType>(params, allocator);
    else
        return new (allocator) BruteForceIndex_Single<DataType, DistType>(params, allocator);
}

VecSimIndex *NewIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator) {
    if (params->type == VecSimType_FLOAT32) {
        return NewIndex_ChooseMultiOrSingle<float>(params, allocator);
    } else if (params->type == VecSimType_FLOAT64) {
        return NewIndex_ChooseMultiOrSingle<double>(params, allocator);
    }

    // If we got here something is wrong.
    return NULL;
}

template <typename DataType, typename DistType = DataType>
inline size_t EstimateInitialSize_ChooseMultiOrSingle(bool is_multi) {
    // check if single and return new bf_index
    if (is_multi)
        return sizeof(BruteForceIndex_Multi<DataType, DistType>);
    else
        return sizeof(BruteForceIndex_Single<DataType, DistType>);
}

size_t EstimateInitialSize(const BFParams *params) {

    // Constant part (not effected by parameters).
    size_t est = sizeof(VecSimAllocator) + sizeof(size_t);
    if (params->type == VecSimType_FLOAT32) {
        est += EstimateInitialSize_ChooseMultiOrSingle<float>(params->multi);
    } else if (params->type == VecSimType_FLOAT64) {
        est += EstimateInitialSize_ChooseMultiOrSingle<double>(params->multi);
    }
    // Parameters related part.

    if (params->initialCapacity) {
        est += params->initialCapacity * sizeof(labelType) + sizeof(size_t);
    }

    return est;
}

size_t EstimateElementSize(const BFParams *params) {
    return params->dim * VecSimType_sizeof(params->type) + sizeof(labelType);
}
}; // namespace BruteForceFactory
