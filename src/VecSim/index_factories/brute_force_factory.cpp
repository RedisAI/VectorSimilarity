/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/brute_force/brute_force_multi.h"

namespace BruteForceFactory {
template <typename DataType, typename DistType = DataType>
inline VecSimIndex *
NewIndex_ChooseMultiOrSingle(const BFParams *params,
                             const AbstractIndexInitParams &abstractInitParams) {

    // check if single and return new bf_index
    if (params->multi)
        return new (abstractInitParams.allocator)
            BruteForceIndex_Multi<DataType, DistType>(params, abstractInitParams);
    else
        return new (abstractInitParams.allocator)
            BruteForceIndex_Single<DataType, DistType>(params, abstractInitParams);
}

static AbstractIndexInitParams NewAbstractInitParams(const VecSimParams *params) {

    const BFParams *bfParams = &params->algoParams.bfParams;
    AbstractIndexInitParams abstractInitParams = {.allocator =
                                                      VecSimAllocator::newVecsimAllocator(),
                                                  .dim = bfParams->dim,
                                                  .vecType = bfParams->type,
                                                  .metric = bfParams->metric,
                                                  .blockSize = bfParams->blockSize,
                                                  .multi = bfParams->multi,
                                                  .logCtx = params->logCtx};
    return abstractInitParams;
}

VecSimIndex *NewIndex(const VecSimParams *params) {
    const BFParams *bfParams = &params->algoParams.bfParams;
    AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(params);
    return NewIndex(bfParams, NewAbstractInitParams(params));
}

VecSimIndex *NewIndex(const BFParams *bfparams, const AbstractIndexInitParams &abstractInitParams) {
    if (bfparams->type == VecSimType_FLOAT32) {
        return NewIndex_ChooseMultiOrSingle<float>(bfparams, abstractInitParams);
    } else if (bfparams->type == VecSimType_FLOAT64) {
        return NewIndex_ChooseMultiOrSingle<double>(bfparams, abstractInitParams);
    }

    // If we got here something is wrong.
    return NULL;
}

VecSimIndex *NewIndex(const BFParams *bfparams) {
    VecSimParams params = {.algoParams{.bfParams = BFParams{*bfparams}}};
    return NewIndex(&params);
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

    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();

    // Constant part (not effected by parameters).
    size_t est = sizeof(VecSimAllocator) + allocations_overhead;

    if (params->type == VecSimType_FLOAT32) {
        est += EstimateInitialSize_ChooseMultiOrSingle<float>(params->multi);
    } else if (params->type == VecSimType_FLOAT64) {
        est += EstimateInitialSize_ChooseMultiOrSingle<double>(params->multi);
    }
    // Parameters related part.

    if (params->initialCapacity) {
        size_t aligned_cap = RoundUpInitialCapacity(params->initialCapacity, params->blockSize);
        est += aligned_cap * sizeof(labelType) + allocations_overhead;

        est += sizeof(DataBlock) * aligned_cap / params->blockSize + allocations_overhead;
    }
    return est;
}

size_t EstimateElementSize(const BFParams *params) {
    // counting the vector size + idToLabel entry + LabelToIds entry (map reservation)
    return params->dim * VecSimType_sizeof(params->type) + sizeof(labelType) + sizeof(void *);
}
}; // namespace BruteForceFactory
