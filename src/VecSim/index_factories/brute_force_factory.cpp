/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/brute_force/brute_force_multi.h"
#include "VecSim/index_factories/factory_utils.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

namespace BruteForceFactory {
template <typename DataType, typename DistType = DataType>
inline VecSimIndex *NewIndex_ChooseMultiOrSingle(const BFParams *params,
                                                 const AbstractIndexInitParams &abstractInitParams,
                                                 IndexComputerAbstract<DistType> *indexComputer) {

    // check if single and return new bf_index
    if (params->multi)
        return new (abstractInitParams.allocator)
            BruteForceIndex_Multi<DataType, DistType>(params, abstractInitParams, indexComputer);
    else
        return new (abstractInitParams.allocator)
            BruteForceIndex_Single<DataType, DistType>(params, abstractInitParams, indexComputer);
}

static AbstractIndexInitParams NewAbstractInitParams(const VecSimParams *params,
                                                     bool is_normalized) {
    const BFParams *bfParams = &params->algoParams.bfParams;

    VecSimMetric metric;
    if (is_normalized && bfParams->metric == VecSimMetric_Cosine) {
        metric = VecSimMetric_IP;
    } else {
        metric = bfParams->metric;
    }
    AbstractIndexInitParams abstractInitParams = {.allocator =
                                                      VecSimAllocator::newVecsimAllocator(),
                                                  .dim = bfParams->dim,
                                                  .vecType = bfParams->type,
                                                  .metric = metric,
                                                  .blockSize = bfParams->blockSize,
                                                  .multi = bfParams->multi,
                                                  .logCtx = params->logCtx};
    return abstractInitParams;
}

VecSimIndex *NewIndex(const VecSimParams *params, bool is_normalized) {
    const BFParams *bfParams = &params->algoParams.bfParams;
    AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(params, is_normalized);
    return NewIndex(bfParams, abstractInitParams);
}

VecSimIndex *NewIndex(const BFParams *bfparams, const AbstractIndexInitParams &abstractInitParams) {
    if (bfparams->type == VecSimType_FLOAT32) {
        IndexComputerAbstract<float> *indexComputer = CreateIndexComputer<float>(
            abstractInitParams.allocator, bfparams->metric, bfparams->dim);
        return NewIndex_ChooseMultiOrSingle<float>(bfparams, abstractInitParams, indexComputer);
    } else if (bfparams->type == VecSimType_FLOAT64) {
        IndexComputerAbstract<double> *indexComputer = CreateIndexComputer<double>(
            abstractInitParams.allocator, bfparams->metric, bfparams->dim);
        return NewIndex_ChooseMultiOrSingle<double>(bfparams, abstractInitParams, indexComputer);
    } else if (bfparams->type == VecSimType_BFLOAT16) {
        IndexComputerAbstract<float> *indexComputer = CreateIndexComputer<bfloat16, float>(
            abstractInitParams.allocator, bfparams->metric, bfparams->dim);
        return NewIndex_ChooseMultiOrSingle<bfloat16, float>(bfparams, abstractInitParams,
                                                             indexComputer);
    } else if (bfparams->type == VecSimType_FLOAT16) {
        IndexComputerAbstract<float> *indexComputer = CreateIndexComputer<float16, float>(
            abstractInitParams.allocator, bfparams->metric, bfparams->dim);
        return NewIndex_ChooseMultiOrSingle<float16, float>(bfparams, abstractInitParams,
                                                            indexComputer);
    }

    // If we got here something is wrong.
    return NULL;
}

VecSimIndex *NewIndex(const BFParams *bfparams, bool is_normalized) {
    VecSimParams params = {.algoParams{.bfParams = BFParams{*bfparams}}};
    return NewIndex(&params, is_normalized);
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
        est += EstimateContainersMemory<float>(params->metric);
        est += EstimateInitialSize_ChooseMultiOrSingle<float>(params->multi);
    } else if (params->type == VecSimType_FLOAT64) {
        est += EstimateContainersMemory<double>(params->metric);
        est += EstimateInitialSize_ChooseMultiOrSingle<double>(params->multi);
    } else if (params->type == VecSimType_BFLOAT16) {
        est += EstimateContainersMemory<bfloat16, float>(params->metric);
        est += EstimateInitialSize_ChooseMultiOrSingle<bfloat16, float>(params->multi);
    } else if (params->type == VecSimType_FLOAT16) {
        est += EstimateContainersMemory<float16, float>(params->metric);
        est += EstimateInitialSize_ChooseMultiOrSingle<float16, float>(params->multi);
    }
    est += sizeof(DataBlocksContainer) + allocations_overhead;
    // Parameters related part.

    if (params->initialCapacity) {
        size_t aligned_cap = RoundUpInitialCapacity(params->initialCapacity, params->blockSize);
        est += aligned_cap * sizeof(labelType) + allocations_overhead;
    }
    return est;
}

size_t EstimateElementSize(const BFParams *params) {
    // counting the vector size + idToLabel entry + LabelToIds entry (map reservation)
    return params->dim * VecSimType_sizeof(params->type) + sizeof(labelType) + sizeof(void *);
}
}; // namespace BruteForceFactory
