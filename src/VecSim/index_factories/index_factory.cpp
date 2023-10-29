/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "index_factory.h"
#include "hnsw_factory.h"
#include "brute_force_factory.h"
#include "tiered_factory.h"
#include "VecSim/vec_sim_index.h"

namespace VecSimFactory {
VecSimIndex *NewIndex(const VecSimParams *params) {
    VecSimIndex *index = NULL;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    try {
        switch (params->algo) {
        case VecSimAlgo_HNSWLIB: {
            index = HNSWFactory::NewIndex(params);
            break;
        }

        case VecSimAlgo_BF: {
            index = BruteForceFactory::NewIndex(params);
            break;
        }
        case VecSimAlgo_TIERED: {
            index = TieredFactory::NewIndex(&params->algoParams.tieredParams);
            break;
        }
        }
    } catch (...) {
        // Index will delete itself. For now, do nothing.
    }
    return index;
}

size_t EstimateInitialSize(const VecSimParams *params) {
    switch (params->algo) {
    case VecSimAlgo_HNSWLIB:
        return HNSWFactory::EstimateInitialSize(&params->algoParams.hnswParams);
    case VecSimAlgo_BF:
        return BruteForceFactory::EstimateInitialSize(&params->algoParams.bfParams);
    case VecSimAlgo_TIERED:
        return TieredFactory::EstimateInitialSize(&params->algoParams.tieredParams);
    }
    return -1;
}

size_t EstimateElementSize(const VecSimParams *params) {
    switch (params->algo) {
    case VecSimAlgo_HNSWLIB:
        return HNSWFactory::EstimateElementSize(&params->algoParams.hnswParams);
    case VecSimAlgo_BF:
        return BruteForceFactory::EstimateElementSize(&params->algoParams.bfParams);
    case VecSimAlgo_TIERED:
        return TieredFactory::EstimateElementSize(&params->algoParams.tieredParams);
    }
    return -1;
}

} // namespace VecSimFactory
