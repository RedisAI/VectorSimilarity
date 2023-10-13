#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/algorithms/raft_ivf/ivf.cuh"

namespace RaftIvfFactory {

static AbstractIndexInitParams NewAbstractInitParams(const VecSimParams *params) {

    const RaftIvfParams *raftIvfParams = &params->algoParams.raftIvfParams;
    AbstractIndexInitParams abstractInitParams = {.allocator =
                                                      VecSimAllocator::newVecsimAllocator(),
                                                  .dim = raftIvfParams->dim,
                                                  .vecType = raftIvfParams->type,
                                                  .metric = raftIvfParams->metric,
                                                  //.multi = raftIvfParams->multi,
                                                  //.logCtx = params->logCtx
                                                };
    return abstractInitParams;
}

VecSimIndex *NewIndex(const RaftIvfParams *raftIvfParams, const AbstractIndexInitParams &abstractInitParams) {
    if (raftIvfParams->type == VecSimType_FLOAT32) {
        return new (abstractInitParams.allocator)
            RaftIVFIndex<float, float>(raftIvfParams, abstractInitParams);
    } else if (raftIvfParams->type == VecSimType_FLOAT64) {
        return new (abstractInitParams.allocator)
            RaftIVFIndex<double, double>(raftIvfParams, abstractInitParams);
    }

    // If we got here something is wrong.
    return NULL;
}

VecSimIndex *NewIndex(const VecSimParams *params) {
    const RaftIvfParams *raftIvfParams = &params->algoParams.raftIvfParams;
    AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(params);
    return NewIndex(raftIvfParams, NewAbstractInitParams(params));
}

VecSimIndex *NewIndex(const RaftIvfParams *raftIvfParams) {
    VecSimParams params = {.algoParams{.raftIvfParams = RaftIvfParams{*raftIvfParams}}};
    return NewIndex(&params);
}

size_t EstimateInitialSize(const RaftIvfParams *raftIvfParams) {

    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();

    // Constant part (not effected by parameters).
    size_t est = sizeof(VecSimAllocator) + allocations_overhead;
    est += sizeof(RaftIVFIndex<float, float>);                                // Object size
    if (!raftIvfParams->usePQ) {
        // Size of each cluster data
        est += raftIvfParams->nLists * sizeof(raft::neighbors::ivf_flat::list_data<float, std::int64_t>);
        // Vector of shared ptr to cluster
        est += raftIvfParams->nLists * sizeof(std::shared_ptr<raft::neighbors::ivf_flat::list_data<float, std::int64_t>>);
    } else {
        // Size of each cluster data
        est += raftIvfParams->nLists * sizeof(raft::neighbors::ivf_pq::list_data<std::int64_t>);
        // accum_sorted_sizes_ Array
        est += raftIvfParams->nLists * sizeof(std::int64_t);
        // vector of shared ptr to cluster
        est += raftIvfParams->nLists * sizeof(std::shared_ptr<raft::neighbors::ivf_pq::list_data<std::int64_t>>);
    }
    return est;
}

size_t EstimateElementSize(const RaftIvfParams *raftIvfParams) {
    // Vectors are stored on the GPU.
    return 0;
}
}; // namespace RaftIvfFactory
