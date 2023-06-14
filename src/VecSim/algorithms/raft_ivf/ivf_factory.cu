#include "ivf_factory.h"
#include "ivf_flat.cuh"
#include "ivf_pq.cuh"
#include "ivf_tiered.cuh"
#include <iostream>

namespace RaftIVFFlatFactory {

VecSimIndex *NewIndex(const RaftIVFFlatParams *params, std::shared_ptr<VecSimAllocator> allocator) {
    assert(params->type == VecSimType_FLOAT32);
    return new (allocator) RaftIVFFlatIndex(params, allocator);
}

VecSimIndex *NewTieredIndex(const TieredRaftIVFFlatParams *params,
                            std::shared_ptr<VecSimAllocator> allocator) {
    assert(params->flatParams.type == VecSimType_FLOAT32);
    auto *flat_index = NewIndex(&params->flatParams, allocator);
    return new (allocator)
        TieredRaftIvfIndex(dynamic_cast<RaftIVFIndex *>(flat_index), params->tieredParams);
}

size_t EstimateInitialSize(const RaftIVFFlatParams *params) { 
    size_t est = sizeof(RaftIVFFlatIndex);                                // Object size
    est += params->nLists * sizeof(raft::neighbors::ivf_flat::list_data<float, std::int64_t>);         // Size of each cluster data
    est += params->nLists * sizeof(std::shared_ptr<raft::neighbors::ivf_flat::list_data<float, std::int64_t>>);                    // vector of shared ptr to cluster
    return est;
}

size_t EstimateElementSize(const RaftIVFFlatParams *params) { return 0; }
} // namespace RaftIVFFlatFactory

namespace RaftIVFPQFactory {
VecSimIndex *NewIndex(const RaftIVFPQParams *params, std::shared_ptr<VecSimAllocator> allocator) {
    assert(params->type == VecSimType_FLOAT32);
    return new (allocator) RaftIVFPQIndex(params, allocator);
}
VecSimIndex *NewTieredIndex(const TieredRaftIVFPQParams *params,
                            std::shared_ptr<VecSimAllocator> allocator) {
    assert(params->PQParams.type == VecSimType_FLOAT32);
    auto *pq_index = NewIndex(&params->PQParams, allocator);
    return new (allocator)
        TieredRaftIvfIndex(dynamic_cast<RaftIVFIndex *>(pq_index), params->tieredParams);
}

size_t EstimateInitialSize(const RaftIVFPQParams *params) {
    size_t est = sizeof(RaftIVFPQIndex);                                // Object size
    est += params->nLists * sizeof(raft::neighbors::ivf_pq::list_data<std::int64_t>);         // Size of each cluster data
    est += params->nLists * sizeof(std::int64_t);                       // accum_sorted_sizes_ Array
    est += params->nLists * sizeof(std::shared_ptr<raft::neighbors::ivf_pq::list_data<std::int64_t>>);                    // vector of shared ptr to cluster

    return est;
}

size_t EstimateElementSize(const RaftIVFPQParams *params) { return 0; }
} // namespace RaftIVFPQFactory
