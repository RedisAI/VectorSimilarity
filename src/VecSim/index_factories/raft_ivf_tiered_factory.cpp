#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/algorithms/raft_ivf/ivf_tiered.h"
#include "VecSim/algorithms/raft_ivf/ivf_interface.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/index_factories/raft_ivf_factory.h"

namespace TieredRaftIvfFactory {

VecSimIndex *NewIndex(const TieredIndexParams *params)
{
    assert(params->primaryIndexParams->algoParams.raftIvfParams.type == VecSimType_FLOAT32 &&
            "Invalid IVF data type algorithm");

    using DataType = float;
    using DistType = float;
    // initialize raft index
    auto *raft_index = reinterpret_cast<RaftIvfInterface<DataType, DistType> *>(
        RaftIvfFactory::NewIndex(params->primaryIndexParams));
    // initialize brute force index
    BFParams bf_params = {
        .type = params->primaryIndexParams->algoParams.raftIvfParams.type,
        .dim = params->primaryIndexParams->algoParams.raftIvfParams.dim,
        .metric = params->primaryIndexParams->algoParams.raftIvfParams.metric,
        .multi = params->primaryIndexParams->algoParams.raftIvfParams.multi,
        //.blockSize = params->primaryIndexParams->algoParams.raftIvfParams.blockSize
    };

    std::shared_ptr<VecSimAllocator> flat_allocator = VecSimAllocator::newVecsimAllocator();
    AbstractIndexInitParams abstractInitParams = {.allocator = flat_allocator,
                                                  .dim = bf_params.dim,
                                                  .vecType = bf_params.type,
                                                  .metric = bf_params.metric,
                                                  .blockSize = bf_params.blockSize,
                                                  .multi = bf_params.multi,
                                                  .logCtx = params->primaryIndexParams->logCtx};
    auto frontendIndex = static_cast<BruteForceIndex<DataType, DistType> *>(
        BruteForceFactory::NewIndex(&bf_params, abstractInitParams));

    // Create new tiered RaftIVF index
    std::shared_ptr<VecSimAllocator> management_layer_allocator =
        VecSimAllocator::newVecsimAllocator();

    return new (management_layer_allocator) TieredRaftIvfIndex<DataType, DistType>(
        raft_index, frontendIndex, *params, management_layer_allocator);
}

// The size estimation is the sum of the buffer (brute force) and main index initial sizes
// estimations, plus the tiered index class size. Note it does not include the size of internal
// containers such as the job queue, as those depend on the user implementation.
size_t EstimateInitialSize(const TieredIndexParams *params) {
    auto raft_ivf_params = params->primaryIndexParams->algoParams.raftIvfParams;

    // Add size estimation of VecSimTieredIndex sub indexes.
    size_t est = RaftIvfFactory::EstimateInitialSize(&raft_ivf_params);

    // Management layer allocator overhead.
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    est += sizeof(VecSimAllocator) + allocations_overhead;

    // Size of the TieredRaftIvfIndex struct.
    if (raft_ivf_params.type == VecSimType_FLOAT32) {
        est += sizeof(TieredRaftIvfIndex<float, float>);
    } else if (raft_ivf_params.type == VecSimType_FLOAT64) {
        est += sizeof(TieredRaftIvfIndex<double, double>);
    }

    return est;
}

}; // namespace TieredRaftIvfFactory
