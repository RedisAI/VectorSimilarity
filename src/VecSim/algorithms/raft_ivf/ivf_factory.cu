//#include "ivf_flat.cuh"
//#include "ivf_pq.cuh"
#include "ivf_index.cuh"
#include "ivf_factory.h"
#include "ivf_tiered.cuh"
#include <iostream>


namespace RaftIVFFlatFactory {

VecSimIndex *NewIndex(const RaftIVFFlatParams *params, std::shared_ptr<VecSimAllocator> allocator)
{
    assert(params->type == VecSimType_FLOAT32);
    return new (allocator) RaftIVFIndex(params, allocator);
}

VecSimIndex *NewTieredIndex(const TieredRaftIVFFlatParams *params,
                            std::shared_ptr<VecSimAllocator> allocator)
{
    assert(params->flatParams.type == VecSimType_FLOAT32);
    auto *flat_index = NewIndex(&params->flatParams, allocator);
    return new (allocator) TieredRaftIvfIndex(dynamic_cast<RaftIvfIndexInterface*>(flat_index), params->tieredParams);
}

size_t EstimateInitialSize(const RaftIVFFlatParams *params)
{
    return sizeof(RaftIVFIndex);
}

size_t EstimateElementSize(const RaftIVFFlatParams *params)
{
    return 0;
}
}

namespace RaftIVFPQFactory {
VecSimIndex *NewIndex(const RaftIVFPQParams *params, std::shared_ptr<VecSimAllocator> allocator)
{
    assert(params->type == VecSimType_FLOAT32);
    return new (allocator) RaftIVFIndex(params, allocator);
}
VecSimIndex *NewTieredIndex(const TieredRaftIVFPQParams *params,
                            std::shared_ptr<VecSimAllocator> allocator)
{
    assert(params->PQParams.type == VecSimType_FLOAT32);
    auto *pq_index = NewIndex(&params->PQParams, allocator);
    return new (allocator) TieredRaftIvfIndex(dynamic_cast<RaftIvfIndexInterface*>(pq_index), params->tieredParams);
}

size_t EstimateInitialSize(const RaftIVFPQParams *params)
{
    return sizeof(RaftIVFIndex);
}

size_t EstimateElementSize(const RaftIVFPQParams *params)
{
    return 0;
}
}
