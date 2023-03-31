#include "raft_flat.cuh"
#include "raft_pq.cuh"
#include "raft_ivf_factory.h"
#include <iostream>


namespace RaftFlatFactory {

VecSimIndex *NewIndex(const RaftFlatParams *params, std::shared_ptr<VecSimAllocator> allocator)
{
    assert(params->type == VecSimType_FLOAT32);
    return new (allocator) RaftFlatIndex(params, allocator);
}
size_t EstimateInitialSize(const RaftFlatParams *params)
{
    return sizeof(RaftFlatIndex);
}
size_t EstimateElementSize(const RaftFlatParams *params)
{
    return 0;
}
}

namespace RaftPQFactory {
VecSimIndex *NewIndex(const RaftPQParams *params, std::shared_ptr<VecSimAllocator> allocator)
{
    return new (allocator) RaftPQIndex(params, allocator);
}
size_t EstimateInitialSize(const RaftPQParams *params)
{
    return sizeof(RaftPQIndex);
}
size_t EstimateElementSize(const RaftPQParams *params)
{
    return 0;
}
}