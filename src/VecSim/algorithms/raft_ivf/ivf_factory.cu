#include "ivf_flat.cuh"
#include "ivf_pq.cuh"
#include "ivf_factory.h"
#include <iostream>


namespace RaftIVFFlatFactory {

VecSimIndex *NewIndex(const RaftIVFFlatParams *params, std::shared_ptr<VecSimAllocator> allocator)
{
    assert(params->type == VecSimType_FLOAT32);
    return new (allocator) RaftIVFFlatIndex(params, allocator);
}
size_t EstimateInitialSize(const RaftIVFFlatParams *params)
{
    return sizeof(RaftIVFFlatIndex);
}
size_t EstimateElementSize(const RaftIVFFlatParams *params)
{
    return 0;
}
}

namespace RaftIVFPQFactory {
VecSimIndex *NewIndex(const RaftIVFPQParams *params, std::shared_ptr<VecSimAllocator> allocator)
{
    return new (allocator) RaftIVFPQIndex(params, allocator);
}
size_t EstimateInitialSize(const RaftIVFPQParams *params)
{
    return sizeof(RaftIVFPQIndex);
}
size_t EstimateElementSize(const RaftIVFPQParams *params)
{
    return 0;
}
}