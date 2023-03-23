#include "raft_flat.cuh"
#include "raft_pq.cuh"
#include "raft_ivf_factory.h"
#include <iostream>


auto GetRaftDistanceType(VecSimMetric vsm){
    raft::distance::DistanceType result;
    switch (vsm) {
        case VecSimMetric::VecSimMetric_L2:
            result = raft::distance::DistanceType::L2Expanded;
            break;
        case VecSimMetric_IP:
            result = raft::distance::DistanceType::InnerProduct;
            break;
        case VecSimMetric_Cosine:
            result = raft::distance::DistanceType::CosineExpanded;
            break;
        default:
            throw std::runtime_error("Metric not supported");
    }
    return result;
}
namespace RaftFlatFactory {

VecSimIndex *NewIndex(const RaftFlatParams *params, std::shared_ptr<VecSimAllocator> allocator)
{
    std::cout << "Test Raft\n";
    if (params->type == VecSimType_FLOAT32) {
        return new (allocator) RaftFlatIndex<float, float>(params, allocator);
    } else if (params->type == VecSimType_FLOAT64) {
        return new (allocator) RaftFlatIndex<double, double>(params, allocator);
    }
}
size_t EstimateInitialSize(const RaftFlatParams *params)
{

}
size_t EstimateElementSize(const RaftFlatParams *params)
{

}
}

namespace RaftPQFactory {

    /*VecSimIndex *NewIndex(const RaftPQParams *params, std::shared_ptr<VecSimAllocator> allocator)
    {
        std::cout << "Test Raft\n";
        return new (allocator) RaftPQIndex(params, allocator);
    }*/
    size_t EstimateInitialSize(const RaftPQParams *params)
    {
    
    }
    size_t EstimateElementSize(const RaftPQParams *params)
    {
    
    }
}