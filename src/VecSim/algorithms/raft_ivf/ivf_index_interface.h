#pragma once

#include <raft/distance/distance_types.hpp>
#include "VecSim/vec_sim_index.h"

raft::distance::DistanceType GetRaftDistanceType(VecSimMetric vsm) {
    raft::distance::DistanceType result;
    switch (vsm) {
    case VecSimMetric::VecSimMetric_L2:
        result = raft::distance::DistanceType::L2Expanded;
        break;
    case VecSimMetric_IP:
        result = raft::distance::DistanceType::InnerProduct;
        break;
    default:
        throw std::runtime_error("Metric not supported");
    }
    return result;
}

class RaftIvfIndexInterface : public VecSimIndexAbstract<float> {
public:
    RaftIvfIndexInterface(std::shared_ptr<VecSimAllocator> allocator, size_t dim,
                          VecSimType vecType, VecSimMetric metric, size_t blockSize, bool multi)
        : VecSimIndexAbstract<float>(allocator, dim, vecType, metric, blockSize, multi) {}
    virtual int addVectorBatch(const void *vector_data, labelType *label, size_t batch_size,
                               bool overwrite_allowed = true) = 0;
    virtual int addVectorBatchGpuBuffer(const void *vector_data, std::int64_t *label,
                                        size_t batch_size, bool overwrite_allowed = true) = 0;
};
