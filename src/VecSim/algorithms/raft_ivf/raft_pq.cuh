#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // BFParams
#include "VecSim/vec_sim_index.h"        // VecSimIndexAbstract
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator

#include "raft/core/device_resources.hpp"
#include "raft/neighbors/ivf_pq.cuh"
#include "raft/neighbors/ivf_pq_types.hpp"


template <typename DataType, typename DistType>
class RaftPQIndex : public VecSimIndexAbstract<DistType> {
    RaftPQIndex(const RaftPQParams *params, std::shared_ptr<VecSimAllocator> allocator);
};

template <typename DataType, typename DistType>
RaftPQIndex<DataType, DistType>::RaftPQIndex(const RaftPQParams *params,
                                               std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndexAbstract<DistType>(allocator, params->dim, params->type, params->metric, params->blockSize, false),
      count(0)
{
    /*assert(VecSimType_sizeof(this->vecType) == sizeof(DataType));
    this->idToLabelMapping.resize(params->initialCapacity);
    */



    raft::device_resources res_;
    auto build_params = raft::neighbors::ivf_pq::index_params{};
    /*build_params.metric = GetRaftDistanceType(params->metric);
    build_params.n_lists = ivf_raft_cfg.nlist;
    build_params.kmeans_n_iters = ivf_raft_cfg.kmeans_n_iters;
    build_params.kmeans_trainset_fraction = ivf_raft_cfg.kmeans_trainset_fraction;
    build_params.adaptive_centers = ivf_raft_cfg.adaptive_centers;
    return raft::neighbors::ivf_pq::build<float, std::int64_t>(*res_, build_params, data_gpu.data(), rows, dim);*/
}
