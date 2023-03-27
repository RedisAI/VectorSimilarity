#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // BFParams
#include "VecSim/vec_sim_index.h"        // VecSimIndexAbstract
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator

#include "raft/core/device_resources.hpp"
#include "raft/neighbors/ivf_pq.cuh"
#include "raft/neighbors/ivf_pq_types.hpp"

extern raft::distance::DistanceType GetRaftDistanceType(VecSimMetric vsm);

template <typename DataType, typename DistType>
class RaftPQIndex : public VecSimIndexAbstract<DistType> {
    RaftPQIndex(const RaftPQParams *params, std::shared_ptr<VecSimAllocator> allocator);
    /*int addVector(const void *vector_data, labelType label, bool overwrite_allowed = true) override;
    int deleteVector(labelType label) override { return 0;}
    double getDistanceFrom(labelType label, const void *vector_data) const override {
        assert(!"getDistanceFrom not implemented");
        return INVALID_SCORE;
    }
    size_t indexSize() const override {
        if (!flat_index_) {
            return 0;
        }
        return counts_;
    }
    size_t indexCapacity() const override {
        assert(!"indexCapacity not implemented");
        return 0;
    }
    void increaseCapacity() override {
        assert(!"increaseCapacity not implemented");
    }
    inline size_t indexLabelCount() const override {
        if (!flat_index_) {
            return 0;
        }
        return counts_; //TODO: Return unique counts
    }
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k, VecSimQueryParams *queryParams) override;
    virtual VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius, VecSimQueryParams *queryParams) override
    {
        assert(!"RangeQuery not implemented");
    }
    virtual VecSimIndexInfo info() const override
    {
        VecSimIndexInfo info;
        info.algo = VecSimAlgo_RaftPQ;
        info.bfInfo.dim = this->dim;
        info.bfInfo.type = this->vecType;
        info.bfInfo.metric = this->metric;
        info.bfInfo.indexSize = this->counts_;
        info.bfInfo.indexLabelCount = this->indexLabelCount();
        info.bfInfo.blockSize = this->blockSize;
        info.bfInfo.memory = this->getAllocationSize();
        info.bfInfo.isMulti = false;
        info.bfInfo.last_mode = this->last_mode;
        return info;
    }
    virtual VecSimInfoIterator *infoIterator() const override
    {
        
    }
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob, VecSimQueryParams *queryParams) const override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;

protected:
    raft::device_resources res_;
    std::unique_ptr<raft::neighbors::ivf_pq::index<std::int64_t>> pq_index_;*/
    idType counts_;
};

template <typename DataType, typename DistType>
RaftPQIndex<DataType, DistType>::RaftPQIndex(const RaftPQParams *params,
                                               std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndexAbstract<DistType>(allocator, params->dim, params->type, params->metric, params->blockSize, false),
      counts_(0)
{
    /*
    auto build_params = raft::neighbors::ivf_pq::index_params{};
    build_params.metric = GetRaftDistanceType(params->metric);
    build_params.n_lists = params->n_lists;
    build_params.pq_bits = params->pq_bits;
    build_params.pq_dims = params->pq_dims;
    build_params.adaptive_centers = params->adaptive_centers;
    build_params.add_data_on_build = false;
    pq_index_ = std::make_unique<raft::neighbors::ivf_pq::index<std::int64_t>>(raft::neighbors::ivf_pq::build<std::int64_t>(res_, build_params,
                                                                           nullptr, 0, this->dim));
    */
}
