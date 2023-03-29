#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // BFParams
#include "VecSim/vec_sim_index.h"        // VecSimIndexAbstract
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator

#include "VecSim/algorithms/brute_force/bfs_batch_iterator.h"   // TODO: Temporary header to remove

#include "raft/core/device_resources.hpp"
#include "raft/neighbors/ivf_pq.cuh"
#include "raft/neighbors/ivf_pq_types.hpp"

extern raft::distance::DistanceType GetRaftDistanceType(VecSimMetric vsm);

template <typename DataType, typename DistType>
class RaftPQIndex : public VecSimIndexAbstract<DistType> {
public:
    RaftPQIndex(const RaftPQParams *params, std::shared_ptr<VecSimAllocator> allocator);
    int addVector(const void *vector_data, labelType label, bool overwrite_allowed = true) override;
    int deleteVector(labelType label) override { 
        assert(!"deleteVector not implemented");
        return 0;
    }
    double getDistanceFrom(labelType label, const void *vector_data) const override {
        assert(!"getDistanceFrom not implemented");
        return INVALID_SCORE;
    }
    size_t indexSize() const override {
        if (!pq_index_) {
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
        if (!pq_index_) {
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
        assert(!"infoIterator not implemented");
        size_t numberOfInfoFields = 12;
        VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields);
        return infoIterator;
    }
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob, VecSimQueryParams *queryParams) const override
    {
        assert(!"newBatchIterator not implemented");
        // TODO: Using BFS_Batch Iterator temporarily for the return type
        return new (this->allocator) BFS_BatchIterator<DataType, DistType>(const_cast<void*>(queryBlob), nullptr, queryParams, this->allocator);
    }
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override
    {
        return true; // TODO: Implement this
    }

protected:
    raft::device_resources res_;
    std::unique_ptr<raft::neighbors::ivf_pq::index<std::int64_t>> pq_index_;
    idType counts_;
    raft::neighbors::ivf_pq::index_params build_params_;
    raft::neighbors::ivf_pq::search_params search_params_;
};

template <typename DataType, typename DistType>
RaftPQIndex<DataType, DistType>::RaftPQIndex(const RaftPQParams *params,
                                               std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndexAbstract<DistType>(allocator, params->dim, params->type, params->metric, params->blockSize, false),
      counts_(0)
{
    build_params_.metric = GetRaftDistanceType(params->metric);
    build_params_.n_lists = params->nLists;
    build_params_.pq_bits = params->pqBits;
    build_params_.pq_dim = params->pqDim;
    build_params_.add_data_on_build = true;
    switch (params->codebookKind) {
        case (RaftPQ_PerCluster):
            build_params_.codebook_kind = raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
            break;
        case (RaftPQ_PerSubspace):
            build_params_.codebook_kind = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
            break;
        default:
            assert(!"Unexpected codebook kind value");
    }

    switch (params->lutType) {
        case (CUDAType_R_32F):
            search_params_.lut_dtype = CUDA_R_32F;
            break;
        case (CUDAType_R_16F):
            search_params_.lut_dtype = CUDA_R_16F;
            break;
        case (CUDAType_R_8U):
            search_params_.lut_dtype = CUDA_R_8U;
            break;
    }
    switch (params->internalDistanceType) {
        case (CUDAType_R_32F):
            search_params_.lut_dtype = CUDA_R_32F;
            break;
        case (CUDAType_R_16F):
            search_params_.lut_dtype = CUDA_R_16F;
            break;
        default:
            assert(!"Unexpected codebook kind value");
    }
    search_params_.preferred_shmem_carveout = params->preferredShmemCarveout;
    //pq_index_ = std::make_unique<raft::neighbors::ivf_pq::index<std::int64_t>>(raft::neighbors::ivf_pq::build<std::int64_t>(res_, build_params,
    //                                                                       nullptr, 0, this->dim));
}

template <typename DataType, typename DistType>
int RaftPQIndex<DataType, DistType>::addVector(const void *vector_data, labelType label, bool overwrite_allowed)
{
    assert(label < static_cast<labelType>(std::numeric_limits<std::int64_t>::max()));
    auto vector_data_gpu = raft::make_device_matrix<DataType, std::int64_t>(res_, 1, this->dim);
    auto label_converted = static_cast<std::int64_t>(label);
    auto label_gpu = raft::make_device_vector<std::int64_t, std::int64_t>(res_, 1);
    raft::copy(vector_data_gpu.data_handle(), (DataType*)vector_data, this->dim, res_.get_stream());
    raft::copy(label_gpu.data_handle(), &label_converted, 1, res_.get_stream());

    // TODO use mdspan ivf_pq functions of 23.04
    if (!pq_index_) {
        pq_index_ = std::make_unique<raft::neighbors::ivf_pq::index<std::int64_t>>(raft::neighbors::ivf_pq::build<DataType, std::int64_t>(
            res_, build_params_, vector_data_gpu.data_handle(), std::int64_t(1), uint32_t(this->dim)));
    } else {
        raft::neighbors::ivf_pq::extend(res_, pq_index_.get(), vector_data_gpu.data_handle(),
            label_gpu.data_handle(), std::int64_t(1));
    }
    // TODO: Verify that label exists already?
    // TODO normalizeVector for cosine?
    this->counts_ += 1;
    return 1;
}

// Search for the k closest vectors to a given vector in the index.
template <typename DataType, typename DistType>
VecSimQueryResult_List RaftPQIndex<DataType, DistType>::topKQuery(
    const void *queryBlob, size_t k, VecSimQueryParams *queryParams)
{
    VecSimQueryResult_List result_list = {0};
    if (!pq_index_) {
        result_list.results = array_new<VecSimQueryResult>(0);
        return result_list;
    }
    auto vector_data_gpu = raft::make_device_matrix<DataType, std::int64_t>(res_, queryParams->batchSize, this->dim);
    auto neighbors_gpu = raft::make_device_matrix<std::int64_t, std::int64_t>(res_, queryParams->batchSize, k);
    auto distances_gpu = raft::make_device_matrix<float, std::int64_t>(res_, queryParams->batchSize, k);
    raft::copy(vector_data_gpu.data_handle(), (const DataType*)queryBlob, this->dim * queryParams->batchSize, res_.get_stream());
    raft::neighbors::ivf_pq::search(res_, search_params_, *pq_index_, vector_data_gpu.data_handle(), 1, k, neighbors_gpu.data_handle(), distances_gpu.data_handle());

    auto result_size = queryParams->batchSize * k;
    auto neighbors = array_new_len<std::int64_t>(result_size, result_size);
    auto distances = array_new_len<float>(result_size, result_size);
    raft::copy(neighbors, neighbors_gpu.data_handle(), result_size, res_.get_stream());
    raft::copy(distances, distances_gpu.data_handle(), result_size, res_.get_stream());
    result_list.results = array_new_len<VecSimQueryResult>(k, k);
    for (size_t i = 0; i < k; ++i) {
        VecSimQueryResult_SetId(result_list.results[i], neighbors[i]);
        VecSimQueryResult_SetScore(result_list.results[i], distances[i]);
    }
    array_free(neighbors);
    array_free(distances);
    return result_list;
}
