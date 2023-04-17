#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // BFParams
#include "VecSim/vec_sim_index.h"        // VecSimIndexAbstract
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator
#include "ivf_index_interface.h"

#include "VecSim/algorithms/brute_force/bfs_batch_iterator.h"   // TODO: Temporary header to remove

#include "raft/core/device_resources.hpp"
#include "raft/neighbors/ivf_pq.cuh"
#include "raft/neighbors/ivf_pq_types.hpp"

#ifdef RAFT_COMPILED
#include <raft/neighbors/specializations.cuh>
#endif

class RaftIVFPQIndex : public RaftIvfIndexInterface {
public:
    using DataType = float;
    using DistType = float;
    using raftIvfPQIndex = raft::neighbors::ivf_pq::index<std::int64_t>;
    RaftIVFPQIndex(const RaftIVFPQParams *params, std::shared_ptr<VecSimAllocator> allocator);
    int addVector(const void *vector_data, labelType label, bool overwrite_allowed = true) override;
    int addVectorBatch(const void *vector_data, labelType *labels, size_t batch_size, bool overwrite_allowed = true);
    int deleteVector(labelType label) override { 
        assert(!"deleteVector not implemented");
        return 0;
    }
    double getDistanceFrom(labelType label, const void *vector_data) const override {
        assert(!"getDistanceFrom not implemented");
        return INVALID_SCORE;
    }
    size_t indexSize() const override { return counts_; }
    size_t indexCapacity() const override {
        assert(!"indexCapacity not implemented");
        return 0;
    }
    void increaseCapacity() override {
        assert(!"increaseCapacity not implemented");
    }
    inline size_t indexLabelCount() const override {
        return counts_; //TODO: Return unique counts
    }
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k, VecSimQueryParams *queryParams) override;
    //virtual VecSimQueryResultBatch_List topKQueryBatch(const void *queryBlob, size_t k, VecSimQueryParams *queryParams);
    virtual VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius, VecSimQueryParams *queryParams) override
    {
        assert(!"RangeQuery not implemented");
    }
    virtual VecSimIndexInfo info() const override
    {
        VecSimIndexInfo info;
        info.algo = VecSimAlgo_RaftIVFPQ;
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
    std::unique_ptr<raftIvfPQIndex> pq_index_;
    idType counts_;
    raft::neighbors::ivf_pq::index_params build_params_;
    raft::neighbors::ivf_pq::search_params search_params_;
};

RaftIVFPQIndex::RaftIVFPQIndex(const RaftIVFPQParams *params, std::shared_ptr<VecSimAllocator> allocator)
    : RaftIvfIndexInterface(allocator, params->dim, params->type, params->metric, params->blockSize, params->multi),
      counts_(0)
{
    build_params_.metric = GetRaftDistanceType(params->metric);
    build_params_.n_lists = params->nLists;
    build_params_.pq_bits = params->pqBits;
    build_params_.pq_dim = params->pqDim;
    build_params_.add_data_on_build = true;
    switch (params->codebookKind) {
        case (RaftIVFPQ_PerCluster):
            build_params_.codebook_kind = raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
            break;
        case (RaftIVFPQ_PerSubspace):
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
    //pq_index_ = std::make_unique<raftIvfPQIndex>(raft::neighbors::ivf_pq::build<std::int64_t>(res_, build_params,
    //                                                                       nullptr, 0, this->dim));
}

int RaftIVFPQIndex::addVector(const void *vector_data, labelType label, bool overwrite_allowed)
{
    assert(label < static_cast<labelType>(std::numeric_limits<std::int64_t>::max()));
    auto vector_data_gpu = raft::make_device_matrix<DataType, std::int64_t>(res_, 1, this->dim);
    auto label_converted = static_cast<std::int64_t>(label);
    auto label_gpu = raft::make_device_matrix<std::int64_t, std::int64_t>(res_, 1, 1);
    RAFT_CUDA_TRY(cudaMemcpyAsync(vector_data_gpu.data_handle(), (DataType*)vector_data,
                                  this->dim * sizeof(float), cudaMemcpyDefault, res_.get_stream()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(label_gpu.data_handle(), &label_converted,
                                  sizeof(std::int64_t), cudaMemcpyDefault, res_.get_stream()));
    if (!pq_index_) {
        pq_index_ = std::make_unique<raftIvfPQIndex>(raft::neighbors::ivf_pq::build<DataType, std::int64_t>(
            res_, build_params_, raft::make_const_mdspan(vector_data_gpu.view())));
    } else {
        raft::neighbors::ivf_pq::extend(res_, raft::make_const_mdspan(vector_data_gpu.view()), std::make_optional(raft::make_const_mdspan(label_gpu.view())), pq_index_.get());
    }
    res_.sync_stream();
    // TODO: Verify that label exists already?
    // TODO normalizeVector for cosine?
    this->counts_ += 1;
    return 1;
}

int RaftIVFPQIndex::addVectorBatch(const void *vector_data, labelType* labels, size_t batch_size, bool overwrite_allowed)
{
    auto vector_data_gpu = raft::make_device_matrix<DataType, std::int64_t>(res_, batch_size, this->dim);
    auto label_original = std::vector<labelType>(labels, labels + batch_size);
    auto label_converted = std::vector<std::int64_t>(label_original.begin(), label_original.end());
    auto label_gpu = raft::make_device_matrix<std::int64_t, std::int64_t>(res_, batch_size, 1);
    RAFT_CUDA_TRY(cudaMemcpyAsync(vector_data_gpu.data_handle(), (DataType*)vector_data,
                                  this->dim * batch_size * sizeof(float), cudaMemcpyDefault, res_.get_stream()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(label_gpu.data_handle(), label_converted.data(),
                                  batch_size * sizeof(std::int64_t), cudaMemcpyDefault, res_.get_stream()));

    if (!pq_index_) {
        pq_index_ = std::make_unique<raftIvfPQIndex>(raft::neighbors::ivf_pq::build<DataType, std::int64_t>(
            res_, build_params_, raft::make_const_mdspan(vector_data_gpu.view())));
    } else {
        raft::neighbors::ivf_pq::extend(res_, raft::make_const_mdspan(vector_data_gpu.view()),
            std::make_optional(raft::make_const_mdspan(label_gpu.view())), pq_index_.get());
    }
    res_.sync_stream();
    // TODO: Verify that label exists already?
    // TODO normalizeVector for cosine?
    this->counts_ += batch_size;
    return batch_size;
}

// Search for the k closest vectors to a given vector in the index.
VecSimQueryResult_List RaftIVFPQIndex::topKQuery(
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
    RAFT_CUDA_TRY(cudaMemcpyAsync(vector_data_gpu.data_handle(), (const DataType*)queryBlob,
                                  this->dim * queryParams->batchSize * sizeof(float), cudaMemcpyDefault, res_.get_stream()));
    raft::neighbors::ivf_pq::search(res_, search_params_, *pq_index_, raft::make_const_mdspan(vector_data_gpu.view()), neighbors_gpu.view(), distances_gpu.view());

    auto result_size = queryParams->batchSize * k;
    auto neighbors = array_new_len<std::int64_t>(result_size, result_size);
    auto distances = array_new_len<float>(result_size, result_size);
    RAFT_CUDA_TRY(cudaMemcpyAsync(neighbors, neighbors_gpu.data_handle(),
                                  result_size * sizeof(std::int64_t), cudaMemcpyDefault, res_.get_stream()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(distances, distances_gpu.data_handle(),
                                  result_size * sizeof(float), cudaMemcpyDefault, res_.get_stream()));
    res_.sync_stream();
    result_list.results = array_new_len<VecSimQueryResult>(k, k);
    for (size_t i = 0; i < k; ++i) {
        VecSimQueryResult_SetId(result_list.results[i], neighbors[i]);
        VecSimQueryResult_SetScore(result_list.results[i], distances[i]);
    }
    array_free(neighbors);
    array_free(distances);
    return result_list;
}
