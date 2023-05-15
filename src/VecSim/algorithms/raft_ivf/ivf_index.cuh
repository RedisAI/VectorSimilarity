#include <optional>
#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "ivf_index_interface.h"

#include <raft/core/device_resources.hpp>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>


class RaftIVFIndex : public RaftIvfIndexInterface {
public:
    using DataType = float;
    using DistType = float;
    using raftIvfFlatIndex = raft::neighbors::ivf_flat::index<DataType, std::int64_t>;
    using raftIvfPQIndex = raft::neighbors::ivf_pq::index<std::int64_t>;

    template <typename T>
    RaftIVFIndex(const T *params, std::shared_ptr<VecSimAllocator> allocator);
    int addVector(const void *vector_data, labelType label, bool overwrite_allowed = true) override;
    int addVectorBatch(const void *vector_data, labelType* label, size_t batch_size, bool overwrite_allowed = true) override;
    int addVectorBatchGpuBuffer(const void *vector_data, std::int64_t* label, size_t batch_size, bool overwrite_allowed = true) override;
    int deleteVector(labelType label) override { return 0;}
    double getDistanceFrom(labelType label, const void *vector_data) const override {
        assert(!"getDistanceFrom not implemented");
        return INVALID_SCORE;
    }
    size_t indexSize() const override { return counts_; }
    size_t indexCapacity() const override {
        // Not implemented
        return 0;
    }
    void increaseCapacity() override {
        // Not implemented
    }
    inline size_t indexLabelCount() const override {
        return counts_; //TODO: Return unique counts
    }
    VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k, VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius, VecSimQueryParams *queryParams) override
    {
        assert(!"RangeQuery not implemented");
    }
    VecSimIndexInfo info() const override
    {
        VecSimIndexInfo info;
        if (is_flat_) {
            info.raftIvfFlatInfo.dim = this->dim;
            info.raftIvfFlatInfo.type = this->vecType;
            info.raftIvfFlatInfo.metric = this->metric;
            info.raftIvfFlatInfo.indexSize = this->counts_;
            info.raftIvfFlatInfo.nLists = build_params_flat_->n_lists;
            
            info.algo = VecSimAlgo_RaftIVFFlat;
        } else {
            info.raftIvfPQInfo.dim = this->dim;
            info.raftIvfPQInfo.type = this->vecType;
            info.raftIvfPQInfo.metric = this->metric;
            info.raftIvfPQInfo.indexSize = this->counts_;
            info.raftIvfPQInfo.nLists = build_params_pq_->n_lists;
            info.raftIvfPQInfo.pqBits = build_params_pq_->pq_bits;
            info.raftIvfPQInfo.pqDim = build_params_pq_->pq_dim;
            info.algo = VecSimAlgo_RaftIVFPQ;
        }
        return info;
    }
    VecSimInfoIterator *infoIterator() const override
    {
        assert(!"infoIterator not implemented");
    }
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob, VecSimQueryParams *queryParams) const override
    {
        assert(!"newBatchIterator not implemented");
    }
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override
    {
        assert(!"preferAdHocSearch not implemented");
    }


protected:
    raft::device_resources res_;
    std::unique_ptr<raftIvfFlatIndex> flat_index_;
    std::unique_ptr<raftIvfPQIndex> pq_index_;
    idType counts_;
    std::unique_ptr<raft::neighbors::ivf_flat::index_params> build_params_flat_;
    std::unique_ptr<raft::neighbors::ivf_flat::search_params> search_params_flat_;
    std::unique_ptr<raft::neighbors::ivf_pq::index_params> build_params_pq_;
    std::unique_ptr<raft::neighbors::ivf_pq::search_params> search_params_pq_;
    bool is_flat_ = true;
};

template <typename T>
RaftIVFIndex::RaftIVFIndex(const T *params, std::shared_ptr<VecSimAllocator> allocator)
    : RaftIvfIndexInterface(allocator, params->dim, params->type, params->metric, params->blockSize, params->multi),
      counts_(0)
{
    if constexpr (std::is_same_v<RaftIVFFlatParams, T>) {
        const RaftIVFFlatParams *params_flat = dynamic_cast<const RaftIVFFlatParams*>(params);
        build_params_flat_ = std::make_unique<raft::neighbors::ivf_flat::index_params>();
        build_params_flat_->metric = GetRaftDistanceType(params_flat->metric);
        build_params_flat_->n_lists = params_flat->nLists;
        build_params_flat_->kmeans_n_iters = params_flat->kmeans_nIters;
        build_params_flat_->kmeans_trainset_fraction = params_flat->kmeans_trainsetFraction;
        build_params_flat_->adaptive_centers = params_flat->adaptiveCenters;
        build_params_flat_->add_data_on_build = false;
        search_params_flat_ = std::make_unique<raft::neighbors::ivf_flat::search_params>();
        search_params_flat_->n_probes = params_flat->nProbes;
    } else if constexpr (std::is_same_v<RaftIVFPQParams, T>) {
        is_flat_ = false;
        const RaftIVFPQParams *params_pq = dynamic_cast<const RaftIVFPQParams*>(params);
        build_params_pq_ = std::make_unique<raft::neighbors::ivf_pq::index_params>();
        build_params_pq_->metric = GetRaftDistanceType(params_pq->metric);
        build_params_pq_->n_lists = params_pq->nLists;
        build_params_pq_->pq_bits = params_pq->pqBits;
        build_params_pq_->pq_dim = params_pq->pqDim;
        build_params_pq_->add_data_on_build = false;
        switch (params_pq->codebookKind) {
            case (RaftIVFPQ_PerCluster):
                build_params_pq_->codebook_kind = raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
                break;
            case (RaftIVFPQ_PerSubspace):
                build_params_pq_->codebook_kind = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
                break;
            default:
                assert(!"Unexpected codebook kind value");
        }

        search_params_pq_ = std::make_unique<raft::neighbors::ivf_pq::search_params>();
        switch (params_pq->lutType) {
            case (CUDAType_R_32F):
                search_params_pq_->lut_dtype = CUDA_R_32F;
                break;
            case (CUDAType_R_16F):
                search_params_pq_->lut_dtype = CUDA_R_16F;
                break;
            case (CUDAType_R_8U):
                search_params_pq_->lut_dtype = CUDA_R_8U;
                break;
        }
        switch (params_pq->internalDistanceType) {
            case (CUDAType_R_32F):
                search_params_pq_->lut_dtype = CUDA_R_32F;
                break;
            case (CUDAType_R_16F):
                search_params_pq_->lut_dtype = CUDA_R_16F;
                break;
            default:
                assert(!"Unexpected codebook kind value");
        }
        search_params_pq_->preferred_shmem_carveout = params_pq->preferredShmemCarveout;
    } else {
        throw std::runtime_error("Unknown params");
    }
}

int RaftIVFIndex::addVector(const void *vector_data, labelType label, bool overwrite_allowed)
{
    assert(label < static_cast<labelType>(std::numeric_limits<std::int64_t>::max()));
    auto vector_data_gpu = raft::make_device_matrix<DataType, std::int64_t>(res_, 1, this->dim);
    auto label_converted = static_cast<std::int64_t>(label);
    auto label_gpu = raft::make_device_vector<std::int64_t, std::int64_t>(res_, 1);

    RAFT_CUDA_TRY(cudaMemcpyAsync(vector_data_gpu.data_handle(), (DataType*)vector_data,
                                  this->dim * sizeof(float), cudaMemcpyDefault, res_.get_stream()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(label_gpu.data_handle(), &label_converted,
                                  sizeof(std::int64_t), cudaMemcpyDefault, res_.get_stream()));

    if (is_flat_) {
        if (!flat_index_) {
            flat_index_ = std::make_unique<raftIvfFlatIndex>(raft::neighbors::ivf_flat::build<DataType, std::int64_t>(
                res_, *build_params_flat_, raft::make_const_mdspan(vector_data_gpu.view())));
        }
        raft::neighbors::ivf_flat::extend(res_, raft::make_const_mdspan(vector_data_gpu.view()),
            std::make_optional(raft::make_const_mdspan(label_gpu.view())), flat_index_.get());
    } else {
        if (!pq_index_) {
            pq_index_ = std::make_unique<raftIvfPQIndex>(raft::neighbors::ivf_pq::build<DataType, std::int64_t>(
                res_, *build_params_pq_, raft::make_const_mdspan(vector_data_gpu.view())));
        }
        raft::neighbors::ivf_pq::extend(res_, raft::make_const_mdspan(vector_data_gpu.view()), std::make_optional(raft::make_const_mdspan(label_gpu.view())), pq_index_.get());
    }
    res_.sync_stream();

    // TODO: Verify that label exists already?
    // TODO normalizeVector for cosine?
    this->counts_ += 1;
    return 1;
}

int RaftIVFIndex::addVectorBatch(const void *vector_data, labelType* labels, size_t batch_size, bool overwrite_allowed)
{
    auto vector_data_gpu = raft::make_device_matrix<DataType, std::int64_t>(res_, batch_size, this->dim);
    auto label_original = std::vector<labelType>(labels, labels + batch_size);
    auto label_converted = std::vector<std::int64_t>(label_original.begin(), label_original.end());
    auto label_gpu = raft::make_device_vector<std::int64_t, std::int64_t>(res_, batch_size);

    RAFT_CUDA_TRY(cudaMemcpyAsync(vector_data_gpu.data_handle(), (DataType*)vector_data,
                                  this->dim * batch_size * sizeof(float), cudaMemcpyDefault, res_.get_stream()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(label_gpu.data_handle(), label_converted.data(),
                                  batch_size * sizeof(std::int64_t), cudaMemcpyDefault, res_.get_stream()));
    
    if (is_flat_) {
        if (!flat_index_) {
            flat_index_ = std::make_unique<raftIvfFlatIndex>(raft::neighbors::ivf_flat::build<DataType, std::int64_t>(
                res_, *build_params_flat_, raft::make_const_mdspan(vector_data_gpu.view())));
        }
        raft::neighbors::ivf_flat::extend(res_, raft::make_const_mdspan(vector_data_gpu.view()),
            std::make_optional(raft::make_const_mdspan(label_gpu.view())), flat_index_.get());
    } else {
        if (!pq_index_) {
            pq_index_ = std::make_unique<raftIvfPQIndex>(raft::neighbors::ivf_pq::build<DataType, std::int64_t>(
                res_, *build_params_pq_, raft::make_const_mdspan(vector_data_gpu.view())));
        }
        raft::neighbors::ivf_pq::extend(res_, raft::make_const_mdspan(vector_data_gpu.view()),
            std::make_optional(raft::make_const_mdspan(label_gpu.view())), pq_index_.get());
    }
    res_.sync_stream();

    this->counts_ += batch_size;
    return batch_size;
}

int RaftIVFIndex::addVectorBatchGpuBuffer(const void *vector_data, std::int64_t* labels, size_t batch_size, bool overwrite_allowed)
{
    auto vector_data_gpu = raft::make_device_matrix_view<const DataType, std::int64_t>((const DataType*)vector_data, batch_size, this->dim);
    auto label_gpu = raft::make_device_vector_view<const std::int64_t, std::int64_t>(labels, batch_size);

    if (is_flat_) {
        if (!flat_index_) {
            flat_index_ = std::make_unique<raftIvfFlatIndex>(raft::neighbors::ivf_flat::build<DataType, std::int64_t>(
                res_, *build_params_flat_, vector_data_gpu));
        }
        raft::neighbors::ivf_flat::extend(res_, vector_data_gpu, std::make_optional(label_gpu), flat_index_.get());
    } else {
        if (!pq_index_) {
            pq_index_ = std::make_unique<raftIvfPQIndex>(raft::neighbors::ivf_pq::build<DataType, std::int64_t>(
                res_, *build_params_pq_, vector_data_gpu));
        }
        raft::neighbors::ivf_pq::extend(res_, vector_data_gpu, std::make_optional(label_gpu), pq_index_.get());
    }
    res_.sync_stream();
    
    this->counts_ += batch_size;
    return batch_size;
}

// Search for the k closest vectors to a given vector in the index.
VecSimQueryResult_List RaftIVFIndex::topKQuery(
    const void *queryBlob, size_t k, VecSimQueryParams *queryParams)
{
    VecSimQueryResult_List result_list = {0};
    if ((is_flat_ && !flat_index_) || (!is_flat_ && !pq_index_)) {
        result_list.results = array_new<VecSimQueryResult>(0);
        return result_list;
    }
    auto vector_data_gpu = raft::make_device_matrix<DataType, std::int64_t>(res_, queryParams->batchSize, this->dim);
    auto neighbors_gpu = raft::make_device_matrix<std::int64_t, std::int64_t>(res_, queryParams->batchSize, k);
    auto distances_gpu = raft::make_device_matrix<float, std::int64_t>(res_, queryParams->batchSize, k);
    RAFT_CUDA_TRY(cudaMemcpyAsync(vector_data_gpu.data_handle(), (const DataType*)queryBlob,
                                  this->dim * queryParams->batchSize * sizeof(float), cudaMemcpyDefault, res_.get_stream()));
    if (is_flat_)
        raft::neighbors::ivf_flat::search(res_, *search_params_flat_, *flat_index_, raft::make_const_mdspan(vector_data_gpu.view()), neighbors_gpu.view(), distances_gpu.view());
    else
        raft::neighbors::ivf_pq::search(res_, *search_params_pq_, *pq_index_, raft::make_const_mdspan(vector_data_gpu.view()), neighbors_gpu.view(), distances_gpu.view());

    auto result_size = queryParams->batchSize * k;
    auto* neighbors = array_new_len<std::int64_t>(result_size, result_size);
    auto* distances = array_new_len<float>(result_size, result_size);
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
