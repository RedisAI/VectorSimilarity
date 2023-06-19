#pragma once

#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>

#include "ivf_index.cuh"

class RaftIVFPQIndex : public RaftIVFIndex {
public:
    using raftIvfPQIndex_t = raft::neighbors::ivf_pq::index<std::int64_t>;

    RaftIVFPQIndex(const RaftIVFPQParams *params_pq, std::shared_ptr<VecSimAllocator> allocator)
        : RaftIVFIndex(params_pq, allocator) {
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
        search_params_pq_->n_probes = params_pq->nProbes;
        search_params_pq_->preferred_shmem_carveout = params_pq->preferredShmemCarveout;
    }

    int addVectorBatchGpuBuffer(const void *vector_data, std::int64_t *labels, size_t batch_size,
                                bool overwrite_allowed = true) override {
        auto vector_data_gpu = raft::make_device_matrix_view<const DataType, std::int64_t>(
            (const DataType *)vector_data, batch_size, this->dim);
        auto label_gpu =
            raft::make_device_vector_view<const std::int64_t, std::int64_t>(labels, batch_size);

        if (!pq_index_) {
            pq_index_ = std::make_unique<raftIvfPQIndex_t>(
                raft::neighbors::ivf_pq::build<DataType, std::int64_t>(res_, *build_params_pq_,
                                                                       vector_data_gpu));
        }
        raft::neighbors::ivf_pq::extend<DataType, std::int64_t>(res_, vector_data_gpu, std::make_optional(label_gpu),
                                        pq_index_.get());
        return batch_size;
    }
    void search(const void *vector_data, void *neighbors, void *distances, size_t batch_size,
                size_t k) override {
        auto vector_data_gpu = raft::make_device_matrix_view<const DataType, std::uint32_t>(
            (const DataType *)vector_data, batch_size, this->dim);
        auto neighbors_gpu = raft::make_device_matrix_view<std::int64_t, std::uint32_t>(
            (std::int64_t *)neighbors, batch_size, k);
        auto distances_gpu =
            raft::make_device_matrix_view<float, std::uint32_t>((float *)distances, batch_size, k);

        raft::neighbors::ivf_pq::search<DataType, std::int64_t>(res_, *search_params_pq_, *pq_index_, vector_data_gpu,
                                        neighbors_gpu, distances_gpu);
    }
    VecSimIndexInfo info() const override {
        VecSimIndexInfo info;
        info.raftIvfPQInfo.dim = this->dim;
        info.raftIvfPQInfo.type = this->vecType;
        info.raftIvfPQInfo.metric = this->metric;
        info.raftIvfPQInfo.indexSize = this->indexSize();
        info.raftIvfPQInfo.nLists = build_params_pq_->n_lists;
        info.raftIvfPQInfo.pqBits = build_params_pq_->pq_bits;
        info.raftIvfPQInfo.pqDim = build_params_pq_->pq_dim;
        info.algo = VecSimAlgo_RaftIVFPQ;
        return info;
    }
    size_t nLists() override { return build_params_pq_->n_lists; }
    size_t indexSize() const override { return pq_index_.get() == nullptr ? 0 : pq_index_->size(); }
protected:
    std::unique_ptr<raftIvfPQIndex_t> pq_index_;
    // Build params are kept as class member because the build step on Raft side happens on
    // the first vector insertion
    std::unique_ptr<raft::neighbors::ivf_pq::index_params> build_params_pq_;
    std::unique_ptr<raft::neighbors::ivf_pq::search_params> search_params_pq_;
};
