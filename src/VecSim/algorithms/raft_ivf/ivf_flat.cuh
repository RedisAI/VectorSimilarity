#pragma once

#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include "ivf_index.cuh"

class RaftIVFFlatIndex : public RaftIVFIndex {
public:
    using raftIvfFlatIndex_t = raft::neighbors::ivf_flat::index<DataType, std::int64_t>;
    RaftIVFFlatIndex(const RaftIVFFlatParams *params_flat,
                     std::shared_ptr<VecSimAllocator> allocator)
        : RaftIVFIndex(params_flat, allocator) {
        build_params_flat_ = std::make_unique<raft::neighbors::ivf_flat::index_params>();
        build_params_flat_->metric = GetRaftDistanceType(params_flat->metric);
        build_params_flat_->n_lists = params_flat->nLists;
        build_params_flat_->kmeans_n_iters = params_flat->kmeans_nIters;
        build_params_flat_->kmeans_trainset_fraction = params_flat->kmeans_trainsetFraction;
        build_params_flat_->adaptive_centers = params_flat->adaptiveCenters;
        build_params_flat_->add_data_on_build = false;
        search_params_flat_ = std::make_unique<raft::neighbors::ivf_flat::search_params>();
        search_params_flat_->n_probes = params_flat->nProbes;
    }
    int addVectorBatchGpuBuffer(const void *vector_data, std::int64_t *labels, size_t batch_size,
                                bool overwrite_allowed) override {
        auto vector_data_gpu = raft::make_device_matrix_view<const DataType, std::int64_t>(
            (const DataType *)vector_data, batch_size, this->dim);
        auto label_gpu =
            raft::make_device_vector_view<const std::int64_t, std::int64_t>(labels, batch_size);

        if (!flat_index_) {
            flat_index_ = std::make_unique<raftIvfFlatIndex_t>(
                raft::neighbors::ivf_flat::build<DataType, std::int64_t>(res_, *build_params_flat_,
                                                                         vector_data_gpu));
        }
        raft::neighbors::ivf_flat::extend<DataType, std::int64_t>(res_, vector_data_gpu, std::make_optional(label_gpu),
                                          flat_index_.get());
        return batch_size;
    }

    void search(const void *vector_data, void *neighbors, void *distances, size_t batch_size,
                size_t k) override {
        auto vector_data_gpu = raft::make_device_matrix_view<const DataType, std::int64_t>(
            (const DataType *)vector_data, batch_size, this->dim);
        auto neighbors_gpu = raft::make_device_matrix_view<std::int64_t, std::int64_t>(
            (std::int64_t *)neighbors, batch_size, k);
        auto distances_gpu =
            raft::make_device_matrix_view<float, std::int64_t>((float *)distances, batch_size, k);

        raft::neighbors::ivf_flat::search<DataType, std::int64_t>(res_, *search_params_flat_, *flat_index_, vector_data_gpu,
                                          neighbors_gpu, distances_gpu);
    }
    VecSimIndexInfo info() const override {
        VecSimIndexInfo info;
        info.raftIvfFlatInfo.dim = this->dim;
        info.raftIvfFlatInfo.type = this->vecType;
        info.raftIvfFlatInfo.metric = this->metric;
        info.raftIvfFlatInfo.indexSize = this->indexSize();
        info.raftIvfFlatInfo.nLists = build_params_flat_->n_lists;
        info.algo = VecSimAlgo_RaftIVFFlat;
        return info;
    }
    size_t nLists() override { return build_params_flat_->n_lists; }
    size_t indexSize() const override { return flat_index_.get() == nullptr ? 0 : flat_index_->size(); }

protected:
    std::unique_ptr<raftIvfFlatIndex_t> flat_index_;
    // Build params are kept as class member because the build step on Raft side happens on
    // the first vector insertion
    std::unique_ptr<raft::neighbors::ivf_flat::index_params> build_params_flat_;
    std::unique_ptr<raft::neighbors::ivf_flat::search_params> search_params_flat_;
};
