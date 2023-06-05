#include <optional>
#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/memory/vecsim_malloc.h"

#include <raft/core/device_resources.hpp>

#pragma once

raft::distance::DistanceType inline GetRaftDistanceType(VecSimMetric vsm) {
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

class RaftIVFIndex : public VecSimIndexAbstract<float> {
public:
    using DataType = float;
    using DistType = float;

    template <typename T>
    RaftIVFIndex(const T *params, std::shared_ptr<VecSimAllocator> allocator)
        : VecSimIndexAbstract<float>(allocator, params->dim, params->type, params->metric,
                                     params->blockSize, params->multi),
          counts_(0) {}
    int addVector(const void *vector_data, labelType label,
                  bool overwrite_allowed = true) override {
        return this->addVectorBatch(vector_data, &label, 1, overwrite_allowed);
    }
    int addVectorBatch(const void *vector_data, labelType *label, size_t batch_size,
                       bool overwrite_allowed = true);
    virtual int addVectorBatchGpuBuffer(const void *vector_data, std::int64_t *label,
                                        size_t batch_size, bool overwrite_allowed = true) = 0;
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
        // Not implemented
        return 0;
    }
    void increaseCapacity() override {
        // Not implemented
    }
    inline size_t indexLabelCount() const override {
        return counts_; // TODO: Return unique counts
    }
    VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                     VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius,
                                      VecSimQueryParams *queryParams) override {
        assert(!"RangeQuery not implemented");
    }
    VecSimInfoIterator *infoIterator() const override { assert(!"infoIterator not implemented"); }
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) const override {
        assert(!"newBatchIterator not implemented");
    }
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override {
        assert(!"preferAdHocSearch not implemented");
    }

    auto get_resources() { return res_; }

    virtual size_t nLists() = 0;

protected:
    virtual void search(const void *vector_data, void *neighbors, void *distances,
                        size_t batch_size, size_t k) = 0;
    raft::device_resources res_;
    idType counts_;
};

int RaftIVFIndex::addVectorBatch(const void *vector_data, labelType *labels, size_t batch_size,
                                 bool overwrite_allowed) {
    auto vector_data_gpu =
        raft::make_device_matrix<DataType, std::int64_t>(res_, batch_size, this->dim);
    auto label_original = std::vector<labelType>(labels, labels + batch_size);
    auto label_converted = std::vector<std::int64_t>(label_original.begin(), label_original.end());
    auto label_gpu = raft::make_device_vector<std::int64_t, std::int64_t>(res_, batch_size);

    RAFT_CUDA_TRY(cudaMemcpyAsync(vector_data_gpu.data_handle(), (DataType *)vector_data,
                                  this->dim * batch_size * sizeof(float), cudaMemcpyDefault,
                                  res_.get_stream()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(label_gpu.data_handle(), label_converted.data(),
                                  batch_size * sizeof(std::int64_t), cudaMemcpyDefault,
                                  res_.get_stream()));

    this->addVectorBatchGpuBuffer(vector_data_gpu.data_handle(), label_gpu.data_handle(),
                                  batch_size, overwrite_allowed);
    res_.sync_stream();

    this->counts_ += batch_size;
    return batch_size;
}

// Search for the k closest vectors to a given vector in the index.
VecSimQueryResult_List RaftIVFIndex::topKQuery(const void *queryBlob, size_t k,
                                               VecSimQueryParams *queryParams) {
    VecSimQueryResult_List result_list = {0};
    if (this->counts_ == 0) {
        result_list.results = array_new<VecSimQueryResult>(0);
        return result_list;
    }
    if (k > this->counts_)
        k = this->counts_; // Safeguard K
    auto vector_data_gpu =
        raft::make_device_matrix<DataType, std::uint32_t>(res_, queryParams->batchSize, this->dim);
    auto neighbors_gpu =
        raft::make_device_matrix<std::int64_t, std::uint32_t>(res_, queryParams->batchSize, k);
    auto distances_gpu =
        raft::make_device_matrix<float, std::uint32_t>(res_, queryParams->batchSize, k);
    RAFT_CUDA_TRY(cudaMemcpyAsync(vector_data_gpu.data_handle(), (const DataType *)queryBlob,
                                  this->dim * queryParams->batchSize * sizeof(float),
                                  cudaMemcpyDefault, res_.get_stream()));

    this->search(vector_data_gpu.data_handle(), neighbors_gpu.data_handle(),
                 distances_gpu.data_handle(), 1, k);

    auto result_size = queryParams->batchSize * k;
    auto *neighbors = array_new_len<std::int64_t>(result_size, result_size);
    auto *distances = array_new_len<float>(result_size, result_size);
    RAFT_CUDA_TRY(cudaMemcpyAsync(neighbors, neighbors_gpu.data_handle(),
                                  result_size * sizeof(std::int64_t), cudaMemcpyDefault,
                                  res_.get_stream()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(distances, distances_gpu.data_handle(),
                                  result_size * sizeof(float), cudaMemcpyDefault,
                                  res_.get_stream()));
    res_.sync_stream();

    result_list.results = array_new_len<VecSimQueryResult>(k, k);
    for (size_t i = 0; i < k; ++i) {
        VecSimQueryResult_SetId(result_list.results[i], neighbors[i]);
        VecSimQueryResult_SetScore(result_list.results[i], distances[i]);
    }
    array_free(neighbors);
    array_free(distances);
    return result_list;
};
