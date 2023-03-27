#include <optional>
#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/memory/vecsim_malloc.h"

#include "raft/core/device_resources.hpp"
#include "raft/neighbors/ivf_flat.cuh"
#include "raft/neighbors/ivf_flat_types.hpp"

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

template <typename DataType, typename DistType>
class RaftFlatIndex : public VecSimIndexAbstract<DistType> {
public:
    RaftFlatIndex(const RaftFlatParams *params, std::shared_ptr<VecSimAllocator> allocator);
    int addVector(const void *vector_data, labelType label, bool overwrite_allowed = true) override;
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
        info.algo = VecSimAlgo_RaftFlat;
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
    std::unique_ptr<raft::neighbors::ivf_flat::index<DataType, std::int64_t>> flat_index_;
    idType counts_;
};

template <typename DataType, typename DistType>
RaftFlatIndex<DataType, DistType>::RaftFlatIndex(const RaftFlatParams *params,
                                               std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndexAbstract<DistType>(allocator, params->dim, params->type, params->metric, params->blockSize, false),
      counts_(0)
{
    auto build_params = raft::neighbors::ivf_flat::index_params{};
    build_params.metric = GetRaftDistanceType(params->metric);
    build_params.n_lists = params->n_lists;
    build_params.kmeans_n_iters = params->kmeans_n_iters;
    build_params.kmeans_trainset_fraction = params->kmeans_trainset_fraction;
    build_params.adaptive_centers = params->adaptive_centers;
    build_params.add_data_on_build = false;
    flat_index_ = std::make_unique<raft::neighbors::ivf_flat::index<DataType, int64_t>>(raft::neighbors::ivf_flat::build<DataType, std::int64_t>(res_, build_params,
                                                                           nullptr, 0, this->dim));
}

template <typename DataType, typename DistType>
int RaftFlatIndex<DataType, DistType>::addVector(const void *vector_data, labelType label, bool overwrite_allowed)
{
    assert(label < static_cast<labelType>(std::numeric_limits<std::int64_t>::max()));
    if (!flat_index_) {
        return -1;
    }
    auto vector_data_gpu = raft::make_device_matrix<DataType, std::int64_t>(res_, 1, this->dim);
    auto label_converted = static_cast<std::int64_t>(label);
    auto label_gpu = raft::make_device_vector<std::int64_t, std::int64_t>(res_, 1);
    raft::copy(vector_data_gpu.data_handle(), (DataType*)vector_data, this->dim, res_.get_stream());
    raft::copy(label_gpu.data_handle(), &label_converted, 1, res_.get_stream());

    raft::neighbors::ivf_flat::extend(res_, flat_index_.get(), raft::make_const_mdspan(vector_data_gpu.view()),
        std::make_optional(raft::make_const_mdspan(label_gpu.view())));
    // TODO: Verify that label exists already?
    // TODO normalizeVector for cosine?
    this->counts_ += 1;
    return 1;
}

// Search for the k closest vectors to a given vector in the index.
template <typename DataType, typename DistType>
VecSimQueryResult_List RaftFlatIndex<DataType, DistType>::topKQuery(
    const void *queryBlob, size_t k, VecSimQueryParams *queryParams)
{
    VecSimQueryResult_List result_list = {0};
    if (!flat_index_) {
        result_list.results = array_new<VecSimQueryResult>(0);
        return result_list;
    }
    raft::neighbors::ivf_flat::search_params raft_search_params{};
    auto vector_data_gpu = raft::make_device_matrix<DataType, std::int64_t>(res_, queryParams->batchSize, this->dim);
    auto neighbors_gpu = raft::make_device_matrix<std::int64_t, std::int64_t>(res_, queryParams->batchSize, k);
    auto distances_gpu = raft::make_device_matrix<float, std::int64_t>(res_, queryParams->batchSize, k);
    raft::copy(vector_data_gpu.data_handle(), (const DataType*)queryBlob, this->dim * queryParams->batchSize, res_.get_stream());
    raft::neighbors::ivf_flat::search(res_, *flat_index_, raft::make_const_mdspan(vector_data_gpu.view()), neighbors_gpu.view(), distances_gpu.view(), raft_search_params, k);

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
