#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_pq.cuh>
#include "ivf.h"

template <>
int RaftIvfIndex<float, float>::addVectorBatchAsync(const void *vector_data, labelType *label,
                                                          size_t batch_size, void *auxiliaryCtx) {
    // Convert labels to internal data type
    auto label_original = std::vector<labelType>(label, label + batch_size);
    auto label_converted =
        std::vector<internal_idx_t>(label_original.begin(), label_original.end());
    // Allocate memory on device to hold vectors to be added
    auto vector_data_gpu =
        raft::make_device_matrix<data_type, internal_idx_t>(res_, batch_size, this->dim);
    // Allocate memory on device to hold vector labels
    auto label_gpu = raft::make_device_vector<internal_idx_t, internal_idx_t>(res_, batch_size);

    // Copy vector data to previously allocated device buffer
    raft::copy(vector_data_gpu.data_handle(), static_cast<float const *>(vector_data),
               this->dim * batch_size, res_.get_stream());
    // Copy label data to previously allocated device buffer
    raft::copy(label_gpu.data_handle(), label_converted.data(), batch_size, res_.get_stream());

    if (std::holds_alternative<raft::neighbors::ivf_flat::index_params>(build_params_)) {
        if (!index_) {
            index_ = raft::neighbors::ivf_flat::build(
                res_, std::get<raft::neighbors::ivf_flat::index_params>(build_params_),
                raft::make_const_mdspan(vector_data_gpu.view()));
        }
        raft::neighbors::ivf_flat::extend(
            res_, raft::make_const_mdspan(vector_data_gpu.view()),
            std::make_optional(raft::make_const_mdspan(label_gpu.view())),
            &std::get<index_flat_t>(*index_));
    } else {
        if (!index_) {
            index_ = raft::neighbors::ivf_pq::build(
                res_, std::get<raft::neighbors::ivf_pq::index_params>(build_params_),
                raft::make_const_mdspan(vector_data_gpu.view()));
        }
        raft::neighbors::ivf_pq::extend(
            res_, raft::make_const_mdspan(vector_data_gpu.view()),
            std::make_optional(raft::make_const_mdspan(label_gpu.view())),
            &std::get<index_pq_t>(*index_));
    }

    return batch_size;
}

template <>
VecSimQueryReply *
RaftIvfIndex<float, float>::topKQuery(const void *queryBlob, size_t k,
                                            VecSimQueryParams *queryParams) const {
    auto result_list = new VecSimQueryReply(this->allocator);
    auto nVectors = this->indexSize();
    if (nVectors == 0 || k == 0 || !index_.has_value()) {
        return result_list;
    }
    // Ensure we are not trying to retrieve more vectors than exist in the
    // index
    k = std::min(k, nVectors);
    // Allocate memory on device for search vector
    auto vector_data_gpu = raft::make_device_matrix<data_type, internal_idx_t>(res_, 1, this->dim);
    // Allocate memory on device for neighbor and distance results
    auto neighbors_gpu = raft::make_device_matrix<internal_idx_t, internal_idx_t>(res_, 1, k);
    auto distances_gpu = raft::make_device_matrix<dist_type, internal_idx_t>(res_, 1, k);
    // Copy query vector to device
    raft::copy(vector_data_gpu.data_handle(), static_cast<const data_type *>(queryBlob), this->dim,
               res_.get_stream());

    // Perform correct search based on index type
    if (std::holds_alternative<index_flat_t>(*index_)) {
        raft::neighbors::ivf_flat::search<data_type, internal_idx_t>(
            res_, std::get<raft::neighbors::ivf_flat::search_params>(search_params_),
            std::get<index_flat_t>(*index_), raft::make_const_mdspan(vector_data_gpu.view()),
            neighbors_gpu.view(), distances_gpu.view());
        // TODO ADD STREAM MANAGER
    } else {
        raft::neighbors::ivf_pq::search<data_type, internal_idx_t>(
            res_, std::get<raft::neighbors::ivf_pq::search_params>(search_params_),
            std::get<index_pq_t>(*index_), raft::make_const_mdspan(vector_data_gpu.view()),
            neighbors_gpu.view(), distances_gpu.view());
        // TODO ADD STREAM MANAGER
    }

    // Allocate host buffers to hold returned results
    auto neighbors = vecsim_stl::vector<internal_idx_t>(k, this->allocator);
    auto distances = vecsim_stl::vector<dist_type>(k, this->allocator);
    // Copy data back from device to host
    raft::copy(neighbors.data(), neighbors_gpu.data_handle(), k, res_.get_stream());
    raft::copy(distances.data(), distances_gpu.data_handle(), k, res_.get_stream());

    // Ensure search is complete and data have been copied back before
    // building query result objects on host
    res_.sync_stream();

    result_list->results.resize(k);
    for (auto i = 0; i < k; ++i) {
        result_list->results[i].id = labelType{neighbors[i]};
        result_list->results[i].score = distances[i];
    }

    return result_list;
}
