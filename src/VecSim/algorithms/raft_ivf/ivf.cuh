#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <cuda_runtime.h>
#include <library_types.h>
#include "VecSim/vec_sim.h"
// For VecSimMetric, RaftIvfParams, labelType
#include "VecSim/vec_sim_common.h"
// For VecSimIndexAbstract
#include "VecSim/vec_sim_index.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/memory/vecsim_malloc.h"

#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>


inline auto constexpr GetRaftDistanceType(VecSimMetric vsm) {
    auto result = raft::distance::DistanceType{};
    switch (vsm) {
    case VecSimMetric_L2:
        result = raft::distance::DistanceType::L2Expanded;
        break;
    case VecSimMetric_IP:
        result = raft::distance::DistanceType::InnerProduct;
        break;
    default:
        throw raft::exception("Metric not supported");
    }
    return result;
}

inline auto constexpr GetRaftCodebookKind(IVFPQCodebookKind vss_codebook) {
    auto result = raft::neighbors::ivf_pq::codebook_gen{};
    switch (vss_codebook) {
    case IVFPQCodebookKind_PerCluster:
        result = raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
        break;
    case IVFPQCodebookKind_PerSubspace:
        result = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
        break;
    default:
        throw raft::exception("Unexpected IVFPQ codebook kind");
    }
    return result;
}

inline auto constexpr GetCudaType(CudaType vss_type) {
    auto result = cudaDataType_t{};
    switch (vss_type) {
    case CUDAType_R_32F:
        result = CUDA_R_32F;
        break;
    case CUDAType_R_16F:
        result = CUDA_R_16F;
        break;
    case CUDAType_R_8U:
        result = CUDA_R_8U;
        break;
    default:
        throw raft::exception("Unexpected CUDA type");
    }
    return result;
}

template <typename DataType, typename DistType = DataType>
struct IVFIndex : public VecSimIndexAbstract<DistType> {
    using data_type = DataType;
    using dist_type = DistType;

private:
    // Allow either IVF-flat or IVFPQ parameters
    using build_params_t = std::variant<raft::neighbors::ivf_flat::index_params,
                                        raft::neighbors::ivf_pq::index_params>;
    using search_params_t = std::variant<raft::neighbors::ivf_flat::search_params,
                                         raft::neighbors::ivf_pq::search_params>;
    using internal_idx_t = std::uint32_t;
    using ann_index_t = std::variant<raft::neighbors::ivf_flat::index<data_type, internal_idx_t>,
                                     raft::neighbors::ivf_pq::index<internal_idx_t>>;

public:
    IVFIndex(const RaftIvfParams *raftIvfParams, const AbstractIndexInitParams &commonParams)
        : VecSimIndexAbstract<dist_type>{commonParams},
          res_{raft::resource_manager::get_device_resources()}, build_params_{[raftIvfParams]() {
              auto result = raftIvfParams->usePQ ? build_params_t{std::in_place_index<1>}
                                                 : build_params_t{std::in_place_index<0>};
              std::visit(
                  [raftIvfParams](auto &&inner) {
                      inner.metric = GetRaftDistanceType(raftIvfParams->metric);
                      inner.n_lists = raftIvfParams->nLists;
                      inner.kmeans_n_iters = raftIvfParams->kmeans_nIters;
                      inner.kmeans_trainset_fraction = raftIvfParams->kmeans_trainsetFraction;
                      inner.conservative_memory_allocation =
                          raftIvfParams->conservativeMemoryAllocation;
                      if constexpr (std::is_same_v<decltype(inner),
                                                   raft::neighbors::ivf_pq::index_params>) {
                          inner.pq_bits = raftIvfParams->pqBits;
                          inner.pq_dim = raftIvfParams->pqDim;
                          inner.codebook_kind = GetRaftCodebookKind(raftIvfParams->codebookKind);
                      } else {
                          inner.adaptive_centers = raftIvfParams->adaptiveCenters;
                      }
                  },
                  result);
              return result;
          }()},
          search_params_{[raftIvfParams]() {
              auto result = raftIvfParams->usePQ ? search_params_t{std::in_place_index<1>}
                                                 : search_params_t{std::in_place_index<0>};
              std::visit(
                  [raftIvfParams](auto &&inner) {
                      inner.n_probes = raftIvfParams->nProbes;
                      if constexpr (std::is_same_v<decltype(inner),
                                                   raft::neighbors::ivf_pq::search_params>) {
                          inner.lut_dtype = GetCudaType(raftIvfParams->lutType);
                          inner.internal_distance_dtype =
                              GetCudaType(raftIvfParams->internalDistanceType);
                          inner.preferred_shmem_carvout = raftIvfParams->preferredShmemCarveout;
                      }
                  },
                  result);
              return result;
          }()},
          index_{std::nullopt} {}
    auto addVector(const void *vector_data, labelType label,
                   bool overwrite_allowed = true) override {
        return addVectorBatch(vector_data, &label, 1, overwrite_allowed);
    }
    auto addVectorBatchAsync(const void *vector_data, labelType *label, size_t batch_size,
                             bool overwrite_allowed = true) {
        // Allocate memory on device to hold vectors to be added
        auto vector_data_gpu =
            raft::make_device_matrix<data_type, internal_idx_t>(res_, batch_size, this->dim);
        // Allocate memory on device to hold vector labels
        auto label_gpu = raft::make_device_vector<labelType>(res_, batch_size);

        // Copy vector data to previously allocated device buffer
        raft::copy(vector_data_gpu.data_handle(), static_cast<DataType const *>(vector_data),
                   this->dim * batch_size, res_.get_stream());
        // Copy label data to previously allocated device buffer
        raft::copy(label_gpu.data_handle(), label, batch_size, res_.get_stream());

        if (std::holds_alternative<raft::neighbors::ivf_flat::index_params>(build_params_)) {
            if (!index_) {
                index_ = raft::neighbors::ivf_flat::build(
                    res_, std::get<raft::neighbors::ivf_flat::index_params>(build_params_),
                    vector_data_gpu.view());
            }
            raft::neighbors::ivf_flat::extend(res_, vector_data_gpu.view(), label_gpu, *index_);
        } else {
            if (!index_) {
                index_ = raft::neighbors::ivf_pq::build(
                    res_, std::get<raft::neighbors::ivf_pq::index_params>(build_params_),
                    vector_data_gpu.view());
            }
            raft::neighbors::ivf_pq::extend(res_, vector_data_gpu.view(), label_gpu, *index_);
        }

        return batch_size;
    }
    auto addVectorBatch(const void *vector_data, labelType *label, size_t batch_size,
                        bool overwrite_allowed = true) {
        auto result = addVectorBatchAsync(vector_data, label, batch_size, overwrite_allowed);
        // Ensure that above operation has executed on device before
        // returning from this function on host
        res_.sync_stream();
        return result;
    }
    auto deleteVector(labelType label) override {
        assert(!"deleteVector not implemented");
        return 0;
    }
    double getDistanceFrom(labelType label, const void *vector_data) const override {
        assert(!"getDistanceFrom not implemented");
        return INVALID_SCORE;
    }
    size_t indexCapacity() const override {
        assert(!"indexCapacity not implemented");
        return 0;
    }
    void increaseCapacity() override { assert(!"increaseCapacity not implemented"); }
    inline auto indexLabelCount() const override {
        return this->indexSize(); // TODO: Return unique counts
    }
    auto topKQuery(const void *queryBlob, size_t k, VecSimQueryParams *queryParams) override {
        auto result_list = VecSimQueryResult_List{0};
        auto nVectors = this->indexSize();
        if (nVectors == 0) {
            result_list.results = array_new<VecSimQueryResult>(0);
        } else {
            // Ensure we are not trying to retrieve more vectors than exist in the
            // index
            k = std::min(k, nVectors);
            // Allocate memory on device for search vector
            auto vector_data_gpu = raft::make_device_matrix<data_type>(res_, 1, this->dim);
            // Allocate memory on device for neighbor results
            auto neighbors_gpu = raft::make_device_vector<labelType>(res_, k);
            // Allocate memory on device for distance results
            auto distances_gpu = raft::make_device_vector<dist_type>(res_, k);
            // Copy query vector to device
            raft::copy(vector_data_gpu.data_handle(), static_cast<data_type>(queryBlob), this->dim,
                       res_.get_stream());

            // Perform correct search based on index type
            if (std::holds_alternative<raft::neighbors::ivf_flat::index>(index_)) {
                raft::neighbors::ivf_flat::search<data_type, internal_idx_t>(
                    res_, std::get<raft::neighbors::ivf_flat::search_params>(search_params_),
                    std::get<raft::neighbors::ivf_flat::index>(*index_), vector_data_gpu.view(),
                    neighbors_gpu.view(), distances_gpu.view())
            } else {
                raft::neighbors::ivf_pq::search<data_type, internal_idx_t>(
                    res_, std::get<raft::neighbors::ivf_flat::search_params>(search_params_),
                    std::get<raft::neighbors::ivf_flat::index>(*index_), vector_data_gpu.view(),
                    neighbors_gpu.view(), distances_gpu.view())
            }

            // Allocate host buffers to hold returned results
            auto neighbors =
                std::unique_ptr(array_new_len<labelType>(k, k), &array_free<labelType>);
            auto distances =
                std::unique_ptr(array_new_len<dist_type>(k, k), &array_free<dist_type>);
            // Copy data back from device to host
            raft::copy(neighbors.get(), neighbors_gpu.data_handle(), this->dim, res_.get_stream());
            raft::copy(distances.get(), distances_gpu.data_handle(), this->dim, res_.get_stream());

            result_list.results = array_new_len<VecSimQueryResult>(k, k);

            // Ensure search is complete and data have been copied back before
            // building query result objects on host
            res_.sync_stream();
            for (size_t i = 0; i < k; ++i) {
                VecSimQueryResult_SetId(result_list.results[i], neighbors[i]);
                VecSimQueryResult_SetScore(result_list.results[i], distances[i]);
            }
        }
        return result_list;
    }

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

    auto &get_resources() const { return res_; }

    auto nLists() {
        return std::visit([](auto &&params) { return params.n_list; }, build_params_);
    }

    auto indexSize() {
        auto result = size_t{};
        if (index_) {
            result = std::visit([](auto &&index) { return index.size(); }, *index_);
        }
        return result;
    }

private:
    // An object used to manage common device resources that may be
    // expensive to build but frequently accessed
    raft::device_resources res_;
    // Store build params to allow for index build on first batch
    // insertion
    build_params_t build_params_;
    // Store search params to use with each search after initializing in
    // constructor
    search_params_t search_params_;
    // Use a std::optional to allow building of the index on first batch
    // insertion
    std::optional<ann_index_t> index_;
};
