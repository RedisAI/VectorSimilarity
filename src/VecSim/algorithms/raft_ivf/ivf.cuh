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
#include "VecSim/query_result_definitions.h" // VecSimQueryResult VecSimQueryReply
#include "VecSim/algorithms/raft_ivf/ivf_interface.h"  // RaftIvfInterface
#include "VecSim/memory/vecsim_malloc.h"

#include <raft/core/bitset.cuh>
#include <raft/core/device_resources_manager.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/init.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/sample_filter.cuh>

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

inline auto constexpr GetRaftCodebookKind(RaftIVFPQCodebookKind vss_codebook) {
    auto result = raft::neighbors::ivf_pq::codebook_gen{};
    switch (vss_codebook) {
    case RaftIVFPQCodebookKind_PerCluster:
        result = raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
        break;
    case RaftIVFPQCodebookKind_PerSubspace:
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

void init_raft_resources() {
    auto static init_flag = std::once_flag{};
    std::call_once(init_flag, []() {
        raft::device_resources_manager::set_streams_per_device(8); // TODO: use env variable
        raft::device_resources_manager::set_stream_pools_per_device(8);
        // Create a memory pool with half of the available GPU memory.
        raft::device_resources_manager::set_mem_pool();
    });
}

template <typename DataType, typename DistType = DataType>
struct RaftIvfIndex : public RaftIvfInterface<DataType, DistType> {
    using data_type = DataType;
    using dist_type = DistType;

private:
    // Allow either IVF-flat or IVFPQ parameters
    using build_params_t = std::variant<raft::neighbors::ivf_flat::index_params,
                                        raft::neighbors::ivf_pq::index_params>;
    using search_params_t = std::variant<raft::neighbors::ivf_flat::search_params,
                                         raft::neighbors::ivf_pq::search_params>;
    using internal_idx_t = std::uint64_t;
    using index_flat_t = raft::neighbors::ivf_flat::index<data_type, internal_idx_t>;
    using index_pq_t = raft::neighbors::ivf_pq::index<internal_idx_t>;
    using ann_index_t = std::variant<index_flat_t, index_pq_t>;

public:
    RaftIvfIndex(const RaftIvfParams *raftIvfParams, const AbstractIndexInitParams &commonParams)
        : RaftIvfInterface<dist_type>{commonParams},
          build_params_{raftIvfParams->usePQ ? build_params_t{std::in_place_index<1>}
                                             : build_params_t{std::in_place_index<0>}},
          search_params_{raftIvfParams->usePQ ? search_params_t{std::in_place_index<1>}
                                              : search_params_t{std::in_place_index<0>}},
          index_{std::nullopt}, deleted_indices_{std::nullopt}, numDeleted_{0},
          idToLabelLookup_{this->allocator}, labelToIdLookup_{this->allocator} {
        std::visit(
            [raftIvfParams](auto &&inner) {
                inner.metric = GetRaftDistanceType(raftIvfParams->metric);
                inner.n_lists = raftIvfParams->nLists;
                inner.kmeans_n_iters = raftIvfParams->kmeans_nIters;
                inner.add_data_on_build = false;
                inner.kmeans_trainset_fraction = raftIvfParams->kmeans_trainsetFraction;
                inner.conservative_memory_allocation = raftIvfParams->conservativeMemoryAllocation;
                if constexpr (std::is_same_v<decltype(inner),
                                             raft::neighbors::ivf_flat::index_params>) {
                    inner.adaptive_centers = raftIvfParams->adaptiveCenters;
                } else if constexpr (std::is_same_v<decltype(inner),
                                                    raft::neighbors::ivf_pq::index_params>) {
                    inner.pq_bits = raftIvfParams->pqBits;
                    inner.pq_dim = raftIvfParams->pqDim;
                    inner.codebook_kind = GetRaftCodebookKind(raftIvfParams->codebookKind);
                }
            },
            build_params_);
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
            search_params_);

        init_raft_resources();
    }
    int addVector(const void *vector_data, labelType label, void *auxiliaryCtx = nullptr) override {
        return addVectorBatch(vector_data, &label, 1, auxiliaryCtx);
    }
    int addVectorBatch(const void *vector_data, labelType *label, size_t batch_size,
                       void *auxiliaryCtx = nullptr) override {
        const auto &res = raft::device_resources_manager::get_device_resources();
        // Allocate memory on device to hold vectors to be added
        auto vector_data_gpu =
            raft::make_device_matrix<data_type, internal_idx_t>(res, batch_size, this->dim);

        // Copy vector data to previously allocated device buffer
        raft::copy(vector_data_gpu.data_handle(), static_cast<float const *>(vector_data),
                   this->dim * batch_size, res.get_stream());

        // Create GPU vector to hold ids
        internal_idx_t first_id = this->indexSize();
        internal_idx_t last_id = first_id + batch_size;
        auto ids = raft::make_device_vector<internal_idx_t, internal_idx_t>(res, batch_size);
        raft::linalg::range(ids.data_handle(), first_id, last_id, res.get_stream());

        // Build index if it does not exist, and extend it with the new vectors and their ids
        if (std::holds_alternative<raft::neighbors::ivf_flat::index_params>(build_params_)) {
            if (!index_) {
                index_ = raft::neighbors::ivf_flat::build(
                    res, std::get<raft::neighbors::ivf_flat::index_params>(build_params_),
                    raft::make_const_mdspan(vector_data_gpu.view()));
                deleted_indices_ = {raft::core::bitset<uint32_t, internal_idx_t>(res, 0)};
            }
            raft::neighbors::ivf_flat::extend(res, raft::make_const_mdspan(vector_data_gpu.view()),
                                              {raft::make_const_mdspan(ids.view())},
                                              &std::get<index_flat_t>(*index_));
        } else {
            if (!index_) {
                index_ = raft::neighbors::ivf_pq::build(
                    res, std::get<raft::neighbors::ivf_pq::index_params>(build_params_),
                    raft::make_const_mdspan(vector_data_gpu.view()));
                deleted_indices_ = {raft::core::bitset<uint32_t, internal_idx_t>(res, 0)};
            }
            raft::neighbors::ivf_pq::extend(res, raft::make_const_mdspan(vector_data_gpu.view()),
                                            {raft::make_const_mdspan(ids.view())},
                                            &std::get<index_pq_t>(*index_));
        }

        // Add labels to internal idToLabelLookup_ mapping
        this->idToLabelLookup_.insert(this->idToLabelLookup_.end(), label, label + batch_size);
        for (auto i = 0; i < batch_size; ++i) {
            this->labelToIdLookup_[label[i]] = first_id + i;
        }

        // Update the size of the deleted indices bitset
        deleted_indices_->resize(res, deleted_indices_->size() + batch_size);

        // Ensure that above operation has executed on device before
        // returning from this function on host
        res.sync_stream();
        return batch_size;
    }
    int deleteVector(labelType label) override {
        // Check if label exists in internal labelToIdLookup_ mapping
        auto search = labelToIdLookup_.find(label);
        if (search == labelToIdLookup_.end()) {
            return 0;
        }
        const auto &res = raft::device_resources_manager::get_device_resources();
        // Create GPU vector to hold ids to mark as deleted
        internal_idx_t id = search->second;
        auto id_gpu = raft::make_device_vector<internal_idx_t, internal_idx_t>(res, 1);
        raft::copy(id_gpu.data_handle(), &id, 1, res.get_stream());
        // Mark the id as deleted
        deleted_indices_->set(res, raft::make_const_mdspan(id_gpu.view()), false);

        // Remove label from internal labelToIdLookup_ mapping
        labelToIdLookup_.erase(search);
        // Ensure that above operation has executed on device before
        // returning from this function on host
        res.sync_stream();
        this->numDeleted_ += 1;
        return 1;
    }
    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        assert(!"getDistanceFrom not implemented");
        return INVALID_SCORE;
    }
    size_t indexCapacity() const override {
        assert(!"indexCapacity not implemented");
        return 0;
    }
    inline vecsim_stl::set<labelType> getLabelsSet() const override {
        vecsim_stl::set<labelType> result(this->allocator);
        for (auto const &pair : labelToIdLookup_) {
            result.insert(pair.first);
        }
        return result;
    }
    // void increaseCapacity() override { assert(!"increaseCapacity not implemented"); }
    inline size_t indexLabelCount() const override { return this->labelToIdLookup_.size(); }
    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override {
        const auto &res = raft::device_resources_manager::get_device_resources();
        auto result_list = new VecSimQueryReply(this->allocator);
        auto nVectors = this->indexSize();
        if (nVectors == 0 || k == 0 || !index_.has_value()) {
            return result_list;
        }
        // Ensure we are not trying to retrieve more vectors than exist in the
        // index
        k = std::min(k, nVectors);
        // Allocate memory on device for search vector
        auto vector_data_gpu =
            raft::make_device_matrix<data_type, internal_idx_t>(res, 1, this->dim);
        // Allocate memory on device for neighbor and distance results
        auto neighbors_gpu = raft::make_device_matrix<internal_idx_t, internal_idx_t>(res, 1, k);
        auto distances_gpu = raft::make_device_matrix<dist_type, internal_idx_t>(res, 1, k);
        // Copy query vector to device
        raft::copy(vector_data_gpu.data_handle(), static_cast<const data_type *>(queryBlob),
                   this->dim, res.get_stream());
        auto bitset_filter = raft::neighbors::filtering::bitset_filter(deleted_indices_->view());

        // Perform correct search based on index type
        if (std::holds_alternative<index_flat_t>(*index_)) {
            raft::neighbors::ivf_flat::search_with_filtering<data_type, internal_idx_t,
                                                             decltype(bitset_filter)>(
                res, std::get<raft::neighbors::ivf_flat::search_params>(search_params_),
                std::get<index_flat_t>(*index_), raft::make_const_mdspan(vector_data_gpu.view()),
                neighbors_gpu.view(), distances_gpu.view(), bitset_filter);
        } else {
            raft::neighbors::ivf_pq::search_with_filtering<data_type, internal_idx_t,
                                                           decltype(bitset_filter)>(
                res, std::get<raft::neighbors::ivf_pq::search_params>(search_params_),
                std::get<index_pq_t>(*index_), raft::make_const_mdspan(vector_data_gpu.view()),
                neighbors_gpu.view(), distances_gpu.view(), bitset_filter);
        }

        // Allocate host buffers to hold returned results
        auto neighbors = vecsim_stl::vector<internal_idx_t>(k, this->allocator);
        auto distances = vecsim_stl::vector<dist_type>(k, this->allocator);
        // Copy data back from device to host
        raft::copy(neighbors.data(), neighbors_gpu.data_handle(), k, res.get_stream());
        raft::copy(distances.data(), distances_gpu.data_handle(), k, res.get_stream());

        // Ensure search is complete and data have been copied back before
        // building query result objects on host
        res.sync_stream();

        result_list->results.resize(k);
        for (auto i = 0; i < k; ++i) {
            result_list->results[i].id = idToLabelLookup_[neighbors[i]];
            result_list->results[i].score = distances[i];
        }

        return result_list;
    }

    virtual VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                         VecSimQueryParams *queryParams) const override {
        assert(!"RangeQuery not implemented");
        return nullptr;
    }
    VecSimInfoIterator *infoIterator() const override {
        assert(!"infoIterator not implemented");
        return nullptr;
    }
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) const override {
        assert(!"newBatchIterator not implemented");
        return nullptr;
    }
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override {
        assert(!"preferAdHocSearch not implemented");
        return false;
    }

    virtual uint32_t nLists() const override {
        return std::visit([](auto &&params) { return params.n_lists; }, build_params_);
    }

    size_t indexSize() const override {
        auto result = size_t{};
        if (index_) {
            result = std::visit([](auto &&index) { return index.size(); }, *index_);
        }
        return result - this->numDeleted_;
    }
    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        if (std::holds_alternative<raft::neighbors::ivf_flat::index_params>(build_params_)) {
            info.algo = VecSimAlgo_RAFT_IVFFLAT;
        } else {
            info.algo = VecSimAlgo_RAFT_IVFPQ;
        }
        info.isTiered = false;
        return info;
    }
    VecSimIndexInfo info() const override {
        VecSimIndexInfo info;
        info.commonInfo = this->getCommonInfo();
        info.raftIvfInfo.nLists = nLists();
        if (std::holds_alternative<raft::neighbors::ivf_pq::index_params>(build_params_)) {
            info.commonInfo.basicInfo.algo = VecSimAlgo_RAFT_IVFPQ;
            const auto build_params_pq =
                std::get<raft::neighbors::ivf_pq::index_params>(build_params_);
            info.raftIvfInfo.pqBits = build_params_pq.pq_bits;
            info.raftIvfInfo.pqDim = build_params_pq.pq_dim;
        } else {
            info.commonInfo.basicInfo.algo = VecSimAlgo_RAFT_IVFFLAT;
        }
        return info;
    }

    virtual inline void setNProbes(uint32_t n_probes) override {
        std::visit([n_probes](auto &&params) { params.n_probes = n_probes; }, search_params_);
    }

private:
    // Store build params to allow for index build on first batch
    // insertion
    build_params_t build_params_;
    // Store search params to use with each search after initializing in
    // constructor
    search_params_t search_params_;
    // Use a std::optional to allow building of the index on first batch
    // insertion
    std::optional<ann_index_t> index_;
    // Bitset used for deleteVectors and search filtering.
    std::optional<raft::core::bitset<std::uint32_t, internal_idx_t>> deleted_indices_;
    internal_idx_t numDeleted_ = 0;

    vecsim_stl::vector<labelType> idToLabelLookup_;
    vecsim_stl::unordered_map<labelType, internal_idx_t> labelToIdLookup_;
};
