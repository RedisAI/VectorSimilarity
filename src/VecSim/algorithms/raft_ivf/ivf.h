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
#include "VecSim/memory/vecsim_malloc.h"

#include <raft/core/device_resources_manager.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/ivf_flat_types.hpp>
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

template <typename DataType, typename DistType = DataType>
struct RaftIvfIndex : public VecSimIndexAbstract<DistType> {
    using data_type = DataType;
    using dist_type = DistType;

private:
    // Allow either IVF-flat or IVFPQ parameters
    using build_params_t = std::variant<raft::neighbors::ivf_flat::index_params,
                                        raft::neighbors::ivf_pq::index_params>;
    using search_params_t = std::variant<raft::neighbors::ivf_flat::search_params,
                                         raft::neighbors::ivf_pq::search_params>;
    using internal_idx_t = std::int64_t;
    using index_flat_t = raft::neighbors::ivf_flat::index<data_type, internal_idx_t>;
    using index_pq_t = raft::neighbors::ivf_pq::index<internal_idx_t>;
    using ann_index_t = std::variant<index_flat_t, index_pq_t>;

public:
    RaftIvfIndex(const RaftIvfParams *raftIvfParams, const AbstractIndexInitParams &commonParams)
        : VecSimIndexAbstract<dist_type>{commonParams},
          res_{raft::device_resources_manager::get_device_resources()},
          build_params_{raftIvfParams->usePQ ? build_params_t{std::in_place_index<1>}
                                             : build_params_t{std::in_place_index<0>}},
          search_params_{raftIvfParams->usePQ ? search_params_t{std::in_place_index<1>}
                                              : search_params_t{std::in_place_index<0>}},
          index_{std::nullopt} {
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
    }
    int addVector(const void *vector_data, labelType label, void *auxiliaryCtx = nullptr) override {
        return addVectorBatch(vector_data, &label, 1, auxiliaryCtx);
    }
    int addVectorBatchAsync(const void *vector_data, labelType *label, size_t batch_size,
                            void *auxiliaryCtx = nullptr);
    int addVectorBatch(const void *vector_data, labelType *label, size_t batch_size,
                       void *auxiliaryCtx = nullptr) {
        auto result = addVectorBatchAsync(vector_data, label, batch_size, auxiliaryCtx);
        // Ensure that above operation has executed on device before
        // returning from this function on host
        res_.sync_stream();
        return result;
    }
    int deleteVector(labelType label) override {
        assert(!"deleteVector not implemented");
        return 0;
    }
    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        assert(!"getDistanceFrom not implemented");
        return INVALID_SCORE;
    }
    size_t indexCapacity() const override {
        assert(!"indexCapacity not implemented");
        return 0;
    }
    // void increaseCapacity() override { assert(!"increaseCapacity not implemented"); }
    inline size_t indexLabelCount() const override {
        return this->indexSize(); // TODO: Return unique counts
    }
    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override;

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

    auto &get_resources() const { return res_; }

    auto nLists() const {
        return std::visit([](auto &&params) { return params.n_lists; }, build_params_);
    }

    size_t indexSize() const override {
        auto result = size_t{};
        if (index_) {
            result = std::visit([](auto &&index) { return index.size(); }, *index_);
        }
        return result;
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
            const auto build_params_pq =
                std::get<raft::neighbors::ivf_pq::index_params>(build_params_);
            info.raftIvfInfo.pqBits = build_params_pq.pq_bits;
            info.raftIvfInfo.pqDim = build_params_pq.pq_dim;
        }
        return info;
    }

    inline void setNProbes(uint32_t n_probes) {
        std::visit([n_probes](auto &&params) { params.n_probes = n_probes; }, search_params_);
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
