/* TODO: change the copyright here */

/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

/* TODO clean the includes */
#pragma once
#include "VecSim/query_results.h"

#include "svs/core/distance.h"
#include "svs/core/query_result.h"
#include "svs/core/logging.h"
#include "spdlog/sinks/callback_sink.h"

namespace details {
// VecSim->SVS data type conversion
template <typename T>
struct vecsim_dtype;

template <>
struct vecsim_dtype<float> {
    using type = float;
};

template <>
struct vecsim_dtype<svs::Float16> {
    using type = vecsim_types::float16;
};

template <typename T>
using vecsim_dt = typename vecsim_dtype<T>::type;

// SVS->VecSim distance conversion
template <typename DistType>
float toVecSimDistance(float);

template <>
inline float toVecSimDistance<svs::distance::DistanceL2>(float v) {
    return v;
}

template <>
inline float toVecSimDistance<svs::distance::DistanceIP>(float v) {
    return 1.f - v;
}

template <>
inline float toVecSimDistance<svs::distance::DistanceCosineSimilarity>(float v) {
    return 1.f - v;
}

template <typename Ea, typename Eb, size_t Da, size_t Db>
float computeVecSimDistance(svs::distance::DistanceL2 dist, std::span<Ea, Da> a,
                            std::span<Eb, Db> b) {
    return toVecSimDistance<svs::distance::DistanceL2>(svs::distance::compute(dist, a, b));
}

template <typename Ea, typename Eb, size_t Da, size_t Db>
float computeVecSimDistance(svs::distance::DistanceIP dist, std::span<Ea, Da> a,
                            std::span<Eb, Db> b) {
    return toVecSimDistance<svs::distance::DistanceIP>(svs::distance::compute(dist, a, b));
}

template <typename Ea, typename Eb, size_t Da, size_t Db>
float computeVecSimDistance(svs::distance::DistanceCosineSimilarity /*dist*/, std::span<Ea, Da> a,
                            std::span<Eb, Db> b) {
    // VecSim uses IP for Cosine distance
    return computeVecSimDistance(svs::distance::DistanceIP{}, a, b);
}

// VecSim allocator wrapper for SVS containers
template <typename T>
struct SVSAllocator {
private:
    std::shared_ptr<VecSimAllocator> allocator_;

public:
    // Type Aliases
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

    // Constructor
    SVSAllocator(std::shared_ptr<VecSimAllocator> vs_allocator)
        : allocator_{std::move(vs_allocator)} {}

    // Construct from another value type allocator.

    // Allocation and Deallocation.
    [[nodiscard]] constexpr value_type *allocate(std::size_t n) {
        return static_cast<value_type *>(allocator_->allocate_aligned(n * sizeof(T), alignof(T)));
    }

    constexpr void deallocate(value_type *ptr, size_t count) noexcept {
        allocator_->deallocate(ptr, count * sizeof(T));
    }

    // Intercept zero-argument construction to do default initialization.
    // template <typename U>
    // void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
    //     ::new (static_cast<void*>(p)) U;
    // }
};

inline svs::index::vamana::VamanaSearchParameters
joinSearchParams(svs::index::vamana::VamanaSearchParameters &&sp,
                 const VecSimQueryParams *queryParams) {
    if (queryParams == nullptr) {
        return std::move(sp);
    }

    auto &rt_params = queryParams->svsRuntimeParams;
    if (rt_params.windowSize > 0) {
        sp.buffer_config({rt_params.windowSize});
    }
    switch (rt_params.searchHistory) {
    case VecSimOption_ENABLE:
        sp.search_buffer_visited_set(true);
        break;
    case VecSimOption_DISABLE:
        sp.search_buffer_visited_set(false);
        break;
    default:
        break;
    }
    return std::move(sp);
}

inline svs::lib::PowerOfTwo SVSBlockSize(size_t bs, size_t elem_size) {
    auto svs_bs = svs::lib::prevpow2(bs * elem_size);
    // block size should not be less than element size
    while (svs_bs.value() < elem_size) {
        svs_bs = svs::lib::PowerOfTwo{svs_bs.raw() + 1};
    }
    return svs_bs;
}

} // namespace details

template <typename DataType, size_t QuantBits, size_t ResidualBits, class Enable = void>
struct SVSStorageTraits {
    using allocator_type = details::SVSAllocator<DataType>;
    using blocked_type = svs::data::Blocked<allocator_type>;
    using index_storage_type = svs::data::BlockedData<DataType, svs::Dynamic, allocator_type>;

    template <svs::data::ImmutableMemoryDataset Dataset>
    static index_storage_type create_storage(const Dataset &data, size_t block_size,
                                             std::shared_ptr<VecSimAllocator> allocator) {
        const auto dim = data.dimensions();
        const auto size = data.size();
        auto svs_bs = details::SVSBlockSize(block_size, element_size(dim));
        allocator_type data_allocator{std::move(allocator)};
        blocked_type blocked_alloc{{svs_bs}, data_allocator};
        index_storage_type init_data{size, dim, blocked_alloc};
        for (const auto &i : data.eachindex()) {
            init_data.set_datum(i, data.get_datum(i));
        }
        return init_data;
    }

    static constexpr size_t element_size(size_t dims, size_t /*alignment*/ = 0) {
        return dims * sizeof(DataType);
    }
};

template <typename Idx>
struct SVSGraphBuilder {
    using allocator_type = details::SVSAllocator<Idx>;
    using blocked_type = svs::data::Blocked<allocator_type>;
    using graph_data_type = svs::data::BlockedData<Idx, svs::Dynamic, allocator_type>;
    using graph_type = svs::graphs::SimpleGraphBase<Idx, graph_data_type>;

    template <class Data, class DistType, class Pool>
    static graph_type build_graph(const svs::index::vamana::VamanaBuildParameters &parameters,
                                  const Data &data, DistType distance, Pool &threadpool,
                                  Idx entry_point, size_t block_size,
                                  std::shared_ptr<VecSimAllocator> allocator) {
        auto svs_bs =
            details::SVSBlockSize(block_size, (parameters.graph_max_degree + 1) * sizeof(Idx));
        // Perform graph construction.
        allocator_type data_allocator{std::move(allocator)};
        blocked_type blocked_alloc{{svs_bs}, data_allocator};
        auto graph = graph_type{data.size(), parameters.graph_max_degree, blocked_alloc};
        auto prefetch_parameters =
            svs::index::vamana::extensions::estimate_prefetch_parameters(data);
        auto builder = svs::index::vamana::VamanaBuilder(
            graph, data, std::move(distance), parameters, threadpool, prefetch_parameters);

        builder.construct(1.0f, entry_point);
        builder.construct(parameters.alpha, entry_point);
        return graph;
    }

    static constexpr size_t element_size(size_t graph_max_degree, size_t alignment = 0) {
        return sizeof(Idx) * (graph_max_degree + 1);
    }
};
