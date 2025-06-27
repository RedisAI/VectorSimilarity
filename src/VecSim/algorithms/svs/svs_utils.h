/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once
#include "VecSim/query_results.h"
#include "VecSim/types/float16.h"

#include "svs/core/distance.h"
#include "svs/lib/float16.h"
#include "svs/index/vamana/dynamic_index.h"

#if HAVE_SVS_LVQ
#include "svs/cpuid.h"
#endif

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

// Maximum training threshold for SVS index, used to limit the size of training data
constexpr size_t SVS_MAX_TRAINING_THRESHOLD = 100 * DEFAULT_BLOCK_SIZE; // 100 * 1024 vectors
// Default batch update threshold for SVS index.
constexpr size_t SVS_DEFAULT_UPDATE_THRESHOLD = 1 * DEFAULT_BLOCK_SIZE; // 1 * 1024 vectors
// Default wait time for the update job in microseconds
constexpr size_t SVS_DEFAULT_UPDATE_JOB_WAIT_TIME = 100; // 0.1 ms

namespace svs_details {
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
double toVecSimDistance(float);

template <>
inline double toVecSimDistance<svs::distance::DistanceL2>(float v) {
    return static_cast<double>(v);
}

template <>
inline double toVecSimDistance<svs::distance::DistanceIP>(float v) {
    return 1.0 - static_cast<double>(v);
}

template <>
inline double toVecSimDistance<svs::distance::DistanceCosineSimilarity>(float v) {
    return 1.0 - static_cast<double>(v);
}

// VecSim allocator wrapper for SVS containers
template <typename T>
using SVSAllocator = VecsimSTLAllocator<T>;

template <typename T, typename U>
static T getOrDefault(T v, U def) {
    return v != T{} ? v : static_cast<T>(def);
}

inline svs::index::vamana::VamanaBuildParameters
makeVamanaBuildParameters(const SVSParams &params) {
    // clang-format off
    // evaluate optimal default parameters; current assumption:
    // * alpha (1.2 or 0.95) depends on metric: L2: > 1.0, IP, Cosine: < 1.0
    //      In the Vamana algorithm implementation in SVS, the choice of alpha value
    //      depends on the type of similarity measure used. For L2, which minimizes distance,
    //      an alpha value greater than 1 is needed, typically around 1.2.
    //      For Inner Product and Cosine, which maximize similarity or distance,
    //      the alpha value should be less than 1, usually 0.9 or 0.95 works.
    // * construction_window_size (200): similar to HNSW_EF_CONSTRUCTION
    // * graph_max_degree (32): similar to HNSW_M * 2
    // * max_candidate_pool_size (600): =~ construction_window_size * 3
    // * prune_to (28): < graph_max_degree, optimal = graph_max_degree - 4
    //      The prune_to parameter is a performance feature designed to enhance build time
    //      by setting a small difference between this value and the maximum graph degree.
    //      This acts as a threshold for how much pruning can reduce the number of neighbors.
    //      Typically, a small gap of 4 or 8 is sufficient to improve build time
    //      without compromising the quality of the graph.
    // * use_search_history (true): now: is enabled if not disabled explicitly
    //                              future: default value based on other index parameters
    const auto construction_window_size = getOrDefault(params.construction_window_size, SVS_VAMANA_DEFAULT_CONSTRUCTION_WINDOW_SIZE);
    const auto graph_max_degree = getOrDefault(params.graph_max_degree, SVS_VAMANA_DEFAULT_GRAPH_MAX_DEGREE);

    // More info about VamanaBuildParameters can be found there:
    // https://intel.github.io/ScalableVectorSearch/python/vamana.html#svs.VamanaBuildParameters
    return svs::index::vamana::VamanaBuildParameters{
        getOrDefault(params.alpha, (params.metric == VecSimMetric_L2 ?
            SVS_VAMANA_DEFAULT_ALPHA_L2 : SVS_VAMANA_DEFAULT_ALPHA_IP)),
        graph_max_degree,
        construction_window_size,
        getOrDefault(params.max_candidate_pool_size, construction_window_size * 3),
        getOrDefault(params.prune_to, graph_max_degree - 4),
        params.use_search_history == VecSimOption_AUTO ? SVS_VAMANA_DEFAULT_USE_SEARCH_HISTORY :
            params.use_search_history == VecSimOption_ENABLE,
    };
    // clang-format on
}

// Join default SVS search parameters with VecSim query runtime parameters
inline svs::index::vamana::VamanaSearchParameters
joinSearchParams(svs::index::vamana::VamanaSearchParameters &&sp,
                 const VecSimQueryParams *queryParams, bool is_two_level_lvq) {
    if (queryParams == nullptr) {
        return std::move(sp);
    }

    auto &rt_params = queryParams->svsRuntimeParams;
    size_t sws = sp.buffer_config_.get_search_window_size();
    size_t sbc = sp.buffer_config_.get_total_capacity();
    if (rt_params.windowSize > 0) {
        sws = rt_params.windowSize;
        if(rt_params.bufferCapacity > 0) {
            sbc = rt_params.bufferCapacity;
        }
        else {
          if(!is_two_level_lvq) {
              // set windowSize as default
              sbc = rt_params.windowSize;
          }
          else {
              // set windowSize * 1.5 as default for Two-level LVQ
              sbc = static_cast<size_t>(rt_params.windowSize * 1.5);
          }
        }
    }
    sp.buffer_config({sws, sbc});
    switch (rt_params.searchHistory) {
    case VecSimOption_ENABLE:
        sp.search_buffer_visited_set(true);
        break;
    case VecSimOption_DISABLE:
        sp.search_buffer_visited_set(false);
        break;
    default: // AUTO mode, let the algorithm decide
        break;
    }
    return std::move(sp);
}

// @brief Block size for SVS storage required to be a power-of-two
// @param bs VecSim block size
// @param elem_size SVS storage element size
// @return block size in type of SVS `PowerOfTwo`
inline svs::lib::PowerOfTwo SVSBlockSize(size_t bs, size_t elem_size) {
    auto svs_bs = svs::lib::prevpow2(bs * elem_size);
    // block size should not be less than element size
    while (svs_bs.value() < elem_size) {
        svs_bs = svs::lib::PowerOfTwo{svs_bs.raw() + 1};
    }
    return svs_bs;
}

// Check if the SVS implementation supports Quantization mode
// @param quant_bits requested SVS quantization mode
// @return pair<fallbackMode, bool>
// @note even if VecSimSvsQuantBits is a simple enum value,
//       in theory, it can be a complex type with a combination of modes:
//       - primary bits, secondary/residual bits, dimesionality reduction, etc.
//       which can be incompatible to each-other.
inline std::pair<VecSimSvsQuantBits, bool> isSVSQuantBitsSupported(VecSimSvsQuantBits quant_bits) {
    switch (quant_bits) {
    // non-quantized mode and scalar quantization are always supported
    case VecSimSvsQuant_NONE:
    case VecSimSvsQuant_Scalar:
        return std::make_pair(quant_bits, true);
    default:
        // fallback to no quantization if we have no LVQ support in code
        // or if the CPU doesn't support it
#if HAVE_SVS_LVQ
        return svs::detail::intel_enabled() ? std::make_pair(quant_bits, true)
                                            : std::make_pair(VecSimSvsQuant_Scalar, true);
#else
        return std::make_pair(VecSimSvsQuant_Scalar, true);
#endif
    }
    assert(false && "Should never reach here");
    // unreachable code, but to avoid compiler warning
    return std::make_pair(VecSimSvsQuant_NONE, false);
}
} // namespace svs_details

template <typename DataType, size_t QuantBits, size_t ResidualBits, bool IsLeanVec,
          class Enable = void>
struct SVSStorageTraits {
    using allocator_type = svs_details::SVSAllocator<DataType>;
    // In SVS, the default allocator is designed for static indices,
    // where the size of the data or graph is known in advance,
    // allowing all structures to be allocated at once. In contrast,
    // the Blocked allocator supports dynamic allocations,
    // enabling memory to be allocated in blocks as needed when the index size grows.
    using blocked_type = svs::data::Blocked<allocator_type>; // Used in creating storage
    // svs::Dynamic means runtime dimensionality in opposite to compile-time dimensionality
    using index_storage_type = svs::data::BlockedData<DataType, svs::Dynamic, allocator_type>;

    static constexpr bool is_compressed() { return false; }

    static constexpr VecSimSvsQuantBits get_compression_mode() {
        return VecSimSvsQuant_NONE; // No compression for this storage
    }

    template <svs::data::ImmutableMemoryDataset Dataset, svs::threads::ThreadPool Pool>
    static index_storage_type create_storage(const Dataset &data, size_t block_size, Pool &pool,
                                             std::shared_ptr<VecSimAllocator> allocator, size_t /* leanvec_dim */) {
        const auto dim = data.dimensions();
        const auto size = data.size();
        // SVS storage element size and block size can be differ than VecSim
        auto svs_bs = svs_details::SVSBlockSize(block_size, element_size(dim));
        // Allocate initial SVS storage for index
        allocator_type data_allocator{std::move(allocator)};
        blocked_type blocked_alloc{{svs_bs}, data_allocator};
        index_storage_type init_data{size, dim, blocked_alloc};
        // Copy data to allocated storage
        svs::threads::parallel_for(pool, svs::threads::StaticPartition(data.eachindex()),
                                   [&](auto is, auto SVS_UNUSED(tid)) {
                                       for (auto i : is) {
                                           init_data.set_datum(i, data.get_datum(i));
                                       }
                                   });
        return init_data;
    }

    // SVS storage element size can be differ than VecSim DataSize
    static constexpr size_t element_size(size_t dims, size_t /*alignment*/ = 0, size_t /*leanvec_dim*/ = 0) {
        return dims * sizeof(DataType);
    }

    static size_t storage_capacity(const index_storage_type &storage) { return storage.capacity(); }

    template <typename Distance, typename E, size_t N>
    static float compute_distance_by_id(const index_storage_type &storage, const Distance &distance,
                                        size_t id, std::span<E, N> query) {
        auto dist_f = svs::index::vamana::extensions::single_search_setup(storage, distance);

        // SVS distance function may require to fix/pre-process one of arguments
        svs::distance::maybe_fix_argument(dist_f, query);

        // Get the datum from the storage using the storage ID
        auto datum = storage.get_datum(id);
        return svs::distance::compute(dist_f, query, datum);
    }
};

template <typename SVSIdType>
struct SVSGraphBuilder {
    using allocator_type = svs_details::SVSAllocator<SVSIdType>;
    using blocked_type = svs::data::Blocked<allocator_type>;
    using graph_data_type = svs::data::BlockedData<SVSIdType, svs::Dynamic, allocator_type>;
    using graph_type = svs::graphs::SimpleGraphBase<SVSIdType, graph_data_type>;

    // Build SVS Graph using custom allocator
    // The logic has been taken from one of `MutableVamanaIndex` constructors
    // See:
    // https://github.com/intel/ScalableVectorSearch/blob/main/include/svs/index/vamana/dynamic_index.h#L189
    template <class Data, class DistType, class Pool>
    static graph_type build_graph(const svs::index::vamana::VamanaBuildParameters &parameters,
                                  const Data &data, DistType distance, Pool &threadpool,
                                  SVSIdType entry_point, size_t block_size,
                                  std::shared_ptr<VecSimAllocator> allocator,
                                  const svs::logging::logger_ptr &logger) {
        auto svs_bs =
            svs_details::SVSBlockSize(block_size, element_size(parameters.graph_max_degree));
        // Perform graph construction.
        allocator_type data_allocator{std::move(allocator)};
        blocked_type blocked_alloc{{svs_bs}, data_allocator};
        auto graph = graph_type{data.size(), parameters.graph_max_degree, blocked_alloc};
        // SVS incorporates an advanced software prefetching scheme with two parameters: step and
        // lookahead. These parameters determine how far ahead to prefetch data vectors
        // and how many items to prefetch at a time. We have set default values for these parameters
        // based on the data types, which we found to perform better through heuristic analysis.
        auto prefetch_parameters =
            svs::index::vamana::extensions::estimate_prefetch_parameters(data);
        auto builder = svs::index::vamana::VamanaBuilder(
            graph, data, std::move(distance), parameters, threadpool, prefetch_parameters);

        // Specific to the Vamana algorithm:
        // It builds in two rounds, one with alpha=1 and the second time with the user/config
        // provided alpha value.
        builder.construct(1.0f, entry_point, svs::logging::Level::Trace, logger);
        builder.construct(parameters.alpha, entry_point, svs::logging::Level::Trace, logger);
        return graph;
    }

    // SVS Vamana graph element size
    static constexpr size_t element_size(size_t graph_max_degree, size_t alignment = 0) {
        // For every Vamana graph node SVS allocates a record with current node ID and
        // graph_max_degree neighbors
        return sizeof(SVSIdType) * (graph_max_degree + 1);
    }
};

// Custom thread pool for SVS index
// Based on svs::threads::NativeThreadPoolBase with changes:
// * Number of threads is fixed on construction time
// * Pool is resizable in bounds of pre-allocated threads
class VecSimSVSThreadPoolImpl {
public:
    // Allocate `num_threads - 1` threads since the main thread participates in the work
    // as well.
    explicit VecSimSVSThreadPoolImpl(size_t num_threads = 1)
        : size_{num_threads}, threads_(num_threads - 1) {}

    size_t capacity() const { return threads_.size() + 1; }
    size_t size() const { return size_; }

    // Support resize - do not modify threads container just limit the size
    void resize(size_t new_size) {
        std::lock_guard lock{use_mutex_};
        size_ = std::clamp(new_size, size_t{1}, threads_.size() + 1);
    }

    void parallel_for(std::function<void(size_t)> f, size_t n) {
        if (n > size_) {
            throw svs::threads::ThreadingException("Number of tasks exceeds the thread pool size");
        }
        if (n == 0) {
            return;
        } else if (n == 1) {
            // Run on the main function.
            try {
                f(0);
            } catch (const std::exception &error) {
                manage_exception_during_run(error.what());
            }
            return;
        } else {
            std::lock_guard lock{use_mutex_};
            for (size_t i = 0; i < n - 1; ++i) {
                threads_[i].assign({&f, i + 1});
            }
            // Run on the main function.
            try {
                f(0);
            } catch (const std::exception &error) {
                manage_exception_during_run(error.what());
            }

            // Wait until all threads are done.
            // If any thread fails, then we're throwing.
            for (size_t i = 0; i < size_ - 1; ++i) {
                auto &thread = threads_[i];
                thread.wait();
                if (!thread.is_okay()) {
                    manage_exception_during_run();
                }
            }
        }
    }

    void manage_exception_during_run(const std::string &thread_0_message = {}) {
        auto message = std::string{};
        auto inserter = std::back_inserter(message);
        if (!thread_0_message.empty()) {
            fmt::format_to(inserter, "Thread 0: {}\n", thread_0_message);
        }

        // Manage all other exceptions thrown, restarting crashed threads.
        for (size_t i = 0; i < size_ - 1; ++i) {
            auto &thread = threads_[i];
            thread.wait();
            if (!thread.is_okay()) {
                try {
                    thread.unsafe_get_exception();
                } catch (const std::exception &error) {
                    fmt::format_to(inserter, "Thread {}: {}\n", i + 1, error.what());
                }
                // Restart the thread.
                threads_[i].shutdown();
                threads_[i] = svs::threads::Thread{};
            }
        }
        throw svs::threads::ThreadingException{std::move(message)};
    }

private:
    std::mutex use_mutex_;
    size_t size_;
    std::vector<svs::threads::Thread> threads_;
};

// Copy-movable wrapper for VecSimSVSThreadPoolImpl
class VecSimSVSThreadPool {
private:
    std::shared_ptr<VecSimSVSThreadPoolImpl> pool_;

public:
    explicit VecSimSVSThreadPool(size_t num_threads = 1)
        : pool_{std::make_shared<VecSimSVSThreadPoolImpl>(num_threads)} {}

    size_t capacity() const { return pool_->capacity(); }
    size_t size() const { return pool_->size(); }

    void parallel_for(std::function<void(size_t)> f, size_t n) {
        pool_->parallel_for(std::move(f), n);
    }

    void resize(size_t new_size) { pool_->resize(new_size); }
};
