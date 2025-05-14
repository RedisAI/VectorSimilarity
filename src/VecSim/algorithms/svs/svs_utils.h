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

#include <cpuid.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

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

    // Support allocator type rebinding in LeanVec
    template <typename U>
    friend class SVSAllocator;

    template <typename U>
    SVSAllocator(SVSAllocator<U> other) : allocator_{other.allocator_} {}
};

// Join default SVS search parameters with VecSim query runtime parameters
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

// clang-format off
inline bool check_cpuid() {
    uint32_t eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);
    std::string vendor_id = std::string((const char*)&ebx, 4) +
                            std::string((const char*)&edx, 4) +
                            std::string((const char*)&ecx, 4);
    return (vendor_id == "GenuineIntel");
}
// clang-format on

// Check if the SVS implementation supports Quantization mode
// @param quant_bits requested SVS quantization mode
// @return pair<fallbackMode, bool>
// @note even if VecSimSvsQuantBits is a simple enum value,
//       in theory, it can be a complex type with a combination of modes:
//       - primary bits, secondary/residual bits, dimesionality reduction, etc.
//       which can be incompatible to each-other.
inline std::pair<VecSimSvsQuantBits, bool> isSVSQuantBitsSupported(VecSimSvsQuantBits quant_bits) {
    // If HAVE_SVS_LVQ is not defined, we don't support any quantization mode
    // else we check if the CPU supports SVS LVQ
    bool supported = quant_bits == VecSimSvsQuant_NONE
#if HAVE_SVS_LVQ
                     || check_cpuid() // Check if the CPU supports SVS LVQ
#endif
        ;

    // If the quantization mode is not supported, we fallback to non-quantized mode
    // - this is temporary solution until we have a basic quantization mode in SVS
    // TODO: use basic SVS quantization as a fallback for unsupported modes
    auto fallBack = supported ? quant_bits : VecSimSvsQuant_NONE;

    // And always return true, as far as fallback mode is always supported
    // Upon further decision changes, some cases should treated as not-supported
    // So we will need return false.
    return std::make_pair(fallBack, true);
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

    template <svs::data::ImmutableMemoryDataset Dataset, svs::threads::ThreadPool Pool>
    static index_storage_type create_storage(const Dataset &data, size_t block_size, Pool &pool,
                                             std::shared_ptr<VecSimAllocator> allocator) {
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
    static constexpr size_t element_size(size_t dims, size_t /*alignment*/ = 0) {
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
                                  std::shared_ptr<VecSimAllocator> allocator) {
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
        builder.construct(1.0f, entry_point);
        builder.construct(parameters.alpha, entry_point);
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
