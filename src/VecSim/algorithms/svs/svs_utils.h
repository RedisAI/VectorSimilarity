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
#include "VecSim/vec_sim_interface.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/types/float16.h"

#include "svs/core/distance.h"
#include "svs/lib/float16.h"
#include "svs/index/vamana/dynamic_index.h"

#if HAVE_SVS_LVQ
#include "svs/cpuid.h"
#endif

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

// Maximum training threshold for SVS index, used to limit the size of training data
constexpr size_t SVS_MAX_TRAINING_THRESHOLD = 100 * DEFAULT_BLOCK_SIZE; // 100 * 1024 vectors
// Default wait time for the update job in microseconds
constexpr size_t SVS_DEFAULT_UPDATE_JOB_WAIT_TIME = 5000; // 5 ms

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

    // buffer capacity is changed only if window size is changed
    if (rt_params.windowSize > 0) {
        sws = rt_params.windowSize;
        if (rt_params.bufferCapacity > 0) {
            // case 1: change both window size and buffer capacity
            sbc = rt_params.bufferCapacity;
        } else {
            // case 2: change only window size
            // In this case, set buffer capacity based on window size
            if (!is_two_level_lvq) {
                // set buffer capacity to windowSize
                sbc = rt_params.windowSize;
            } else {
                // set buffer capacity to windowSize * 1.5 for Two-level LVQ
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

    static blocked_type make_blocked_allocator(size_t block_size, size_t dim,
                                               std::shared_ptr<VecSimAllocator> allocator) {
        // SVS storage element size and block size can be differ than VecSim
        auto svs_bs = svs_details::SVSBlockSize(block_size, element_size(dim));
        allocator_type data_allocator{std::move(allocator)};
        return blocked_type{{svs_bs}, data_allocator};
    }

    template <svs::data::ImmutableMemoryDataset Dataset, svs::threads::ThreadPool Pool>
    static index_storage_type create_storage(const Dataset &data, size_t block_size, Pool &pool,
                                             std::shared_ptr<VecSimAllocator> allocator,
                                             size_t /* leanvec_dim */) {
        const auto dim = data.dimensions();
        const auto size = data.size();
        // Allocate initial SVS storage for index
        auto blocked_alloc = make_blocked_allocator(block_size, dim, std::move(allocator));
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

    static index_storage_type load(const svs::lib::LoadTable &table, size_t block_size, size_t dim,
                                   std::shared_ptr<VecSimAllocator> allocator) {
        auto blocked_alloc = make_blocked_allocator(block_size, dim, std::move(allocator));
        // Load the data from disk
        return index_storage_type::load(table, blocked_alloc);
    }

    static index_storage_type load(const std::string &path, size_t block_size, size_t dim,
                                   std::shared_ptr<VecSimAllocator> allocator) {
        auto blocked_alloc = make_blocked_allocator(block_size, dim, std::move(allocator));
        // Load the data from disk
        return index_storage_type::load(path, blocked_alloc);
    }

    // SVS storage element size can be differ than VecSim DataSize
    static constexpr size_t element_size(size_t dims, size_t /*alignment*/ = 0,
                                         size_t /*leanvec_dim*/ = 0) {
        return dims * sizeof(DataType);
    }

    static size_t storage_capacity(const index_storage_type &storage) { return storage.capacity(); }
};

template <typename SVSIdType>
struct SVSGraphBuilder {
    using allocator_type = svs_details::SVSAllocator<SVSIdType>;
    using blocked_type = svs::data::Blocked<allocator_type>;
    using graph_data_type = svs::data::BlockedData<SVSIdType, svs::Dynamic, allocator_type>;
    using graph_type = svs::graphs::SimpleGraph<SVSIdType, blocked_type>;

    static blocked_type make_blocked_allocator(size_t block_size, size_t graph_max_degree,
                                               std::shared_ptr<VecSimAllocator> allocator) {
        // SVS block size is a power of two, so we can use it directly
        auto svs_bs = svs_details::SVSBlockSize(block_size, element_size(graph_max_degree));
        allocator_type data_allocator{std::move(allocator)};
        return blocked_type{{svs_bs}, data_allocator};
    }

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
        // Perform graph construction.
        auto blocked_alloc =
            make_blocked_allocator(block_size, parameters.graph_max_degree, std::move(allocator));
        auto graph = graph_type{data.size(), parameters.graph_max_degree, blocked_alloc};
        // SVS incorporates an advanced software prefetching scheme with two parameters: step and
        // lookahead. These parameters determine how far ahead to prefetch data vectors
        // and how many items to prefetch at a time. We have set default values for these parameters
        // based on the data types, which we found to perform better through heuristic analysis.
        auto prefetch_parameters =
            svs::index::vamana::extensions::estimate_prefetch_parameters(data);
        auto builder = svs::index::vamana::VamanaBuilder(
            graph, data, std::move(distance), parameters, threadpool, prefetch_parameters, logger);

        // Specific to the Vamana algorithm:
        // It builds in two rounds, one with alpha=1 and the second time with the user/config
        // provided alpha value.
        builder.construct(1.0f, entry_point, svs::logging::Level::Trace, logger);
        builder.construct(parameters.alpha, entry_point, svs::logging::Level::Trace, logger);
        return graph;
    }

    static graph_type load(const svs::lib::LoadTable &table, size_t block_size,
                           const svs::index::vamana::VamanaBuildParameters &parameters,
                           std::shared_ptr<VecSimAllocator> allocator) {
        auto blocked_alloc =
            make_blocked_allocator(block_size, parameters.graph_max_degree, std::move(allocator));
        // Load the graph from disk
        return graph_type::load(table, blocked_alloc);
    }

    static graph_type load(const std::string &path, size_t block_size,
                           const svs::index::vamana::VamanaBuildParameters &parameters,
                           std::shared_ptr<VecSimAllocator> allocator) {
        auto blocked_alloc =
            make_blocked_allocator(block_size, parameters.graph_max_degree, std::move(allocator));
        // Load the graph from disk
        return graph_type::load(path, blocked_alloc);
    }

    // SVS Vamana graph element size
    static constexpr size_t element_size(size_t graph_max_degree, size_t alignment = 0) {
        // For every Vamana graph node SVS allocates a record with current node ID and
        // graph_max_degree neighbors
        return sizeof(SVSIdType) * (graph_max_degree + 1);
    }
};

// A slot in the shared SVS thread pool. Wraps an SVS Thread with an occupancy flag
// used by the rental mechanism. Slots are individually heap-allocated and are never
// destroyed (the pool only grows physically; shrink is logical), so renters can hold
// raw pointers safely.
//
// The OS thread is spawned lazily on first rent, not at slot creation. Growing the
// pool (VecSim_UpdateThreadPoolSize on `CONFIG SET WORKERS N`) runs on the Redis main
// thread; spawning N threads there is O(N) serialized thread-creates + boot handshakes
// (each svs::threads::Thread ctor blocks until its worker reaches the Spinning state),
// and freshly booted threads burn their idle-spin budget with no work, oversubscribing
// the CPU so the handshakes stretch superlinearly — seconds for large N (MOD-16610).
// Slot allocation is trivial, so resize returns immediately and each thread's spawn
// cost is paid by the renter on the first job that actually uses the slot — at which
// point the new thread's first spin window immediately catches its assigned partition.
struct ThreadSlot {
    // Engaged on first ensureThread() call. Slots that are never rented never spawn
    // an OS thread (and are destroyed for free on shrink — no shutdown handshake).
    std::optional<svs::threads::Thread> thread;
    std::atomic<bool> occupied{false};

    ThreadSlot() = default;

    // Non-copyable, non-movable (atomic is not movable)
    ThreadSlot(const ThreadSlot &) = delete;
    ThreadSlot &operator=(const ThreadSlot &) = delete;
    ThreadSlot(ThreadSlot &&) = delete;
    ThreadSlot &operator=(ThreadSlot &&) = delete;

    // Spawn the OS thread if not spawned yet. Must only be called by the renter that
    // won the `occupied` compare-and-swap — that renter has exclusive ownership of the
    // slot until it releases it, so no synchronization on `thread` is needed.
    // May throw std::system_error if the OS refuses a new thread; callers must treat
    // that as a degraded-execution trigger, not a fatal error.
    svs::threads::Thread &ensureThread() {
        if (!thread.has_value()) {
            thread.emplace();
        }
        return *thread;
    }
};

// Shared thread pool for SVS indexes with rental model.
// Based on svs::threads::NativeThreadPoolBase with changes:
// * Pool is physically resizable (creates/destroys OS threads)
// * Threads are rented for the duration of a parallel_for call
// * Multiple callers can rent disjoint subsets of threads concurrently
// * Shrinking while threads are rented is safe (shared_ptr lifecycle)
class VecSimSVSThreadPoolImpl {
    using SlotPtr = std::shared_ptr<ThreadSlot>;

    // RAII guard for threads rented from the shared pool. On destruction, marks all
    // rented slots as unoccupied (lock-free atomic stores). Holds raw pointers —
    // safe because slots are never destroyed: the pool only grows physically (shrink
    // is logical), slots are individually heap-allocated (stable addresses across the
    // vector's reallocation on grow), and the pool singleton is deliberately leaked
    // at exit.
    class RentedThreads {
    public:
        RentedThreads() = default;

        // Move-only
        RentedThreads(RentedThreads &&other) noexcept : slots_(std::move(other.slots_)) {}
        RentedThreads(const RentedThreads &) = delete;
        RentedThreads &operator=(const RentedThreads &) = delete;
        RentedThreads &operator=(RentedThreads &&) = delete;

        ~RentedThreads() { release(); }

        void add(ThreadSlot *slot) { slots_.push_back(slot); }

        size_t count() const { return slots_.size(); }

        // Lazily spawn the OS thread for rented slot `i` (see ThreadSlot::ensureThread).
        // May throw std::system_error on thread-resource exhaustion.
        svs::threads::Thread &ensureThreadAt(size_t i) {
            assert(i < slots_.size());
            return slots_[i]->ensureThread();
        }

        // Destroy slot `i`'s OS thread (joins it); the slot reverts to the unspawned
        // state and will lazily respawn on a future rent. Used to retire crashed threads.
        void resetThreadAt(size_t i) {
            assert(i < slots_.size());
            slots_[i]->thread.reset();
        }

        svs::threads::Thread &operator[](size_t i) {
            assert(i < slots_.size());
            assert(slots_[i]->thread.has_value() && "Rented slot must have a spawned thread");
            return *slots_[i]->thread;
        }

    private:
        void release() {
            for (auto *slot : slots_) {
                slot->occupied.store(false, std::memory_order_release);
            }
            slots_.clear();
        }

        std::vector<ThreadSlot *> slots_;
    };

    // Create a pool with `num_threads` total parallelism (including the calling thread).
    // Allocates `num_threads - 1` worker slots; OS threads are spawned lazily on first
    // rent (see ThreadSlot). num_threads must be >= 1.
    // In write-in-place mode, the pool is created with num_threads == 1 (0 worker threads,
    // only the calling thread participates).
    // Private — use instance() to access the shared singleton.
    explicit VecSimSVSThreadPoolImpl(size_t num_threads = 1)
        : allocator_(VecSimAllocator::newVecsimAllocator()), slots_(allocator_),
          logical_size_(num_threads - 1) {
        assert(num_threads && "VecSimSVSThreadPoolImpl should not be created with 0 threads");
        slots_.reserve(num_threads - 1);
        for (size_t i = 0; i < num_threads - 1; ++i) {
            slots_.push_back(
                std::allocate_shared<ThreadSlot>(VecsimSTLAllocator<ThreadSlot>(allocator_)));
        }
    }

    // Set to true the first time instance() constructs the singleton. Allows other
    // code paths (e.g., global stats reporting) to query whether the pool has been
    // touched without forcing its lazy construction.
    static std::atomic<bool> &initialized_flag() {
        static std::atomic<bool> flag{false};
        return flag;
    }

public:
    // Singleton accessor for the shared SVS thread pool.
    // Always valid — initialized with size 1 (write-in-place mode: 0 worker threads,
    // only the calling thread participates). Resized on VecSim_UpdateThreadPoolSize() calls.
    static std::shared_ptr<VecSimSVSThreadPoolImpl> instance() {
        static auto shared_pool = [] {
            auto p = std::shared_ptr<VecSimSVSThreadPoolImpl>(
                new VecSimSVSThreadPoolImpl(1),
                [](VecSimSVSThreadPoolImpl *) { /* leak at exit */ });
            initialized_flag().store(true, std::memory_order_release);
            return p;
        }();
        return shared_pool;
    }

    // Returns true iff instance() has ever been called (singleton constructed).
    static bool isInitialized() { return initialized_flag().load(std::memory_order_acquire); }

    // Total parallelism: logical worker count + 1 (the calling thread always
    // participates). May be smaller than the physical slot capacity after a shrink —
    // slots above the logical limit are retired lazily (see reclaimExcessThreads).
    size_t size() const {
        std::lock_guard lock{pool_mutex_};
        return logical_size_ + 1;
    }

    // Bytes currently allocated through the pool's internal allocator (the slots vector
    // and the ThreadSlot objects). Does not include allocations performed by SVS itself
    // outside of the pool, nor per-index wrapper state.
    size_t getAllocationSize() const { return allocator_->getAllocationSize(); }

    // Bytes allocated by the shared pool singleton. Returns 0 until the first SVS index
    // attaches, for either reason:
    //   * the singleton was never constructed (no SVS index created and
    //     VecSim_UpdateThreadPoolSize never called), or
    //   * it was constructed to record a requested size (VecSim_UpdateThreadPoolSize at
    //     module init) but no SVS index has attached yet.
    // This keeps process-wide vector memory reported as 0 on deployments that only use
    // non-SVS indexes (or none at all). Safe to call from any context; does not force
    // singleton construction.
    static size_t getSharedAllocationSize() {
        if (!isInitialized()) {
            return 0;
        }
        auto pool = instance();
        std::lock_guard lock{pool->pool_mutex_};
        if (!pool->has_attached_index_) {
            return 0;
        }
        return pool->getAllocationSize();
    }

    // Resize the shared pool. in all cases the requested size is stored in
    // deferred_size_ and applied later if not applied immediately:
    //   * no SVS index attached yet → recorded; applied on first
    //     onIndexAttached() (so no OS threads are spawned in deployments that
    //     never create an SVS index).
    //   * shrink while jobs are in flight (pending_jobs_ > 0) → recorded;
    //     applied when endScheduledJob() drops pending_jobs_ back to 0 (avoids
    //     destroying slots that already-reserved RediSearch workers will rent).
    //   * otherwise (grows; shrinks with no jobs in flight) → applied
    //     immediately via resize_locked().
    // The two deferral cases never overlap — no jobs can be in flight before
    // the first index attaches. Clamps to a minimum of 1.
    void resize(size_t new_size) {
        new_size = std::max(new_size, size_t{1});
        std::lock_guard lock{pool_mutex_};
        if (has_attached_index_) {
            resize_locked(new_size);
        } else {
            deferred_size_ = new_size;
        }
    }

    // Called from the per-index VecSimSVSThreadPool ctor. The first call flips
    // has_attached_index_ and applies any size requested earlier via resize()
    // — this is where OS threads are actually spawned in the lazy path.
    void onIndexAttached() {
        std::lock_guard lock{pool_mutex_};
        if (has_attached_index_)
            return;
        has_attached_index_ = true;
        if (deferred_size_)
            resize_locked(deferred_size_.value());
    }

#ifdef BUILD_TESTS
    // Restore the singleton to its as-constructed state: drop all worker slots
    // (releasing vector capacity so getAllocationSize() returns 0), clear
    // has_attached_index_, deferred_size_, and pending_jobs_. Intended for unit
    // tests that need a clean baseline (the singleton itself is process-wide
    // and cannot be torn down). Caller must ensure no jobs are in flight.
    // Number of slots whose OS thread is currently spawned (parked or in use).
    // Test-only introspection: lets integration tests assert spawn/park/reclaim
    // transitions without counting process-wide OS threads (which is noisy in a
    // binary that runs thousands of unrelated tests).
    size_t spawnedThreadCountForTest() const {
        std::lock_guard lock{pool_mutex_};
        size_t count = 0;
        for (const auto &slot : slots_) {
            count += slot->thread.has_value() ? 1 : 0;
        }
        return count;
    }

    void resetForTest() {
        std::lock_guard lock{pool_mutex_};
        assert(pending_jobs_ == 0 && "resetForTest called with jobs in flight");
        // Swap with a fresh empty vector to release the capacity allocation
        // (clear() destroys elements but retains capacity). Destroying the slots here
        // joins any spawned threads — acceptable in tests.
        vecsim_stl::vector<SlotPtr>(allocator_).swap(slots_);
        logical_size_ = 0;
        deferred_size_.reset();
        has_attached_index_ = false;
    }
#endif

    // Atomically mark a logical job as pending and snapshot the current shared pool size.
    size_t beginScheduledJob() {
        std::lock_guard lock{pool_mutex_};
        ++pending_jobs_;
        return logical_size_ + 1;
    }

    // Decrement the pending-jobs counter. When it reaches zero, apply any deferred
    // resize. Counter-and-size bookkeeping ONLY — never joins threads: this also runs
    // from JobsRegistry teardown of unexecuted jobs (FT.DROPINDEX, main thread).
    // Thread reclamation happens separately, on the worker-executed job completion
    // path (see reclaimExcessThreads).
    void endScheduledJob() {
        std::lock_guard lock{pool_mutex_};
        assert(pending_jobs_ > 0 && "endScheduledJob called without matching beginScheduledJob");
        if (--pending_jobs_ == 0 && deferred_size_.has_value()) {
            resize_locked(deferred_size_.value());
            deferred_size_.reset();
        }
    }

    // Join and destroy the OS threads of slots parked above the logical limit — the
    // "GC" that trims the pool back to its configured size after a shrink. The joins
    // happen on the calling thread; call only from background contexts (a worker that
    // just completed a scheduled job), never from the resize caller (the Redis main
    // thread). Cheap no-op when nothing is parked.
    //
    // Safety without a pending-jobs gate: rent() only hands out slots below
    // logical_size_, and a lowered logical_size_ only takes effect at a point where no
    // job holds an older, larger size snapshot (the deferred-shrink protocol). So an
    // unoccupied slot at index >= logical_size_ can never become rented; the set is
    // stable once observed under the lock. Slots still occupied by a pre-shrink rental
    // are skipped and reclaimed on a later pass.
    //
    // Best-effort and non-throwing: runs after the job completion guard, where an
    // exception would unwind into the C worker-thread boundary. The local container is
    // reserved before any thread is detached, so an allocation failure aborts the pass
    // cleanly (threads simply stay parked for the next pass).
    void reclaimExcessThreads() noexcept {
        try {
            std::vector<svs::threads::Thread> doomed;
            {
                std::lock_guard lock{pool_mutex_};
                if (slots_.size() <= logical_size_) {
                    return;
                }
                doomed.reserve(slots_.size() - logical_size_);
                for (size_t i = logical_size_; i < slots_.size(); ++i) {
                    auto &slot = *slots_[i];
                    if (slot.thread.has_value() && !slot.occupied.load(std::memory_order_acquire)) {
                        doomed.push_back(std::move(*slot.thread));
                        slot.thread.reset();
                    }
                }
            }
            // ~Thread joins each detached thread (a parked worker wakes, sees
            // RequestShutdown and exits promptly — no idle-spin burn), outside
            // pool_mutex_ so a concurrent CONFIG SET resize is never blocked.
        } catch (...) {
            // Allocation failure — skip this pass; threads stay parked.
        }
    }

    // Execute `f` in parallel with `n` partitions. The calling thread runs partition 0,
    // and up to `n-1` worker threads are rented for partitions 1..n-1.
    // Same signature as the SVS ThreadPool concept.
    void parallel_for(std::function<void(size_t)> f, size_t n, void *log_ctx = nullptr) {
        if (n == 0) {
            return;
        }
        if (n == 1) {
            // Single partition: run on the calling thread, no rental needed.
            try {
                f(0);
            } catch (const std::exception &error) {
                // No workers to check — just rethrow with formatted message.
                auto msg = fmt::format("Thread 0: {}\n", error.what());
                throw svs::threads::ThreadingException{std::move(msg)};
            }
            return;
        }

        // Rent n-1 worker threads. A shortfall (fewer slots than requested) is handled
        // below by the degraded-execution path, like every other dispatch failure.
        auto rented = rent(n - 1, log_ctx);

        // Dispatch partitions 1..n-1 to rented workers: for each slot, lazily spawn its
        // OS thread and immediately assign its partition (interleaved, so the fresh
        // thread's first spin window catches the work; runs outside pool_mutex_ so
        // background spawns never block a concurrent CONFIG SET resize).
        //
        // Degraded execution: if a spawn throws (std::system_error — thread-resource
        // exhaustion) or an assign throws (worker crashed between rents), stop
        // dispatching. Partitions [dispatched+1, n) run on the calling thread below —
        // partitions are never dropped, whatever the trigger (spawn failure, assign
        // failure, or rent shortfall).
        size_t dispatched = 0;
        std::string dispatch_error;
        for (size_t i = 0; i < rented.count(); ++i) {
            try {
                rented.ensureThreadAt(i);
                rented[i].assign({&f, i + 1});
            } catch (const std::exception &error) {
                dispatch_error = error.what();
                // If the failure came from a crashed worker, retire it so the slot
                // lazily respawns a healthy thread on a future rent. Never throws
                // past this point (joining an exited thread is cheap and safe).
                rented.resetThreadAt(i);
                break;
            }
            ++dispatched;
        }

        if (!dispatch_error.empty() || dispatched < n - 1) {
            auto msg =
                fmt::format("SVS thread pool: dispatched {} of {} partitions to "
                            "workers (rented {}); running the rest on the calling "
                            "thread.{}{}",
                            dispatched, n - 1, rented.count(),
                            dispatch_error.empty() ? "" : " Dispatch error: ", dispatch_error);
            if (VecSimIndexInterface::logCallback && log_ctx) {
                VecSimIndexInterface::logCallback(log_ctx, "warning", msg.c_str());
            }
        }

        // Run partition 0 on the calling thread, then any partitions that were not
        // dispatched. Errors are collected per-partition (parity with worker behavior:
        // one failing partition does not prevent the others from running).
        auto message = std::string{};
        auto inserter = std::back_inserter(message);
        bool has_error = false;
        try {
            f(0);
        } catch (const std::exception &error) {
            has_error = true;
            fmt::format_to(inserter, "Thread 0: {}\n", error.what());
        }
        for (size_t p = dispatched + 1; p < n; ++p) {
            try {
                f(p);
            } catch (const std::exception &error) {
                has_error = true;
                fmt::format_to(inserter, "Partition {} (on calling thread): {}\n", p, error.what());
            }
        }

        // Wait for all dispatched workers and collect errors.
        // RentedThreads destructor will release the slots after this block.
        manage_workers_after_run(has_error, std::move(message), dispatched, rented);
    }

private:
    // Rent up to `count` worker threads from the pool. Returns an RAII guard that
    // automatically releases the threads when destroyed.
    // The SVS pool is sized to match the RediSearch thread pool, and RediSearch controls
    // scheduling via reserve jobs, so all requested slots should always be available.
    // Getting fewer threads than requested indicates a bug in the scheduling logic.
    RentedThreads rent(size_t count, void *log_ctx = nullptr) {
        RentedThreads rented;
        if (count == 0) {
            return rented;
        }

        std::lock_guard lock{pool_mutex_};
        size_t rented_count = 0;
        // Only slots below the logical limit are rentable — slots above it (parked
        // after a shrink) are reserved for reclamation and must not gain new renters.
        for (size_t i = 0; i < logical_size_; ++i) {
            auto &slot = slots_[i];
            bool expected = false;
            if (slot->occupied.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                rented.add(slot.get());
                if (++rented_count >= count) {
                    break;
                }
            }
        }

        if (rented.count() < count) {
            auto msg = fmt::format("SVS thread pool: rented {} threads out of {} requested "
                                   "(pool has {} rentable slots). This should not happen.",
                                   rented.count(), count, logical_size_);
            if (VecSimIndexInterface::logCallback) {
                assert(log_ctx && "Log context must be provided when logging is available");
                VecSimIndexInterface::logCallback(log_ctx, "warning", msg.c_str());
            }
            assert(false && "Failed to rent the expected number of SVS threads");
        }
        return rented;
    }

    // Wait for the first `dispatched` rented workers to finish (only those actually got
    // a partition assigned). If any worker (or the calling thread) threw, retire crashed
    // workers to the unspawned state and throw a combined exception.
    void manage_workers_after_run(bool has_error, std::string message, size_t dispatched,
                                  RentedThreads &rented) {
        auto inserter = std::back_inserter(message);

        for (size_t i = 0; i < dispatched; ++i) {
            auto &thread = rented[i];
            thread.wait();
            if (!thread.is_okay()) {
                has_error = true;
                try {
                    thread.unsafe_get_exception();
                } catch (const std::exception &error) {
                    fmt::format_to(inserter, "Thread {}: {}\n", i + 1, error.what());
                }
                // Retire the crashed thread; the slot lazily respawns on a future rent.
                rented.resetThreadAt(i);
            }
        }

        if (has_error) {
            throw svs::threads::ThreadingException{std::move(message)};
        }
    }

    // Actual resize logic. Caller must hold pool_mutex_.
    // Grow is always applied immediately. Shrink is deferred if pending_jobs_ > 0.
    //
    // Resize is LOGICAL: it moves logical_size_ (the limit rent() enforces) and only
    // ever extends the physical slot vector — never destroys slots. Classification is
    // against logical_size_, not slots_.size(): a grow from logical 1 to 4 inside a
    // 2000-slot high-water pool is a grow. Shrinking parks the threads of slots above
    // the new limit (they sleep; one final idle-spin window, no periodic wakeup);
    // they are joined later by reclaimExcessThreads() on a background thread, or
    // reused warm if the pool grows again first. This keeps both resize directions
    // O(slots) on the Redis main thread — never O(thread create/join).
    void resize_locked(size_t new_size) {
        size_t target_workers = new_size - 1;

        if (target_workers >= logical_size_) {
            // Grow (or same size): apply immediately, cancel any pending deferred shrink.
            deferred_size_.reset();
            for (size_t i = slots_.size(); i < target_workers; ++i) {
                slots_.push_back(
                    std::allocate_shared<ThreadSlot>(VecsimSTLAllocator<ThreadSlot>(allocator_)));
            }
            logical_size_ = target_workers;
        } else {
            // Shrink.
            if (pending_jobs_ > 0) {
                // Defer shrink — jobs in flight snapshotted the old size and may
                // still rent up to it.
                deferred_size_ = new_size;
            } else {
                // Safe to shrink now — no jobs in flight. Purely logical: slots above
                // the limit become unrentable; their threads (if spawned) stay parked
                // until reclaimExcessThreads() joins them off the main thread.
                logical_size_ = target_workers;
            }
        }
    }

    std::shared_ptr<VecSimAllocator> allocator_; // pool's own allocator for memory tracking
    mutable std::mutex pool_mutex_;
    // Physical slots. Only grows (high-water mark); heap-allocated slots, so raw
    // ThreadSlot* held by renters stay valid across vector reallocation on grow.
    vecsim_stl::vector<SlotPtr> slots_;
    // Logical worker count = the rent() limit = configured pool size - 1. Slots at
    // index >= logical_size_ are unrentable and eligible for thread reclamation.
    size_t logical_size_ = 0;
    size_t pending_jobs_ = 0; // jobs currently scheduled / in-flight
    // Pending pool size to apply at the next safe point: either the first SVS index
    // attaches (onIndexAttached()) or pending_jobs_ drops to 0 (endScheduledJob()).
    // The two cases are sequential and never overlap.
    std::optional<size_t> deferred_size_;
    // Flips true on the first onIndexAttached() call; gates resize() between
    // "record only" and "apply immediately".
    bool has_attached_index_ = false;
};

// Per-index wrapper around the shared VecSimSVSThreadPoolImpl singleton.
// Lightweight, copyable (SVS stores a copy via ThreadPoolHandle). Both the original
// and SVS's copy share the same pool_ and parallelism_ via shared_ptr, so state
// changes propagate automatically.
// Satisfies the svs::threads::ThreadPool concept (size() + parallel_for).
// The pool is always valid — in write-in-place mode it has size 1 (0 worker threads).
class VecSimSVSThreadPool {
private:
    std::shared_ptr<VecSimSVSThreadPoolImpl> pool_; // shared across all indexes
    // Per-index parallelism, shared across copies (SVS stores a copy of VecSimSVSThreadPool).
    // SVS reads this value via size() during parallel_for to decide how many threads to use
    // for task partitioning. Because SVS reads size() internally — not under our control —
    // the caller must ensure that parallelism_ is stable for the entire duration of any SVS
    // operation (search, build, consolidate, add, etc.). In practice this means:
    //   setParallelism(n) and the subsequent SVS call must be protected by the same lock,
    //   and no other code path may call setParallelism() on the same index concurrently.
    // Currently, mainIndexGuard (exclusive) or updateJobMutex fulfills this role.
    std::shared_ptr<std::atomic<size_t>> parallelism_;
    void *log_ctx_ = nullptr; // per-index log context

public:
    // Construct using the shared pool singleton.
    // parallelism_ starts at 1 (the calling thread always participates), matching the
    // pool's minimum size. Safe for immediate use in write-in-place mode without an
    // explicit setParallelism() call.
    // parallelism_ is allocated through the provided VecsimAllocator so that the
    // allocation is tracked by the index's memory accounting.
    // Notifies the shared pool that an index has attached — on the very first call this
    // applies any size requested earlier via resize() (lazy thread spawn).
    explicit VecSimSVSThreadPool(const std::shared_ptr<VecSimAllocator> &allocator,
                                 void *log_ctx = nullptr)
        : pool_(VecSimSVSThreadPoolImpl::instance()),
          parallelism_(std::allocate_shared<std::atomic<size_t>>(
              VecsimSTLAllocator<std::atomic<size_t>>(allocator), size_t{1})),
          log_ctx_(log_ctx) {
        pool_->onIndexAttached();
    }

    // Resize the shared pool singleton. Delegates to VecSimSVSThreadPoolImpl::instance().
    static void resize(size_t new_size) { VecSimSVSThreadPoolImpl::instance()->resize(new_size); }

    // Set the degree of parallelism for this index's next operation.
    // n must be the number of threads actually reserved by the caller (i.e., the
    // RediSearch workers that checked in via ReserveThreadJob). This is what allows
    // us to assert n <= pool size: reserved workers are occupied RediSearch threads,
    // so the pool cannot shrink while they are held, and n cannot exceed the pool size.
    //
    // IMPORTANT: The caller must hold a lock that prevents any concurrent SVS operation
    // on this index from reading size() between setParallelism() and the operation that
    // depends on it. SVS internally calls pool.size() (which returns parallelism_) during
    // parallel_for — if another thread calls setParallelism() concurrently, the operation
    // may see an inconsistent value.
    void setParallelism(size_t n) {
        assert(n >= 1 && "Parallelism must be at least 1 (the calling thread)");
        assert(n <= pool_->size() && "Parallelism exceeds shared pool size");
        parallelism_->store(n);
    }
    size_t getParallelism() const { return parallelism_->load(); }

    // Returns per-index parallelism. SVS uses this for task partitioning (ThreadPool concept).
    size_t size() const { return parallelism_->load(); }

    // Shared pool size — used by scheduling to decide how many reserve jobs to submit.
    static size_t poolSize() { return VecSimSVSThreadPoolImpl::instance()->size(); }

    // See VecSimSVSThreadPoolImpl::getSharedAllocationSize().
    static size_t getSharedAllocationSize() {
        return VecSimSVSThreadPoolImpl::getSharedAllocationSize();
    }

    // Delegates to the shared pool's parallel_for, passing the per-index log context.
    // n may be less than parallelism_ when the problem size is smaller than the
    // thread count (SVS computes n = min(arg.size(), pool.size())).
    void parallel_for(std::function<void(size_t)> f, size_t n) {
        pool_->parallel_for(std::move(f), n, log_ctx_);
    }
};
