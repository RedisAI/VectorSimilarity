/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

// #include "graph_data.h"
// #include "visited_nodes_handler.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/utils/vec_utils.h"
#include <vector>
// #include "VecSim/containers/data_block.h"
// #include "VecSim/containers/raw_data_container_interface.h"
// #include "VecSim/containers/data_blocks_container.h"
// #include "VecSim/containers/vecsim_results_container.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/vec_sim_index.h"
// #include "VecSim/tombstone_interface.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "VecSim/containers/data_block.h"
// #include "../RediSearch/.install/boost/boost/unordered/concurrent_flat_map.hpp"

// Includes that should be in the inherited class
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/spaces/computer/preprocessors.h"
#include "VecSim/algorithms/hnsw/visited_nodes_handler.h"
#include "VecSim/algorithms/hnsw/hnsw.h" // For HNSWAddVectorState definition
#include "VecSim/utils/updatable_heap.h"

#ifdef BUILD_TESTS
#include "hnsw_serialization_utils.h"
#include "VecSim/utils/serializer.h"
#endif

// #include <deque>
#include <memory>
// #include <cassert>
#include <climits>
// #include <queue>
#include <random>
// #include <iostream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
// #include <sys/resource.h>
// #include <fstream>
#include <shared_mutex>
#include <atomic>
#include <array>

// Forward declaration for AsyncJob
#include "VecSim/vec_sim_tiered_index.h"

using std::pair;

template <typename DistType>
using candidatesMaxHeap = vecsim_stl::max_priority_queue<DistType, idType>;
template <typename DistType>
using candidatesList = vecsim_stl::vector<std::pair<DistType, idType>>;
template <typename DistType>
using candidatesLabelsMaxHeap = vecsim_stl::abstract_priority_queue<DistType, labelType>;
using graphNodeType = std::pair<idType, unsigned short>; // represented as: (element_id, level)

// Hash function for graphNodeType
struct GraphNodeHash {
    std::size_t operator()(const graphNodeType &k) const {
        return std::hash<idType>()(k.first) ^ (std::hash<unsigned short>()(k.second) << 1);
    }
};

////////////////////////////////////// Auxiliary HNSW structs //////////////////////////////////////

struct DiskElementMetaData {
    labelType label;
    size_t topLevel;
    elementFlags flags;

    DiskElementMetaData(labelType label = INVALID_LABEL) noexcept
        : label(label), topLevel(0), flags(0) {}
    DiskElementMetaData(labelType label, size_t topLevel) noexcept
        : label(label), topLevel(topLevel), flags(0) {}
};

// The state of the index and the newly stored vector to be passed to indexVector.
// Note: This is already defined in hnsw.h, so we'll use that one

static constexpr char GraphKeyPrefix[3] = "GK";

#pragma pack(1)
struct GraphKey {
    // uint8_t version;
    uint16_t level;
    idType id;

    GraphKey(idType id, size_t level) : level(static_cast<uint16_t>(level)), id(id) {}

    rocksdb::Slice asSlice() const {
        // Create a key with the "GK" prefix followed by the struct data
        static thread_local std::vector<char> key_buffer;
        key_buffer.resize(3 + sizeof(*this)); // 3 bytes for "GK" + struct size

        // Copy the "GK" prefix
        key_buffer[0] = 'G';
        key_buffer[1] = 'K';
        key_buffer[2] = '\0';

        // Copy the struct data
        std::memcpy(key_buffer.data() + 3, this, sizeof(*this));

        return rocksdb::Slice(key_buffer.data(), key_buffer.size());
    }

    graphNodeType node() const { return graphNodeType(id, level); }
};

#pragma pack()

////////////////////////////////////// HNSW Disk Job Structures //////////////////////////////////////

/**
 * Forward declaration of HNSWDiskIndex for job structures.
 */
template <typename DataType, typename DistType>
class HNSWDiskIndex;

/**
 * Definition of a job that inserts a single vector into the HNSW disk graph.
 * Each job handles one vector independently and can run in parallel with other insert jobs.
 */
struct HNSWDiskInsertJob : public AsyncJob {
    idType vectorId;
    size_t elementMaxLevel;

    HNSWDiskInsertJob(std::shared_ptr<VecSimAllocator> allocator, idType vectorId_,
                      size_t elementMaxLevel_, JobCallback insertCb, VecSimIndex *index_)
        : AsyncJob(allocator, HNSW_DISK_INSERT_VECTOR_JOB, insertCb, index_), vectorId(vectorId_),
          elementMaxLevel(elementMaxLevel_) {}
};


/**
 * Definition of the job that inserts a single vector completely from start to end.
 * Each job is self-contained and writes directly to disk upon completion.
 * No batching or staging - optimized for workloads where disk write is cheap
 * but reading (searching for neighbors) is the bottleneck.
 *
 * The job holds copies of the vector data to avoid external references and race conditions.
 */
struct HNSWDiskSingleInsertJob : public AsyncJob {
    idType vectorId;
    size_t elementMaxLevel;
    // Store vector data directly in the job (no external references)
    std::string rawVectorData;       // Original float32 vector
    std::string processedVectorData; // Preprocessed/quantized vector for distance calculations

    HNSWDiskSingleInsertJob(std::shared_ptr<VecSimAllocator> allocator, idType vectorId_,
                            size_t elementMaxLevel_, std::string &&rawVector,
                            std::string &&processedVector, JobCallback insertCb,
                            VecSimIndex *index_)
        : AsyncJob(allocator, HNSW_DISK_SINGLE_INSERT_JOB, insertCb, index_),
          vectorId(vectorId_), elementMaxLevel(elementMaxLevel_),
          rawVectorData(std::move(rawVector)), processedVectorData(std::move(processedVector)) {}
};

//////////////////////////////////// HNSW index implementation ////////////////////////////////////

template <typename DataType, typename DistType>
class HNSWDiskIndex : public VecSimIndexAbstract<DataType, DistType>
#ifdef BUILD_TESTS
    ,
                      public Serializer
#endif
{
protected:
    // Index build parameters
    // size_t maxElements;
    size_t M;
    size_t M0;
    size_t efConstruction;

    // Index search parameter
    size_t ef;
    double epsilon;

    // Index meta-data (based on the data dimensionality and index parameters)
    // size_t elementGraphDataSize;
    // size_t levelDataSize;
    double mult;
    // uint8_t version; // version of the graph.

    // Index level generator of the top level for a new element
    mutable std::default_random_engine levelGenerator;

    // multithreaded scenario.
    std::atomic<size_t> curElementCount;
    size_t numMarkedDeleted;

    idType entrypointNode;
    size_t maxLevel; // this is the top level of the entry point's element

    // Index data
    // vecsim_stl::vector<DataBlock> graphDataBlocks;
    vecsim_stl::vector<DiskElementMetaData> idToMetaData;
    vecsim_stl::unordered_map<labelType, idType> labelToIdMap;
    rocksdb::DB *db;                 // RocksDB database, not owned by the index
    rocksdb::Options dbOptions;      // RocksDB options, not owned by the index
    rocksdb::ColumnFamilyHandle *cf; // RocksDB column family handle, not owned by the index
    std::string dbPath;              // Path where RocksDB data is stored

    mutable std::shared_mutex indexDataGuard;
    mutable VisitedNodesHandlerPool visitedNodesHandlerPool;

    /**
     * Threshold for batching delete operations.
     * When the number of pending deletions reaches this value, the deletions are processed in a batch.
     */
    size_t deleteBatchThreshold = 10;
    vecsim_stl::vector<idType> pendingDeleteIds;

    bool useRawData = true;

public:
    // Search metrics counters (for benchmarking)
    mutable std::atomic_size_t num_visited_nodes;
    mutable std::atomic_size_t num_visited_nodes_higher_levels;

protected:
    // In-memory graph updates staging (for delayed disk operations)
    struct GraphUpdate {
        idType node_id;
        size_t level;
        vecsim_stl::vector<idType> neighbors;

        GraphUpdate(idType node_id, size_t level, const vecsim_stl::vector<idType> &neighbors,
                    std::shared_ptr<VecSimAllocator> allocator)
            : node_id(node_id), level(level), neighbors(allocator) {
            this->neighbors = neighbors;
        }
        // eq operator
        bool operator==(const GraphUpdate &other) const {
            return node_id == other.node_id && level == other.level;
        }
    };


    // Staging area for graph updates during batch processing
    // Separate staging areas for insertions and deletions to avoid conflicts
    vecsim_stl::vector<GraphUpdate> stagedInsertUpdates;
    vecsim_stl::vector<GraphUpdate> stagedDeleteUpdates;

    // Staged repair updates: opportunistic cleanup when stale edges are filtered during reads
    // Mutable to allow staging from const search methods.
    // IMPORTANT: This class is NOT thread-safe. All operations (including const methods like
    // getNeighbors and search) must be called from a single thread. The mutable fields below
    // are modified during read operations for opportunistic graph cleanup.
    // TODO: For multi-threaded support, these fields need proper synchronization or a different
    // approach (e.g., returning repair suggestions instead of staging them directly).
    mutable vecsim_stl::vector<GraphUpdate> stagedRepairUpdates;

    // Hash maps for O(1) lookups in staged updates
    // Key: (node_id << 32) | level - combines node_id and level into a single uint64_t
    // Value: index into the corresponding staged updates vector
    std::unordered_map<uint64_t, size_t> stagedInsertMap;
    std::unordered_map<uint64_t, size_t> stagedDeleteMap;

    // Hash map for O(1) lookups and duplicate detection in stagedRepairUpdates
    // Key: (node_id << 32) | level - combines node_id and level into a single uint64_t
    // Value: index into stagedRepairUpdates vector
    // Mutable - see thread-safety note above for stagedRepairUpdates
    mutable std::unordered_map<uint64_t, size_t> stagedRepairMap;

    // Track which nodes need their neighbor lists updated (for bidirectional connections)
    struct NeighborUpdate {
        idType node_id;
        size_t level;
        idType new_neighbor_id;

        NeighborUpdate(idType node_id, size_t level, idType new_neighbor_id)
            : node_id(node_id), level(level), new_neighbor_id(new_neighbor_id) {}
    };

    // TODO: Consider merging with stagedInsertUpdates
    vecsim_stl::vector<NeighborUpdate> stagedInsertNeighborUpdates;

    // Temporary storage for raw vectors in RAM (until flush batch)
    // Maps idType -> raw vector data (stored as string for simplicity)
    std::unordered_map<idType, std::string> rawVectorsInRAM;


    /********************************** Multi-threading Support **********************************/

    // Job queue parameters (similar to tiered index)
    void *jobQueue = nullptr;
    void *jobQueueCtx = nullptr;
    SubmitCB SubmitJobsToQueue = nullptr;

    // Lock for protecting staged updates during merge from parallel insert jobs
    // Uses shared_mutex for better read concurrency - multiple readers can access simultaneously
    mutable std::shared_mutex stagedUpdatesGuard;

    // Lock for protecting vectors container during concurrent access
    // Needed because addElement can resize the container, invalidating pointers
    mutable std::shared_mutex vectorsGuard;

    // Note: metadataGuard was consolidated into indexDataGuard
    // indexDataGuard now protects: entrypointNode, maxLevel, idToMetaData, labelToIdMap

    // Lock for protecting rawVectorsInRAM during concurrent access
    // Needed because unordered_map can rehash during insert, invalidating iterators
    mutable std::shared_mutex rawVectorsGuard;

    /********************************** Batchless Mode Support **********************************/

    // Segmented neighbor cache for reduced lock contention in multi-threaded scenarios
    // Cache is partitioned into NUM_CACHE_SEGMENTS independent segments
    // Each segment has its own lock, cache map, and dirty set
    // This allows threads accessing different segments to proceed in parallel

    static constexpr size_t NUM_CACHE_SEGMENTS = 64;  // Power of 2 for efficient modulo

    // Cache segment structure - each segment is cache-line aligned to prevent false sharing
    struct alignas(64) CacheSegment {
        std::shared_mutex guard;
        std::unordered_map<uint64_t, std::vector<idType>> cache;
        std::unordered_set<uint64_t> dirty;
        // Track nodes created in current batch (never written to disk yet)
        // This helps avoid disk lookups for new nodes
        std::unordered_set<uint64_t> newNodes;

        CacheSegment() = default;
        CacheSegment(const CacheSegment&) = delete;
        CacheSegment& operator=(const CacheSegment&) = delete;
        CacheSegment(CacheSegment&&) = delete;
        CacheSegment& operator=(CacheSegment&&) = delete;
    };

    // Array of cache segments - using unique_ptr for lazy initialization
    mutable std::unique_ptr<CacheSegment[]> cacheSegments_;

    // Atomic counter for total dirty nodes (for fast threshold check without locking)
    mutable std::atomic<size_t> totalDirtyCount_{0};

    // Helper function to compute segment index from cache key
    // Uses mixing function for better distribution
    static size_t getSegmentIndex(uint64_t key) {
        // Mix the bits for better distribution (splitmix64-style mixing)
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdULL;
        key ^= key >> 33;
        return key % NUM_CACHE_SEGMENTS;
    }

    // Threshold for flushing dirty nodes to disk, expressed as a count of dirty nodes.
    // - Unit: number of distinct dirty nodes tracked in `totalDirtyCount_` across all cache segments.
    // - Behavior:
    //     * 0  : flush after each insert/update (no batching).
    //     * >0 : accumulate dirty nodes and flush once the global dirty count reaches this threshold.
    // - Tradeoffs:
    //     * Lower values (including 0) reduce the amount of data lost on crash and smooth memory usage,
    //       but increase the number of RocksDB writes and flush operations (higher write amplification
    //       and potentially higher latency per insert).
    //     * Higher values reduce the frequency of disk writes and can improve bulk-insert throughput,
    //       but increase peak memory usage for cached/dirty nodes and may cause longer flush pauses
    //       when the batch is written, as well as more data at risk between flushes.
    size_t diskWriteBatchThreshold = 1000;

    // Lock for protecting dirty nodes flush operations (global flush serialization)
    mutable std::mutex diskWriteGuard;

    // Atomic counter for pending single insert jobs (batchless mode)
    std::atomic<size_t> pendingSingleInsertJobs_{0};

protected:
    HNSWDiskIndex() = delete; // default constructor is disabled.
    // default (shallow) copy constructor is disabled.
    HNSWDiskIndex(const HNSWDiskIndex &) = delete;

    auto getNeighborhoods(const vecsim_stl::vector<idType> &ids) const;

    idType getNeighborsByHeuristic2(candidatesList<DistType> &top_candidates, size_t M) const;
    void getNeighborsByHeuristic2(candidatesList<DistType> &top_candidates, size_t M,
                                  vecsim_stl::vector<idType> &not_chosen_candidates) const;
    template <bool record_removed>
    void getNeighborsByHeuristic2_internal(candidatesList<DistType> &top_candidates, size_t M,
                                           vecsim_stl::vector<idType> *removed_candidates) const;

    // Helper methods for GraphKey value serialization/deserialization
    // GraphKey value format: [raw_vector_data][neighbor_count][neighbor_ids...]
    std::string serializeGraphValue(const void* vector_data, const vecsim_stl::vector<idType>& neighbors) const;
    void deserializeGraphValue(const std::string& value, vecsim_stl::vector<idType>& neighbors) const;
    const void* getVectorFromGraphValue(const std::string& value) const;

public:
    // Pure virtual methods from VecSimIndexInterface
    int addVector(const void *blob, labelType label) override;

public:
    // Core vector addition methods
    void insertElementToGraph(idType element_id, size_t element_max_level,
                              idType entry_point, size_t global_max_level,
                              const void *raw_vector_data, const void *vector_data,
                              vecsim_stl::vector<uint64_t> &modifiedNodes);
    idType mutuallyConnectNewElement(idType new_node_id,
                                     vecsim_stl::updatable_max_heap<DistType, idType> &top_candidates,
                                     size_t level, vecsim_stl::vector<uint64_t> &modifiedNodes);

    // Delete batch processing methods
    void processDeleteBatch();
    void flushDeleteBatch(); // Force flush current delete batch

    // Job submission helpers
    void submitSingleJob(AsyncJob *job);
    void submitJobs(vecsim_stl::vector<AsyncJob *> &jobs);

    // Job execution
    static void executeSingleInsertJobWrapper(AsyncJob *job);
    void executeSingleInsertJob(HNSWDiskSingleInsertJob *job);

    // Cache management methods
    void addNeighborToCache(idType nodeId, size_t level, idType newNeighborId);
    void setNeighborsInCache(idType nodeId, size_t level, const vecsim_stl::vector<idType> &neighbors, bool isNewNode = false);
    void getNeighborsFromCache(idType nodeId, size_t level, vecsim_stl::vector<idType> &result) const;
    // Atomic check-and-add for cache - returns true if added, false if at capacity
    bool tryAddNeighborToCacheIfCapacity(idType nodeId, size_t level, idType newNeighborId, size_t maxCapacity);

private:
    // Helper to load neighbors from disk into cache if not already present
    // Returns true if data is available in cache after call, false otherwise
    // Caller must hold lock on cacheSegment.guard (will be temporarily released during disk I/O)
    void loadNeighborsFromDiskIfNeeded(uint64_t key, CacheSegment& cacheSegment,
                                        std::unique_lock<std::shared_mutex>& lock);

    // Write dirty cache entries to disk
    void writeDirtyNodesToDisk(const vecsim_stl::vector<uint64_t> &modifiedNodes,
                               const void *newVectorRawData, idType newVectorId);

    // Unified core function for graph insertion - used by both single-threaded and multi-threaded paths
    // Batching is controlled by diskWriteBatchThreshold (0 = no batching, >0 = batch size)
    void executeGraphInsertionCore(idType vectorId, size_t elementMaxLevel,
                                   idType entryPoint, size_t globalMaxLevel,
                                   const void *rawVectorData, const void *processedVectorData);

    // Helper to write a single vector's entry to disk
    void writeVectorToDisk(idType vectorId, const void *rawVectorData,
                           const vecsim_stl::vector<idType> &neighbors);

    // Helper methods
    void emplaceHeap(vecsim_stl::abstract_priority_queue<DistType, idType> &heap, DistType dist,
                       idType id) const;
    void emplaceHeap(vecsim_stl::abstract_priority_queue<DistType, labelType> &heap, DistType dist,
                       idType id) const;
    void getNeighbors(idType nodeId, size_t level, vecsim_stl::vector<idType>& result) const;
    size_t getNeighborsCount(idType nodeId, size_t level) const;

    // Manual control of staged delete updates
    void flushStagedDeleteUpdates(); // Manually flush any pending staged delete updates

protected:
    // Helper method to filter deleted or invalid nodes from a neighbor list (DRY principle)
    // Returns true if any nodes were filtered out
    // Filters out: nodes marked as deleted, and nodes with invalid IDs (>= curElementCount)
    inline bool filterDeletedNodes(vecsim_stl::vector<idType>& neighbors) const {
        size_t original_size = neighbors.size();
        size_t elementCount = curElementCount.load(std::memory_order_acquire);
        // Hold shared lock to prevent idToMetaData resize during access
        std::shared_lock<std::shared_mutex> lock(indexDataGuard);
        size_t metadataSize = idToMetaData.size();
        auto new_end = std::remove_if(neighbors.begin(), neighbors.end(),
            [this, elementCount, metadataSize](idType id) {
                // Use lock-free check: bounds check first, then atomic flag read
                return id >= elementCount || id >= metadataSize ||
                       isMarkedAsUnsafe<DELETE_MARK>(id);
            });
        neighbors.erase(new_end, neighbors.end());
        return neighbors.size() < original_size;
    }

    // Helper to create a unique key for (node_id, level) pair for hash map
    inline uint64_t makeRepairKey(idType node_id, size_t level) const {
        return (static_cast<uint64_t>(node_id) << 32) | static_cast<uint64_t>(level);
    }

    // New method for flushing staged graph updates to disk
    void flushStagedGraphUpdates(vecsim_stl::vector<GraphUpdate>& graphUpdates,
                                  vecsim_stl::vector<NeighborUpdate>& neighborUpdates);

    // Re-evaluate neighbor connections when a neighbor's list is full
    // Applies heuristic to select best neighbors including the new node
    void revisitNeighborConnections(idType new_node_id, idType selected_neighbor,
                                    size_t level, DistType distance,
                                    vecsim_stl::vector<uint64_t> &modifiedNodes);

public:
    // Methods needed by benchmark framework
    const void *getDataByInternalId(idType id) const;
    vecsim_stl::updatable_max_heap<DistType, idType> searchLayer(idType ep_id, const void *data_point, size_t level,
                                            size_t ef) const;
    vecsim_stl::updatable_max_heap<DistType, labelType> searchLayerLabels(idType ep_id, const void *data_point, size_t level,
                                            size_t ef) const;
    template <bool running_query>
    void greedySearchLevel(const void *data_point, size_t level, idType &curr_element,
                           DistType &cur_dist) const;
    std::pair<idType, size_t> safeGetEntryPointState() const;
    VisitedNodesHandler *getVisitedList() const;
    void returnVisitedList(VisitedNodesHandler *visited_nodes_handler) const;
    candidatesLabelsMaxHeap<DistType> *getNewMaxPriorityQueue() const;
    bool isMarkedDeleted(idType id) const;
    bool isMarkedDeleted(labelType id) const;
    bool isMarkedDeletedUnsafe(idType id) const;  // Lock-free version for hot paths
    labelType getExternalLabel(idType id) const;

    // Helper methods for emplacing to heaps (overloaded for idType and labelType)
    void emplaceToHeap(vecsim_stl::abstract_priority_queue<DistType, idType> &heap, DistType dist,
                       idType id) const;
    void emplaceToHeap(vecsim_stl::abstract_priority_queue<DistType, labelType> &heap,
                       DistType dist, idType id) const;

    template <typename Identifier>
    void processCandidate(idType curNodeId, const void *data_point, size_t level, size_t ef,
                          std::unordered_set<idType> *visited_set,
                          vecsim_stl::updatable_max_heap<DistType, Identifier> &top_candidates,
                          candidatesMaxHeap<DistType> &candidate_set, DistType &lowerBound) const;

    // Raw vector storage and retrieval methods
    bool getRawVector(idType id, void* output_buffer) const;

    // Re-rank candidates using raw float32 distances for improved recall
    void rerankWithRawDistances(vecsim_stl::updatable_max_heap<DistType, labelType>& candidates,
                                const void* query_data, size_t k) const;

protected:
    // Internal version that assumes caller already holds the lock (or is inside a locked section)
    bool getRawVectorInternal(idType id, void* output_buffer) const;
    idType searchBottomLayerEP(const void *query_data, void *timeoutCtx = nullptr,
                               VecSimQueryReply_Code *rc = nullptr) const;


public:
    HNSWDiskIndex(const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
                  const IndexComponents<DataType, DistType> &components, rocksdb::DB *db,
                  rocksdb::ColumnFamilyHandle *cf, const std::string &dbPath = "",
                  size_t random_seed = 100, void *jobQueue = nullptr, void *jobQueueCtx = nullptr,
                  SubmitCB submitCb = nullptr);
    virtual ~HNSWDiskIndex();

    /*************************** Index API ***************************/
    // void batchUpdate(const std::vector<pair<labelType, const void *>> &new_elements,
    //                  const std::vector<labelType> &deleted_labels);

    VecSimQueryReply *topKQuery(const void *query_data, size_t k,
                                VecSimQueryParams *queryParams) const override;
    VecSimQueryReply *rangeQuery(const void *query_data, double radius,
                                 VecSimQueryParams *queryParams) const override;
    VecSimIndexDebugInfo debugInfo() const override;
    VecSimDebugInfoIterator *debugInfoIterator() const override;
    VecSimIndexBasicInfo basicInfo() const override;
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override;

    // Methods needed by benchmark framework
    void fitMemory() override;
    void getDataByLabel(labelType label,
                        std::vector<std::vector<DataType>> &vectors_output) const override;

    // Missing virtual method implementations
    VecSimIndexStatsInfo statisticInfo() const override;
    void setLastSearchMode(VecSearchMode mode) override;
    void runGC() override;
    void acquireSharedLocks() override;
    void releaseSharedLocks() override;
    std::vector<std::vector<char>> getStoredVectorDataByLabel(labelType label) const override;
    vecsim_stl::set<labelType> getLabelsSet() const override;
    int deleteVector(labelType label) override;
    double getDistanceFrom_Unsafe(labelType id, const void *blob) const override;

    uint64_t getAllocationSize() const override;
    uint64_t getDBMemorySize() const;
    uint64_t getDiskSize() const;
    std::shared_ptr<rocksdb::Statistics> getDBStatistics() const;

    // Get detailed RocksDB memory breakdown
    struct RocksDBMemoryBreakdown {
        uint64_t memtables;
        uint64_t table_readers;
        uint64_t block_cache;
        uint64_t pinned_blocks;
        uint64_t total;
    };
    RocksDBMemoryBreakdown getDBMemoryBreakdown() const;

public:
    // Public methods for testing
    size_t indexSize() const override;
    size_t indexCapacity() const override;
    size_t indexLabelCount() const override;
    size_t getRandomLevel(double reverse_size);
    size_t getMaxLevel() const { return maxLevel; }

    // Flagging API
    template <Flags FLAG>
    void markAs(idType internalId) {
        std::shared_lock<std::shared_mutex> lock(indexDataGuard);
        __atomic_fetch_or(&idToMetaData[internalId].flags, FLAG, 0);
    }
    template <Flags FLAG>
    void unmarkAs(idType internalId) {
        std::shared_lock<std::shared_mutex> lock(indexDataGuard);
        __atomic_fetch_and(&idToMetaData[internalId].flags, ~FLAG, 0);
    }
    template <Flags FLAG>
    bool isMarkedAs(idType internalId) const {
        std::shared_lock<std::shared_mutex> lock(indexDataGuard);
        return __atomic_load_n(&idToMetaData[internalId].flags, 0) & FLAG;
    }

    // Lock-free version for hot paths where we know the ID is valid and in bounds
    // (caller must ensure internalId < idToMetaData.size())
    template <Flags FLAG>
    bool isMarkedAsUnsafe(idType internalId) const {
        return __atomic_load_n(&idToMetaData[internalId].flags, 0) & FLAG;
    }

    // Mark delete API
    vecsim_stl::vector<idType> markDelete(labelType label);
    size_t getNumMarkedDeleted() const { return numMarkedDeleted; }

    // Batch deletion control (for benchmarking)
    void setDeleteBatchThreshold(size_t threshold) { deleteBatchThreshold = threshold; }
    size_t getDeleteBatchThreshold() const { return deleteBatchThreshold; }
    size_t getPendingDeleteCount() const { return pendingDeleteIds.size(); }

    // Disk write batching control
    void setDiskWriteBatchThreshold(size_t threshold) { diskWriteBatchThreshold = threshold; }
    size_t getDiskWriteBatchThreshold() const { return diskWriteBatchThreshold; }
    size_t getPendingDiskWriteCount() const { return totalDirtyCount_.load(std::memory_order_relaxed); }
    void flushDirtyNodesToDisk(); // Flush all pending dirty nodes to disk

    // Job queue configuration (for multi-threaded processing)
    void setJobQueue(void *jobQueue_, void *jobQueueCtx_, SubmitCB submitCb_) {
        jobQueue = jobQueue_;
        jobQueueCtx = jobQueueCtx_;
        SubmitJobsToQueue = submitCb_;
    }

    // Debug methods to inspect graph structure
    void debugPrintGraphStructure() const;
    void debugPrintNodeNeighbors(idType node_id) const;
    void debugPrintAllGraphKeys() const;
    size_t debugCountGraphEdges() const;
    void debugValidateGraphConnectivity() const;

    // Debug methods for staged updates
    void debugPrintStagedUpdates() const;

private:
    // HNSW helper methods
    void replaceEntryPoint();

    // Helper to stage a delete update, merging with existing entry if present
    inline void stageDeleteUpdate(idType node_id, size_t level, vecsim_stl::vector<idType> &neighbors) {
        uint64_t key = makeRepairKey(node_id, level);
        auto existing_it = stagedDeleteMap.find(key);
        if (existing_it != stagedDeleteMap.end()) {
            stagedDeleteUpdates[existing_it->second].neighbors = neighbors;
        } else {
            stagedDeleteMap[key] = stagedDeleteUpdates.size();
            stagedDeleteUpdates.emplace_back(node_id, level, neighbors, this->allocator);
        }
    }

    // Graph repair helper for deletion - repairs a neighbor's connections after a node is deleted.
    // This maintains graph quality and navigability using a heuristic-based approach.
    // Parameters:
    //   neighbor_id: The neighbor whose connections need repair
    //   level: The graph level being repaired
    //   deleted_id: The internal ID of the node being deleted
    //   deleted_node_neighbors: Neighbors of the deleted node (potential repair candidates)
    //   neighbor_neighbors: Current neighbors of neighbor_id (will be modified)
    void repairNeighborConnections(idType neighbor_id, size_t level, idType deleted_id,
                                   const vecsim_stl::vector<idType> &deleted_node_neighbors,
                                   vecsim_stl::vector<idType> &neighbor_neighbors);

    /*****************************************************************/

#ifdef BUILD_TESTS
#include "hnsw_disk_serializer_declarations.h"
#endif
};

constexpr size_t INITIAL_CAPACITY = 1000;

/********************************** Constructors & Destructor **********************************/

template <typename DataType, typename DistType>
HNSWDiskIndex<DataType, DistType>::HNSWDiskIndex(
    const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
    const IndexComponents<DataType, DistType> &components, rocksdb::DB *db,
    rocksdb::ColumnFamilyHandle *cf, const std::string &dbPath, size_t random_seed,
    void *jobQueue_, void *jobQueueCtx_, SubmitCB submitCb_)
    : VecSimIndexAbstract<DataType, DistType>(abstractInitParams, components),
      idToMetaData(INITIAL_CAPACITY, this->allocator), labelToIdMap(this->allocator), db(db),
      cf(cf), dbPath(dbPath), indexDataGuard(),
      visitedNodesHandlerPool(INITIAL_CAPACITY, this->allocator),
      pendingDeleteIds(this->allocator), num_visited_nodes(0),
      num_visited_nodes_higher_levels(0), stagedInsertUpdates(this->allocator),
      stagedDeleteUpdates(this->allocator), stagedRepairUpdates(this->allocator),
      stagedInsertNeighborUpdates(this->allocator),
      jobQueue(jobQueue_), jobQueueCtx(jobQueueCtx_), SubmitJobsToQueue(submitCb_),
      cacheSegments_(new CacheSegment[NUM_CACHE_SEGMENTS]) {

    M = params->M ? params->M : HNSW_DEFAULT_M;
    M0 = M * 2;
    if (M0 > UINT16_MAX)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");

    efConstruction = params->efConstruction ? params->efConstruction : HNSW_DEFAULT_EF_C;
    efConstruction = std::max(efConstruction, M);
    ef = params->efRuntime ? params->efRuntime : HNSW_DEFAULT_EF_RT;
    epsilon = params->epsilon > 0.0 ? params->epsilon : HNSW_DEFAULT_EPSILON;
    dbOptions = db->GetOptions();
    curElementCount = 0;
    numMarkedDeleted = 0;
    entrypointNode = INVALID_ID;
    maxLevel = 0;

    if (M <= 1)
        throw std::runtime_error("HNSW index parameter M cannot be 1");
    mult = 1 / log(1.0 * M);
    levelGenerator.seed(random_seed);
}

template <typename DataType, typename DistType>
HNSWDiskIndex<DataType, DistType>::~HNSWDiskIndex() {
    // Clear any staged updates before destruction
    stagedInsertUpdates.clear();
    stagedInsertMap.clear();
    stagedDeleteUpdates.clear();
    stagedDeleteMap.clear();
    stagedRepairUpdates.clear();
    stagedRepairMap.clear();
    stagedInsertNeighborUpdates.clear();

    // Clear pending vectors
    pendingDeleteIds.clear();

    // Clear raw vectors in RAM
    rawVectorsInRAM.clear();

    // Clear main data structures
    idToMetaData.clear();
    labelToIdMap.clear();

    // Ensure all memory is properly released
    idToMetaData.shrink_to_fit();

    // Note: db and cf are not owned by this class, so we don't delete them
    // Base class destructor will handle indexCalculator and preprocessors
}


template <typename DataType, typename DistType>
VecSimQueryReply *
HNSWDiskIndex<DataType, DistType>::topKQuery(const void *query_data, size_t k,
                                             VecSimQueryParams *queryParams) const {

    auto rep = new VecSimQueryReply(this->allocator);
    this->lastMode = STANDARD_KNN;

    // Check if index is empty
    if (curElementCount.load(std::memory_order_acquire) == 0 || k == 0) {
        return rep;
    }

    // Preprocess the query
    auto processed_query_ptr = this->preprocessQuery(query_data);
    const void *processed_query = processed_query_ptr.get();

    // const float* v_data = reinterpret_cast<const float*>(query_data);
    // std::cout << "v_data[0]: " << v_data[0] << std::endl;
    // std::cout << "v_data[n]: " << v_data[this->dim - 1] << std::endl;

    // const int8_t* p_data = reinterpret_cast<const int8_t*>(processed_query);
    // std::cout << "p_data[0]: " << static_cast<int>(p_data[0]) << std::endl;
    // std::cout << "p_data[n]: " << static_cast<int>(p_data[this->dim - 1]) << std::endl << std::endl;


    // Get search parameters
    size_t query_ef = this->ef;
    void *timeoutCtx = nullptr;
    if (queryParams) {
        timeoutCtx = queryParams->timeoutCtx;
        if (queryParams->hnswRuntimeParams.efRuntime != 0) {
            query_ef = queryParams->hnswRuntimeParams.efRuntime;
        }
    }

    // Step 1: Find the entry point by searching from top level to bottom level
    idType bottom_layer_ep = searchBottomLayerEP(processed_query, timeoutCtx, &rep->code);
    if (VecSim_OK != rep->code || bottom_layer_ep == INVALID_ID) {
        return rep; // Empty index or error
    }

    // Step 2: Search bottom layer using quantized distances
    auto results = searchLayerLabels(bottom_layer_ep, processed_query, 0, query_ef);

    // Step 3: Re-rank candidates using raw float32 distances for improved recall
    if (useRawData && !results.empty()) {
        rerankWithRawDistances(results, query_data, k);
    }

    while (results.size() > k) {
        results.pop();
    }
    if (!results.empty()) {
        rep->results.resize(results.size());
        for (auto result = rep->results.rbegin(); result != rep->results.rend(); result++) {
            std::tie(result->score, result->id) = results.top();
            results.pop();
        }
        rep->code = VecSim_QueryReply_OK; // Mark as successful since we found results
    }
    return rep;
}

/********************************** Helpers **********************************/

template <typename DataType, typename DistType>
auto HNSWDiskIndex<DataType, DistType>::getNeighborhoods(const vecsim_stl::vector<idType> &ids) const {
    // Create a map to store the neighbors for each label
    std::unordered_map<graphNodeType, vecsim_stl::vector<idType>, GraphNodeHash> neighbors_map;
    // Create a vector of slices to store the keys
    std::vector<GraphKey> graphKeys;
    std::vector<rocksdb::Slice> keys;
    graphKeys.reserve(ids.size() * maxLevel);
    keys.reserve(ids.size() * maxLevel);
    // Iterate over the ids and create the keys
    for (const auto &id : ids) {
        const size_t curMaxLevel = idToMetaData[id].topLevel;
        for (size_t level = 0; level <= curMaxLevel; ++level) {
            graphKeys.emplace_back(id, level);
            keys.emplace_back(graphKeys.back().asSlice());
        }
    }

    // Perform a multi-get operation to retrieve the values for the keys
    std::vector<std::string> values;
    std::vector<rocksdb::ColumnFamilyHandle *> cfs(keys.size(), cf);
    this->db->MultiGet(rocksdb::ReadOptions(), cfs, keys, &values);

    // Iterate over the values and fill the neighbors map
    for (size_t i = 0; i < graphKeys.size(); ++i) {
        const auto &key = graphKeys[i];
        const auto &value = values[i];

        // Parse the value using new format: [vector_data][neighbor_count][neighbor_ids...]
        vecsim_stl::vector<idType> neighbors(this->allocator);
        deserializeGraphValue(value, neighbors);

        // Store the neighbors in the map
        neighbors_map[key.node()] = std::move(neighbors);
    }
    return neighbors_map;
}

template <typename DataType, typename DistType>
idType HNSWDiskIndex<DataType, DistType>::getNeighborsByHeuristic2(
    candidatesList<DistType> &top_candidates, const size_t M) const {
    if (top_candidates.size() < M) {
        return std::min_element(top_candidates.begin(), top_candidates.end(),
                                [](const auto &a, const auto &b) { return a.first < b.first; })
            ->second;
    }
    getNeighborsByHeuristic2_internal<false>(top_candidates, M, nullptr);
    return top_candidates.front().second;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::getNeighborsByHeuristic2(
    candidatesList<DistType> &top_candidates, const size_t M,
    vecsim_stl::vector<idType> &removed_candidates) const {
    getNeighborsByHeuristic2_internal<true>(top_candidates, M, &removed_candidates);
}

template <typename DataType, typename DistType>
template <bool record_removed>
void HNSWDiskIndex<DataType, DistType>::getNeighborsByHeuristic2_internal(
    candidatesList<DistType> &top_candidates, const size_t M,
    vecsim_stl::vector<idType> *removed_candidates) const {
    if (top_candidates.size() < M) {
        return;
    }

    candidatesList<DistType> return_list(this->allocator);
    vecsim_stl::vector<const void *> cached_vectors(this->allocator);
    return_list.reserve(M);
    cached_vectors.reserve(M);
    if constexpr (record_removed) {
        removed_candidates->reserve(top_candidates.size());
    }

    // Sort the candidates by their distance (we don't mind the secondary order (the internal id))
    std::sort(top_candidates.begin(), top_candidates.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    auto current_pair = top_candidates.begin();
    for (; current_pair != top_candidates.end() && return_list.size() < M; ++current_pair) {
        DistType candidate_to_query_dist = current_pair->first;
        bool good = true;
        const void *curr_vector = getDataByInternalId(current_pair->second); // TODO: get quantized

        // a candidate is "good" to become a neighbour, unless we find
        // another item that was already selected to the neighbours set which is closer
        // to both q and the candidate than the distance between the candidate and q.
        for (size_t i = 0; i < return_list.size(); i++) {
            DistType candidate_to_selected_dist =
                this->calcDistance(cached_vectors[i], curr_vector); // TODO: quantized
            if (candidate_to_selected_dist < candidate_to_query_dist) {
                if constexpr (record_removed) {
                    removed_candidates->push_back(current_pair->second);
                }
                good = false;
                break;
            }
        }
        if (good) {
            cached_vectors.push_back(curr_vector);
            return_list.push_back(*current_pair);
        }
    }

    if constexpr (record_removed) {
        for (; current_pair != top_candidates.end(); ++current_pair) {
            removed_candidates->push_back(current_pair->second);
        }
    }

    top_candidates.swap(return_list);
}

template <typename DataType, typename DistType>
int HNSWDiskIndex<DataType, DistType>::addVector(
    const void *vector, labelType label
) {

    // Atomically get a unique element ID - this is the critical fix for Race #1
    // fetch_add returns the OLD value before incrementing, giving us a unique ID
    idType newElementId = static_cast<idType>(curElementCount.fetch_add(1, std::memory_order_acq_rel));

    // Store raw vector in RAM first (until written to disk)
    // We need to store the original vector before preprocessing
    // NOTE: In batchless mode, we still use rawVectorsInRAM so other concurrent jobs can access
    // the raw vectors of vectors that haven't been written to disk yet
    const char* raw_data = reinterpret_cast<const char*>(vector);
    {
        std::lock_guard<std::shared_mutex> lock(rawVectorsGuard);
        rawVectorsInRAM[newElementId] = std::string(raw_data, this->inputBlobSize);
    }
    // Preprocess the vector
    ProcessedBlobs processedBlobs = this->preprocess(vector);

    // Store the processed vector in memory (protected by vectorsGuard)
    {
        std::lock_guard<std::shared_mutex> lock(vectorsGuard);
        size_t containerId = this->vectors->size();
        this->vectors->addElement(processedBlobs.getStorageBlob(), containerId);
    }

    // Create new element ID and metadata
    size_t elementMaxLevel = getRandomLevel(mult);
    DiskElementMetaData new_element(label, elementMaxLevel);

    // Ensure capacity for the new element ID (protected by indexDataGuard)
    {
        std::lock_guard<std::shared_mutex> lock(indexDataGuard);
        if (newElementId >= indexCapacity()) {
            size_t new_cap = ((newElementId + this->blockSize) / this->blockSize) * this->blockSize;
            visitedNodesHandlerPool.resize(new_cap);
            idToMetaData.resize(new_cap);
            labelToIdMap.reserve(new_cap);
        }

        // Store metadata immediately
        idToMetaData[newElementId] = new_element;
        labelToIdMap[label] = newElementId;
    }

    // Resize visited nodes handler pool to accommodate new elements
    // Use load() to read atomic value
    visitedNodesHandlerPool.resize(curElementCount.load(std::memory_order_acquire));

    // Each vector is processed immediately and written to disk
    // Get entry point info
    idType currentEntryPoint;
    size_t currentMaxLevel;
    {
        std::shared_lock<std::shared_mutex> lock(indexDataGuard);
        currentEntryPoint = entrypointNode;
        currentMaxLevel = maxLevel;
    }

    // Handle first vector (becomes entry point)
    if (currentEntryPoint == INVALID_ID) {
        bool becameEntryPoint = false;
        std::unique_lock<std::shared_mutex> lock(indexDataGuard);
        if (entrypointNode == INVALID_ID) {
            entrypointNode = newElementId;
            maxLevel = elementMaxLevel;
            becameEntryPoint = true;
        } else {
            // Another thread already set the entry point - update our local state
            currentEntryPoint = entrypointNode;
            currentMaxLevel = maxLevel;
        }
        lock.unlock();
        // If we became the entry point, write initial vector to disk
        if (becameEntryPoint) {
            // Write initial vector to disk with empty neighbors
            vecsim_stl::vector<idType> emptyNeighbors(this->allocator);
            for (size_t level = 0; level <= elementMaxLevel; level++) {
                GraphKey graphKey(newElementId, level);
                std::string value = serializeGraphValue(vector, emptyNeighbors);
                auto writeOptions = rocksdb::WriteOptions();
                writeOptions.disableWAL = true;
                db->Put(writeOptions, cf, graphKey.asSlice(), value);
            }
            // Remove raw vector from RAM now that it's on disk
            {
                std::lock_guard<std::shared_mutex> rawLock(rawVectorsGuard);
                rawVectorsInRAM.erase(newElementId);
            }
            return 1;
        }
        // Fall through to normal graph insertion with updated entry point
    }

    // Check if we have a job queue for async processing
    if (SubmitJobsToQueue != nullptr) {
        // Multi-threaded: submit job for async processing
        std::string rawVectorCopy(raw_data, this->inputBlobSize);
        std::string processedVectorCopy(
            reinterpret_cast<const char *>(processedBlobs.getStorageBlob()),
            this->dataSize);

        auto *job = new (this->allocator) HNSWDiskSingleInsertJob(
            this->allocator, newElementId, elementMaxLevel, std::move(rawVectorCopy),
            std::move(processedVectorCopy),
            HNSWDiskIndex<DataType, DistType>::executeSingleInsertJobWrapper, this);

        submitSingleJob(job);
    } else {
        // Single-threaded: execute inline (batching controlled by diskWriteBatchThreshold)
        executeGraphInsertionCore(newElementId, elementMaxLevel, currentEntryPoint,
                                  currentMaxLevel, vector, processedBlobs.getStorageBlob());
    }

    return 1;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::flushStagedGraphUpdates(
    vecsim_stl::vector<GraphUpdate>& graphUpdates,
    vecsim_stl::vector<NeighborUpdate>& neighborUpdates) {
    if (graphUpdates.empty() && neighborUpdates.empty()) {
        return;
    }

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;
    // Write graph updates first (so they're available when processing neighbor updates)
    rocksdb::WriteBatch graphBatch;

    // Buffer for retrieving raw vectors (in case they're on disk)
    std::vector<char> raw_vector_buffer(this->inputBlobSize);

    // First, handle new node insertions
    for (const auto &update : graphUpdates) {
        auto newKey = GraphKey(update.node_id, update.level);

        // If neighbors list is empty, this is a deletion - remove the key from disk
        if (update.neighbors.empty()) {
            graphBatch.Delete(cf, newKey.asSlice());
            continue;
        }

        // Get raw vector data (use internal version since we already hold the lock or
        // are being called from a context where locking is managed by the caller)
        if (!getRawVectorInternal(update.node_id, raw_vector_buffer.data())) {
            this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                        "WARNING: Skipping graph update for node %u at level %zu - no raw vector data available",
                        update.node_id, update.level);
            continue;
        }

        // Serialize with new format: [raw_vector_data][neighbor_count][neighbor_ids...]
        std::string graph_value = serializeGraphValue(raw_vector_buffer.data(), update.neighbors);

        graphBatch.Put(cf, newKey.asSlice(), graph_value);
    }

    // Write graph updates to disk first
    rocksdb::Status graph_status = this->db->Write(writeOptions, &graphBatch);
    if (!graph_status.ok()) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING, "ERROR: Failed to write graph updates batch: %s",
                 graph_status.ToString().c_str());
        return;
    }

    // Then, handle neighbor list updates for existing nodes
    // Group neighbor updates by node and level for efficient processing
    std::unordered_map<idType, std::unordered_map<size_t, vecsim_stl::vector<idType>>> neighborUpdatesByNode;

    for (const auto& update : neighborUpdates) {
        auto& levelMap = neighborUpdatesByNode[update.node_id];
        auto it = levelMap.find(update.level);
        if (it == levelMap.end()) {
            levelMap.emplace(update.level, vecsim_stl::vector<idType>(this->allocator));
            it = levelMap.find(update.level);
        }
        it->second.push_back(update.new_neighbor_id);
    }

    // Use a separate batch for neighbor updates
    rocksdb::WriteBatch neighborBatch;

    // Process each node's neighbor updates (reuse the buffer from above)
    for (const auto& [node_id, levelMap] : neighborUpdatesByNode) {
        for (const auto& [level, newNeighbors] : levelMap) {
            // Read existing graph value from disk
            GraphKey neighborKey(node_id, level);
            std::string existing_graph_value;
            rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, neighborKey.asSlice(), &existing_graph_value);

            vecsim_stl::vector<idType> updated_neighbors(this->allocator);

            if (status.ok()) {
                // Parse existing neighbors using new format
                deserializeGraphValue(existing_graph_value, updated_neighbors);
            }

            // Add new neighbors (avoiding duplicates)
            for (idType new_neighbor : newNeighbors) {
                bool found = false;
                for (idType existing : updated_neighbors) {
                    if (existing == new_neighbor) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    updated_neighbors.push_back(new_neighbor);
                }
            }

        getRawVectorInternal(node_id, raw_vector_buffer.data());

        // Serialize with new format and add to batch
        std::string graph_value = serializeGraphValue(raw_vector_buffer.data(), updated_neighbors);
        neighborBatch.Put(cf, neighborKey.asSlice(), graph_value);
        }
    }

    // Write neighbor updates batch
    rocksdb::Status neighbor_status = this->db->Write(writeOptions, &neighborBatch);
    if (!neighbor_status.ok()) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING, "ERROR: Failed to write neighbor updates batch: %s",
                 neighbor_status.ToString().c_str());
    }

    // Clear staged updates after successful flush
    graphUpdates.clear();
    neighborUpdates.clear();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::revisitNeighborConnections(
    idType new_node_id, idType selected_neighbor, size_t level, DistType distance,
    vecsim_stl::vector<uint64_t> &modifiedNodes) {
    // Read the neighbor's current neighbor list from cache (source of truth)
    vecsim_stl::vector<idType> existing_neighbors(this->allocator);
    getNeighborsFromCache(selected_neighbor, level, existing_neighbors);

    // Collect all candidates: existing neighbors + new node
    candidatesList<DistType> candidates(this->allocator);
    candidates.reserve(existing_neighbors.size() + 1);

    // Add the new node with its pre-calculated distance
    candidates.emplace_back(distance, new_node_id);

    // Add existing neighbors with their distances to the selected neighbor
    const void *selected_neighbor_data = getDataByInternalId(selected_neighbor);
    for (idType existing_neighbor_id : existing_neighbors) {
        const void *existing_data = getDataByInternalId(existing_neighbor_id);
        DistType existing_dist = this->calcDistance(existing_data, selected_neighbor_data);
        candidates.emplace_back(existing_dist, existing_neighbor_id);
    }

    // Apply the neighbor selection heuristic
    size_t max_M_cur = level ? M : M0;
    vecsim_stl::vector<idType> removed_candidates(this->allocator);
    getNeighborsByHeuristic2(candidates, max_M_cur, removed_candidates);

    // Extract selected neighbor IDs
    vecsim_stl::vector<idType> new_neighbors(this->allocator);
    new_neighbors.reserve(candidates.size());
    for (const auto &candidate : candidates) {
        new_neighbors.push_back(candidate.second);
    }

    // Update cache with the new neighbor list
    setNeighborsInCache(selected_neighbor, level, new_neighbors);
    modifiedNodes.push_back(makeRepairKey(selected_neighbor, level));
}

template <typename DataType, typename DistType>
idType HNSWDiskIndex<DataType, DistType>::searchBottomLayerEP(const void *query_data,
                                                              void *timeoutCtx,
                                                              VecSimQueryReply_Code *rc) const {
    if (rc) *rc = VecSim_QueryReply_OK;

    // Use safeGetEntryPointState to read entry point with proper locking
    auto [curr_element, max_level] = safeGetEntryPointState();
    if (curr_element == INVALID_ID)
        return curr_element; // index is empty.

    DistType cur_dist = this->calcDistance(query_data, getDataByInternalId(curr_element));
    for (size_t level = max_level; level > 0 && curr_element != INVALID_ID; --level) {
        greedySearchLevel<true>(query_data, level, curr_element, cur_dist);
    }
    return curr_element;
}

/********************************** Helper Methods **********************************/

// Serialize GraphKey value: [raw_vector_data][neighbor_count][neighbor_ids...]
template <typename DataType, typename DistType>
std::string HNSWDiskIndex<DataType, DistType>::serializeGraphValue(
    const void* vector_data, const vecsim_stl::vector<idType>& neighbors) const {

    size_t neighbor_count = neighbors.size();
    size_t total_size = this->inputBlobSize + sizeof(size_t) + neighbor_count * sizeof(idType);

    std::string result;
    result.resize(total_size);

    char* ptr = result.data();

    // Copy raw vector data (original, unprocessed)
    std::memcpy(ptr, vector_data, this->inputBlobSize);
    ptr += this->inputBlobSize;

    // Copy neighbor count
    std::memcpy(ptr, &neighbor_count, sizeof(size_t));
    ptr += sizeof(size_t);

    // Copy neighbor IDs
    if (neighbor_count > 0) {
        std::memcpy(ptr, neighbors.data(), neighbor_count * sizeof(idType));
    }

    return result;
}



// Deserialize GraphKey value to extract neighbors
template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::deserializeGraphValue(
    const std::string& value, vecsim_stl::vector<idType>& neighbors) const {

    neighbors.clear();

    if (value.size() < this->inputBlobSize + sizeof(size_t)) {
        // Invalid value format
        return;
    }

    const char* ptr = value.data();
    
    // Skip raw vector data
    ptr += this->inputBlobSize;

    // Read neighbor count
    size_t neighbor_count;
    std::memcpy(&neighbor_count, ptr, sizeof(size_t));
    ptr += sizeof(size_t);

    // Read neighbor IDs
    if (neighbor_count > 0 && value.size() >= this->inputBlobSize + sizeof(size_t) + neighbor_count * sizeof(idType)) {
        neighbors.resize(neighbor_count);
        std::memcpy(neighbors.data(), ptr, neighbor_count * sizeof(idType));
    }
}

// Extract raw vector data pointer from GraphKey value
template <typename DataType, typename DistType>
const void* HNSWDiskIndex<DataType, DistType>::getVectorFromGraphValue(const std::string& value) const {
    if (value.size() < this->inputBlobSize) {
        return nullptr;
    }
    return value.data();
}

/********************************** Stub Implementations **********************************/

template <typename DataType, typename DistType>
const void *HNSWDiskIndex<DataType, DistType>::getDataByInternalId(idType id) const {
    assert(id < curElementCount.load(std::memory_order_acquire));

    // Acquire shared lock to prevent concurrent resize of vectors container
    std::shared_lock<std::shared_mutex> lock(vectorsGuard);
    const void* result = this->vectors->getElement(id);
    if (result != nullptr) {
        return result;
    }

    throw std::runtime_error("Vector data not found for id " + std::to_string(id));
}

template <typename DataType, typename DistType>
bool HNSWDiskIndex<DataType, DistType>::getRawVector(idType id, void* output_buffer) const {
    size_t elementCount = curElementCount.load(std::memory_order_acquire);
    if (id >= elementCount) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                 "WARNING: getRawVector called with invalid id %u (current count: %zu)",
                 id, elementCount);
                 return false;
    }

    // First check RAM (for vectors not yet flushed) - protected by shared lock
    {
        std::shared_lock<std::shared_mutex> lock(rawVectorsGuard);
        auto it = rawVectorsInRAM.find(id);
        if (it != rawVectorsInRAM.end()) {
            const char* data_ptr = it->second.data();
            std::memcpy(output_buffer, data_ptr, this->inputBlobSize);
            return true;
        }
    }

    // If not in RAM or cache, retrieve from disk (RocksDB is thread-safe)
    GraphKey graphKey(id, 0);
    std::string level0_graph_value;
    rocksdb::Status status =
        db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &level0_graph_value);

    if (status.ok()) {
        // Extract vector data
        const void* vector_data = getVectorFromGraphValue(level0_graph_value);
        if (vector_data != nullptr) {
            // Must copy to output buffer since level0_graph_value will be destroyed
            std::memcpy(output_buffer, vector_data, this->inputBlobSize);
            return true;
        }
    } else if (status.IsNotFound()) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                 "WARNING: Raw vector not found in RAM or on disk for id %u (isMarkedDeleted: %d)",
                 id, isMarkedDeleted(id));
    } else {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                  "WARNING: Failed to retrieve raw vector for id %u: %s", id,
                  status.ToString().c_str());
    }
    return false;

}

// Internal version for use during flush operations
template <typename DataType, typename DistType>
bool HNSWDiskIndex<DataType, DistType>::getRawVectorInternal(idType id, void* output_buffer) const {
    size_t elementCount = curElementCount.load(std::memory_order_acquire);
    if (id >= elementCount) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                 "WARNING: getRawVectorInternal called with invalid id %u (current count: %zu)",
                 id, elementCount);
                 return false;
    }

    // First check RAM (for vectors not yet flushed) - protected by shared lock
    {
        std::shared_lock<std::shared_mutex> lock(rawVectorsGuard);
        auto it = rawVectorsInRAM.find(id);
        if (it != rawVectorsInRAM.end()) {
            const char* data_ptr = it->second.data();
            std::memcpy(output_buffer, data_ptr, this->inputBlobSize);
            return true;
        }
    }

    // If not in RAM, retrieve from disk (NO LOCK - caller is expected to hold lock)
    GraphKey graphKey(id, 0);
    std::string level0_graph_value;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &level0_graph_value);

    if (status.ok()) {
        // Extract vector data
        const void* vector_data = getVectorFromGraphValue(level0_graph_value);
        if (vector_data != nullptr) {
            std::memcpy(output_buffer, vector_data, this->inputBlobSize);
            return true;
        }
    } else if (status.IsNotFound()) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                 "WARNING: Raw vector not found in RAM or on disk for id %u (isMarkedDeleted: %d)",
                 id, isMarkedDeleted(id));
    } else {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                  "WARNING: Failed to retrieve raw vector for id %u: %s", id,
                  status.ToString().c_str());
    }
    return false;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::rerankWithRawDistances(
    vecsim_stl::updatable_max_heap<DistType, labelType>& candidates,
    const void* query_data, size_t k) const {

    if (candidates.empty()) {
        return;
    }

    // Extract all candidates from the heap
    std::vector<std::pair<DistType, labelType>> candidate_list;
    candidate_list.reserve(candidates.size());
    while (!candidates.empty()) {
        candidate_list.push_back(candidates.top());
        candidates.pop();
    }

    // Buffer for raw vector data
    std::vector<char> raw_vector_buffer(this->inputBlobSize);

    // Recalculate distances using raw float32 vectors
    for (auto& candidate : candidate_list) {
        labelType label = candidate.second;

        // Find internal ID for this label
        auto it = labelToIdMap.find(label);
        if (it == labelToIdMap.end()) {
            // Label not found (might have been deleted), keep original distance
            continue;
        }
        idType internal_id = it->second;

        // Get raw vector and recalculate distance
        if (getRawVector(internal_id, raw_vector_buffer.data())) {
            DistType raw_dist = this->calcDistanceRaw(query_data, raw_vector_buffer.data());
            candidate.first = raw_dist;
        } else {
            this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                "Failed to fetch all raw vectors during reranking, skipping reranking.");
            return;
        }
    }

    // Re-insert all candidates with updated distances
    for (const auto& candidate : candidate_list) {
        candidates.emplace(candidate.first, candidate.second);
    }
}

template <typename DataType, typename DistType>
vecsim_stl::updatable_max_heap<DistType, idType>
HNSWDiskIndex<DataType, DistType>::searchLayer(idType ep_id, const void *data_point, size_t layer,
                                           size_t ef) const {

    std::unordered_set<idType> visited_set;
    visited_set.reserve(10000);

    vecsim_stl::updatable_max_heap<DistType, idType> top_candidates(this->allocator);
    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    DistType lowerBound;
    if (!isMarkedDeleted(ep_id)) {
        DistType dist = this->calcDistance(data_point, getDataByInternalId(ep_id));
        lowerBound = dist;
        top_candidates.emplace(dist, ep_id);
        candidate_set.emplace(-dist, ep_id);
    } else {
        lowerBound = std::numeric_limits<DistType>::max();
        candidate_set.emplace(-lowerBound, ep_id);
    }

    visited_set.insert(ep_id);
    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();

        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() >= ef) {
            break;
        }
        candidate_set.pop();

        processCandidate(curr_el_pair.second, data_point, layer, ef,
                         &visited_set, top_candidates,
                         candidate_set, lowerBound);
    }

    return top_candidates;
}

template <typename DataType, typename DistType>
vecsim_stl::updatable_max_heap<DistType, labelType>
HNSWDiskIndex<DataType, DistType>::searchLayerLabels(idType ep_id, const void *data_point, size_t layer,
                                           size_t ef) const {
    std::unordered_set<idType> visited_set;
    visited_set.reserve(10000);

    vecsim_stl::updatable_max_heap<DistType, labelType> top_candidates(this->allocator);
    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    DistType lowerBound;
    if (!isMarkedDeleted(ep_id)) {
        DistType dist = this->calcDistance(data_point, getDataByInternalId(ep_id));
        lowerBound = dist;
        top_candidates.emplace(dist, getExternalLabel(ep_id));
        candidate_set.emplace(-dist, ep_id);
    } else {
        lowerBound = std::numeric_limits<DistType>::max();
        candidate_set.emplace(-lowerBound, ep_id);
    }

    visited_set.insert(ep_id);
    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();

        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() >= ef) {
            break;
        }
        candidate_set.pop();

        // Pass internal ID to processCandidate - getNeighbors expects idType, not labelType
        // The emplaceHeap overload handles conversion to labelType for top_candidates
        processCandidate(curr_el_pair.second, data_point, layer, ef,
                         &visited_set, top_candidates,
                         candidate_set, lowerBound);
    }

    return top_candidates;
}

template <typename DataType, typename DistType>
labelType HNSWDiskIndex<DataType, DistType>::getExternalLabel(idType id) const {
    if (id >= idToMetaData.size()) {
        return INVALID_LABEL;
    }
    return idToMetaData[id].label;
}

template <typename DataType, typename DistType>
bool HNSWDiskIndex<DataType, DistType>::isMarkedDeleted(idType id) const {
    return id < idToMetaData.size() && isMarkedAs<DELETE_MARK>(id);
}

// Lock-free version for hot paths - still needs bounds checking
template <typename DataType, typename DistType>
bool HNSWDiskIndex<DataType, DistType>::isMarkedDeletedUnsafe(idType id) const {
    // Must check bounds even in unsafe version - accessing out of bounds is UB
    if (id >= idToMetaData.size()) {
        return true;  // Treat out-of-bounds as deleted (won't be added to results)
    }
    return isMarkedAsUnsafe<DELETE_MARK>(id);
}

template <typename DataType, typename DistType>
bool HNSWDiskIndex<DataType, DistType>::isMarkedDeleted(labelType id) const {
    auto it = labelToIdMap.find(id);
    if (it == labelToIdMap.end()) {
        return true;
    }
    return isMarkedAs<DELETE_MARK>(it->second);
}

template <typename DataType, typename DistType>
std::pair<idType, size_t> HNSWDiskIndex<DataType, DistType>::safeGetEntryPointState() const {
    std::shared_lock<std::shared_mutex> lock(indexDataGuard);
    return std::make_pair(entrypointNode, maxLevel);
}

template <typename DataType, typename DistType>
template <bool running_query>
void HNSWDiskIndex<DataType, DistType>::greedySearchLevel(const void *data_point, size_t level,
                                                          idType &curr_element,
                                                          DistType &cur_dist) const {
    // Hold shared lock for entire search to prevent idToMetaData resize during isMarkedDeletedUnsafe calls
    std::shared_lock<std::shared_mutex> indexLock(indexDataGuard);

    bool changed;
    idType bestCand = curr_element;
    idType bestNonDeletedCand = bestCand;
    size_t visited_count = 0;

    do {
        changed = false;

        // Read neighbors (checks staged updates first, then RocksDB)
        vecsim_stl::vector<idType> neighbors(this->allocator);
        getNeighbors(bestCand, level, neighbors);

        if (neighbors.empty()) {
            // No neighbors found for this node at this level, stop searching
            break;
        }

        // Check each neighbor to find a better candidate
        for (size_t i = 0; i < neighbors.size(); i++) {
            idType candidate = neighbors[i];

            assert (candidate < curElementCount && "candidate error: out of index range");
            if (running_query) {
                visited_count++;
            }
            // Calculate distance to this candidate
            DistType d = this->calcDistance(data_point, getDataByInternalId(candidate));

            // If this candidate is closer, update our best candidate
            if (d < cur_dist) {
                cur_dist = d;
                bestCand = candidate;
                changed = true;
                // Use lock-free version - candidate was already validated by getNeighbors
                if (!running_query && !isMarkedDeletedUnsafe(candidate)) {
                    bestNonDeletedCand = bestCand;
                }
            }
        }

    } while (changed);

    // Update the current element to the best candidate found
    if (!running_query) {
        curr_element = bestNonDeletedCand;
    } else {
        curr_element = bestCand;
        // Update the counter for higher level visited nodes
        num_visited_nodes_higher_levels.fetch_add(visited_count, std::memory_order_relaxed);
    }
}

template <typename DataType, typename DistType>
candidatesLabelsMaxHeap<DistType> *
HNSWDiskIndex<DataType, DistType>::getNewMaxPriorityQueue() const {
    // Use updatable_max_heap to allow updating distances for labels
    return new (this->allocator)
            vecsim_stl::updatable_max_heap<DistType, labelType>(this->allocator);
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::emplaceToHeap(
    vecsim_stl::abstract_priority_queue<DistType, idType> &heap, DistType dist, idType id) const {
    heap.emplace(dist, id);
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::emplaceToHeap(
    vecsim_stl::abstract_priority_queue<DistType, labelType> &heap, DistType dist,
    idType id) const {
    heap.emplace(dist, getExternalLabel(id));
}

template <typename DataType, typename DistType>
VisitedNodesHandler *HNSWDiskIndex<DataType, DistType>::getVisitedList() const {
    return visitedNodesHandlerPool.getAvailableVisitedNodesHandler();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::returnVisitedList(
    VisitedNodesHandler *visited_nodes_handler) const {
    visitedNodesHandlerPool.returnVisitedNodesHandlerToPool(visited_nodes_handler);
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(levelGenerator)) * reverse_size;
    return (size_t)r;
}

template <typename DataType, typename DistType>
template <typename Identifier>
void HNSWDiskIndex<DataType, DistType>::processCandidate(
    idType curNodeId, const void *data_point, size_t level, size_t ef, std::unordered_set<idType> *visited_set,
    vecsim_stl::updatable_max_heap<DistType, Identifier> &top_candidates,
    candidatesMaxHeap<DistType> &candidate_set, DistType &lowerBound) const {
    assert(visited_set != nullptr);
    // Add neighbors to candidate set for further exploration
    vecsim_stl::vector<idType> neighbors(this->allocator);
    getNeighbors(curNodeId, level, neighbors);

    if (!neighbors.empty()) {
        num_visited_nodes.fetch_add(1, std::memory_order_relaxed);

        for (idType candidate_id : neighbors) {
            // Skip invalid neighbors
            assert(candidate_id < curElementCount.load(std::memory_order_acquire));

            if (visited_set->find(candidate_id) != visited_set->end()) {
                continue;
            }
            visited_set->insert(candidate_id);
            // TODO: possibly use cached raw vectors
            DistType cur_dist =
                this->calcDistance(data_point, getDataByInternalId(candidate_id));
            if (lowerBound > cur_dist || top_candidates.size() < ef) {

                candidate_set.emplace(-cur_dist, candidate_id);

                // Insert the candidate to the top candidates heap only if it is not marked as
                // deleted. Use lock-free version since candidate_id was already bounds-checked
                // by getNeighbors (which filters invalid IDs).
                if (!isMarkedDeletedUnsafe(candidate_id))
                    emplaceHeap(top_candidates, cur_dist, candidate_id);

                if (top_candidates.size() > ef)
                    top_candidates.pop();

                // If we have marked deleted elements, we need to verify that `top_candidates` is
                // not empty (since we might have not added any non-deleted element yet).
                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }
}

template <typename DataType, typename DistType>
VecSimQueryReply *
HNSWDiskIndex<DataType, DistType>::rangeQuery(const void *query_data, double radius,
                                              VecSimQueryParams *queryParams) const {
    // TODO: Implement range query
    auto rep = new VecSimQueryReply(this->allocator);
    return rep;
}

template <typename DataType, typename DistType>
VecSimIndexDebugInfo HNSWDiskIndex<DataType, DistType>::debugInfo() const {
    // TODO: Implement debug info
    VecSimIndexDebugInfo info = {};
    return info;
}

template <typename DataType, typename DistType>
VecSimDebugInfoIterator *HNSWDiskIndex<DataType, DistType>::debugInfoIterator() const {
    // TODO: Implement debug info iterator
    return nullptr;
}

template <typename DataType, typename DistType>
VecSimIndexBasicInfo HNSWDiskIndex<DataType, DistType>::basicInfo() const {
    // TODO: Implement basic info
    VecSimIndexBasicInfo info = {};
    return info;
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
HNSWDiskIndex<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                    VecSimQueryParams *queryParams) const {
    // TODO: Implement batch iterator
    return nullptr;
}

template <typename DataType, typename DistType>
bool HNSWDiskIndex<DataType, DistType>::preferAdHocSearch(size_t subsetSize, size_t k,
                                                          bool initial_check) const {
    // TODO: Implement ad-hoc search preference logic
    return false;
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::indexCapacity() const {
    return idToMetaData.size();
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::indexSize() const {
    // Return active element count (curElementCount never decreases in disk mode,
    // so we subtract numMarkedDeleted to get the actual number of active elements)
    return curElementCount.load(std::memory_order_acquire) - this->numMarkedDeleted;
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::indexLabelCount() const {
    return labelToIdMap.size();
}

/********************************** Helper Methods **********************************/

template <typename DataType, typename DistType>
inline void HNSWDiskIndex<DataType, DistType>::emplaceHeap(
    vecsim_stl::abstract_priority_queue<DistType, idType> &heap, DistType dist,
    idType id) const {
        heap.emplace(dist, id);
    }

template <typename DataType, typename DistType>
inline void HNSWDiskIndex<DataType, DistType>::emplaceHeap(
    vecsim_stl::abstract_priority_queue<DistType, labelType> &heap, DistType dist,
    idType id) const {
        heap.emplace(dist, getExternalLabel(id));
    }

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::getNeighbors(idType nodeId, size_t level,
                                                     vecsim_stl::vector<idType> &result) const {
    // Clear the result vector first
    result.clear();

    uint64_t lookup_key = makeRepairKey(nodeId, level);
    bool foundInCache = false;

    // Step 1: Check cache first (source of truth for pending updates)
    {
        size_t segment = getSegmentIndex(lookup_key);
        auto& cacheSegment = cacheSegments_[segment];
        std::shared_lock<std::shared_mutex> cacheLock(cacheSegment.guard);
        auto it = cacheSegment.cache.find(lookup_key);
        if (it != cacheSegment.cache.end()) {
            result.reserve(it->second.size());
            for (idType id : it->second) {
                result.push_back(id);
            }
            foundInCache = true;
        } else if (cacheSegment.newNodes.find(lookup_key) != cacheSegment.newNodes.end()) {
            // New node not yet connected - return empty
            foundInCache = true;  // Treat as found (empty is valid)
        }
    }

    // Step 2: Check delete staging area (for deletion operations)
    if (!foundInCache) {
        std::shared_lock<std::shared_mutex> lock(stagedUpdatesGuard);
        auto delete_it = stagedDeleteMap.find(lookup_key);
        if (delete_it != stagedDeleteMap.end()) {
            const auto &update = stagedDeleteUpdates[delete_it->second];
            result.reserve(update.neighbors.size());
            for (size_t i = 0; i < update.neighbors.size(); i++) {
                result.push_back(update.neighbors[i]);
            }
            foundInCache = true;
        }

        // Also check staged repair updates (already cleaned neighbors waiting to be flushed)
        if (!foundInCache) {
            auto repair_it = stagedRepairMap.find(lookup_key);
            if (repair_it != stagedRepairMap.end()) {
                const auto &update = stagedRepairUpdates[repair_it->second];
                result.reserve(update.neighbors.size());
                for (size_t i = 0; i < update.neighbors.size(); i++) {
                    result.push_back(update.neighbors[i]);
                }
                foundInCache = true;
            }
        }
    }

    // Step 3: If not found in cache or staging, read from disk
    if (!foundInCache) {
        GraphKey graphKey(nodeId, level);
        std::string graph_value;
        rocksdb::Status status =
            db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &graph_value);

        if (status.ok()) {
            deserializeGraphValue(graph_value, result);
        }
    }

    // Note: We do NOT filter deleted nodes here. During search, we need to explore
    // through deleted nodes to find non-deleted neighbors. The filtering happens in
    // processCandidate when adding to top_candidates (via isMarkedDeletedUnsafe check).
    //
    // Lazy repair for deleted nodes is handled separately when we detect stale edges
    // during graph maintenance operations, not during search.
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::getNeighborsCount(idType nodeId, size_t level) const {
    uint64_t lookup_key = makeRepairKey(nodeId, level);

    // Step 1: Check cache first (source of truth for pending updates)
    {
        size_t segment = getSegmentIndex(lookup_key);
        auto& cacheSegment = cacheSegments_[segment];
        std::shared_lock<std::shared_mutex> cacheLock(cacheSegment.guard);
        auto it = cacheSegment.cache.find(lookup_key);
        if (it != cacheSegment.cache.end()) {
            return it->second.size();
        }
        // Check if this is a new node (never written to disk)
        if (cacheSegment.newNodes.find(lookup_key) != cacheSegment.newNodes.end()) {
            return 0;  // New node not yet connected
        }
    }

    // Step 2: If not found in cache, check disk
    GraphKey graphKey(nodeId, level);
    std::string graph_value;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &graph_value);

    if (status.ok()) {
        const char* ptr = graph_value.data();
        ptr += this->inputBlobSize;
        size_t neighbor_count = *reinterpret_cast<const size_t*>(ptr);
        return neighbor_count;
    }

    return 0;
}

/********************************** Multi-threaded Job Execution **********************************/

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::submitSingleJob(AsyncJob *job) {
    this->SubmitJobsToQueue(this->jobQueue, this->jobQueueCtx, &job, &job->Execute, 1);
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::submitJobs(vecsim_stl::vector<AsyncJob *> &jobs) {
    vecsim_stl::vector<JobCallback> callbacks(jobs.size(), this->allocator);
    for (size_t i = 0; i < jobs.size(); i++) {
        callbacks[i] = jobs[i]->Execute;
    }
    this->SubmitJobsToQueue(this->jobQueue, this->jobQueueCtx, jobs.data(), callbacks.data(),
                            jobs.size());
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::repairNeighborConnections(
    idType neighbor_id, size_t level, idType deleted_id,
    const vecsim_stl::vector<idType> &deleted_node_neighbors,
    vecsim_stl::vector<idType> &neighbor_neighbors) {

    // ===== Graph Repair Strategy =====
    // When deleting a node, we need to repair its neighbors' connections to maintain
    // graph quality and navigability. We use a heuristic-based approach similar to
    // the regular HNSW implementation (see hnsw.h::repairConnectionsForDeletion).
    //
    // Strategy:
    // 1. Collect candidates: existing neighbors + deleted node's neighbors
    // 2. Calculate distances using quantized vectors (fast, in-memory)
    // 3. Apply getNeighborsByHeuristic2 to select best neighbors
    // 4. This ensures high-quality connections that maintain search performance

    size_t max_M = (level == 0) ? M0 : M;

    // Build candidate set with distances
    // Candidates include: existing neighbors (minus deleted) + deleted node's neighbors
    candidatesList<DistType> candidates(this->allocator);
    const void *neighbor_data = getDataByInternalId(neighbor_id);

    // Use a hash set to track candidate IDs for O(1) duplicate detection
    std::unordered_set<idType> candidate_ids;
    size_t elementCount = curElementCount.load(std::memory_order_acquire);

    // Add existing neighbors (excluding the deleted node) with their distances
    for (idType nn : neighbor_neighbors) {
        if (nn != deleted_id && nn < elementCount && !isMarkedDeleted(nn)) {
            const void *nn_data = getDataByInternalId(nn);
            DistType dist = this->calcDistance(nn_data, neighbor_data);
            candidates.emplace_back(dist, nn);
            candidate_ids.insert(nn);
        }
    }

    // Add deleted node's neighbors (excluding current neighbor) as repair candidates
    for (idType candidate_id : deleted_node_neighbors) {
        if (candidate_id != neighbor_id && candidate_id < elementCount &&
            !isMarkedDeleted(candidate_id)) {
            // Check if already in candidates to avoid duplicates using O(1) hash set lookup
            if (candidate_ids.find(candidate_id) == candidate_ids.end()) {
                const void *candidate_data = getDataByInternalId(candidate_id);
                DistType dist = this->calcDistance(candidate_data, neighbor_data);
                candidates.emplace_back(dist, candidate_id);
                candidate_ids.insert(candidate_id);
            }
        }
    }

    // Track original neighbors for bidirectional edge updates
    vecsim_stl::unordered_set<idType> original_neighbors_set(this->allocator);
    original_neighbors_set.reserve(neighbor_neighbors.size());
    for (idType nn : neighbor_neighbors) {
        if (nn != deleted_id && nn < elementCount) {
            original_neighbors_set.insert(nn);
        }
    }

    // Apply heuristic to select best neighbors if we have more than max_M
    vecsim_stl::vector<idType> updated_neighbors(this->allocator);
    if (candidates.size() > max_M) {
        vecsim_stl::vector<idType> removed_candidates(this->allocator);
        getNeighborsByHeuristic2(candidates, max_M, removed_candidates);
    }
    // Extract selected neighbor IDs (works for both cases)
    updated_neighbors.reserve(candidates.size());
    for (const auto &[dist, id] : candidates) {
        updated_neighbors.push_back(id);
    }

    // Stage the update for this neighbor
    stageDeleteUpdate(neighbor_id, level, updated_neighbors);

}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::processDeleteBatch() {
    if (pendingDeleteIds.empty()) return;

    // Clear any previous staged updates (for deletions)
    stagedDeleteUpdates.clear();
    stagedDeleteMap.clear();

    // Create a set of IDs being deleted in this batch for quick lookup
    std::unordered_set<idType> deletingIds(pendingDeleteIds.begin(), pendingDeleteIds.end());
    size_t elementCount = curElementCount.load(std::memory_order_acquire);

    // Process each deleted node
    for (idType deleted_id : pendingDeleteIds) {
        // Skip if already processed or invalid
        if (deleted_id >= elementCount || deleted_id >= idToMetaData.size()) {
            continue;
        }

        const DiskElementMetaData &metadata = idToMetaData[deleted_id];
        if (metadata.label == INVALID_LABEL) {
            continue; // Already deleted
        }

        size_t topLevel = metadata.topLevel;

        // Process each level of the deleted node
        for (size_t level = 0; level <= topLevel; level++) {
            // Get the deleted node's neighbors at this level
            vecsim_stl::vector<idType> deleted_node_neighbors(this->allocator);
            getNeighbors(deleted_id, level, deleted_node_neighbors);

            // For each neighbor of the deleted node
            for (idType neighbor_id : deleted_node_neighbors) {
                // Skip if neighbor is also deleted, invalid, or in the current deletion batch
                if (neighbor_id >= elementCount || isMarkedDeleted(neighbor_id) ||
                    deletingIds.find(neighbor_id) != deletingIds.end()) {
                    continue;
                }

                // Get the neighbor's current neighbor list
                vecsim_stl::vector<idType> neighbor_neighbors(this->allocator);
                getNeighbors(neighbor_id, level, neighbor_neighbors);

                // Check if this is a bidirectional edge
                bool is_bidirectional = false;
                for (idType nn : neighbor_neighbors) {
                    if (nn == deleted_id) {
                        is_bidirectional = true;
                        break;
                    }
                }

                if (is_bidirectional) {
                    // Repair connections using the dedicated helper method
                    repairNeighborConnections(neighbor_id, level, deleted_id,
                                              deleted_node_neighbors, neighbor_neighbors);
                }
            }

            // Delete the node's graph entry at this level by staging an empty neighbor list
            // (or we could use a Delete operation in the batch)
            vecsim_stl::vector<idType> empty_neighbors(this->allocator);
            uint64_t del_key = makeRepairKey(deleted_id, level);
            // For deletion entries, always overwrite - the node is being deleted
            stagedDeleteMap[del_key] = stagedDeleteUpdates.size();
            stagedDeleteUpdates.emplace_back(deleted_id, level, empty_neighbors, this->allocator);
        }
    }

    // Mark metadata as invalid and clean up raw vectors AFTER processing all nodes
    // This ensures getNeighbors() and other methods work correctly during graph repair
    for (idType deleted_id : pendingDeleteIds) {
        if (deleted_id >= elementCount || deleted_id >= idToMetaData.size()) {
            continue;
        }
        // Mark the metadata as invalid
        idToMetaData[deleted_id].label = INVALID_LABEL;

        // Remove raw vector from RAM if it exists
        auto ram_it = rawVectorsInRAM.find(deleted_id);
        if (ram_it != rawVectorsInRAM.end()) {
            rawVectorsInRAM.erase(ram_it);
        }
    }

    // Flush all staged graph updates to disk in a single batch operation
    vecsim_stl::vector<NeighborUpdate> emptyNeighborUpdates(this->allocator);
    flushStagedGraphUpdates(stagedDeleteUpdates, emptyNeighborUpdates);
    stagedDeleteMap.clear();

    // Flush staged repair updates (opportunistic cleanup from getNeighbors)
    // But first, filter out any repairs for nodes that were just deleted
    if (!stagedRepairUpdates.empty()) {
        vecsim_stl::vector<GraphUpdate> filteredRepairUpdates(this->allocator);
        for (const auto &update : stagedRepairUpdates) {
            // Skip repairs for nodes that are in the deletion batch
            if (deletingIds.find(update.node_id) == deletingIds.end()) {
                filteredRepairUpdates.push_back(update);
            }
        }
        if (!filteredRepairUpdates.empty()) {
            flushStagedGraphUpdates(filteredRepairUpdates, emptyNeighborUpdates);
        }
        // Clear all staged repairs (including filtered ones)
        stagedRepairUpdates.clear();
        stagedRepairMap.clear();
    }

    // Clear the pending delete IDs
    pendingDeleteIds.clear();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::flushDeleteBatch() {
    processDeleteBatch();
}

/********************************** Debug Methods **********************************/

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::debugPrintGraphStructure() const {
    size_t elementCount = curElementCount.load(std::memory_order_acquire);
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "=== HNSW Disk Index Graph Structure ===");
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Total elements: %zu", elementCount);
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Entry point: %u", entrypointNode);
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Max level: %zu", maxLevel);
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "M: %zu, M0: %zu", M, M0);

    // Count total edges
    size_t total_edges = debugCountGraphEdges();
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Total graph edges: %zu", total_edges);

    // Print metadata for each element
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Element metadata:");
    for (size_t i = 0; i < std::min(elementCount, idToMetaData.size()); ++i) {
        if (idToMetaData[i].label != INVALID_LABEL) {
            this->log(VecSimCommonStrings::LOG_DEBUG_STRING,
                      "  Element %zu: label=%u, topLevel=%zu", i, idToMetaData[i].label,
                      idToMetaData[i].topLevel);
        }
    }

    // Print graph keys and their neighbors
    debugPrintAllGraphKeys();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::debugPrintNodeNeighbors(idType node_id) const {
    size_t elementCount = curElementCount.load(std::memory_order_acquire);
    if (node_id >= elementCount) {
        this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Node %u is out of range (max: %zu)",
                  node_id, (elementCount - 1));
        return;
    }

    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "=== Neighbors for Node %u ===", node_id);
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Label: %u", getExternalLabel(node_id));
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Top level: %zu",
              idToMetaData[node_id].topLevel);

    // Check each level
    for (size_t level = 0; level <= idToMetaData[node_id].topLevel; ++level) {
        GraphKey graphKey(node_id, level);
        std::string graph_value;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &graph_value);

        if (status.ok()) {
            // Parse using new format
            vecsim_stl::vector<idType> neighbors(this->allocator);
            deserializeGraphValue(graph_value, neighbors);
            this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "  Level %zu (%zu neighbors): ", level, neighbors.size());
            for (size_t i = 0; i < neighbors.size(); i++) {
                this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "%u ", neighbors[i]);
            }
        } else {
            this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "  Level %zu: No neighbors found",
                      level);
        }
    }
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::debugPrintAllGraphKeys() const {
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "=== All Graph Keys in RocksDB ===");

    auto readOptions = rocksdb::ReadOptions();
    readOptions.fill_cache = false;
    readOptions.prefix_same_as_start = true;

    auto it = db->NewIterator(readOptions, cf);
    size_t key_count = 0;
    size_t total_neighbors = 0;

    for (it->Seek(GraphKeyPrefix); it->Valid(); it->Next()) {
        auto key = it->key();

        // Parse the key: "GK" + GraphKey struct
        if (key.size() >= 9 && key.starts_with("GK")) { // 3 bytes for "GK" + 2 for level + 4 for id
            const GraphKey *graphKey = reinterpret_cast<const GraphKey *>(key.data() + 3);
            this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "  Key %zu: node=%u, level=%u",
                      key_count, graphKey->id, graphKey->level);
        } else {
            this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "  Key %zu: invalid format (size=%zu)",
                      key_count, key.size());
        }
        
        // Parse graph value using new format
        std::string graph_value = it->value().ToString();
        vecsim_stl::vector<idType> neighbors(this->allocator);
        deserializeGraphValue(graph_value, neighbors);
        size_t num_neighbors = neighbors.size();
        total_neighbors += num_neighbors;

        this->log(VecSimCommonStrings::LOG_DEBUG_STRING, " (%zu neighbors)", num_neighbors);

        // Print first few neighbors
        if (num_neighbors > 0) {
            this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "    Neighbors: ");
            for (size_t i = 0; i < std::min(num_neighbors, size_t(5)); i++) {
                this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "%u ", neighbors[i]);
            }
            if (num_neighbors > 5) {
                this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "... (and %zu more)",
                          (num_neighbors - 5));
            }
        }

        key_count++;
    }

    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Total graph keys: %zu", key_count);
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Total neighbor connections: %zu",
              total_neighbors);

    delete it;
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::debugCountGraphEdges() const {
    auto readOptions = rocksdb::ReadOptions();
    readOptions.fill_cache = false;
    readOptions.prefix_same_as_start = true;

    auto it = db->NewIterator(readOptions, cf);
    size_t total_edges = 0;

    for (it->Seek(GraphKeyPrefix); it->Valid(); it->Next()) {
        // Parse graph value using new format
        std::string graph_value = it->value().ToString();
        vecsim_stl::vector<idType> neighbors(this->allocator);
        deserializeGraphValue(graph_value, neighbors);
        total_edges += neighbors.size();
    }

    delete it;
    return total_edges;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::debugValidateGraphConnectivity() const {
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "=== Graph Connectivity Validation ===");
    size_t elementCount = curElementCount.load(std::memory_order_acquire);

    if (elementCount == 0) {
        this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Index is empty, nothing to validate");
        return;
    }

    // Check if entry point exists and has neighbors
    if (entrypointNode != INVALID_ID) {
        this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Entry point %u exists", entrypointNode);
        debugPrintNodeNeighbors(entrypointNode);
    } else {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING, "WARNING: No entry point set!");
    }

    // Check connectivity for first few elements
    size_t elements_to_check = std::min(elementCount, size_t(5));
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING,
              "Checking connectivity for first %zu elements:", elements_to_check);

    for (size_t i = 0; i < elements_to_check; ++i) {
        if (idToMetaData[i].label != INVALID_LABEL) {
            this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Element %zu:", i);
            debugPrintNodeNeighbors(i);
        }
    }

    // Check for isolated nodes (nodes with no neighbors at any level)
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Checking for isolated nodes...");
    size_t isolated_count = 0;

    for (size_t i = 0; i < elementCount; ++i) {
        if (idToMetaData[i].label == INVALID_LABEL)
            continue;

        bool has_neighbors = false;
        for (size_t level = 0; level <= idToMetaData[i].topLevel; ++level) {
            GraphKey graphKey(i, level);
            std::string graph_value;
            rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &graph_value);

            if (status.ok()) {
                // Parse using new format
                vecsim_stl::vector<idType> neighbors(this->allocator);
                deserializeGraphValue(graph_value, neighbors);
                if (neighbors.size() > 0) {
                    has_neighbors = true;
                    break;
                }
            }
        }

        if (!has_neighbors) {
            this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                      "  WARNING: Element %zu (label %u) has no neighbors at any level!", i,
                      idToMetaData[i].label);
            isolated_count++;
        }
    }

    if (isolated_count == 0) {
        this->log(VecSimCommonStrings::LOG_DEBUG_STRING,
                  "  All elements have at least some neighbors");
    } else {
        this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "  Found %zu isolated elements",
                  isolated_count);
    }
}


template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::flushStagedDeleteUpdates() {
    // Flush staged delete updates to disk
    // Note: This is a non-const method that modifies the staging areas
    vecsim_stl::vector<NeighborUpdate> emptyNeighborUpdates(this->allocator);
    flushStagedGraphUpdates(stagedDeleteUpdates, emptyNeighborUpdates);
    stagedDeleteMap.clear();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::debugPrintStagedUpdates() const {
    // Print both insert and delete staged updates
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "=== Staged Updates ===");
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Staged Insert Graph Updates: %zu",
              stagedInsertUpdates.size());
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Staged Insert Neighbor Updates: %zu",
              stagedInsertNeighborUpdates.size());
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Staged Delete Graph Updates: %zu",
              stagedDeleteUpdates.size());
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Staged Repair Updates: %zu",
              stagedRepairUpdates.size());
}

// Add missing method implementations for benchmark framework
template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::fitMemory() {
    // TODO: Implement memory fitting
    // For now, just a stub implementation
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::getDataByLabel(
    labelType label, std::vector<std::vector<DataType>> &vectors_output) const {
    vectors_output.clear();

    std::shared_lock<std::shared_mutex> index_data_lock(indexDataGuard);

    // Check if label exists in the map
    auto it = labelToIdMap.find(label);
    if (it == labelToIdMap.end()) {
        return; // Label not found
    }

    idType id = it->second;

    // Get the raw vector data
    std::vector<DataType> raw_vector(this->dim);
    if (!getRawVector(id, raw_vector.data())) {
        return; // Vector not found
    }

    vectors_output.push_back(std::move(raw_vector));
}

template <typename DataType, typename DistType>
std::vector<std::vector<char>>
HNSWDiskIndex<DataType, DistType>::getStoredVectorDataByLabel(labelType label) const {
    // TODO: Implement stored vector data retrieval
    // For now, just a stub implementation
    return {};
}

template <typename DataType, typename DistType>
vecsim_stl::set<labelType> HNSWDiskIndex<DataType, DistType>::getLabelsSet() const {
    std::shared_lock<std::shared_mutex> index_data_lock(indexDataGuard);
    vecsim_stl::set<labelType> labels(this->allocator);
    for (const auto &it : labelToIdMap) {
        labels.insert(it.first);
    }
    return labels;
}

template <typename DataType, typename DistType>
int HNSWDiskIndex<DataType, DistType>::deleteVector(labelType label) {

    vecsim_stl::vector<idType> deleted_ids = markDelete(label);
    if (deleted_ids.empty()) {
        return 0; // Label not found or already deleted
    }

    pendingDeleteIds.insert(pendingDeleteIds.end(), deleted_ids.begin(), deleted_ids.end());

    if (pendingDeleteIds.size() >= deleteBatchThreshold) {
        processDeleteBatch();
    }

    return 1;
}

template <typename DataType, typename DistType>
double HNSWDiskIndex<DataType, DistType>::getDistanceFrom_Unsafe(labelType id,
                                                                 const void *blob) const {
    // TODO: Implement distance calculation
    // For now, just a stub implementation
    return 0.0;
}

template <typename DataType, typename DistType>
uint64_t HNSWDiskIndex<DataType, DistType>::getAllocationSize() const {

    return this->allocator->getAllocationSize();
}

template <typename DataType, typename DistType>
typename HNSWDiskIndex<DataType, DistType>::RocksDBMemoryBreakdown
HNSWDiskIndex<DataType, DistType>::getDBMemoryBreakdown() const {
    RocksDBMemoryBreakdown breakdown;
    breakdown.memtables = 0;
    breakdown.table_readers = 0;
    breakdown.block_cache = 0;
    breakdown.pinned_blocks = 0;

    this->db->GetIntProperty(rocksdb::DB::Properties::kSizeAllMemTables, &breakdown.memtables);
    this->db->GetIntProperty(rocksdb::DB::Properties::kEstimateTableReadersMem, &breakdown.table_readers);
    this->db->GetIntProperty(rocksdb::DB::Properties::kBlockCacheUsage, &breakdown.block_cache);
    this->db->GetIntProperty(rocksdb::DB::Properties::kBlockCachePinnedUsage, &breakdown.pinned_blocks);
    
    breakdown.total = breakdown.memtables + breakdown.table_readers +
                      breakdown.block_cache + breakdown.pinned_blocks;
    return breakdown;
}

template <typename DataType, typename DistType>
uint64_t HNSWDiskIndex<DataType, DistType>::getDBMemorySize() const {
    // Get comprehensive RocksDB memory usage by summing all components:
    // 1. Memtables (active, unflushed immutable, and pinned immutable)
    // 2. Table readers (filter and index blocks not in block cache)
    // 3. Block cache (uncompressed data blocks)
    // 4. Pinned blocks (blocks pinned by iterators)

    uint64_t memtables = 0;
    uint64_t table_readers = 0;
    uint64_t block_cache = 0;
    uint64_t pinned_blocks = 0;

    this->db->GetIntProperty(rocksdb::DB::Properties::kSizeAllMemTables, &memtables);
    this->db->GetIntProperty(rocksdb::DB::Properties::kEstimateTableReadersMem, &table_readers);
    this->db->GetIntProperty(rocksdb::DB::Properties::kBlockCacheUsage, &block_cache);
    this->db->GetIntProperty(rocksdb::DB::Properties::kBlockCachePinnedUsage, &pinned_blocks);

    return memtables + table_readers + block_cache + pinned_blocks;
}

template <typename DataType, typename DistType>
uint64_t HNSWDiskIndex<DataType, DistType>::getDiskSize() const {
    uint64_t disk_size = 0;
    this->db->GetIntProperty(rocksdb::DB::Properties::kTotalSstFilesSize, &disk_size);
    return disk_size;
}

template <typename DataType, typename DistType>
std::shared_ptr<rocksdb::Statistics> HNSWDiskIndex<DataType, DistType>::getDBStatistics() const {
    // Get statistics directly from the database instead of from the cached dbOptions copy
    // because GetOptions() returns a copy that doesn't preserve the shared_ptr to statistics
    return this->db->GetOptions().statistics;
}

// Missing virtual method implementations for HNSWDiskIndex
template <typename DataType, typename DistType>
VecSimIndexStatsInfo HNSWDiskIndex<DataType, DistType>::statisticInfo() const {
    VecSimIndexStatsInfo info = {};
    info.memory = this->getAllocationSize();

    
    // Processed vectors memory (stored in this->vectors container)
    info.vectors_memory = this->vectors->size() * this->dataSize;

    // RocksDB memory and disk usage
    info.db_memory = this->getDBMemorySize();
    info.db_disk = this->getDiskSize();

    // Number of marked deleted elements
    info.numberOfMarkedDeleted = this->numMarkedDeleted;

    return info;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::setLastSearchMode(VecSearchMode mode) {
    // TODO: Implement if needed for disk-based HNSW
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::runGC() {
    // TODO: Implement garbage collection for disk-based HNSW
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::acquireSharedLocks() {
    // TODO: Implement if needed for disk-based HNSW
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::releaseSharedLocks() {
    // TODO: Implement if needed for disk-based HNSW
}

/********************************** Mark Delete Implementation **********************************/

template <typename DataType, typename DistType>
vecsim_stl::vector<idType> HNSWDiskIndex<DataType, DistType>::markDelete(labelType label) {
    vecsim_stl::vector<idType> internal_ids(this->allocator);

    // Find the internal ID for this label
    auto it = labelToIdMap.find(label);
    if (it == labelToIdMap.end()) {
        // Label doesn't exist, return empty vector
        return internal_ids;
    }

    const idType internalId = it->second;

    // Check if already marked deleted
    if (idToMetaData[internalId].flags & DELETE_MARK) {
        // Already deleted, return empty vector
        return internal_ids;
    }

    // Mark as deleted (but don't clean up raw vectors yet - they're needed for graph repair
    // in processDeleteBatch. Cleanup happens there after repair is complete.)
    idToMetaData[internalId].flags |= DELETE_MARK;
    this->numMarkedDeleted++;

    // If this is the entrypoint, we need to replace it
    if (internalId == entrypointNode) {
        replaceEntryPoint();
    }

    // Remove from label lookup
    labelToIdMap.erase(it);

    // Return the internal ID that was marked deleted
    internal_ids.push_back(internalId);
    return internal_ids;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::replaceEntryPoint() {
    // This method is called when the current entrypoint is marked as deleted
    // We need to find a new entrypoint from the remaining non-deleted nodes
    idType old_entry_point_id = entrypointNode;
    size_t currentMaxLevel = maxLevel;

    // Try to find a new entrypoint at the current max level (including level 0)
    // Use a do-while or check >= 0 to include level 0
    while (true) {
        // First, try to find a neighbor of the old entrypoint at this level
        GraphKey graphKey(old_entry_point_id, currentMaxLevel);
        std::string graph_value;
        rocksdb::Status status =
            db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &graph_value);

        if (status.ok() && !graph_value.empty()) {
            // Correctly deserialize the graph value to get neighbors
            vecsim_stl::vector<idType> neighbors(this->allocator);
            deserializeGraphValue(graph_value, neighbors);

            // Try to find a non-deleted neighbor
            for (size_t i = 0; i < neighbors.size(); i++) {
                if (neighbors[i] < idToMetaData.size() &&
                    !(idToMetaData[neighbors[i]].flags & DELETE_MARK)) {
                    entrypointNode = neighbors[i];
                    maxLevel = currentMaxLevel;
                    return;
                }
            }
        }

        // If no suitable neighbor found, search for any non-deleted node at this level
        size_t elementCount = curElementCount.load();
        for (idType id = 0; id < elementCount; id++) {
            if (id != old_entry_point_id && id < idToMetaData.size() &&
                idToMetaData[id].label != INVALID_LABEL &&
                idToMetaData[id].topLevel >= currentMaxLevel &&
                !(idToMetaData[id].flags & DELETE_MARK)) {
                entrypointNode = id;
                maxLevel = currentMaxLevel;
                return;
            }
        }

        // No non-deleted nodes at this level, decrease maxLevel and try again
        if (currentMaxLevel == 0) break;
        currentMaxLevel--;
    }

    // If we get here, the index is empty or all nodes are deleted
    entrypointNode = INVALID_ID;
    maxLevel = 0;
}

/********************************** Batchless Mode Implementation **********************************/

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::getNeighborsFromCache(
    idType nodeId, size_t level, vecsim_stl::vector<idType> &result) const {

    result.clear();
    uint64_t key = makeRepairKey(nodeId, level);
    size_t segment = getSegmentIndex(key);
    auto& cacheSegment = cacheSegments_[segment];

    // Step 1: Check in-memory cache (includes pending writes from all jobs)
    {
        std::shared_lock<std::shared_mutex> lock(cacheSegment.guard);
        auto it = cacheSegment.cache.find(key);
        if (it != cacheSegment.cache.end()) {
            // Copy from std::vector to vecsim_stl::vector
            result.reserve(it->second.size());
            for (idType id : it->second) {
                result.push_back(id);
            }
            filterDeletedNodes(result);
            return;
        }

        // Check if this is a new node (never written to disk) - return empty
        if (cacheSegment.newNodes.find(key) != cacheSegment.newNodes.end()) {
            // New node not yet in cache - return empty (will be populated when connected)
            return;
        }
    }

    // Step 2: Not in cache and not a new node - read from disk
    GraphKey graphKey(nodeId, level);
    std::string graph_value;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &graph_value);

    if (status.ok()) {
        deserializeGraphValue(graph_value, result);
    }

    // Add to cache for future reads (convert to unique_lock for write)
    {
        std::unique_lock<std::shared_mutex> lock(cacheSegment.guard);
        // Double-check: another thread may have populated it
        auto it = cacheSegment.cache.find(key);
        if (it == cacheSegment.cache.end()) {
            // Copy from vecsim_stl::vector to std::vector
            std::vector<idType> cacheEntry;
            cacheEntry.reserve(result.size());
            for (idType id : result) {
                cacheEntry.push_back(id);
            }
            cacheSegment.cache[key] = std::move(cacheEntry);
        }
    }

    filterDeletedNodes(result);
}

// Helper to load neighbors from disk into cache if not already present
// Caller must hold unique_lock on cacheSegment.guard (will be temporarily released during disk I/O)
template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::loadNeighborsFromDiskIfNeeded(
    uint64_t key, CacheSegment& cacheSegment, std::unique_lock<std::shared_mutex>& lock) {

    auto it = cacheSegment.cache.find(key);
    if (it != cacheSegment.cache.end()) {
        return; // Already in cache
    }

    // Check if this is a new node (never written to disk) - skip disk lookup
    bool isNewNode = (cacheSegment.newNodes.find(key) != cacheSegment.newNodes.end());

    if (!isNewNode) {
        // First modification of existing node - need to load current state from disk
        idType nodeId = static_cast<idType>(key >> 32);
        size_t level = static_cast<size_t>(key & 0xFFFFFFFF);

        lock.unlock();
        vecsim_stl::vector<idType> diskNeighbors(this->allocator);
        GraphKey graphKey(nodeId, level);
        std::string graph_value;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &graph_value);
        if (status.ok()) {
            deserializeGraphValue(graph_value, diskNeighbors);
        }
        lock.lock();

        // Re-check: another thread might have populated it
        if (cacheSegment.cache.find(key) == cacheSegment.cache.end()) {
            // Convert vecsim_stl::vector to std::vector
            std::vector<idType> cacheEntry;
            cacheEntry.reserve(diskNeighbors.size());
            for (idType id : diskNeighbors) {
                cacheEntry.push_back(id);
            }
            cacheSegment.cache[key] = std::move(cacheEntry);
        }
    } else {
        // New node - initialize with empty neighbor list
        cacheSegment.cache[key] = std::vector<idType>();
    }
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::addNeighborToCache(
    idType nodeId, size_t level, idType newNeighborId) {

    uint64_t key = makeRepairKey(nodeId, level);
    size_t segment = getSegmentIndex(key);
    auto& cacheSegment = cacheSegments_[segment];

    std::unique_lock<std::shared_mutex> lock(cacheSegment.guard);

    // Load from disk if not in cache (handles newNodes check internally)
    loadNeighborsFromDiskIfNeeded(key, cacheSegment, lock);

    // Add new neighbor (avoid duplicates)
    auto &neighbors = cacheSegment.cache[key];
    if (std::find(neighbors.begin(), neighbors.end(), newNeighborId) == neighbors.end()) {
        neighbors.push_back(newNeighborId);
    }

    // Mark as dirty (needs disk write) and increment atomic counter
    auto insertResult = cacheSegment.dirty.insert(key);
    if (insertResult.second) {  // Only increment if newly inserted
        totalDirtyCount_.fetch_add(1, std::memory_order_relaxed);
    }
}

template <typename DataType, typename DistType>
bool HNSWDiskIndex<DataType, DistType>::tryAddNeighborToCacheIfCapacity(
    idType nodeId, size_t level, idType newNeighborId, size_t maxCapacity) {

    uint64_t key = makeRepairKey(nodeId, level);
    size_t segment = getSegmentIndex(key);
    auto& cacheSegment = cacheSegments_[segment];

    std::unique_lock<std::shared_mutex> lock(cacheSegment.guard);

    // Load from disk if not in cache (handles newNodes check internally)
    loadNeighborsFromDiskIfNeeded(key, cacheSegment, lock);

    // Atomic check-and-add under the lock
    auto &neighbors = cacheSegment.cache[key];

    // Check if already present (avoid duplicates)
    if (std::find(neighbors.begin(), neighbors.end(), newNeighborId) != neighbors.end()) {
        return true; // Already added, consider it success
    }

    // Check capacity
    if (neighbors.size() >= maxCapacity) {
        return false; // No capacity
    }

    // Has capacity - add the neighbor
    neighbors.push_back(newNeighborId);
    auto insertResult = cacheSegment.dirty.insert(key);
    if (insertResult.second) {  // Only increment if newly inserted
        totalDirtyCount_.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::setNeighborsInCache(
    idType nodeId, size_t level, const vecsim_stl::vector<idType> &neighbors, bool isNewNode) {

    uint64_t key = makeRepairKey(nodeId, level);
    size_t segment = getSegmentIndex(key);
    auto& cacheSegment = cacheSegments_[segment];

    // Convert vecsim_stl::vector to std::vector
    std::vector<idType> cacheEntry;
    cacheEntry.reserve(neighbors.size());
    for (idType id : neighbors) {
        cacheEntry.push_back(id);
    }

    std::unique_lock<std::shared_mutex> lock(cacheSegment.guard);
    cacheSegment.cache[key] = std::move(cacheEntry);

    // If this is a new node, track it to avoid disk lookups
    if (isNewNode) {
        cacheSegment.newNodes.insert(key);
    }

    auto insertResult = cacheSegment.dirty.insert(key);
    if (insertResult.second) {  // Only increment if newly inserted
        totalDirtyCount_.fetch_add(1, std::memory_order_relaxed);
    }
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::writeVectorToDisk(
    idType vectorId, const void *rawVectorData,
    const vecsim_stl::vector<idType> &neighbors) {

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;

    // Write all levels for this vector
    size_t topLevel = idToMetaData[vectorId].topLevel;
    rocksdb::WriteBatch batch;

    for (size_t level = 0; level <= topLevel; level++) {
        GraphKey graphKey(vectorId, level);
        std::string value = serializeGraphValue(rawVectorData, neighbors);
        batch.Put(cf, graphKey.asSlice(), value);
    }

    db->Write(writeOptions, &batch);
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::writeDirtyNodesToDisk(
    const vecsim_stl::vector<uint64_t> &modifiedNodes,
    const void *newVectorRawData, idType newVectorId) {

    if (modifiedNodes.empty()) {
        return;
    }

    // ==================== SWAP-AND-FLUSH PATTERN ====================
    // To avoid the "Lost Update" race condition, we atomically clear the dirty flag
    // BEFORE writing to disk (not after). If a new insert occurs while we're writing,
    // it will simply re-add the node to the dirty set. No updates are lost.
    // ================================================================

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;

    rocksdb::WriteBatch batch;
    std::vector<char> rawVectorBuffer(this->inputBlobSize);
    vecsim_stl::vector<uint64_t> nodesToWrite(this->allocator);

    // ==================== SWAP-AND-FLUSH PATTERN ====================
    // Two-phase approach to minimize lock contention while preventing lost updates:
    // Phase 1: Read cache data under SHARED lock (allows concurrent writers)
    // Phase 2: Clear dirty flags under EXCLUSIVE lock (brief, after all reads done)
    // If writing fails, we re-add nodes to dirty set.
    // ================================================================

    // Phase 1: Read cache data under shared locks (allows other writers to proceed)
    for (uint64_t key : modifiedNodes) {
        idType nodeId = static_cast<idType>(key >> 32);
        size_t level = static_cast<size_t>(key & 0xFFFFFFFF);
        size_t segment = getSegmentIndex(key);
        auto& cacheSegment = cacheSegments_[segment];

        // Get neighbors from cache under SHARED lock (concurrent reads allowed)
        vecsim_stl::vector<idType> neighbors(this->allocator);
        {
            std::shared_lock<std::shared_mutex> segmentLock(cacheSegment.guard);
            auto it = cacheSegment.cache.find(key);
            if (it == cacheSegment.cache.end()) {
                continue;
            }
            const std::vector<idType> &cacheNeighbors = it->second;
            neighbors.reserve(cacheNeighbors.size());
            for (idType id : cacheNeighbors) {
                neighbors.push_back(id);
            }
        }
        // Shared lock released - other threads can add to cache

        // Get raw vector data (no cache lock held during disk I/O)
        const void *rawData;
        if (nodeId == newVectorId) {
            rawData = newVectorRawData;
        } else {
            // Read from RAM or disk
            if (!getRawVectorInternal(nodeId, rawVectorBuffer.data())) {
                this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                          "WARNING: Could not get raw vector for node %u", nodeId);
                continue;
            }
            rawData = rawVectorBuffer.data();
        }

        GraphKey graphKey(nodeId, level);
        std::string value = serializeGraphValue(rawData, neighbors);
        batch.Put(cf, graphKey.asSlice(), value);
        nodesToWrite.push_back(key);
    }

    // Phase 2: Clear dirty flags BEFORE writing (swap-and-flush pattern)
    // Brief exclusive locks - one segment at a time
    for (uint64_t key : nodesToWrite) {
        size_t segment = getSegmentIndex(key);
        auto& cacheSegment = cacheSegments_[segment];
        std::unique_lock<std::shared_mutex> segmentLock(cacheSegment.guard);
        if (cacheSegment.dirty.erase(key) > 0) {
            totalDirtyCount_.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    if (nodesToWrite.empty()) {
        return;
    }

    // Phase 3: Write to disk (no locks held)
    // Dirty flags already cleared in Phase 2 - any concurrent modifications
    // will re-add to dirty set and be picked up in next flush
    rocksdb::Status status = db->Write(writeOptions, &batch);
    if (!status.ok()) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                  "ERROR: Failed to write dirty nodes to disk: %s", status.ToString().c_str());

        // On failure, re-add nodes to dirty set so they'll be retried
        for (uint64_t key : nodesToWrite) {
            size_t segment = getSegmentIndex(key);
            auto& cacheSegment = cacheSegments_[segment];
            std::unique_lock<std::shared_mutex> segmentLock(cacheSegment.guard);
            if (cacheSegment.dirty.insert(key).second) {
                totalDirtyCount_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    // Success case: dirty flags already cleared, nothing more to do
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::flushDirtyNodesToDisk() {
    std::lock_guard<std::mutex> flushLock(diskWriteGuard);

    // Check if there are any dirty nodes using atomic counter (fast path)
    if (totalDirtyCount_.load(std::memory_order_relaxed) == 0) {
        return;
    }

    // ==================== SWAP-AND-FLUSH PATTERN ====================
    // Process one segment at a time to minimize lock contention:
    // 1. Acquire exclusive lock on segment (blocks inserts briefly - nanoseconds for swap)
    // 2. Swap dirty set with empty set (atomic ownership transfer)
    // 3. Release lock immediately (segment is free for new inserts)
    // 4. Process swapped nodes and write to disk (no locks held)
    //
    // If a new insert happens after the swap, it sees an empty dirty set
    // and correctly adds itself. No updates are lost.
    // ================================================================

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;

    rocksdb::WriteBatch batch;
    std::vector<char> rawVectorBuffer(this->inputBlobSize);
    std::unordered_set<idType> successfulLevel0Ids;

    // Track all nodes we're trying to write (for error recovery)
    vecsim_stl::vector<uint64_t> allNodesToFlush(this->allocator);

    // Process each segment independently
    for (size_t s = 0; s < NUM_CACHE_SEGMENTS; ++s) {
        auto& cacheSegment = cacheSegments_[s];

        // Local container to take ownership of dirty nodes
        std::unordered_set<uint64_t> nodesToFlush;

        {
            // 1. Acquire exclusive lock (brief - just for the swap)
            std::unique_lock<std::shared_mutex> segmentLock(cacheSegment.guard);

            if (cacheSegment.dirty.empty()) {
                continue;
            }

            // 2. Atomic swap - move dirty set contents to local, leave dirty empty
            // New inserts after this will add to the now-empty dirty set
            nodesToFlush.swap(cacheSegment.dirty);

            // 3. Update global counter - we've taken ownership of these nodes
            totalDirtyCount_.fetch_sub(nodesToFlush.size(), std::memory_order_relaxed);
        }
        // 4. Lock released here - segment is free for inserts

        // 5. Process nodes and add to batch (no locks held)
        for (uint64_t key : nodesToFlush) {
            idType nodeId = static_cast<idType>(key >> 32);
            size_t level = static_cast<size_t>(key & 0xFFFFFFFF);

            // Get neighbors from cache (need shared lock briefly)
            vecsim_stl::vector<idType> neighbors(this->allocator);
            {
                std::shared_lock<std::shared_mutex> segmentLock(cacheSegment.guard);
                auto it = cacheSegment.cache.find(key);
                if (it == cacheSegment.cache.end()) {
                    // Node not in cache - shouldn't happen, but skip if it does
                    continue;
                }
                const std::vector<idType> &cacheNeighbors = it->second;
                neighbors.reserve(cacheNeighbors.size());
                for (idType id : cacheNeighbors) {
                    neighbors.push_back(id);
                }
            }

            // Get raw vector data (no segment lock needed)
            if (!getRawVectorInternal(nodeId, rawVectorBuffer.data())) {
                // Raw vector not available - this shouldn't happen for dirty nodes
                this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                          "WARNING: Raw vector not found for dirty node %u at level %zu", nodeId, level);
                continue;
            }

            GraphKey graphKey(nodeId, level);
            std::string value = serializeGraphValue(rawVectorBuffer.data(), neighbors);
            batch.Put(cf, graphKey.asSlice(), value);
            allNodesToFlush.push_back(key);

            if (level == 0) {
                successfulLevel0Ids.insert(nodeId);
            }
        }
    }

    // Write entire batch to RocksDB (no locks held during I/O)
    if (batch.Count() > 0) {
        rocksdb::Status status = db->Write(writeOptions, &batch);
        if (!status.ok()) {
            this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                      "ERROR: Failed to flush dirty nodes to disk: %s", status.ToString().c_str());

            // On failure, re-add nodes to dirty set so they'll be retried
            for (uint64_t key : allNodesToFlush) {
                size_t segment = getSegmentIndex(key);
                auto& cacheSegment = cacheSegments_[segment];
                std::unique_lock<std::shared_mutex> segmentLock(cacheSegment.guard);
                if (cacheSegment.dirty.insert(key).second) {
                    totalDirtyCount_.fetch_add(1, std::memory_order_relaxed);
                }
            }
            return; // Don't clear newNodes or remove raw vectors on failure
        }
    }

    // Success: Clear newNodes flags for all flushed nodes
    for (uint64_t key : allNodesToFlush) {
        size_t segment = getSegmentIndex(key);
        auto& cacheSegment = cacheSegments_[segment];
        std::unique_lock<std::shared_mutex> segmentLock(cacheSegment.guard);
        cacheSegment.newNodes.erase(key);
    }

    // Remove raw vectors for level 0 nodes that were written to disk
    if (!successfulLevel0Ids.empty()) {
        std::lock_guard<std::shared_mutex> lock(rawVectorsGuard);
        for (idType vectorId : successfulLevel0Ids) {
            rawVectorsInRAM.erase(vectorId);
        }
    }
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::insertElementToGraph(
    idType element_id, size_t element_max_level, idType entry_point, size_t global_max_level,
    const void *raw_vector_data, const void *vector_data,
    vecsim_stl::vector<uint64_t> &modifiedNodes) {

    idType curr_element = entry_point;
    DistType cur_dist = std::numeric_limits<DistType>::max();
    size_t max_common_level;

    if (element_max_level < global_max_level) {
        max_common_level = element_max_level;
        cur_dist = this->calcDistance(vector_data, getDataByInternalId(curr_element));
        for (auto level = static_cast<int>(global_max_level);
             level > static_cast<int>(element_max_level); level--) {
            greedySearchLevel<false>(vector_data, level, curr_element, cur_dist);
        }
    } else {
        max_common_level = global_max_level;
    }

    for (auto level = static_cast<int>(max_common_level); level >= 0; level--) {
        vecsim_stl::updatable_max_heap<DistType, idType> top_candidates =
            searchLayer(curr_element, vector_data, level, efConstruction);

        if (!top_candidates.empty()) {
            curr_element = mutuallyConnectNewElement(element_id, top_candidates, level,
                                                     modifiedNodes);
        } else {
            this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                      "WARNING: No candidates found at level %d!", level);
        }
    }
}

template <typename DataType, typename DistType>
idType HNSWDiskIndex<DataType, DistType>::mutuallyConnectNewElement(
    idType new_node_id, vecsim_stl::updatable_max_heap<DistType, idType> &top_candidates,
    size_t level, vecsim_stl::vector<uint64_t> &modifiedNodes) {

    size_t max_M_cur = level ? M : M0;

    // Copy candidates to list for heuristic processing
    candidatesList<DistType> top_candidates_list(this->allocator);
    top_candidates_list.insert(top_candidates_list.end(), top_candidates.begin(),
                               top_candidates.end());

    // Use heuristic to filter candidates
    idType next_closest_entry_point = getNeighborsByHeuristic2(top_candidates_list, M);

    // Extract selected neighbor IDs
    vecsim_stl::vector<idType> neighbor_ids(this->allocator);
    neighbor_ids.reserve(top_candidates_list.size());
    for (size_t i = 0; i < top_candidates_list.size(); ++i) {
        neighbor_ids.push_back(top_candidates_list[i].second);
    }

    // Update cache for the new node's neighbors
    setNeighborsInCache(new_node_id, level, neighbor_ids);
    modifiedNodes.push_back(makeRepairKey(new_node_id, level));

    // Update existing nodes to include the new node in their neighbor lists
    for (const auto &neighbor_data : top_candidates_list) {
        idType selected_neighbor = neighbor_data.second;
        DistType distance = neighbor_data.first;

        // Use atomic check-and-add to prevent race conditions
        if (tryAddNeighborToCacheIfCapacity(selected_neighbor, level, new_node_id, max_M_cur)) {
            // Successfully added - mark as modified
            modifiedNodes.push_back(makeRepairKey(selected_neighbor, level));
        } else {
            // Full - need to re-evaluate using heuristic
            revisitNeighborConnections(new_node_id, selected_neighbor, level, distance,
                                       modifiedNodes);
        }
    }

    return next_closest_entry_point;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::executeSingleInsertJobWrapper(AsyncJob *job) {
    auto *insertJob = static_cast<HNSWDiskSingleInsertJob *>(job);
    auto *index = static_cast<HNSWDiskIndex<DataType, DistType> *>(job->index);
    index->executeSingleInsertJob(insertJob);
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::executeSingleInsertJob(HNSWDiskSingleInsertJob *job) {
    if (!job->isValid) {
        delete job;
        return;
    }

    // Get current entry point
    idType currentEntryPoint;
    size_t currentMaxLevel;
    {
        std::shared_lock<std::shared_mutex> lock(indexDataGuard);
        currentEntryPoint = entrypointNode;
        currentMaxLevel = maxLevel;
    }

    // Use unified core function (batching controlled by diskWriteBatchThreshold)
    executeGraphInsertionCore(job->vectorId, job->elementMaxLevel, currentEntryPoint,
                              currentMaxLevel, job->rawVectorData.data(),
                              job->processedVectorData.data());

    delete job;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::executeGraphInsertionCore(
    idType vectorId, size_t elementMaxLevel,
    idType entryPoint, size_t globalMaxLevel,
    const void *rawVectorData, const void *processedVectorData) {

    if (entryPoint == INVALID_ID || vectorId == entryPoint) {
        // Entry point or first vector - nothing to connect
        // Raw vector cleanup happens in addVector for entry point case
        return;
    }

    // Track modified nodes for disk write
    vecsim_stl::vector<uint64_t> modifiedNodes(this->allocator);

    // Hold shared lock during graph insertion to prevent idToMetaData resize
    // while we're reading from it in searchLayer/processCandidate
    {
        std::shared_lock<std::shared_mutex> lock(indexDataGuard);
        // Insert into graph
        insertElementToGraph(vectorId, elementMaxLevel, entryPoint, globalMaxLevel,
                             rawVectorData, processedVectorData, modifiedNodes);
    }

    // Handle disk writes - both single-threaded and multi-threaded support batching
    // diskWriteBatchThreshold controls when to flush:
    //   0 = flush after every insert (no batching)
    //   >0 = flush when dirty count reaches threshold
    if (diskWriteBatchThreshold == 0) {
        // No batching: write immediately
        writeDirtyNodesToDisk(modifiedNodes, rawVectorData, vectorId);
        std::lock_guard<std::shared_mutex> lock(rawVectorsGuard);
        rawVectorsInRAM.erase(vectorId);
    } else {
        // Check if flush is needed based on dirty count OR raw vector memory accumulation
        // This prevents unbounded memory growth when vectors are added but threshold isn't reached
        bool shouldFlush = totalDirtyCount_.load(std::memory_order_relaxed) >= diskWriteBatchThreshold;
        if (!shouldFlush) {
            std::shared_lock<std::shared_mutex> lock(rawVectorsGuard);
            shouldFlush = rawVectorsInRAM.size() >= diskWriteBatchThreshold;
        }
        if (shouldFlush) {
            flushDirtyNodesToDisk();
        }
    }
    // else: batching enabled but threshold not reached, keep accumulating

    // Update entry point if this vector has higher level
    if (elementMaxLevel > globalMaxLevel) {
        std::unique_lock<std::shared_mutex> lock(indexDataGuard);
        if (elementMaxLevel > maxLevel) {
            entrypointNode = vectorId;
            maxLevel = elementMaxLevel;
        }
    }
}

#ifdef BUILD_TESTS
#include "hnsw_disk_serializer.h"
#endif
