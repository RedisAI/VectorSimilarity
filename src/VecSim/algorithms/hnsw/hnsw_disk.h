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

    // Index global state - these should be guarded by the indexDataGuard lock in
    // multithreaded scenario.
    size_t curElementCount;
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

    // Global batch operation state
    mutable std::unordered_map<idType, std::vector<idType>> delta_list;
    mutable vecsim_stl::vector<DiskElementMetaData> new_elements_meta_data;

    // Batch processing state
    size_t batchThreshold; // Number of vectors to accumulate before batch update
    vecsim_stl::vector<idType> pendingVectorIds;             // Vector IDs waiting to be indexed
    vecsim_stl::vector<DiskElementMetaData> pendingMetadata; // Metadata for pending vectors
    size_t pendingVectorCount;                               // Count of vectors in memory

    /**
     * Threshold for batching delete operations.
     * When the number of pending deletions reaches this value, the deletions are processed in a batch.
     */
    size_t deleteBatchThreshold = 10;
    vecsim_stl::vector<idType> pendingDeleteIds;

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

    // Can maybe merge with stagedInsertUpdates
    vecsim_stl::vector<NeighborUpdate> stagedInsertNeighborUpdates;

    // Temporary storage for raw vectors in RAM (until flush batch)
    // Maps idType -> raw vector data (stored as string for simplicity)
    std::unordered_map<idType, std::string> rawVectorsInRAM;

    // Cache for raw vectors retrieved from disk (mutable to allow caching in const methods)
    mutable std::unordered_map<idType, std::string> rawVectorsDiskCache;

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
    void insertElementToGraph(idType element_id, size_t element_max_level, idType entry_point,
                              size_t global_max_level, const void *vector_data);
    idType mutuallyConnectNewElement(idType new_node_id,
                                     candidatesMaxHeap<DistType> &top_candidates, size_t level);

    // Batch processing methods
    void processBatch();
    void flushBatch(); // Force flush current batch

    void processDeleteBatch();
    void flushDeleteBatch(); // Force flush current delete batch

    // Helper methods
    void getNeighbors(idType nodeId, size_t level, vecsim_stl::vector<idType>& result) const;
    void searchPendingVectors(const void* query_data, candidatesLabelsMaxHeap<DistType>& top_candidates, size_t k) const;

    // Manual control of staged updates
    void flushStagedUpdates(); // Manually flush any pending staged updates

protected:
    // Helper method to filter deleted or invalid nodes from a neighbor list (DRY principle)
    // Returns true if any nodes were filtered out
    // Filters out: nodes marked as deleted, and nodes with invalid IDs (>= curElementCount)
    inline bool filterDeletedNodes(vecsim_stl::vector<idType>& neighbors) const {
        size_t original_size = neighbors.size();
        auto new_end = std::remove_if(neighbors.begin(), neighbors.end(),
            [this](idType id) { return id >= curElementCount || isMarkedDeleted(id); });
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

    // New method for handling neighbor connection updates when neighbor lists are full
    void stageRevisitNeighborConnections(idType new_node_id, idType selected_neighbor,
                                       size_t level, DistType distance);

    // void patchDeltaList(std::unordered_map<idType, std::vector<idType>> &delta_list,
    //                     vecsim_stl::vector<DiskElementMetaData> &new_elements_meta_data,
    //                     std::unordered_map<idType, idType> &new_ids_mapping);

public:
    // Methods needed by benchmark framework
    const void *getDataByInternalId(idType id) const;
    candidatesMaxHeap<DistType> searchLayer(idType ep_id, const void *data_point, size_t level,
                                            size_t ef) const;
    void greedySearchLevel(const void *data_point, size_t level, idType &curr_element,
                           DistType &cur_dist) const;
    std::pair<idType, size_t> safeGetEntryPointState() const;
    VisitedNodesHandler *getVisitedList() const;
    void returnVisitedList(VisitedNodesHandler *visited_nodes_handler) const;
    candidatesLabelsMaxHeap<DistType> *getNewMaxPriorityQueue() const;
    bool isMarkedDeleted(idType id) const;
    labelType getExternalLabel(idType id) const;
    void processCandidate(idType candidate_id, const void *data_point, size_t level, size_t ef,
                          void *visited_tags, size_t visited_tag,
                          candidatesLabelsMaxHeap<DistType> &top_candidates,
                          candidatesMaxHeap<DistType> &candidate_set, DistType &lowerBound) const;

    // Raw vector storage and retrieval methods
    const char* getRawVector(idType id) const;

protected:
    idType searchBottomLayerEP(const void *query_data, void *timeoutCtx = nullptr,
                               VecSimQueryReply_Code *rc = nullptr) const;

    candidatesLabelsMaxHeap<DistType> *
    searchBottomLayer_WithTimeout(idType ep_id, const void *data_point, size_t ef, size_t k,
                                  void *timeoutCtx = nullptr,
                                  VecSimQueryReply_Code *rc = nullptr) const;

    // New hierarchical search method
    candidatesLabelsMaxHeap<DistType> *
    hierarchicalSearch(const void *data_point, idType ep_id, size_t ef, size_t k,
                       void *timeoutCtx = nullptr, VecSimQueryReply_Code *rc = nullptr) const;

public:
    HNSWDiskIndex(const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
                  const IndexComponents<DataType, DistType> &components, rocksdb::DB *db,
                  rocksdb::ColumnFamilyHandle *cf, const std::string &dbPath = "",
                  size_t random_seed = 100);
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

public:
    // Public methods for testing
    size_t indexSize() const override;
    size_t indexCapacity() const override;
    size_t indexLabelCount() const override;
    size_t getRandomLevel(double reverse_size);

    // Flagging API
    template <Flags FLAG>
    void markAs(idType internalId) {
        __atomic_fetch_or(&idToMetaData[internalId].flags, FLAG, 0);
    }
    template <Flags FLAG>
    void unmarkAs(idType internalId) {
        __atomic_fetch_and(&idToMetaData[internalId].flags, ~FLAG, 0);
    }
    template <Flags FLAG>
    bool isMarkedAs(idType internalId) const {
        return __atomic_load_n(&idToMetaData[internalId].flags, 0) & FLAG;
    }

    // Mark delete API
    vecsim_stl::vector<idType> markDelete(labelType label);
    size_t getNumMarkedDeleted() const { return numMarkedDeleted; }

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
    rocksdb::ColumnFamilyHandle *cf, const std::string &dbPath, size_t random_seed)
    : VecSimIndexAbstract<DataType, DistType>(abstractInitParams, components),
      idToMetaData(INITIAL_CAPACITY, this->allocator), labelToIdMap(this->allocator), db(db),
      cf(cf), dbPath(dbPath), indexDataGuard(),
      visitedNodesHandlerPool(INITIAL_CAPACITY, this->allocator), delta_list(),
      new_elements_meta_data(this->allocator), batchThreshold(10),
      pendingVectorIds(this->allocator), pendingMetadata(this->allocator), pendingVectorCount(0),
      pendingDeleteIds(this->allocator),
      stagedInsertUpdates(this->allocator),
      stagedDeleteUpdates(this->allocator), stagedRepairUpdates(this->allocator),
      stagedInsertNeighborUpdates(this->allocator) {

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

    // initializations for special treatment of the first node
    entrypointNode = INVALID_ID;
    maxLevel = HNSW_INVALID_LEVEL;

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
    pendingVectorIds.clear();
    pendingMetadata.clear();
    pendingDeleteIds.clear();

    // Clear delta list and new elements metadata
    delta_list.clear();
    new_elements_meta_data.clear();

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

    if ((curElementCount == 0 && pendingVectorCount == 0) || k == 0) {
        return rep;
    }

    // Preprocess the query
    auto processed_query_ptr = this->preprocessQuery(query_data);
    const void *processed_query = processed_query_ptr.get();

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

    // Step 2: Perform hierarchical search from top level down to bottom level
    // Use a more sophisticated search that properly traverses the HNSW hierarchy
    auto *results = hierarchicalSearch(processed_query, bottom_layer_ep, std::max(query_ef, k), k,
                                       timeoutCtx, &rep->code);

    if (VecSim_OK == rep->code && results) {
        // Step 3: Also search pending batch vectors and merge results
        if (pendingVectorCount > 0) {
            searchPendingVectors(processed_query, *results, k);
        }

        rep->results.resize(results->size());
        for (auto result = rep->results.rbegin(); result != rep->results.rend(); result++) {
            std::tie(result->score, result->id) = results->top();
            results->pop();
        }
    } else {
        // Even if main search failed, still search pending vectors
        if (pendingVectorCount > 0) {
            // Create a simple vector to store pending results
            std::vector<std::pair<DistType, labelType>> pending_results;
            pending_results.reserve(pendingVectorCount);

            // Search pending vectors manually
            for (size_t i = 0; i < pendingVectorCount; i++) {
                idType vectorId = pendingVectorIds[i];
                const void *vector_data = this->vectors->getElement(vectorId);
                const DiskElementMetaData &metadata = idToMetaData[vectorId];
                labelType label = metadata.label;
                DistType dist = this->calcDistance(processed_query, vector_data);

                pending_results.emplace_back(dist, label);
            }

            // Sort by distance and take top k
            std::sort(pending_results.begin(), pending_results.end());
            if (pending_results.size() > k) {
                pending_results.resize(k);
            }

            if (!pending_results.empty()) {
                rep->results.resize(pending_results.size());
                for (size_t i = 0; i < pending_results.size(); i++) {
                    rep->results[i].score = pending_results[i].first;
                    rep->results[i].id = pending_results[i].second;
                }
                rep->code = VecSim_QueryReply_OK; // Mark as successful since we found results
            }
        }
    }

    delete results;
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

    // Store raw vector in RAM first (until flush batch)
    // We need to store the original vector before preprocessing
    idType newElementId = curElementCount;
    const char* raw_data = reinterpret_cast<const char*>(vector);
    rawVectorsInRAM[newElementId] = std::string(raw_data, this->inputBlobSize);

    // Preprocess the vector
    ProcessedBlobs processedBlobs = this->preprocess(vector);

    // Store the processed vector in memory
    size_t containerId = this->vectors->size();
    this->vectors->addElement(processedBlobs.getStorageBlob(), containerId);

    // Create new element ID and metadata
    size_t elementMaxLevel = getRandomLevel(mult);
    DiskElementMetaData new_element(label, elementMaxLevel);

    // Ensure capacity for the new element ID
    if (newElementId >= indexCapacity()) {
        size_t new_cap = ((newElementId + this->blockSize) / this->blockSize) * this->blockSize;
        visitedNodesHandlerPool.resize(new_cap);
        idToMetaData.resize(new_cap);
        labelToIdMap.reserve(new_cap);
    }

    // Store metadata immediately
    idToMetaData[newElementId] = new_element;
    labelToIdMap[label] = newElementId;

    // Increment vector count immediately
    curElementCount++;


    // Resize visited nodes handler pool to accommodate new elements
    visitedNodesHandlerPool.resize(curElementCount);

    // Add only the vector ID to pending vectors for indexing
    pendingVectorIds.push_back(newElementId);
    pendingVectorCount++;

    // Process batch if threshold reached
    if (pendingVectorCount >= batchThreshold) {
        processBatch();
    }

    return 1; // Success
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::insertElementToGraph(idType element_id,
                                                             size_t element_max_level,
                                                             idType entry_point,
                                                             size_t global_max_level,
                                                             const void *vector_data) {

    idType curr_element = entry_point;
    DistType cur_dist = std::numeric_limits<DistType>::max();
    size_t max_common_level;
    if (element_max_level < global_max_level) {
        max_common_level = element_max_level;
        cur_dist = this->calcDistance(vector_data, getDataByInternalId(curr_element));
        for (auto level = static_cast<int>(global_max_level);
             level > static_cast<int>(element_max_level); level--) {
            // this is done for the levels which are above the max level
            // to which we are going to insert the new element. We do
            // a greedy search in the graph starting from the entry point
            // at each level, and move on with the closest element we can find.
            // When there is no improvement to do, we take a step down.
            greedySearchLevel(vector_data, level, curr_element, cur_dist);
        }
    } else {
        max_common_level = global_max_level;
    }

    for (auto level = static_cast<int>(max_common_level); level >= 0; level--) {
        candidatesMaxHeap<DistType> top_candidates =
            searchLayer(curr_element, vector_data, level, efConstruction);

        // If the entry point was marked deleted between iterations, we may receive an empty
        // candidates set.
        if (!top_candidates.empty()) {
            curr_element = mutuallyConnectNewElement(element_id, top_candidates, level);
        } else {
            this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                      "WARNING: No candidates found at level %d!", level);
        }
    }
}

template <typename DataType, typename DistType>
idType HNSWDiskIndex<DataType, DistType>::mutuallyConnectNewElement(
    idType new_node_id, candidatesMaxHeap<DistType> &top_candidates, size_t level) {

    // The maximum number of neighbors allowed for an existing neighbor (not new).
    size_t max_M_cur = level ? M : M0;

    // Filter the top candidates to the selected neighbors by the algorithm heuristics.
    // First, we need to copy the top candidates to a vector.
    candidatesList<DistType> top_candidates_list(this->allocator);
    top_candidates_list.insert(top_candidates_list.end(), top_candidates.begin(),
                               top_candidates.end());
    // Use the heuristic to filter the top candidates, and get the next closest entry point.
    idType next_closest_entry_point = getNeighborsByHeuristic2(top_candidates_list, M);
    assert(top_candidates_list.size() <= M &&
           "Should be not be more than M candidates returned by the heuristic");

    // Instead of writing to disk immediately, stage the updates in memory
    // Stage the new node's neighbors
    vecsim_stl::vector<idType> neighbor_ids(this->allocator);
    neighbor_ids.reserve(top_candidates_list.size());
    for (size_t i = 0; i < top_candidates_list.size(); ++i) {
        neighbor_ids.push_back(top_candidates_list[i].second);
    }

    // Add to staged graph updates (for insertions)
    uint64_t insert_key = makeRepairKey(new_node_id, level);
    stagedInsertMap[insert_key] = stagedInsertUpdates.size();
    stagedInsertUpdates.emplace_back(new_node_id, level, neighbor_ids, this->allocator);

    // Stage updates to existing nodes to include the new node in their neighbor lists
    for (const auto &neighbor_data : top_candidates_list) {
        idType selected_neighbor = neighbor_data.second;
        DistType distance = neighbor_data.first;

        // Check if the neighbor's neighbor list has capacity
        // For disk-based implementation, we need to determine if we need to re-evaluate the
        // neighbor's connections

        // Read the neighbor's current neighbor count from disk to check capacity
        GraphKey neighborKey(selected_neighbor, level);
        std::string existing_neighbors_data;
        rocksdb::Status status =
            db->Get(rocksdb::ReadOptions(), cf, neighborKey.asSlice(), &existing_neighbors_data);

        size_t current_neighbor_count = 0;
        if (status.ok()) {
            current_neighbor_count = existing_neighbors_data.size() / sizeof(idType);
        }

        if (current_neighbor_count < max_M_cur) {
            // Neighbor has capacity, just add the new node
            stagedInsertNeighborUpdates.emplace_back(selected_neighbor, level, new_node_id);
        } else {
            // Neighbor is full, need to re-evaluate connections using revisitNeighborConnections
            // logic
            stageRevisitNeighborConnections(new_node_id, selected_neighbor, level, distance);
        }
    }

    return next_closest_entry_point;
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

    // First, handle new node insertions and updates
    for (const auto &update : graphUpdates) {
        auto newKey = GraphKey(update.node_id, update.level);

        // If neighbors list is empty, this is a deletion - remove the key from disk
        if (update.neighbors.empty()) {
            graphBatch.Delete(cf, newKey.asSlice());
            continue;
        }

        // Get raw vector data
        const void* raw_vector_data = getRawVector(update.node_id);
        if (raw_vector_data == nullptr) {
            this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                        "WARNING: Skipping graph update for node %u at level %zu - no raw vector data available",
                        update.node_id, update.level);
            continue;
        }

        // Serialize with format: [raw_vector_data][neighbor_count][neighbor_ids...]
        std::string graph_value = serializeGraphValue(raw_vector_data, update.neighbors);
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

    // Process each node's neighbor updates
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

            // Add new neighbors (avoiding duplicates) using a hash set for O(1) lookup
            std::unordered_set<idType> neighbor_set(updated_neighbors.begin(), updated_neighbors.end());
            for (idType new_neighbor : newNeighbors) {
                if (neighbor_set.find(new_neighbor) == neighbor_set.end()) {
                    updated_neighbors.push_back(new_neighbor);
                    neighbor_set.insert(new_neighbor);
                }
            }

            const void* raw_vector_data = getRawVector(node_id);

            // Serialize with new format and add to batch
            std::string graph_value = serializeGraphValue(raw_vector_data, updated_neighbors);
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
void HNSWDiskIndex<DataType, DistType>::stageRevisitNeighborConnections(idType new_node_id,
                                                                        idType selected_neighbor,
                                                                        size_t level,
                                                                        DistType distance) {
    // Read the neighbor's current neighbor list from disk
    // TODO: perhaps cache the neigbhors for stage update
    GraphKey neighborKey(selected_neighbor, level);
    std::string graph_value;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, neighborKey.asSlice(), &graph_value);

    if (!status.ok()) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                  "  WARNING: Could not read existing neighbors for node %u at level %zu",
                  selected_neighbor, level);
        // Fall back to simple neighbor update
        stagedInsertNeighborUpdates.emplace_back(selected_neighbor, level, new_node_id);
        return;
    }

    // Parse existing neighbors using new format
    vecsim_stl::vector<idType> existing_neighbors(this->allocator);
    deserializeGraphValue(graph_value, existing_neighbors);

    // Collect all candidates: existing neighbors + new node
    candidatesList<DistType> candidates(this->allocator);
    candidates.reserve(existing_neighbors.size() + 1);

    // Add the new node with its pre-calculated distance
    candidates.emplace_back(distance, new_node_id);

    // Add existing neighbors with their distances to the selected neighbor
    const void* selected_neighbor_data = getDataByInternalId(selected_neighbor);
    for (size_t j = 0; j < existing_neighbors.size(); j++) {
        idType existing_neighbor_id = existing_neighbors[j];
        const void *existing_neighbor_data = getDataByInternalId(existing_neighbor_id);
        DistType existing_distance =
            this->calcDistance(existing_neighbor_data, selected_neighbor_data);
        candidates.emplace_back(existing_distance, existing_neighbor_id);
    }

    // Use the heuristic to select the best neighbors (similar to revisitNeighborConnections in
    // hnsw.h)
    size_t max_M_cur = level ? M : M0;

    // Apply the neighbor selection heuristic
    vecsim_stl::vector<idType> removed_candidates(this->allocator);
    getNeighborsByHeuristic2(candidates, max_M_cur, removed_candidates);

    // Check if the new node was selected as a neighbor
    bool new_node_selected = false;
    for (const auto &candidate : candidates) {
        if (candidate.second == new_node_id) {
            new_node_selected = true;
            break;
        }
    }

    if (new_node_selected) {
        // The new node was selected, so we need to update the neighbor's neighbor list
        // Extract the selected neighbor IDs
        vecsim_stl::vector<idType> selected_neighbor_ids(this->allocator);
        selected_neighbor_ids.reserve(candidates.size());
        for (const auto &candidate : candidates) {
            selected_neighbor_ids.push_back(candidate.second);
        }

        // Stage this update - the neighbor's neighbor list will be completely replaced
        // We'll need to handle this specially in flushStagedGraphUpdates
        uint64_t insert_key = makeRepairKey(selected_neighbor, level);
        stagedInsertMap[insert_key] = stagedInsertUpdates.size();
        stagedInsertUpdates.emplace_back(selected_neighbor, level, selected_neighbor_ids,
                                        this->allocator);

        // Also stage the bidirectional connection from new node to selected neighbor
        stagedInsertNeighborUpdates.emplace_back(new_node_id, level, selected_neighbor);

    } else {
        // The new node was not selected, so we only need to stage the unidirectional connection
        // from new node to selected neighbor
        stagedInsertNeighborUpdates.emplace_back(new_node_id, level, selected_neighbor);
    }
}

template <typename DataType, typename DistType>
idType HNSWDiskIndex<DataType, DistType>::searchBottomLayerEP(const void *query_data,
                                                              void *timeoutCtx,
                                                              VecSimQueryReply_Code *rc) const {
    if (rc)
        *rc = VecSim_QueryReply_OK;

    auto [curr_element, max_level] = safeGetEntryPointState();
    if (curr_element == INVALID_ID)
        return curr_element; // index is empty.

    DistType cur_dist = this->calcDistance(query_data, getDataByInternalId(curr_element));
    for (size_t level = max_level; level > 0 && curr_element != INVALID_ID; --level) {
        greedySearchLevel(query_data, level, curr_element, cur_dist);
    }
    return curr_element;
}

template <typename DataType, typename DistType>
candidatesLabelsMaxHeap<DistType> *HNSWDiskIndex<DataType, DistType>::searchBottomLayer_WithTimeout(
    idType ep_id, const void *data_point, size_t ef, size_t k, void *timeoutCtx,
    VecSimQueryReply_Code *rc) const {

    // Use a simple set for visited nodes tracking
    std::unordered_set<idType> visited_set;

    candidatesLabelsMaxHeap<DistType> *top_candidates = getNewMaxPriorityQueue();
    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    DistType lowerBound;
    if (!isMarkedDeleted(ep_id)) {
        // If ep is not marked as deleted, get its distance and set lower bound and heaps
        // accordingly
        DistType dist = this->calcDistance(data_point, getDataByInternalId(ep_id));
        lowerBound = dist;
        top_candidates->emplace(dist, getExternalLabel(ep_id));
        candidate_set.emplace(-dist, ep_id);
    } else {
        // If ep is marked as deleted, set initial lower bound to max, and don't insert to top
        // candidates heap
        lowerBound = std::numeric_limits<DistType>::max();
        candidate_set.emplace(-lowerBound, ep_id);
    }

    visited_set.insert(ep_id);

    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();

        if ((-curr_el_pair.first) > lowerBound && top_candidates->size() >= ef) {
            break;
        }
        if (timeoutCtx && VECSIM_TIMEOUT(timeoutCtx)) {
            if (rc)
                *rc = VecSim_QueryReply_TimedOut;
            return top_candidates;
        }
        candidate_set.pop();

        processCandidate(curr_el_pair.second, data_point, 0, ef,
                         reinterpret_cast<void *>(&visited_set), 0, *top_candidates, candidate_set,
                         lowerBound);
    }

    while (top_candidates->size() > k) {
        top_candidates->pop();
    }
    if (rc)
        *rc = VecSim_QueryReply_OK;
    return top_candidates;
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
    assert(id < curElementCount);

    if (id < this->vectors->size()) {
        const void* result = this->vectors->getElement(id);
        if (result != nullptr) {
            return result;
        }
    }

    this->log(VecSimCommonStrings::LOG_WARNING_STRING,
             "WARNING: Vector data not found for id %u", id);
    return nullptr;
}

template <typename DataType, typename DistType>
const char* HNSWDiskIndex<DataType, DistType>::getRawVector(idType id) const {

    if (id >= curElementCount) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                 "WARNING: getRawVector called with invalid id %u (current count: %zu)",
                 id, curElementCount);
        return nullptr;
    }

    // First check RAM (for vectors not yet flushed)
    auto it = rawVectorsInRAM.find(id);
    if (it != rawVectorsInRAM.end()) {
        const char* data_ptr = it->second.data();
        return data_ptr;
    }

    // If not in RAM or cache, retrieve from disk
    GraphKey graphKey(id, 0);
    std::string level0_graph_value;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &level0_graph_value);

    if (status.ok()) {
        // Extract vector data
        const void* vector_data = getVectorFromGraphValue(level0_graph_value);
        if (vector_data != nullptr) {
            // Cache the raw vector data
            const char* data_ptr = reinterpret_cast<const char*>(vector_data);
            rawVectorsDiskCache[id] = std::string(data_ptr, this->inputBlobSize);
            return rawVectorsDiskCache[id].data();
        } else {
            this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                     "WARNING: getVectorFromGraphValue returned nullptr for id %u (graph value size: %zu)",
                     id, level0_graph_value.size());
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

    return nullptr;
}

template <typename DataType, typename DistType>
candidatesMaxHeap<DistType>
HNSWDiskIndex<DataType, DistType>::searchLayer(idType ep_id, const void *data_point, size_t level,
                                               size_t ef) const {
    candidatesMaxHeap<DistType> top_candidates(this->allocator);
    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    // Get visited list
    auto *visited_nodes_handler = getVisitedList();
    tag_t visited_tag = visited_nodes_handler->getFreshTag();

    // Start with the entry point and initialize lowerBound
    DistType dist = this->calcDistance(data_point, getDataByInternalId(ep_id));
    DistType lowerBound = dist;
    top_candidates.emplace(dist, ep_id);
    candidate_set.emplace(-dist, ep_id);
    visited_nodes_handler->tagNode(ep_id, visited_tag);

    // Search for candidates
    while (!candidate_set.empty()) {
        auto curr_pair = candidate_set.top();
        DistType curr_dist = -curr_pair.first;

        // Early termination: if we have enough candidates and current distance is worse than
        // lowerBound, stop
        if (curr_dist > lowerBound && top_candidates.size() >= ef) {
            break;
        }

        idType curr_id = curr_pair.second;
        candidate_set.pop();

        // Get neighbors of current node at this level
        vecsim_stl::vector<idType> neighbors(this->allocator);
        getNeighbors(curr_id, level, neighbors);

        for (idType neighbor_id : neighbors) {
            if (visited_nodes_handler->getNodeTag(neighbor_id) == visited_tag) {
                continue;
            }

            visited_nodes_handler->tagNode(neighbor_id, visited_tag);
            DistType neighbor_dist =
                this->calcDistance(data_point, getDataByInternalId(neighbor_id));

            // Add to top candidates if it's good enough
            if (neighbor_dist < lowerBound || top_candidates.size() < ef) {
                top_candidates.emplace(neighbor_dist, neighbor_id);
                candidate_set.emplace(-neighbor_dist, neighbor_id);

                // Update lowerBound if we have enough candidates
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
                if (top_candidates.size() >= ef) {
                    lowerBound = top_candidates.top().first;
                }
            }
        }
    }

    returnVisitedList(visited_nodes_handler);
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

template <typename DataType, typename DistType>
std::pair<idType, size_t> HNSWDiskIndex<DataType, DistType>::safeGetEntryPointState() const {
    std::shared_lock<std::shared_mutex> lock(indexDataGuard);
    return std::make_pair(entrypointNode, maxLevel);
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::greedySearchLevel(const void *data_point, size_t level,
                                                          idType &curr_element,
                                                          DistType &cur_dist) const {
    bool changed;
    idType bestCand = curr_element;

    do {
        changed = false;

        // Read neighbors from RocksDB for the current node at this level
        GraphKey graphKey(bestCand, level);
        std::string graph_value;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &graph_value);

        if (!status.ok()) {
            // No neighbors found for this node at this level, stop searching
            break;
        }

        // Parse the neighbors using new format
        vecsim_stl::vector<idType> neighbors(this->allocator);
        deserializeGraphValue(graph_value, neighbors);

        // Check each neighbor to find a better candidate
        for (size_t i = 0; i < neighbors.size(); i++) {
            idType candidate = neighbors[i];

            // Skip invalid candidates
            if (candidate >= curElementCount) {
                continue;
            }

            // Skip deleted candidates
            if (isMarkedDeleted(candidate)) {
                continue;
            }

            // Calculate distance to this candidate
            DistType d = this->calcDistance(data_point, getDataByInternalId(candidate));

            // If this candidate is closer, update our best candidate
            if (d < cur_dist) {
                cur_dist = d;
                bestCand = candidate;
                changed = true;
            }
        }

    } while (changed);

    // Update the current element to the best candidate found
    curr_element = bestCand;
}

template <typename DataType, typename DistType>
candidatesLabelsMaxHeap<DistType> *
HNSWDiskIndex<DataType, DistType>::getNewMaxPriorityQueue() const {
    // Use max_priority_queue for single-label disk index
    return new (this->allocator)
        vecsim_stl::max_priority_queue<DistType, labelType>(this->allocator);
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
void HNSWDiskIndex<DataType, DistType>::processCandidate(
    idType candidate_id, const void *data_point, size_t level, size_t ef, void *visited_tags,
    size_t visited_tag, candidatesLabelsMaxHeap<DistType> &top_candidates,
    candidatesMaxHeap<DistType> &candidate_set, DistType &lowerBound) const {
    // Use a simple set-based approach for now to avoid visited nodes handler issues
    auto *visited_set = reinterpret_cast<std::unordered_set<idType> *>(visited_tags);
    if (!visited_set) {
        return; // Safety check
    }

    if (visited_set->find(candidate_id) != visited_set->end()) {
        return;
    }

    visited_set->insert(candidate_id);

    // Calculate distance to candidate
    DistType dist = this->calcDistance(data_point, getDataByInternalId(candidate_id));

    // Add to top candidates if it's one of the best and not marked deleted
    if (!isMarkedDeleted(candidate_id)) {
        if (top_candidates.size() < ef || dist < lowerBound) {
            top_candidates.emplace(dist, getExternalLabel(candidate_id));

            // Update lower bound if we have enough candidates
            if (top_candidates.size() >= ef) {
                lowerBound = top_candidates.top().first;
            }
        }
    }

    // Add neighbors to candidate set for further exploration
    vecsim_stl::vector<idType> neighbors(this->allocator);
    getNeighbors(candidate_id, level, neighbors);

    if (!neighbors.empty()) {
        for (idType neighbor_id : neighbors) {
            // Skip invalid neighbors
            if (neighbor_id >= curElementCount) {
                continue;
            }

            if (visited_set->find(neighbor_id) == visited_set->end()) {
                DistType neighbor_dist =
                    this->calcDistance(data_point, getDataByInternalId(neighbor_id));
                candidate_set.emplace(-neighbor_dist, neighbor_id);
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
    return this->curElementCount;
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::indexLabelCount() const {
    return labelToIdMap.size();
}

/********************************** Helper Methods **********************************/

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::getNeighbors(idType nodeId, size_t level, vecsim_stl::vector<idType>& result) const {
    // Clear the result vector first
    result.clear();

    // First check staged graph updates using hash maps for O(1) lookup
    uint64_t lookup_key = makeRepairKey(nodeId, level);

    // Check insert staging area
    auto insert_it = stagedInsertMap.find(lookup_key);
    if (insert_it != stagedInsertMap.end()) {
        const auto &update = stagedInsertUpdates[insert_it->second];
        result.reserve(update.neighbors.size());
        for (size_t i = 0; i < update.neighbors.size(); i++) {
            result.push_back(update.neighbors[i]);
        }
        // Filter out deleted nodes using helper
        filterDeletedNodes(result);
        return;
    }

    // Check delete staging area
    auto delete_it = stagedDeleteMap.find(lookup_key);
    if (delete_it != stagedDeleteMap.end()) {
        const auto &update = stagedDeleteUpdates[delete_it->second];
        result.reserve(update.neighbors.size());
        for (size_t i = 0; i < update.neighbors.size(); i++) {
            result.push_back(update.neighbors[i]);
        }
        // Filter out deleted nodes using helper
        filterDeletedNodes(result);
        return;
    }

    // Also check staged repair updates (already cleaned neighbors waiting to be flushed)
    auto repair_it = stagedRepairMap.find(lookup_key);
    if (repair_it != stagedRepairMap.end()) {
        auto &update = stagedRepairUpdates[repair_it->second];
        result.reserve(update.neighbors.size());
        for (size_t i = 0; i < update.neighbors.size(); i++) {
            result.push_back(update.neighbors[i]);
        }
        // Filter in case nodes were deleted after this repair was staged
        if (filterDeletedNodes(result)) {
            // Update the existing repair entry with the more up-to-date cleaned list
            update.neighbors = result;
        }
        return;
    }

    // If not found in staged updates, check disk
    GraphKey graphKey(nodeId, level);

    std::string graph_value;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &graph_value);

    if (status.ok()) {
        deserializeGraphValue(graph_value, result);

        // Filter out deleted nodes and check if any were filtered
        if (filterDeletedNodes(result)) {
            // Lazy repair: if we filtered any deleted nodes, stage for cleanup
            // Use hash map for O(1) duplicate detection
            uint64_t repair_key = makeRepairKey(nodeId, level);
            if (stagedRepairMap.find(repair_key) == stagedRepairMap.end()) {
                stagedRepairMap[repair_key] = stagedRepairUpdates.size();
                stagedRepairUpdates.emplace_back(nodeId, level, result, this->allocator);
            }
        }
    }
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::searchPendingVectors(
    const void *query_data, candidatesLabelsMaxHeap<DistType> &top_candidates, size_t k) const {
    for (size_t i = 0; i < pendingVectorCount; i++) {
        idType vectorId = pendingVectorIds[i];
        if (isMarkedDeleted(vectorId)) {
            // Skip deleted vectors
            continue;
        }

        // Get the vector data from memory
        const void *vector_data = this->vectors->getElement(vectorId);

        // Get metadata for this vector
        const DiskElementMetaData &metadata = idToMetaData[vectorId];
        labelType label = metadata.label;

        // Calculate distance
        DistType dist = this->calcDistance(query_data, vector_data);

        // Add to candidates if it's good enough
        if (top_candidates.size() < k) {
            top_candidates.emplace(dist, label);
        } else if (dist < top_candidates.top().first) {
            top_candidates.pop();
            top_candidates.emplace(dist, label);
        }
    }
}

/********************************** Batch Processing Methods **********************************/

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::processBatch() {
    if (pendingVectorCount == 0) {
        return;
    }

    // Clear any previous staged updates (for insertions)
    stagedInsertUpdates.clear();
    stagedInsertMap.clear();
    stagedInsertNeighborUpdates.clear();

    // Process each pending vector ID (vectors are already stored in memory)
    for (size_t i = 0; i < pendingVectorCount; i++) {
        idType vectorId = pendingVectorIds[i];
        if (isMarkedDeleted(vectorId)) {
            // Skip deleted vectors
            continue;
        }

        // Get the vector data from memory
        const void *vector_data = this->vectors->getElement(vectorId);

        // Get metadata for this vector
        DiskElementMetaData &metadata = idToMetaData[vectorId];
        size_t elementMaxLevel = metadata.topLevel;

        // Insert into graph if not the first element
        if (entrypointNode != INVALID_ID) {
            insertElementToGraph(vectorId, elementMaxLevel, entrypointNode, maxLevel, vector_data);
        } else {
            // First element becomes the entry point
            entrypointNode = vectorId;
            maxLevel = elementMaxLevel;
        }
    }

    // Now flush all staged graph updates to disk in a single batch operation
    flushStagedGraphUpdates(stagedInsertUpdates, stagedInsertNeighborUpdates);
    stagedInsertMap.clear();

    // Clear the pending vector IDs
    pendingVectorIds.clear();
    rawVectorsInRAM.clear();
    pendingMetadata.clear();
    pendingVectorCount = 0;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::flushBatch() {
    processBatch();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::processDeleteBatch() {
    if (pendingDeleteIds.empty()) return;

    // Clear any previous staged updates (for deletions)
    stagedDeleteUpdates.clear();
    stagedDeleteMap.clear();

    // Create a set of IDs being deleted in this batch for quick lookup
    std::unordered_set<idType> deletingIds(pendingDeleteIds.begin(), pendingDeleteIds.end());

    // Process each deleted node
    for (idType deleted_id : pendingDeleteIds) {
        // Skip if already processed or invalid
        if (deleted_id >= curElementCount || deleted_id >= idToMetaData.size()) {
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
                if (neighbor_id >= curElementCount || isMarkedDeleted(neighbor_id) ||
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
                    const void* neighbor_data = getDataByInternalId(neighbor_id);

                    // Use a hash set to track candidate IDs for O(1) duplicate detection
                    std::unordered_set<idType> candidate_ids;

                    // Add existing neighbors (excluding the deleted node) with their distances
                    for (idType nn : neighbor_neighbors) {
                        if (nn != deleted_id && nn < curElementCount && !isMarkedDeleted(nn)) {
                            const void* nn_data = getDataByInternalId(nn);
                            DistType dist = this->calcDistance(nn_data, neighbor_data);
                            candidates.emplace_back(dist, nn);
                            candidate_ids.insert(nn);
                        }
                    }

                    // Add deleted node's neighbors (excluding current neighbor) as repair candidates
                    for (idType candidate_id : deleted_node_neighbors) {
                        if (candidate_id != neighbor_id &&
                            candidate_id < curElementCount &&
                            !isMarkedDeleted(candidate_id)) {

                            // Check if already in candidates to avoid duplicates using O(1) hash set lookup
                            if (candidate_ids.find(candidate_id) == candidate_ids.end()) {
                                const void* candidate_data = getDataByInternalId(candidate_id);
                                DistType dist = this->calcDistance(candidate_data, neighbor_data);
                                candidates.emplace_back(dist, candidate_id);
                                candidate_ids.insert(candidate_id);
                            }
                        }
                    }

                    vecsim_stl::unordered_set<idType> original_neighbors_set(this->allocator);
                    original_neighbors_set.reserve(neighbor_neighbors.size());
                    for (idType nn : neighbor_neighbors) {
                        if (nn != deleted_id && nn < curElementCount) {
                            original_neighbors_set.insert(nn);
                        }
                    }

                    // Apply heuristic to select best neighbors if we have more than max_M
                    vecsim_stl::vector<idType> updated_neighbors(this->allocator);
                    if (candidates.size() > max_M) {
                        // Use the same heuristic as during insertion for consistency
                        vecsim_stl::vector<idType> removed_candidates(this->allocator);
                        getNeighborsByHeuristic2(candidates, max_M, removed_candidates);

                        // Extract selected neighbor IDs
                        updated_neighbors.reserve(candidates.size());
                        for (const auto& [dist, id] : candidates) {
                            updated_neighbors.push_back(id);
                        }
                    } else {
                        // If we have fewer candidates than max_M, use them all
                        updated_neighbors.reserve(candidates.size());
                        for (const auto& [dist, id] : candidates) {
                            updated_neighbors.push_back(id);
                        }
                    }

                    // Stage the update for this neighbor (for deletions)
                    // Use helper lambda to safely stage updates (merge if key exists)
                    auto stageDeleteUpdate = [this](idType node_id, size_t lvl,
                                                     vecsim_stl::vector<idType>& neighbors) {
                        uint64_t key = makeRepairKey(node_id, lvl);
                        auto existing_it = stagedDeleteMap.find(key);
                        if (existing_it != stagedDeleteMap.end()) {
                            // Update existing entry in place
                            stagedDeleteUpdates[existing_it->second].neighbors = neighbors;
                        } else {
                            // Add new entry
                            stagedDeleteMap[key] = stagedDeleteUpdates.size();
                            stagedDeleteUpdates.emplace_back(node_id, lvl, neighbors, this->allocator);
                        }
                    };

                    stageDeleteUpdate(neighbor_id, level, updated_neighbors);

                    // Handle bidirectional edge updates for new connections
                    // For each new neighbor that wasn't originally connected, we need to add
                    // the repaired neighbor to their neighbor list (if bidirectional)
                    for (idType new_neighbor_id : updated_neighbors) {
                        if (original_neighbors_set.find(new_neighbor_id) == original_neighbors_set.end()) {
                            // This is a new connection created by repair
                            // Check if the new neighbor already points back to the repaired neighbor
                            vecsim_stl::vector<idType> new_neighbor_neighbors(this->allocator);
                            getNeighbors(new_neighbor_id, level, new_neighbor_neighbors);

                            bool already_bidirectional = false;
                            for (idType nn : new_neighbor_neighbors) {
                                if (nn == neighbor_id) {
                                    already_bidirectional = true;
                                    break;
                                }
                            }

                            // If not bidirectional, add the reverse connection
                            if (!already_bidirectional) {
                                // Stage a neighbor update to add the bidirectional connection
                                // This will be handled by adding neighbor_id to new_neighbor_id's list
                                // We need to check if new_neighbor_id's list is full and apply heuristic
                                // if needed

                                size_t max_neighbors = (level == 0) ? M0 : M;
                                if (new_neighbor_neighbors.size() < max_neighbors) {
                                    // Space available - simply add the reverse connection
                                    new_neighbor_neighbors.push_back(neighbor_id);
                                    stageDeleteUpdate(new_neighbor_id, level, new_neighbor_neighbors);
                                } else {
                                    // List is full - apply heuristic to decide if we should replace
                                    // an existing neighbor with the new repair edge.
                                    // This maintains bidirectionality which is critical for HNSW
                                    // recall quality (avoids "trap" nodes that are easy to enter
                                    // but hard to exit during greedy search).

                                    // Build candidate list: existing neighbors + the new repair edge
                                    candidatesList<DistType> reverse_candidates(this->allocator);
                                    reverse_candidates.reserve(new_neighbor_neighbors.size() + 1);

                                    const void* new_neighbor_data = getDataByInternalId(new_neighbor_id);

                                    // Add existing neighbors with their distances
                                    for (idType nn : new_neighbor_neighbors) {
                                        if (nn < curElementCount && !isMarkedDeleted(nn)) {
                                            const void* nn_data = getDataByInternalId(nn);
                                            DistType dist = this->calcDistance(nn_data, new_neighbor_data);
                                            reverse_candidates.emplace_back(dist, nn);
                                        }
                                    }

                                    // Add the repair edge (neighbor_id -> new_neighbor_id's reverse)
                                    DistType repair_dist = this->calcDistance(neighbor_data, new_neighbor_data);
                                    reverse_candidates.emplace_back(repair_dist, neighbor_id);

                                    // Apply heuristic to select the best neighbors
                                    vecsim_stl::vector<idType> removed_from_reverse(this->allocator);
                                    getNeighborsByHeuristic2(reverse_candidates, max_neighbors, removed_from_reverse);

                                    // Check if the repair edge was selected
                                    bool repair_edge_selected = false;
                                    for (const auto& [dist, id] : reverse_candidates) {
                                        if (id == neighbor_id) {
                                            repair_edge_selected = true;
                                            break;
                                        }
                                    }

                                    if (repair_edge_selected) {
                                        // The heuristic chose the repair edge - update the neighbor list
                                        vecsim_stl::vector<idType> updated_reverse_neighbors(this->allocator);
                                        updated_reverse_neighbors.reserve(reverse_candidates.size());
                                        for (const auto& [dist, id] : reverse_candidates) {
                                            updated_reverse_neighbors.push_back(id);
                                        }
                                        stageDeleteUpdate(new_neighbor_id, level, updated_reverse_neighbors);
                                    }
                                    // If repair edge was not selected by heuristic, we accept the
                                    // unidirectional edge - the heuristic determined that the existing
                                    // neighbors are better for search quality
                                }
                            }
                        }
                    }
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
        if (deleted_id >= curElementCount || deleted_id >= idToMetaData.size()) {
            continue;
        }
        // Mark the metadata as invalid
        idToMetaData[deleted_id].label = INVALID_LABEL;

        // Remove raw vector from RAM if it exists
        auto ram_it = rawVectorsInRAM.find(deleted_id);
        if (ram_it != rawVectorsInRAM.end()) {
            rawVectorsInRAM.erase(ram_it);
        }

        // Also remove from disk cache to prevent stale data access
        auto cache_it = rawVectorsDiskCache.find(deleted_id);
        if (cache_it != rawVectorsDiskCache.end()) {
            rawVectorsDiskCache.erase(cache_it);
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
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "=== HNSW Disk Index Graph Structure ===");
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Total elements: %zu", curElementCount);
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Entry point: %u", entrypointNode);
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Max level: %zu", maxLevel);
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "M: %zu, M0: %zu", M, M0);

    // Count total edges
    size_t total_edges = debugCountGraphEdges();
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Total graph edges: %zu", total_edges);

    // Print metadata for each element
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Element metadata:");
    for (size_t i = 0; i < std::min(curElementCount, idToMetaData.size()); ++i) {
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
    if (node_id >= curElementCount) {
        this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Node %u is out of range (max: %zu)",
                  node_id, (curElementCount - 1));
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

    if (curElementCount == 0) {
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
    size_t elements_to_check = std::min(curElementCount, size_t(5));
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

    for (size_t i = 0; i < curElementCount; ++i) {
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
candidatesLabelsMaxHeap<DistType> *
HNSWDiskIndex<DataType, DistType>::hierarchicalSearch(const void *data_point, idType ep_id,
                                                      size_t ef, size_t k, void *timeoutCtx,
                                                      VecSimQueryReply_Code *rc) const {
    if (rc)
        *rc = VecSim_QueryReply_OK;

    // Get the current entry point state
    auto [curr_entry_point, max_level] = safeGetEntryPointState();
    if (curr_entry_point == INVALID_ID) {
        if (rc)
            *rc = VecSim_QueryReply_OK; // Just return OK but no results
        return nullptr;
    }

    // Initialize result containers
    candidatesLabelsMaxHeap<DistType> *top_candidates = getNewMaxPriorityQueue();
    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    // Use a simple set for visited nodes tracking
    std::unordered_set<idType> visited_set;

    // Start from the provided entry point (not the global entry point)
    idType curr_element = ep_id;
    DistType curr_dist = this->calcDistance(data_point, getDataByInternalId(curr_element));

    // Add entry point to results
    if (!isMarkedDeleted(curr_element)) {
        top_candidates->emplace(curr_dist, getExternalLabel(curr_element));
        visited_set.insert(curr_element);
    }

    // Phase 1: Search from top level down to level 1 (hierarchical traversal)
    for (size_t level = max_level; level > 0; --level) {
        // Search at this level using the current element as entry point
        candidatesMaxHeap<DistType> level_candidates =
            searchLayer(curr_element, data_point, level, ef);

        if (!level_candidates.empty()) {
            // Find the closest element at this level to continue the search
            curr_element = level_candidates.top().second;
            curr_dist = level_candidates.top().first;
        }

        // Check timeout
        if (timeoutCtx && VECSIM_TIMEOUT(timeoutCtx)) {
            if (rc)
                *rc = VecSim_QueryReply_TimedOut;
            delete top_candidates;
            return nullptr;
        }
    }

    // Phase 2: Search at the bottom layer (level 0) with beam search
    // Reset visited set for bottom layer search
    visited_set.clear();
    visited_set.insert(curr_element);

    // Initialize candidate set with current element and its neighbors at level 0
    // Since candidatesMaxHeap doesn't have clear(), we'll create a new one
    candidate_set = candidatesMaxHeap<DistType>(this->allocator);
    candidate_set.emplace(-curr_dist, curr_element);

    // Add neighbors of the current element at level 0 to get started
    vecsim_stl::vector<idType> start_neighbors(this->allocator);
    getNeighbors(curr_element, 0, start_neighbors);

    if (!start_neighbors.empty()) {
        for (idType neighbor_id : start_neighbors) {
            if (neighbor_id < curElementCount &&
                visited_set.find(neighbor_id) == visited_set.end()) {
                DistType neighbor_dist =
                    this->calcDistance(data_point, getDataByInternalId(neighbor_id));
                candidate_set.emplace(-neighbor_dist, neighbor_id);
            }
        }
    }

    // Beam search at bottom layer
    DistType lower_bound = curr_dist;

    while (!candidate_set.empty()) {
        // Check timeout
        if (timeoutCtx && VECSIM_TIMEOUT(timeoutCtx)) {
            if (rc)
                *rc = VecSim_QueryReply_TimedOut;
            break;
        }

        auto curr_pair = candidate_set.top();
        DistType curr_candidate_dist = -curr_pair.first;
        idType curr_candidate_id = curr_pair.second;
        candidate_set.pop();

        // If we have enough candidates and current distance is worse, stop
        if (top_candidates->size() >= ef && curr_candidate_dist > lower_bound) {
            break;
        }

        // Process this candidate
        processCandidate(curr_candidate_id, data_point, 0, ef,
                         reinterpret_cast<void *>(&visited_set), 0, *top_candidates, candidate_set,
                         lower_bound);

        // Update lower bound based on current top candidates
        if (top_candidates->size() >= ef) {
            lower_bound = top_candidates->top().first;
        }

        // Continue searching until we have enough candidates or exhaust all possibilities
        if (top_candidates->size() >= ef && candidate_set.empty()) {
            break;
        }
    }

    // Trim results to k
    while (top_candidates->size() > k) {
        top_candidates->pop();
    }

    if (rc)
        *rc = VecSim_QueryReply_OK;
    return top_candidates;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::flushStagedUpdates() {
    // Flush both insert and delete staged updates
    // Note: This is a non-const method that modifies the staging areas
    flushStagedGraphUpdates(stagedInsertUpdates, stagedInsertNeighborUpdates);
    stagedInsertMap.clear();
    vecsim_stl::vector<NeighborUpdate> emptyNeighborUpdates(this->allocator);
    flushStagedGraphUpdates(stagedDeleteUpdates, emptyNeighborUpdates);
    stagedDeleteMap.clear();

    // Also flush staged repair updates (opportunistic cleanup from getNeighbors)
    if (!stagedRepairUpdates.empty()) {
        flushStagedGraphUpdates(stagedRepairUpdates, emptyNeighborUpdates);
        stagedRepairMap.clear();
    }
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
    const void *raw_data = getRawVector(id);
    if (raw_data == nullptr) {
        return; // Vector not found
    }

    // Copy the vector data
    const DataType *data_ptr = static_cast<const DataType *>(raw_data);
    std::vector<DataType> vec(data_ptr, data_ptr + this->dim);
    vectors_output.push_back(std::move(vec));
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
uint64_t HNSWDiskIndex<DataType, DistType>::getDBMemorySize() const {
    uint64_t db_mem_size = 0;
    this->db->GetIntProperty(rocksdb::DB::Properties::kSizeAllMemTables, &db_mem_size);
    return db_mem_size;
}

template <typename DataType, typename DistType>
uint64_t HNSWDiskIndex<DataType, DistType>::getDiskSize() const {
    uint64_t disk_size = 0;
    this->db->GetIntProperty(rocksdb::DB::Properties::kTotalSstFilesSize, &disk_size);
    return disk_size;
}

template <typename DataType, typename DistType>
std::shared_ptr<rocksdb::Statistics> HNSWDiskIndex<DataType, DistType>::getDBStatistics() const {
    return this->dbOptions.statistics;
}

// Missing virtual method implementations for HNSWDiskIndex
template <typename DataType, typename DistType>
VecSimIndexStatsInfo HNSWDiskIndex<DataType, DistType>::statisticInfo() const {
    VecSimIndexStatsInfo info = {};
    info.memory = this->getAllocationSize();
    info.db_memory = this->getDBMemorySize();
    info.db_disk = this->getDiskSize();
    info.numberOfMarkedDeleted = 0; // TODO: Implement if needed
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
    std::unique_lock<std::shared_mutex> index_data_lock(indexDataGuard);

    vecsim_stl::vector<idType> internal_ids(this->allocator);

    // Find the internal ID for this label
    auto it = labelToIdMap.find(label);
    if (it == labelToIdMap.end()) {
        // Label doesn't exist, return empty vector
        return internal_ids;
    }

    const idType internalId = it->second;

    // Check if already marked deleted
    if (isMarkedDeleted(internalId)) {
        // Already deleted, return empty vector
        return internal_ids;
    }

    // Mark as deleted
    markAs<DELETE_MARK>(internalId);
    
    auto raw_it = rawVectorsInRAM.find(internalId);
    if (raw_it != rawVectorsInRAM.end()) {
        rawVectorsInRAM.erase(raw_it);
    }

    auto disk_it = rawVectorsDiskCache.find(internalId);
    if (disk_it != rawVectorsDiskCache.end()) {
        rawVectorsDiskCache.erase(disk_it);
    }
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

    // Try to find a new entrypoint at the current max level
    while (maxLevel != HNSW_INVALID_LEVEL) {
        // First, try to find a neighbor of the old entrypoint at the top level
        GraphKey graphKey(old_entry_point_id, maxLevel);
        std::string graph_value;
        rocksdb::Status status =
            db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &graph_value);

        if (status.ok() && !graph_value.empty()) {
            // Correctly deserialize the graph value to get neighbors
            vecsim_stl::vector<idType> neighbors(this->allocator);
            deserializeGraphValue(graph_value, neighbors);

            // Try to find a non-deleted neighbor
            for (size_t i = 0; i < neighbors.size(); i++) {
                if (!isMarkedDeleted(neighbors[i])) {
                    entrypointNode = neighbors[i];
                    return;
                }
            }
        }

        // If no suitable neighbor found, search for any non-deleted node at this level
        for (idType id = 0; id < curElementCount; id++) {
            if (id != old_entry_point_id && id < idToMetaData.size() &&
                idToMetaData[id].label != INVALID_LABEL && idToMetaData[id].topLevel == maxLevel &&
                !isMarkedDeleted(id)) {
                entrypointNode = id;
                return;
            }
        }

        // No non-deleted nodes at this level, decrease maxLevel and try again
        maxLevel--;
    }

    // If we get here, the index is empty or all nodes are deleted
    entrypointNode = INVALID_ID;
    maxLevel = HNSW_INVALID_LEVEL;
}

#ifdef BUILD_TESTS
#include "hnsw_disk_serializer.h"
#endif
