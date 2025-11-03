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
// #include "hnsw_serialization_utils.h"
// #include "VecSim/utils/serializer.h"
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
static constexpr char RawVectorKeyPrefix[3] = "RV";

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

#pragma pack(1)
struct RawVectorKey {
    idType id;

    RawVectorKey(idType id) : id(id) {}

    rocksdb::Slice asSlice() const {
        // Create a key with the "RV" prefix followed by the id
        static thread_local std::vector<char> key_buffer;
        key_buffer.resize(3 + sizeof(idType)); // 3 bytes for "RV" + id size

        // Copy the "RV" prefix
        key_buffer[0] = 'R';
        key_buffer[1] = 'V';
        key_buffer[2] = '\0';

        // Copy the id
        std::memcpy(key_buffer.data() + 3, &id, sizeof(idType));

        return rocksdb::Slice(key_buffer.data(), key_buffer.size());
    }
};
#pragma pack()

//////////////////////////////////// HNSW index implementation ////////////////////////////////////

template <typename DataType, typename DistType>
class HNSWDiskIndex : public VecSimIndexAbstract<DataType, DistType> {
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
    vecsim_stl::vector<GraphUpdate> stagedGraphUpdates;

    // Track which nodes need their neighbor lists updated (for bidirectional connections)
    struct NeighborUpdate {
        idType node_id;
        size_t level;
        idType new_neighbor_id;

        NeighborUpdate(idType node_id, size_t level, idType new_neighbor_id)
            : node_id(node_id), level(level), new_neighbor_id(new_neighbor_id) {}
    };

    vecsim_stl::vector<NeighborUpdate> stagedNeighborUpdates;

protected:
    HNSWDiskIndex() = delete; // default constructor is disabled.
    // default (shallow) copy constructor is disabled.
    HNSWDiskIndex(const HNSWDiskIndex &) = delete;

    auto getNeighborhoods(const std::vector<idType> &ids) const;

    idType getNeighborsByHeuristic2(candidatesList<DistType> &top_candidates, size_t M) const;
    void getNeighborsByHeuristic2(candidatesList<DistType> &top_candidates, size_t M,
                                  vecsim_stl::vector<idType> &not_chosen_candidates) const;
    template <bool record_removed>
    void getNeighborsByHeuristic2_internal(candidatesList<DistType> &top_candidates, size_t M,
                                           vecsim_stl::vector<idType> *removed_candidates) const;

    std::unordered_map<idType, idType>
    pruneDeleted(const std::vector<idType> &deleted_ids,
                 const std::unordered_map<graphNodeType, std::vector<idType>, GraphNodeHash>
                     &deleted_neighborhoods,
                 idType &entrypointNode);

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

    // Helper methods
    void getNeighbors(idType nodeId, size_t level, std::vector<idType> &result) const;
    void searchPendingVectors(const void *query_data,
                              candidatesLabelsMaxHeap<DistType> &top_candidates, size_t k) const;

    // Manual control of staged updates
    void flushStagedUpdates(); // Manually flush any pending staged updates

protected:
    // New method for flushing staged graph updates to disk
    void flushStagedGraphUpdates();

    // New method for handling neighbor connection updates when neighbor lists are full
    void stageRevisitNeighborConnections(idType new_node_id, idType selected_neighbor, size_t level,
                                         DistType distance);

    void patchDeltaList(std::unordered_map<idType, std::vector<idType>> &delta_list,
                        vecsim_stl::vector<DiskElementMetaData> &new_elements_meta_data,
                        std::unordered_map<idType, idType> &new_ids_mapping);

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
    std::vector<char> getRawVector(idType id) const;

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
                  rocksdb::ColumnFamilyHandle *cf, size_t random_seed = 100);
    virtual ~HNSWDiskIndex();

    /*************************** Index API ***************************/
    void batchUpdate(const std::vector<pair<labelType, const void *>> &new_elements,
                     const std::vector<labelType> &deleted_labels);

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
    uint64_t getDiskSize() const override;
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
};

constexpr size_t INITIAL_CAPACITY = 1000;

/********************************** Constructors & Destructor **********************************/

template <typename DataType, typename DistType>
HNSWDiskIndex<DataType, DistType>::HNSWDiskIndex(
    const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
    const IndexComponents<DataType, DistType> &components, rocksdb::DB *db,
    rocksdb::ColumnFamilyHandle *cf, size_t random_seed)
    : VecSimIndexAbstract<DataType, DistType>(abstractInitParams, components),
      idToMetaData(INITIAL_CAPACITY, this->allocator), labelToIdMap(this->allocator), db(db),
      cf(cf), indexDataGuard(), visitedNodesHandlerPool(INITIAL_CAPACITY, this->allocator),
      delta_list(), new_elements_meta_data(this->allocator), batchThreshold(10),
      pendingVectorIds(this->allocator), pendingMetadata(this->allocator), pendingVectorCount(0),
      stagedGraphUpdates(this->allocator), stagedNeighborUpdates(this->allocator) {

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
    stagedGraphUpdates.clear();
    stagedNeighborUpdates.clear();

    // Clear pending vectors
    pendingVectorIds.clear();
    pendingMetadata.clear();

    // Clear delta list and new elements metadata
    delta_list.clear();
    new_elements_meta_data.clear();

    // Clear main data structures
    idToMetaData.clear();
    labelToIdMap.clear();

    // Ensure all memory is properly released
    idToMetaData.shrink_to_fit();
    labelToIdMap.clear();

    // Note: db and cf are not owned by this class, so we don't delete them
    // Base class destructor will handle indexCalculator and preprocessors
}

/********************************** Index API **********************************/

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::batchUpdate(
    const std::vector<std::pair<labelType, const void *>> &new_elements,
    const std::vector<labelType> &deleted_labels) {
    if (!indexDataGuard.try_lock()) {
        // Cannot acquire lock, another operation is in progress
        return;
    }
    // Use RAII to ensure the lock is released
    std::lock_guard<std::shared_mutex> guard(indexDataGuard, std::adopt_lock);

    std::vector<idType> deleted_ids;
    deleted_ids.reserve(deleted_labels.size());
    for (const auto &label : deleted_labels) {
        auto it = labelToIdMap.find(label);
        if (it != labelToIdMap.end()) {
            deleted_ids.push_back(it->second);
        }
        // If label doesn't exist, skip it (don't add to deleted_ids)
    }

    // Phase 0: fetch and cache the deleted labels neighbors (before reaching here?)
    auto deleted_neighborhoods = getNeighborhoods(deleted_ids);

    // Phase 1: Delete elements
    // iterate over the graph nodes and delete the deleted labels
    // 1. for each not-deleted label with a deleted neighbor, re-choose the neighbors
    //    from the remaining nodes + the deleted nodes neighbors (by the heuristic)
    // 2. for each not-deleted label, write it to the new graph
    auto curEntryPoint = this->entrypointNode;
    auto new_ids_mapping = pruneDeleted(deleted_ids, deleted_neighborhoods, curEntryPoint);
    auto curMaxLevel = maxLevel; // TODO: take `curEntryPoint` maxLevel, address mapping?

    // Phase 2: Insert new elements
    // 1. for each new element, find its neighbors by the heuristic
    // 2. For each neighbor, add to the temporary delta list the new element
    std::unordered_map<idType, std::vector<idType>> delta_list;
    vecsim_stl::vector<DiskElementMetaData> new_elements_meta_data(this->allocator);
    new_elements_meta_data.reserve(new_elements.size());
    // std::unordered_map<labelType, idType> new_labels_mapping;
    for (const auto &[label, vector] : new_elements) {
        // Add vector to memory immediately
        ProcessedBlobs processedBlobs = this->preprocess(vector);
        size_t containerId = this->vectors->size();
        this->vectors->addElement(processedBlobs.getStorageBlob(), containerId);

        // Create metadata and store immediately
        idType newElementId = curElementCount;
        size_t elementMaxLevel = getRandomLevel(mult);
        DiskElementMetaData new_element(label, elementMaxLevel);
        idToMetaData[newElementId] = new_element;
        labelToIdMap[label] = newElementId;
        curElementCount++;

        // Resize visited nodes handler pool to accommodate new elements
        visitedNodesHandlerPool.resize(curElementCount);

        // Add only the vector ID to pending vectors for indexing
        pendingVectorIds.push_back(newElementId);
        pendingVectorCount++;
    }

    // Phase 3: Patch the graph
    // 1. Iterate over the entire graph
    // 2. for each id in the delta list, re-choose the neighbors (by the heuristic)
    patchDeltaList(delta_list, new_elements_meta_data, new_ids_mapping);

    // Phase 4: Set new version, entry point and max level
    // TODO: some lock
    this->entrypointNode = curEntryPoint;
    this->maxLevel = curMaxLevel;
    this->curElementCount += new_elements.size() - deleted_labels.size();
    for (const auto &[old_id, new_id] : new_ids_mapping) {
        idToMetaData[new_id] = idToMetaData[old_id];
        labelToIdMap[idToMetaData[new_id].label] = new_id;
    }
    this->labelToIdMap.reserve(this->curElementCount);
    this->idToMetaData.reserve(this->curElementCount);
    this->idToMetaData.resize(this->curElementCount - deleted_labels.size());
    this->idToMetaData.insert(this->idToMetaData.end(), new_elements_meta_data.begin(),
                              new_elements_meta_data.end());
    for (idType i = 0; i < new_elements_meta_data.size(); ++i) {
        labelToIdMap[new_elements_meta_data[i].label] =
            this->curElementCount + i - deleted_labels.size();
    }
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
auto HNSWDiskIndex<DataType, DistType>::getNeighborhoods(const std::vector<idType> &ids) const {
    // Create a map to store the neighbors for each label
    std::unordered_map<graphNodeType, std::vector<idType>, GraphNodeHash> neighbors_map;
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

        size_t num_neighbors = value.size() / sizeof(idType);
        const idType *neighbor_ids = reinterpret_cast<const idType *>(value.data());
        // Parse the value to get the neighbors
        std::vector<idType> neighbors(neighbor_ids, neighbor_ids + num_neighbors);
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
std::unordered_map<idType, idType> HNSWDiskIndex<DataType, DistType>::pruneDeleted(
    const std::vector<idType> &deleted_ids,
    const std::unordered_map<graphNodeType, std::vector<idType>, GraphNodeHash>
        &deleted_neighborhoods,
    idType &entrypointNode) {

    // Higher ids will reuse the deleted ids
    auto newElementCount = this->curElementCount - deleted_neighborhoods.size();

    auto readOptions = rocksdb::ReadOptions();
    readOptions.fill_cache = false;
    readOptions.prefix_same_as_start = true;

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;

    auto it = this->db->NewIterator(readOptions, cf);
    for (it->Seek(GraphKeyPrefix); it->Valid(); it->Next()) {
        auto key = it->key();
        auto graphKey = reinterpret_cast<const GraphKey *>(key.data());

        if (deleted_neighborhoods.find(graphKey->node()) != deleted_neighborhoods.end()) {
            this->db->Delete(writeOptions, cf, it->key());
            continue;
        }
        auto neighborsSlice = it->value();
        auto neighborsData = reinterpret_cast<const idType *>(neighborsSlice.data());
        size_t num_neighbors = neighborsSlice.size() / sizeof(idType);
        std::vector<idType> neighbors(neighborsData, neighborsData + num_neighbors);

        bool has_deleted_neighbors = false;
        bool has_moving_neighbors = false;
        for (auto &neighbor : neighbors) {
            graphNodeType neighbor_node{neighbor, graphKey->level};
            if (deleted_neighborhoods.find(neighbor_node) != deleted_neighborhoods.end()) {
                has_deleted_neighbors = true;
                if (has_moving_neighbors)
                    break;
            }
            if (neighbor >= newElementCount) {
                has_moving_neighbors = true;
                if (has_deleted_neighbors)
                    break;
            }
        }
        auto new_id = graphKey->id < newElementCount ? graphKey->id
                                                     : deleted_ids[graphKey->id - newElementCount];
        if (has_deleted_neighbors) {
            // Collect candidates for the new neighbors
            candidatesList<DistType> new_neighbors(this->allocator);
            std::unordered_set<idType> neighbors_set;
            auto add_candidate = [&](idType id) {
                if (neighbors_set.insert(id).second) {
                    DistType dist = this->calcDistance(getDataByInternalId(id),
                                                       getDataByInternalId(graphKey->id));
                    new_neighbors.emplace_back(dist, id);
                }
            };
            for (size_t i = 0; i < num_neighbors; ++i) {
                graphNodeType neighbor_node{neighbors[i], graphKey->level};
                auto it = deleted_neighborhoods.find(neighbor_node);
                if (it != deleted_neighborhoods.end()) {
                    // Add the deleted node's neighbors to the new candidate list, if they are not
                    // deleted themself
                    for (const auto &neighbor : it->second) {
                        if (deleted_neighborhoods.find(neighbor_node) ==
                            deleted_neighborhoods.end()) {
                            add_candidate(neighbor);
                        }
                    }
                } else {
                    // Add the current neighbor to the new candidate list
                    add_candidate(neighbors[i]);
                }
            }
            getNeighborsByHeuristic2(new_neighbors, graphKey->level == 0 ? M0 : M);

            // Extract the ids from the new_neighbors list
            neighbors.resize(new_neighbors.size());
            for (size_t i = 0; i < new_neighbors.size(); ++i) {
                neighbors[i] = new_neighbors[i].second;
            }
            // Write new node to the new version, after fixing the deleted ids
        }
        if (has_moving_neighbors || has_deleted_neighbors) {
            for (auto &neighbor : neighbors) {
                if (neighbor >= newElementCount) {
                    neighbor = deleted_ids[neighbor - newElementCount];
                }
            }
        }
        if (has_moving_neighbors || has_deleted_neighbors || graphKey->id < newElementCount) {
            // Create a new key for the new node
            auto newKey = GraphKey(new_id, graphKey->level);
            // Create a new slice for the new neighbors
            size_t bytes = neighbors.size() * sizeof(idType);
            auto neighbors_slice =
                rocksdb::Slice(reinterpret_cast<const char *>(neighbors.data()), bytes);
            this->db->Put(writeOptions, cf, newKey.asSlice(), neighbors_slice);
        }
    }
    delete it;

    if (deleted_neighborhoods.find({entrypointNode, 0}) != deleted_neighborhoods.end()) {
        // Find a new entry point, update the `entrypointNode` reference
    }

    std::unordered_map<idType, idType> new_ids(deleted_ids.size());
    for (size_t i = 0; i < deleted_ids.size(); ++i) {
        new_ids[newElementCount + i] = deleted_ids[i];
    }
    return new_ids;
}

template <typename DataType, typename DistType>
int HNSWDiskIndex<DataType, DistType>::addVector(const void *vector, labelType label) {
    // Preprocess the vector
    ProcessedBlobs processedBlobs = this->preprocess(vector);

    // Store the processed vector in memory immediately
    size_t containerId = this->vectors->size();
    this->vectors->addElement(processedBlobs.getStorageBlob(), containerId);

    // Create new element ID and metadata
    idType newElementId = curElementCount;
    size_t elementMaxLevel = getRandomLevel(mult);
    DiskElementMetaData new_element(label, elementMaxLevel);

    if (newElementId >= indexCapacity()) {
        size_t new_cap = ((newElementId + this->blockSize) / this->blockSize) * this->blockSize;
        visitedNodesHandlerPool.resize(new_cap);
        idToMetaData.resize(new_cap);
        labelToIdMap.reserve(new_cap);
    }

    // Store metadata immediately
    idToMetaData[newElementId] = new_element;
    labelToIdMap[label] = newElementId;

    // Store raw vector to disk
    RawVectorKey rawVectorKey(newElementId);
    const void *storageBlob = processedBlobs.getStorageBlob();
    auto rawVectorSlice =
        rocksdb::Slice(reinterpret_cast<const char *>(storageBlob), this->dataSize);

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;
    rocksdb::Status status = db->Put(writeOptions, cf, rawVectorKey.asSlice(), rawVectorSlice);
    if (!status.ok()) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                  "WARNING: Failed to store raw vector for id %u: %s", newElementId,
                  status.ToString().c_str());
    }

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

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;

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

    // Add to staged graph updates
    stagedGraphUpdates.emplace_back(new_node_id, level, neighbor_ids, this->allocator);

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
            stagedNeighborUpdates.emplace_back(selected_neighbor, level, new_node_id);
        } else {
            // Neighbor is full, need to re-evaluate connections using revisitNeighborConnections
            // logic
            stageRevisitNeighborConnections(new_node_id, selected_neighbor, level, distance);
        }
    }

    return next_closest_entry_point;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::flushStagedGraphUpdates() {
    if (stagedGraphUpdates.empty() && stagedNeighborUpdates.empty()) {
        return;
    }

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;

    // First, handle new node insertions
    for (const auto &update : stagedGraphUpdates) {
        auto newKey = GraphKey(update.node_id, update.level);
        size_t bytes = update.neighbors.size() * sizeof(idType);
        auto neighbors_slice =
            rocksdb::Slice(reinterpret_cast<const char *>(update.neighbors.data()), bytes);

        rocksdb::Status put_status =
            this->db->Put(writeOptions, cf, newKey.asSlice(), neighbors_slice);
        if (!put_status.ok()) {
            this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                      "ERROR: Failed to store graph connection for node %u at level %zu: %s",
                      update.node_id, update.level, put_status.ToString().c_str());
        }
    }

    // Then, handle neighbor list updates for existing nodes
    // Group neighbor updates by node and level for efficient processing
    std::unordered_map<idType, std::unordered_map<size_t, std::vector<idType>>>
        neighborUpdatesByNode;

    for (const auto &update : stagedNeighborUpdates) {
        neighborUpdatesByNode[update.node_id][update.level].push_back(update.new_neighbor_id);
    }

    // Use a single WriteBatch for all neighbor updates to avoid memory issues
    rocksdb::WriteBatch batch;

    // Process each node's neighbor updates
    for (const auto &[node_id, levelMap] : neighborUpdatesByNode) {
        for (const auto &[level, newNeighbors] : levelMap) {
            // Read existing neighbors from disk
            GraphKey neighborKey(node_id, level);
            std::string existing_neighbors_data;
            rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, neighborKey.asSlice(),
                                             &existing_neighbors_data);

            vecsim_stl::vector<idType> updated_neighbors(this->allocator);
            if (status.ok()) {
                // Parse existing neighbors
                const idType *existing_neighbors =
                    reinterpret_cast<const idType *>(existing_neighbors_data.data());
                size_t num_existing = existing_neighbors_data.size() / sizeof(idType);
                updated_neighbors.reserve(num_existing);
                for (size_t i = 0; i < num_existing; i++) {
                    updated_neighbors.push_back(existing_neighbors[i]);
                }
            }

            // Add new neighbors (avoiding duplicates)
            for (idType new_neighbor : newNeighbors) {
                bool found = false;
                for (size_t i = 0; i < updated_neighbors.size(); i++) {
                    if (updated_neighbors[i] == new_neighbor) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    updated_neighbors.push_back(new_neighbor);
                }
            }

            // Add to batch instead of writing immediately
            size_t updated_bytes = updated_neighbors.size() * sizeof(idType);
            auto updated_neighbors_slice = rocksdb::Slice(
                reinterpret_cast<const char *>(updated_neighbors.data()), updated_bytes);
            batch.Put(cf, neighborKey.asSlice(), updated_neighbors_slice);
        }
    }

    // Write all neighbor updates in a single batch
    rocksdb::Status batch_status = this->db->Write(writeOptions, &batch);
    if (!batch_status.ok()) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                  "ERROR: Failed to write neighbor updates batch: %s",
                  batch_status.ToString().c_str());
    }

    // Clear staged updates after successful flush
    stagedGraphUpdates.clear();
    stagedNeighborUpdates.clear();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::stageRevisitNeighborConnections(idType new_node_id,
                                                                        idType selected_neighbor,
                                                                        size_t level,
                                                                        DistType distance) {
    // Read the neighbor's current neighbor list from disk
    // TODO: perhaps cache the neigbhors for stage update
    GraphKey neighborKey(selected_neighbor, level);
    std::string existing_neighbors_data;
    rocksdb::Status status =
        db->Get(rocksdb::ReadOptions(), cf, neighborKey.asSlice(), &existing_neighbors_data);

    if (!status.ok()) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                  "  WARNING: Could not read existing neighbors for node %u at level %zu",
                  selected_neighbor, level);
        // Fall back to simple neighbor update
        stagedNeighborUpdates.emplace_back(selected_neighbor, level, new_node_id);
        return;
    }

    // Parse existing neighbors
    const idType *existing_neighbors =
        reinterpret_cast<const idType *>(existing_neighbors_data.data());
    size_t num_existing = existing_neighbors_data.size() / sizeof(idType);

    // Collect all candidates: existing neighbors + new node
    candidatesList<DistType> candidates(this->allocator);
    candidates.reserve(num_existing + 1);

    // Add the new node with its pre-calculated distance
    candidates.emplace_back(distance, new_node_id);

    // Add existing neighbors with their distances to the selected neighbor
    const void *selected_neighbor_data = getDataByInternalId(selected_neighbor);
    for (size_t j = 0; j < num_existing; j++) {
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
        stagedGraphUpdates.emplace_back(selected_neighbor, level, selected_neighbor_ids,
                                        this->allocator);

        // Also stage the bidirectional connection from new node to selected neighbor
        stagedNeighborUpdates.emplace_back(new_node_id, level, selected_neighbor);

    } else {
        // The new node was not selected, so we only need to stage the unidirectional connection
        // from new node to selected neighbor
        stagedNeighborUpdates.emplace_back(new_node_id, level, selected_neighbor);
    }
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::patchDeltaList(
    std::unordered_map<idType, std::vector<idType>> &delta_list,
    vecsim_stl::vector<DiskElementMetaData> &new_elements_meta_data,
    std::unordered_map<idType, idType> &new_ids_mapping) {

    auto readOptions = rocksdb::ReadOptions();
    readOptions.fill_cache = false;
    readOptions.prefix_same_as_start = true;

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;

    auto it = this->db->NewIterator(readOptions, cf);
    for (it->Seek(GraphKeyPrefix); it->Valid(); it->Next()) {
        auto key = it->key();
        auto graphKey = reinterpret_cast<const GraphKey *>(key.data());

        auto it2 = delta_list.find(graphKey->node().first);
        if (it2 == delta_list.end()) {
            // No need to update this node, move to the next one
            continue;
        }
        auto neighborsSlice = it->value();
        auto neighborsData = reinterpret_cast<const idType *>(neighborsSlice.data());
        size_t num_neighbors = neighborsSlice.size() / sizeof(idType);
        candidatesList<DistType> new_neighbors(this->allocator);
        new_neighbors.reserve(num_neighbors + it2->second.size());
        auto vector = getDataByInternalId(graphKey->id);
        for (const auto &neighbor : it2->second) {
            DistType dist = this->calcDistance(getDataByInternalId(neighbor), vector);
            new_neighbors.emplace_back(dist, neighbor);
        }
        for (size_t i = 0; i < num_neighbors; ++i) {
            DistType dist = this->calcDistance(getDataByInternalId(neighborsData[i]), vector);
            new_neighbors.emplace_back(dist, neighborsData[i]);
        }
        getNeighborsByHeuristic2(new_neighbors, graphKey->level == 0 ? M0 : M);

        // Extract the ids from the new_neighbors list
        std::vector<idType> neighbors(new_neighbors.size());
        for (size_t i = 0; i < new_neighbors.size(); ++i) {
            neighbors[i] = new_neighbors[i].second;
        }
        // Create a new slice for the new neighbors
        size_t bytes = neighbors.size() * sizeof(idType);
        auto neighbors_slice =
            rocksdb::Slice(reinterpret_cast<const char *>(neighbors.data()), bytes);
        this->db->Put(writeOptions, cf, it->key(), neighbors_slice);
    }

    delete it;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

/********************************** Stub Implementations **********************************/

template <typename DataType, typename DistType>
const void *HNSWDiskIndex<DataType, DistType>::getDataByInternalId(idType id) const {
    assert(id < curElementCount);
    // For now, we'll use the memory storage from the base class
    // In a production implementation, you might want to implement a proper cache
    // or use a different strategy for handling disk-based vector retrieval
    const void *result = this->vectors->getElement(id);
    return result;
}

template <typename DataType, typename DistType>
std::vector<char> HNSWDiskIndex<DataType, DistType>::getRawVector(idType id) const {
    std::vector<char> result;

    if (id >= curElementCount) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                  "WARNING: getRawVector called with invalid id %u (current count: %zu)", id,
                  curElementCount);
        return result;
    }

    // Try to retrieve from disk
    RawVectorKey rawVectorKey(id);
    std::string vectorData;
    rocksdb::Status status =
        db->Get(rocksdb::ReadOptions(), cf, rawVectorKey.asSlice(), &vectorData);

    if (status.ok()) {
        // Copy the data from string to vector
        result.assign(vectorData.begin(), vectorData.end());
    } else if (status.IsNotFound()) {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                  "WARNING: Raw vector not found on disk for id %u", id);
    } else {
        this->log(VecSimCommonStrings::LOG_WARNING_STRING,
                  "WARNING: Failed to retrieve raw vector for id %u: %s", id,
                  status.ToString().c_str());
    }

    return result;
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
        std::vector<idType> neighbors;
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
        std::string neighbors_data;
        rocksdb::Status status =
            db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &neighbors_data);

        if (!status.ok()) {
            // No neighbors found for this node at this level, stop searching
            break;
        }

        // Parse the neighbors data
        const idType *neighbors = reinterpret_cast<const idType *>(neighbors_data.data());
        size_t num_neighbors = neighbors_data.size() / sizeof(idType);

        // Check each neighbor to find a better candidate
        for (size_t i = 0; i < num_neighbors; i++) {
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
    std::vector<idType> neighbors;
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
    return this->curElementCount;
}

/********************************** Helper Methods **********************************/

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::getNeighbors(idType nodeId, size_t level,
                                                     std::vector<idType> &result) const {
    // Clear the result vector first
    result.clear();

    // First check staged graph updates
    for (const auto &update : stagedGraphUpdates) {
        if (update.node_id == nodeId && update.level == level) {
            result.reserve(update.neighbors.size());
            for (size_t i = 0; i < update.neighbors.size(); i++) {
                result.push_back(update.neighbors[i]);
            }
            return;
        }
    }

    // If not found in staged updates, check disk
    GraphKey graphKey(nodeId, level);

    // Use a temporary buffer to avoid std::string corruption
    std::string neighbors_data;
    rocksdb::Status status =
        db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &neighbors_data);

    if (status.ok()) {
        size_t num_neighbors = neighbors_data.size() / sizeof(idType);
        result.reserve(num_neighbors);

        // Direct memory assignment - avoid memcpy by casting the data pointer
        const idType *data_ptr = reinterpret_cast<const idType *>(neighbors_data.data());
        result.assign(data_ptr, data_ptr + num_neighbors);
    }
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::searchPendingVectors(
    const void *query_data, candidatesLabelsMaxHeap<DistType> &top_candidates, size_t k) const {
    for (size_t i = 0; i < pendingVectorCount; i++) {
        idType vectorId = pendingVectorIds[i];

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

    // Clear any previous staged updates
    stagedGraphUpdates.clear();
    stagedNeighborUpdates.clear();

    // Process each pending vector ID (vectors are already stored in memory)
    for (size_t i = 0; i < pendingVectorCount; i++) {
        idType vectorId = pendingVectorIds[i];

        // Get the vector data from memory
        const void *vector_data = this->vectors->getElement(vectorId);

        // Get metadata for this vector
        DiskElementMetaData &metadata = idToMetaData[vectorId];
        labelType label = metadata.label;
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
    flushStagedGraphUpdates();

    // Clear the pending vector IDs
    pendingVectorIds.clear();
    pendingMetadata.clear();
    pendingVectorCount = 0;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::flushBatch() {
    processBatch();
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
        std::string neighbors_data;
        rocksdb::Status status =
            db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &neighbors_data);

        if (status.ok()) {
            const idType *neighbors = reinterpret_cast<const idType *>(neighbors_data.data());
            size_t num_neighbors = neighbors_data.size() / sizeof(idType);
            this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "  Level %zu (%zu neighbors): ", level,
                      num_neighbors);
            for (size_t i = 0; i < num_neighbors; i++) {
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

        // Get neighbors count
        auto neighborsSlice = it->value();
        size_t num_neighbors = neighborsSlice.size() / sizeof(idType);
        total_neighbors += num_neighbors;

        this->log(VecSimCommonStrings::LOG_DEBUG_STRING, " (%zu neighbors)", num_neighbors);

        // Print first few neighbors
        if (num_neighbors > 0) {
            const idType *neighbors = reinterpret_cast<const idType *>(neighborsSlice.data());
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
        auto neighborsSlice = it->value();
        size_t num_neighbors = neighborsSlice.size() / sizeof(idType);
        total_edges += num_neighbors;
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
            std::string neighbors_data;
            rocksdb::Status status =
                db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &neighbors_data);

            if (status.ok() && neighbors_data.size() > 0) {
                has_neighbors = true;
                break;
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
    std::vector<idType> start_neighbors;
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
    // Implement the logic to manually flush any pending staged updates
    // This could involve writing any pending staged updates to disk
    // or clearing the staged updates if they are no longer needed
    // For example, you might want to call flushStagedGraphUpdates() here
    flushStagedGraphUpdates();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::debugPrintStagedUpdates() const {
    // Implement the logic to debug print staged updates
    // This could involve logging the contents of stagedGraphUpdates and stagedNeighborUpdates
    // or any other relevant information
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "=== Staged Updates ===");
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Staged Graph Updates: %zu",
              stagedGraphUpdates.size());
    this->log(VecSimCommonStrings::LOG_DEBUG_STRING, "Staged Neighbor Updates: %zu",
              stagedNeighborUpdates.size());
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
    // TODO: Implement data retrieval by label
    // For now, just a stub implementation
    vectors_output.clear();
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
    // TODO: Implement labels set retrieval
    // For now, just a stub implementation
    vecsim_stl::set<labelType> labels(this->allocator);
    return labels;
}

template <typename DataType, typename DistType>
int HNSWDiskIndex<DataType, DistType>::deleteVector(labelType label) {
    // TODO: Implement vector deletion
    // For now, just a stub implementation
    return 0;
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
    uint64_t db_mem_size = 0;
    this->db->GetIntProperty(rocksdb::DB::Properties::kSizeAllMemTables, &db_mem_size);
    // std::cout << "Disk mem size: " << db_mem_size << std::endl;
    return this->allocator->getAllocationSize() + db_mem_size;
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
    info.disk = this->getDiskSize();
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
    __atomic_fetch_add(&numMarkedDeleted, 1, 0);

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
        std::string neighbors_data;
        rocksdb::Status status =
            db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &neighbors_data);

        if (status.ok() && !neighbors_data.empty()) {
            size_t num_neighbors = neighbors_data.size() / sizeof(idType);
            const idType *neighbors = reinterpret_cast<const idType *>(neighbors_data.data());

            // Try to find a non-deleted neighbor
            for (size_t i = 0; i < num_neighbors; i++) {
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
