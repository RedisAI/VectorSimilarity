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
#include "VecSim/algorithms/hnsw/hnsw.h"  // For HNSWAddVectorState definition

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
    std::size_t operator()(const graphNodeType& k) const {
        return std::hash<idType>()(k.first) ^ (std::hash<unsigned short>()(k.second) << 1);
    }
};

////////////////////////////////////// Auxiliary HNSW structs //////////////////////////////////////

struct DiskElementMetaData {
    labelType label;
    size_t topLevel;
    // elementFlags flags;

    DiskElementMetaData(labelType label = INVALID_LABEL) noexcept : label(label) {}
    DiskElementMetaData(labelType label, size_t topLevel) noexcept : label(label), topLevel(topLevel) {}
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

    graphNodeType node() const {
        return graphNodeType(id, level);
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
    idType entrypointNode;
    size_t maxLevel; // this is the top level of the entry point's element

    // Index data
    // vecsim_stl::vector<DataBlock> graphDataBlocks;
    vecsim_stl::vector<DiskElementMetaData> idToMetaData;
    vecsim_stl::unordered_map<labelType, idType> labelToIdMap;
    rocksdb::DB *db;                 // RocksDB database, not owned by the index
    rocksdb::ColumnFamilyHandle *cf; // RocksDB column family handle, not owned by the index

    mutable std::shared_mutex indexDataGuard;
    mutable VisitedNodesHandlerPool visitedNodesHandlerPool;

    // Global batch operation state
    mutable std::unordered_map<idType, std::vector<idType>> delta_list;
    mutable vecsim_stl::vector<DiskElementMetaData> new_elements_meta_data;

    // Batch processing state
    size_t batchThreshold;  // Number of vectors to accumulate before batch update
    vecsim_stl::vector<std::pair<labelType, std::vector<char>>> pendingVectors;  // Vectors waiting to be batched
    vecsim_stl::vector<DiskElementMetaData> pendingMetadata;  // Metadata for pending vectors
    size_t pendingVectorCount;  // Count of vectors in memory

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

    std::unordered_map<idType, idType> pruneDeleted(
        const std::vector<idType> &deleted_ids,
        const std::unordered_map<graphNodeType, std::vector<idType>, GraphNodeHash> &deleted_neighborhoods,
        idType &entrypointNode);
    
public:
    // Pure virtual methods from VecSimIndexInterface
    int addVector(const void *blob, labelType label) override;
    
public:
    // New separated methods for disk-based HNSW (public for testing)
    HNSWAddVectorState storeVector(const void *vector_data, const labelType label);
    void indexVector(const void *vector_data, const labelType label, const HNSWAddVectorState &state);
    void appendVector(const void *vector_data, const labelType label);
    void insertElementToGraph(idType element_id, size_t element_max_level, idType entry_point,
                             size_t global_max_level, const void *vector_data);
    idType mutuallyConnectNewElement(idType new_node_id, candidatesMaxHeap<DistType> &top_candidates, size_t level);
    
    // Batch processing methods
    void addVectorToBatch(const void *vector_data, const labelType label);
    void processBatch();
    void flushBatch();  // Force flush current batch

protected:
    
    void patchDeltaList(std::unordered_map<idType, std::vector<idType>> &delta_list,
                        vecsim_stl::vector<DiskElementMetaData> &new_elements_meta_data,
                        std::unordered_map<idType, idType> &new_ids_mapping);

    // Missing method declarations - adding stub implementations
    const void* getDataByInternalId(idType id) const;
    candidatesMaxHeap<DistType> searchLayer(idType ep_id, const void* data_point, size_t level, size_t ef) const;
    void greedySearchLevel(const void* data_point, size_t level, idType& curr_element, DistType& cur_dist) const;
    std::pair<idType, size_t> safeGetEntryPointState() const;
    VisitedNodesHandler* getVisitedList() const;
    void returnVisitedList(VisitedNodesHandler* visited_nodes_handler) const;
    candidatesLabelsMaxHeap<DistType>* getNewMaxPriorityQueue() const;
    bool isMarkedDeleted(idType id) const;
    labelType getExternalLabel(idType id) const;
    void processCandidate(idType candidate_id, const void* data_point, size_t level, size_t ef,
                         void* visited_tags, size_t visited_tag, candidatesLabelsMaxHeap<DistType>& top_candidates,
                         candidatesMaxHeap<DistType>& candidate_set, DistType& lowerBound) const;

    idType searchBottomLayerEP(const void *query_data, void *timeoutCtx = nullptr,
                               VecSimQueryReply_Code *rc = nullptr) const;

    candidatesLabelsMaxHeap<DistType> *
    searchBottomLayer_WithTimeout(idType ep_id, const void *data_point, size_t ef, size_t k, void *timeoutCtx = nullptr,
                                 VecSimQueryReply_Code *rc = nullptr) const;

    // New hierarchical search method
    candidatesLabelsMaxHeap<DistType> *
    hierarchicalSearch(const void *data_point, idType ep_id, size_t ef, size_t k, void *timeoutCtx = nullptr,
                      VecSimQueryReply_Code *rc = nullptr) const;

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

public:
    // Public methods for testing
    size_t indexSize() const override;
    size_t indexCapacity() const override;
    size_t indexLabelCount() const override;
    size_t getRandomLevel(double reverse_size);

    // Debug methods to inspect graph structure
    void debugPrintGraphStructure() const;
    void debugPrintNodeNeighbors(idType node_id) const;
    void debugPrintAllGraphKeys() const;
    size_t debugCountGraphEdges() const;
    void debugValidateGraphConnectivity() const;

private:
    // HNSW helper methods
    int deleteVector(labelType label) override;
    double getDistanceFrom_Unsafe(labelType id, const void *blob) const override;
    void fitMemory() override;
    void getDataByLabel(labelType label, std::vector<std::vector<DataType>>& vectors_output) const override;
    std::vector<std::vector<char>> getStoredVectorDataByLabel(labelType label) const override;
    vecsim_stl::set<labelType> getLabelsSet() const override;


    /*****************************************************************/
};

/********************************** Constructors & Destructor **********************************/

template <typename DataType, typename DistType>
HNSWDiskIndex<DataType, DistType>::HNSWDiskIndex(
    const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
    const IndexComponents<DataType, DistType> &components, rocksdb::DB *db,
    rocksdb::ColumnFamilyHandle *cf, size_t random_seed)
    : VecSimIndexAbstract<DataType, DistType>(abstractInitParams, components),
      idToMetaData(1000, this->allocator), labelToIdMap(this->allocator),
      db(db), cf(cf), indexDataGuard(), visitedNodesHandlerPool(10, this->allocator),
      delta_list(), new_elements_meta_data(this->allocator),
      batchThreshold(10), pendingVectors(this->allocator), pendingMetadata(this->allocator), pendingVectorCount(0){

    M = params->M ? params->M : HNSW_DEFAULT_M;
    M0 = M * 2;
    if (M0 > UINT16_MAX)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");

    efConstruction = params->efConstruction ? params->efConstruction : HNSW_DEFAULT_EF_C;
    efConstruction = std::max(efConstruction, M);
    ef = params->efRuntime ? params->efRuntime : HNSW_DEFAULT_EF_RT;
    epsilon = params->epsilon > 0.0 ? params->epsilon : HNSW_DEFAULT_EPSILON;

    curElementCount = 0;

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
        appendVector(vector, label);
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

    std::cout << "topKQuery called with k=" << k << ", curElementCount=" << curElementCount << std::endl;
    
    // Debug: Print graph structure before search
    std::cout << "\n=== Graph Structure Before Search ===" << std::endl;
    debugPrintGraphStructure();
    
    auto rep = new VecSimQueryReply(this->allocator);
    this->lastMode = STANDARD_KNN;

    if ((curElementCount == 0 && pendingVectorCount == 0) || k == 0) {
        std::cout << "Empty index or k=0, returning empty results" << std::endl;
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
    std::cout << "\nSearching for entry point..." << std::endl;
    idType bottom_layer_ep = searchBottomLayerEP(processed_query, timeoutCtx, &rep->code);
    std::cout << "Entry point found: " << bottom_layer_ep << std::endl;
    if (VecSim_OK != rep->code || bottom_layer_ep == INVALID_ID) {
        std::cout << "No entry point found or error occurred" << std::endl;
        return rep; // Empty index or error
    }

    // Step 2: Perform hierarchical search from top level down to bottom level
    std::cout << "\nStarting hierarchical search with curElementCount: " << curElementCount << std::endl;
    std::cout << "Search parameters: ef=" << std::max(query_ef, k) << ", k=" << k << std::endl;
    
    // Use a more sophisticated search that properly traverses the HNSW hierarchy
    auto *results = hierarchicalSearch(processed_query, bottom_layer_ep, std::max(query_ef, k), k, timeoutCtx, &rep->code);

    if (VecSim_OK == rep->code && results) {
        std::cout << "Search returned " << results->size() << " results" << std::endl;
        rep->results.resize(results->size());
        for (auto result = rep->results.rbegin(); result != rep->results.rend(); result++) {
            std::tie(result->score, result->id) = results->top();
            std::cout << "Result: id=" << result->id << ", score=" << result->score << std::endl;
            results->pop();
        }
    } else {
        std::cout << "Search failed or returned no results" << std::endl;
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
    std::vector<rocksdb::ColumnFamilyHandle*> cfs(keys.size(), cf);
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
    const std::unordered_map<graphNodeType, std::vector<idType>, GraphNodeHash> &deleted_neighborhoods,
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
                if (has_moving_neighbors) break;
            }
            if (neighbor >= newElementCount) {
                has_moving_neighbors = true;
                if (has_deleted_neighbors) break;
            }
        }
        auto new_id = graphKey->id < newElementCount ? graphKey->id : deleted_ids[graphKey->id - newElementCount];
        if (has_deleted_neighbors) {
            // Collect candidates for the new neighbors
            candidatesList<DistType> new_neighbors(this->allocator);
            std::unordered_set<idType> neighbors_set;
            auto add_candidate = [&](idType id) {
                if (neighbors_set.insert(id).second) {
                    DistType dist =
                        this->calcDistance(getDataByInternalId(id), getDataByInternalId(graphKey->id));
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
int HNSWDiskIndex<DataType, DistType>::addVector(
    const void *vector, labelType label
) {
    std::cout << "\n=== addVector called with label " << label << " ===" << std::endl;
    std::cout << "Current index size: " << curElementCount << std::endl;
    std::cout << "Pending vectors: " << pendingVectorCount << std::endl;
    
    addVectorToBatch(vector, label);
    
    std::cout << "After addVectorToBatch - pending vectors: " << pendingVectorCount << std::endl;
    
    // If batch threshold reached, process immediately for debugging
    if (pendingVectorCount >= batchThreshold) {
        std::cout << "Batch threshold reached, processing immediately..." << std::endl;
        processBatch();
    }
    
    return 1; // Success
}

template <typename DataType, typename DistType>
HNSWAddVectorState HNSWDiskIndex<DataType, DistType>::storeVector(
    const void *vector_data, const labelType label) {
    HNSWAddVectorState state{};

    // Choose randomly the maximum level in which the new element will be in the index.
    state.elementMaxLevel = getRandomLevel(mult);

    // Access and update the index global data structures with the new element meta-data.
    state.newElementId = curElementCount++;

    // Store the vector data in both memory and RocksDB
    // 1. Store in memory for fast access (use processed data)
    ProcessedBlobs processedBlobs = this->preprocess(vector_data);
    this->vectors->addElement(processedBlobs.getStorageBlob(), state.newElementId);
    
    // 2. Store in RocksDB for persistence
    try {
        std::string vector_key = std::to_string(state.newElementId);
        rocksdb::Slice data_slice(reinterpret_cast<const char*>(vector_data), this->dataSize);
        rocksdb::Status vector_status = db->Put(rocksdb::WriteOptions(), cf, vector_key, data_slice);

        if (!vector_status.ok()) {
            std::cout << "Failed to store vector in RocksDB: " << vector_status.ToString() << std::endl;
            // Failed to store vector in RocksDB, but we still have it in memory
            // For now, continue anyway (could add error handling here)
        }
    } catch (const std::exception& e) {
        std::cout << "Exception during RocksDB Put: " << e.what() << std::endl;
    }

    // Create the new element's metadata
    DiskElementMetaData new_element(label, state.elementMaxLevel);
    idToMetaData[state.newElementId] = new_element;

    // Update label mapping
    labelToIdMap[label] = state.newElementId;

    state.currMaxLevel = (int)maxLevel;
    state.currEntryPoint = entrypointNode;
    if (state.elementMaxLevel > state.currMaxLevel) {
        if (entrypointNode == INVALID_ID && maxLevel != HNSW_INVALID_LEVEL) {
            throw std::runtime_error("Internal error - inserting the first element to the graph,"
                                     " but the current max level is not INVALID");
        }
        // If the new elements max level is higher than the maximum level the currently exists in
        // the graph, update the max level and set the new element as entry point.
        entrypointNode = state.newElementId;
        maxLevel = state.elementMaxLevel;
    }
    std::cout << "New element added: " << state.newElementId << " with max level: " << state.elementMaxLevel << std::endl;
    return state;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::indexVector(
    const void *vector_data, const labelType label, const HNSWAddVectorState &state) {
    // Deconstruct the state variables from the auxiliaryCtx. prev_entry_point and prev_max_level
    // are the entry point and index max level at the point of time when the element was stored, and
    // they may (or may not) have changed due to the insertion.
    auto [new_element_id, element_max_level, prev_entry_point, prev_max_level] = state;

    // This condition only means that we are not inserting the first (non-deleted) element (for the
    // first element we do nothing - we don't need to connect to it).
    if (prev_entry_point != INVALID_ID) {
        // Start scanning the graph from the current entry point.
        insertElementToGraph(new_element_id, element_max_level, prev_entry_point, prev_max_level,
                             vector_data);
    }
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::appendVector(
    const void *vector_data, const labelType label) {

    std::cout << "\n=== Adding Vector with Label " << label << " ===" << std::endl;
    
    ProcessedBlobs processedBlobs = this->preprocess(vector_data);
    std::cout << "Vector preprocessed successfully" << std::endl;
    
    HNSWAddVectorState state = this->storeVector(processedBlobs.getStorageBlob(), label);
    std::cout << "Vector stored with ID " << state.newElementId << ", max level " << state.elementMaxLevel << std::endl;

    this->indexVector(processedBlobs.getQueryBlob(), label, state);
    std::cout << "Vector indexed successfully" << std::endl;
    
    // Debug: Print graph structure after adding this vector
    std::cout << "\n=== Graph Structure After Adding Vector " << label << " ===" << std::endl;
    debugPrintGraphStructure();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::insertElementToGraph(
    idType element_id, size_t element_max_level, idType entry_point,
    size_t global_max_level, const void *vector_data) {

    std::cout << "\n=== Inserting Element " << element_id << " to Graph ===" << std::endl;
    std::cout << "Element max level: " << element_max_level << std::endl;
    std::cout << "Global max level: " << global_max_level << std::endl;
    std::cout << "Entry point: " << entry_point << std::endl;

    idType curr_element = entry_point;
    DistType cur_dist = std::numeric_limits<DistType>::max();
    size_t max_common_level;
    if (element_max_level < global_max_level) {
        max_common_level = element_max_level;
        cur_dist = this->calcDistance(vector_data, getDataByInternalId(curr_element));
        std::cout << "Element level < global level, searching from level " << global_max_level 
                  << " down to " << element_max_level << std::endl;
        for (auto level = static_cast<int>(global_max_level);
             level > static_cast<int>(element_max_level); level--) {
            // this is done for the levels which are above the max level
            // to which we are going to insert the new element. We do
            // a greedy search in the graph starting from the entry point
            // at each level, and move on with the closest element we can find.
            // When there is no improvement to do, we take a step down.
            std::cout << "Greedy search at level " << level << std::endl;
            greedySearchLevel(vector_data, level, curr_element, cur_dist);
            std::cout << "After greedy search: curr_element=" << curr_element << ", cur_dist=" << cur_dist << std::endl;
        }
    } else {
        max_common_level = global_max_level;
    }

    std::cout << "Max common level: " << max_common_level << std::endl;
    std::cout << "Starting level-by-level insertion from level " << max_common_level << " down to 0" << std::endl;

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;

    for (auto level = static_cast<int>(max_common_level); level >= 0; level--) {
        std::cout << "\n--- Processing Level " << level << " ---" << std::endl;
        std::cout << "Searching for candidates starting from element " << curr_element << std::endl;
        
        candidatesMaxHeap<DistType> top_candidates =
            searchLayer(curr_element, vector_data, level, efConstruction);
        
        std::cout << "Found " << top_candidates.size() << " candidates at level " << level << std::endl;
        
        // If the entry point was marked deleted between iterations, we may receive an empty
        // candidates set.
        if (!top_candidates.empty()) {
            std::cout << "Mutually connecting element " << element_id << " at level " << level << std::endl;
            curr_element = mutuallyConnectNewElement(element_id, top_candidates, level);
            std::cout << "Next closest entry point: " << curr_element << std::endl;
        } else {
            std::cout << "WARNING: No candidates found at level " << level << "!" << std::endl;
        }
    }
    
    std::cout << "=== Finished inserting Element " << element_id << " ===\n" << std::endl;
}

template <typename DataType, typename DistType>
idType HNSWDiskIndex<DataType, DistType>::mutuallyConnectNewElement(
    idType new_node_id, candidatesMaxHeap<DistType> &top_candidates, size_t level) {

    std::cout << "  mutuallyConnectNewElement: Starting for node " << new_node_id << " at level " << level << std::endl;
    
    // Test RocksDB connection first
    std::cout << "  mutuallyConnectNewElement: Testing RocksDB connection..." << std::endl;
    std::string test_key = "TEST_CONNECTION";
    std::string test_value = "TEST_VALUE";
    rocksdb::Status test_status = this->db->Put(rocksdb::WriteOptions(), cf, test_key, test_value);
    if (!test_status.ok()) {
        std::cout << "  ERROR: RocksDB test write failed: " << test_status.ToString() << std::endl;
        std::cout << "  This indicates a serious RocksDB connection issue!" << std::endl;
    } else {
        std::cout << "  SUCCESS: RocksDB test write succeeded" << std::endl;
        // Clean up test key
        this->db->Delete(rocksdb::WriteOptions(), cf, test_key);
    }

    // Filter the top candidates to the selected neighbors by the algorithm heuristics.
    // First, we need to copy the top candidates to a vector.
    candidatesList<DistType> top_candidates_list(this->allocator);
    top_candidates_list.insert(top_candidates_list.end(), top_candidates.begin(),
                               top_candidates.end());
    // Use the heuristic to filter the top candidates, and get the next closest entry point.
    idType next_closest_entry_point = getNeighborsByHeuristic2(top_candidates_list, M);
    assert(top_candidates_list.size() <= M &&
           "Should be not be more than M candidates returned by the heuristic");

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;

    // Add the new element to the graph at this level
    auto newKey = GraphKey(new_node_id, level);
    std::cout << "  mutuallyConnectNewElement: Created GraphKey for node " << new_node_id 
              << " at level " << level << std::endl;
    std::cout << "  mutuallyConnectNewElement: GraphKey size: " << sizeof(GraphKey) << " bytes" << std::endl;
    std::cout << "  mutuallyConnectNewElement: GraphKey asSlice size: " << newKey.asSlice().size() << " bytes" << std::endl;
    
    std::vector<idType> neighbor_ids(top_candidates_list.size());
    for (size_t i = 0; i < top_candidates_list.size(); ++i) {
        neighbor_ids[i] = top_candidates_list[i].second;
    }
    size_t bytes = neighbor_ids.size() * sizeof(idType);
    auto neighbors_slice =
        rocksdb::Slice(reinterpret_cast<const char *>(neighbor_ids.data()), bytes);
    
    std::cout << "  mutuallyConnectNewElement: Neighbors data size: " << bytes << " bytes" << std::endl;
    
    std::cout << "Storing graph connection: node " << new_node_id << " at level " << level 
              << " with " << neighbor_ids.size() << " neighbors: ";
    for (size_t i = 0; i < neighbor_ids.size(); ++i) {
        std::cout << neighbor_ids[i] << " ";
    }
    std::cout << std::endl;
    
    // Store the new node's neighbors
    rocksdb::Status put_status = this->db->Put(writeOptions, cf, newKey.asSlice(), neighbors_slice);
    if (!put_status.ok()) {
        std::cout << "ERROR: Failed to store graph connection for node " << new_node_id 
                  << " at level " << level << ": " << put_status.ToString() << std::endl;
    } else {
        std::cout << "SUCCESS: Stored graph connection for node " << new_node_id << " at level " << level << std::endl;
    }

    // Update existing nodes to include the new node in their neighbor lists
    for (const auto &neighbor_data : top_candidates_list) {
        idType selected_neighbor = neighbor_data.second;
        
        // Read existing neighbors for this node
        GraphKey neighborKey(selected_neighbor, level);
        std::string existing_neighbors_data;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, neighborKey.asSlice(), &existing_neighbors_data);
        
        std::vector<idType> updated_neighbors;
        if (status.ok()) {
            // Parse existing neighbors
            const idType* existing_neighbors = reinterpret_cast<const idType*>(existing_neighbors_data.data());
            size_t num_existing = existing_neighbors_data.size() / sizeof(idType);
            updated_neighbors.assign(existing_neighbors, existing_neighbors + num_existing);
        }
        
        // Add the new node to the neighbor's list
        updated_neighbors.push_back(new_node_id);
        
        // Write back the updated neighbor list
        size_t updated_bytes = updated_neighbors.size() * sizeof(idType);
        auto updated_neighbors_slice =
            rocksdb::Slice(reinterpret_cast<const char *>(updated_neighbors.data()), updated_bytes);
        rocksdb::Status update_status = this->db->Put(writeOptions, cf, neighborKey.asSlice(), updated_neighbors_slice);
        
        if (!update_status.ok()) {
            std::cout << "ERROR: Failed to update neighbors for node " << selected_neighbor 
                      << " at level " << level << ": " << update_status.ToString() << std::endl;
        } else {
            std::cout << "SUCCESS: Updated node " << selected_neighbor << " to include neighbor " << new_node_id << std::endl;
        }
    }

    return next_closest_entry_point;
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
idType HNSWDiskIndex<DataType, DistType>::searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                                              VecSimQueryReply_Code *rc) const {
    if (rc) *rc = VecSim_QueryReply_OK;
    
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
candidatesLabelsMaxHeap<DistType> *
HNSWDiskIndex<DataType, DistType>::searchBottomLayer_WithTimeout(idType ep_id, const void *data_point,
                                                             size_t ef, size_t k, void *timeoutCtx,
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
            if (rc) *rc = VecSim_QueryReply_TimedOut;
            return top_candidates;
        }
        candidate_set.pop();

        processCandidate(curr_el_pair.second, data_point, 0, ef,
                         reinterpret_cast<void*>(&visited_set), 0, *top_candidates,
                         candidate_set, lowerBound);
    }
    
    while (top_candidates->size() > k) {
        top_candidates->pop();
    }
    if (rc) *rc = VecSim_QueryReply_OK;
    return top_candidates;
}

/********************************** Stub Implementations **********************************/

template <typename DataType, typename DistType>
const void* HNSWDiskIndex<DataType, DistType>::getDataByInternalId(idType id) const {
    // Check if the id is valid
    if (id >= curElementCount) {
        return nullptr;
    }

    // First, try to get from memory (fast path)
    if (id < this->vectors->size() && this->vectors->getElement(id) != nullptr) {
        return this->vectors->getElement(id);
    }

    // Fallback to RocksDB (slower path, for persistence/recovery)
    std::string key = std::to_string(id);
    std::string value;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, key, &value);

    if (!status.ok()) {
        // Vector data not found in either memory or RocksDB
        return nullptr;
    }

    // For now, we'll use the memory storage from the base class
    // In a production implementation, you might want to implement a proper cache
    // or use a different strategy for handling disk-based vector retrieval
    return this->vectors->getElement(id);
}

template <typename DataType, typename DistType>
candidatesMaxHeap<DistType> HNSWDiskIndex<DataType, DistType>::searchLayer(idType ep_id, const void* data_point, size_t level, size_t ef) const {
    std::cout << "  searchLayer: Starting search at level " << level << " from entry point " << ep_id << std::endl;
    
    candidatesMaxHeap<DistType> candidates(this->allocator);
    candidatesMaxHeap<DistType> candidates_set(this->allocator);
    
    // Get visited list
    auto *visited_nodes_handler = getVisitedList();
    tag_t visited_tag = visited_nodes_handler->getFreshTag();
    
    // Start with the entry point
    DistType dist = this->calcDistance(data_point, getDataByInternalId(ep_id));
    std::cout << "  searchLayer: Entry point " << ep_id << " distance: " << dist << std::endl;
    candidates.emplace(dist, ep_id);
    candidates_set.emplace(-dist, ep_id);
    visited_nodes_handler->tagNode(ep_id, visited_tag);
    
    size_t iterations = 0;
    // Search for candidates
    while (!candidates_set.empty()) {
        iterations++;
        auto curr_pair = candidates_set.top();
        DistType curr_dist = -curr_pair.first;
        idType curr_id = curr_pair.second;
        candidates_set.pop();
        
        std::cout << "  searchLayer: Iteration " << iterations << ", processing node " << curr_id 
                  << " with distance " << curr_dist << std::endl;
        
        // If we have enough candidates and the current distance is worse than our best, stop
        if (candidates.size() >= ef && curr_dist > candidates.top().first) {
            std::cout << "  searchLayer: Stopping search - have " << candidates.size() 
                      << " candidates and current distance " << curr_dist 
                      << " > best distance " << candidates.top().first << std::endl;
            break;
        }
        
        // Get neighbors of current node at this level
        GraphKey graphKey(curr_id, level);
        std::string neighbors_data;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &neighbors_data);
        
        if (status.ok()) {
            const idType* neighbors = reinterpret_cast<const idType*>(neighbors_data.data());
            size_t num_neighbors = neighbors_data.size() / sizeof(idType);
            std::cout << "  searchLayer: Node " << curr_id << " has " << num_neighbors << " neighbors at level " << level << std::endl;
            
            for (size_t i = 0; i < num_neighbors; i++) {
                idType neighbor_id = neighbors[i];
                
                if (visited_nodes_handler->getNodeTag(neighbor_id) == visited_tag) {
                    std::cout << "  searchLayer: Skipping already visited neighbor " << neighbor_id << std::endl;
                    continue;
                }
                
                visited_nodes_handler->tagNode(neighbor_id, visited_tag);
                DistType neighbor_dist = this->calcDistance(data_point, getDataByInternalId(neighbor_id));
                std::cout << "  searchLayer: Adding neighbor " << neighbor_id << " with distance " << neighbor_dist << std::endl;
                
                candidates.emplace(neighbor_dist, neighbor_id);
                candidates_set.emplace(-neighbor_dist, neighbor_id);
            }
        } else {
            std::cout << "  searchLayer: WARNING - No neighbors found for node " << curr_id << " at level " << level << std::endl;
            std::cout << "  searchLayer: RocksDB status: " << status.ToString() << std::endl;
        }
    }
    
    std::cout << "  searchLayer: Search completed in " << iterations << " iterations" << std::endl;
    std::cout << "  searchLayer: Found " << candidates.size() << " candidates" << std::endl;
    
    returnVisitedList(visited_nodes_handler);
    return candidates;
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
    // For now, no elements are marked as deleted
    // In a real implementation, this would check a deletion flag
    return false;
}

template <typename DataType, typename DistType>
std::pair<idType, size_t> HNSWDiskIndex<DataType, DistType>::safeGetEntryPointState() const {
    std::shared_lock<std::shared_mutex> lock(indexDataGuard);
    return std::make_pair(entrypointNode, maxLevel);
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::greedySearchLevel(const void* data_point, size_t level, idType& curr_element, DistType& cur_dist) const {
    bool changed;
    idType bestCand = curr_element;

    std::cout << "Greedy search at level " << level << " starting from element " << bestCand << std::endl;

    do {
        changed = false;

        // Read neighbors from RocksDB for the current node at this level
        GraphKey graphKey(bestCand, level);
        std::string neighbors_data;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &neighbors_data);

        if (!status.ok()) {
            // No neighbors found for this node at this level, stop searching
            std::cout << "No neighbors found for element " << bestCand << " at level " << level << std::endl;
            break;
        }

        // Parse the neighbors data
        const idType* neighbors = reinterpret_cast<const idType*>(neighbors_data.data());
        size_t num_neighbors = neighbors_data.size() / sizeof(idType);

        // Check each neighbor to find a better candidate
        for (size_t i = 0; i < num_neighbors; i++) {
            idType candidate = neighbors[i];

            // Skip invalid candidates
            if (candidate >= curElementCount) {
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
candidatesLabelsMaxHeap<DistType>* HNSWDiskIndex<DataType, DistType>::getNewMaxPriorityQueue() const {
    // Use max_priority_queue for single-label disk index
    return new (this->allocator) vecsim_stl::max_priority_queue<DistType, labelType>(this->allocator);
}

template <typename DataType, typename DistType>
VisitedNodesHandler* HNSWDiskIndex<DataType, DistType>::getVisitedList() const {
    return visitedNodesHandlerPool.getAvailableVisitedNodesHandler();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::returnVisitedList(VisitedNodesHandler* visited_nodes_handler) const {
    visitedNodesHandlerPool.returnVisitedNodesHandlerToPool(visited_nodes_handler);
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(levelGenerator)) * reverse_size;
    return (size_t)r;
}



template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::processCandidate(idType candidate_id, const void* data_point, size_t level, size_t ef,
                     void* visited_tags, size_t visited_tag, candidatesLabelsMaxHeap<DistType>& top_candidates,
                     candidatesMaxHeap<DistType>& candidate_set, DistType& lowerBound) const {
    // Use a simple set-based approach for now to avoid visited nodes handler issues
    auto* visited_set = reinterpret_cast<std::unordered_set<idType>*>(visited_tags);
    if (!visited_set) {
        return; // Safety check
    }
    
    if (visited_set->find(candidate_id) != visited_set->end()) {
        return;
    }
    
    std::cout << "    processCandidate: Processing candidate " << candidate_id << std::endl;
    visited_set->insert(candidate_id);
    
    // Calculate distance to candidate
    DistType dist = this->calcDistance(data_point, getDataByInternalId(candidate_id));
    std::cout << "    processCandidate: Candidate " << candidate_id << " distance: " << dist << std::endl;
    
    // Add to top candidates if it's one of the best
    if (top_candidates.size() < ef || dist < lowerBound) {
        top_candidates.emplace(dist, getExternalLabel(candidate_id));
        std::cout << "    processCandidate: Added candidate " << candidate_id << " to results" << std::endl;
        
        // Update lower bound if we have enough candidates
        if (top_candidates.size() >= ef) {
            lowerBound = top_candidates.top().first;
            std::cout << "    processCandidate: Updated lower bound to: " << lowerBound << std::endl;
        }
    } else {
        std::cout << "    processCandidate: Candidate " << candidate_id << " not good enough (dist=" << dist 
                  << " >= lowerBound=" << lowerBound << ")" << std::endl;
    }
    
    // Add neighbors to candidate set for further exploration
    GraphKey graphKey(candidate_id, level);
    std::string neighbors_data;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &neighbors_data);
    
    if (status.ok()) {
        const idType* neighbors = reinterpret_cast<const idType*>(neighbors_data.data());
        size_t num_neighbors = neighbors_data.size() / sizeof(idType);
        std::cout << "    processCandidate: Found " << num_neighbors << " neighbors for candidate " << candidate_id << std::endl;
        
        for (size_t i = 0; i < num_neighbors; i++) {
            idType neighbor_id = neighbors[i];
            
            // Skip invalid neighbors
            if (neighbor_id >= curElementCount) {
                std::cout << "    processCandidate: Skipping invalid neighbor " << neighbor_id << std::endl;
                continue;
            }
            
            if (visited_set->find(neighbor_id) == visited_set->end()) {
                DistType neighbor_dist = this->calcDistance(data_point, getDataByInternalId(neighbor_id));
                candidate_set.emplace(-neighbor_dist, neighbor_id);
                std::cout << "    processCandidate: Added neighbor " << neighbor_id << " to candidate set with distance " << neighbor_dist << std::endl;
            } else {
                std::cout << "    processCandidate: Skipping already visited neighbor " << neighbor_id << std::endl;
            }
        }
    } else {
        std::cout << "    processCandidate: No neighbors found for candidate " << candidate_id << std::endl;
    }
}

template <typename DataType, typename DistType>
VecSimQueryReply *HNSWDiskIndex<DataType, DistType>::rangeQuery(const void *query_data, double radius,
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
VecSimBatchIterator *HNSWDiskIndex<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                                         VecSimQueryParams *queryParams) const {
    // TODO: Implement batch iterator
    return nullptr;
}

template <typename DataType, typename DistType>
bool HNSWDiskIndex<DataType, DistType>::preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const {
    // TODO: Implement ad-hoc search preference logic
    return false;
}


template <typename DataType, typename DistType>
int HNSWDiskIndex<DataType, DistType>::deleteVector(labelType label) {
    // TODO: Implement vector deletion
    return 0;
}

template <typename DataType, typename DistType>
double HNSWDiskIndex<DataType, DistType>::getDistanceFrom_Unsafe(labelType id, const void *blob) const {
    // TODO: Implement distance calculation
    return 0.0;
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::indexSize() const {
    return this->curElementCount + pendingVectorCount;
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::indexCapacity() const {
    return idToMetaData.size();
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::indexLabelCount() const {
    return this->curElementCount + pendingVectorCount;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::fitMemory() {
    // TODO: Implement memory fitting
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::getDataByLabel(labelType label, std::vector<std::vector<DataType>>& vectors_output) const {
    // TODO: Implement data retrieval by label
}

template <typename DataType, typename DistType>
std::vector<std::vector<char>> HNSWDiskIndex<DataType, DistType>::getStoredVectorDataByLabel(labelType label) const {
    // TODO: Implement stored vector data retrieval
    return {};
}

template <typename DataType, typename DistType>
vecsim_stl::set<labelType> HNSWDiskIndex<DataType, DistType>::getLabelsSet() const {
    // TODO: Implement labels set retrieval
    vecsim_stl::set<labelType> labels(this->allocator);
    return labels;
}

/********************************** Batch Processing Methods **********************************/

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::addVectorToBatch(const void *vector_data, const labelType label) {
    std::cout << "  addVectorToBatch: Processing vector with label " << label << std::endl;
    
    // Preprocess the vector
    ProcessedBlobs processedBlobs = this->preprocess(vector_data);
    std::cout << "  addVectorToBatch: Vector preprocessed successfully" << std::endl;
    
    // Store the processed vector in memory
    std::vector<char> vector_copy(this->dataSize);
    std::memcpy(vector_copy.data(), processedBlobs.getStorageBlob(), this->dataSize);
    std::cout << "  addVectorToBatch: Vector copied to memory, size: " << this->dataSize << " bytes" << std::endl;
    
    // Add to pending vectors
    pendingVectors.emplace_back(label, std::move(vector_copy));
    
    // Create metadata for this vector
    size_t elementMaxLevel = getRandomLevel(mult);
    DiskElementMetaData metadata(label, elementMaxLevel);
    pendingMetadata.push_back(metadata);
    
    pendingVectorCount++;
    
    std::cout << "  addVectorToBatch: Added to pending batch. Total pending: " << pendingVectorCount 
              << ", threshold: " << batchThreshold << std::endl;
    std::cout << "  addVectorToBatch: Generated level: " << elementMaxLevel << std::endl;
    
    // Check if we should process the batch
    if (pendingVectorCount >= batchThreshold) {
        std::cout << "  addVectorToBatch: Batch threshold reached, calling processBatch()" << std::endl;
        processBatch();
    }
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::processBatch() {
    if (pendingVectorCount == 0) {
        std::cout << "  processBatch: No pending vectors to process" << std::endl;
        return; // Nothing to process
    }
    
    std::cout << "\n=== Processing Batch ===" << std::endl;
    std::cout << "  processBatch: Processing " << pendingVectorCount << " vectors" << std::endl;
    std::cout << "  processBatch: Current index size: " << curElementCount << std::endl;
    
    // Process each pending vector individually using appendVector
    for (size_t i = 0; i < pendingVectorCount; i++) {
        std::cout << "  processBatch: Processing vector " << i << " with label " << pendingVectors[i].first << std::endl;
        appendVector(pendingVectors[i].second.data(), pendingVectors[i].first);
        std::cout << "  processBatch: Vector " << i << " processed" << std::endl;
    }
    
    // Note: Resize is not needed since we start with a small pool size
    
    // Clear the pending vectors
    pendingVectors.clear();
    pendingMetadata.clear();
    pendingVectorCount = 0;
    
    std::cout << "  processBatch: Batch processing completed" << std::endl;
    std::cout << "  processBatch: Final index size: " << curElementCount << std::endl;
    std::cout << "=== Batch Processing Complete ===\n" << std::endl;
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::flushBatch() {
    processBatch();
}

/********************************** Debug Methods **********************************/

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::debugPrintGraphStructure() const {
    std::cout << "\n=== HNSW Disk Index Graph Structure ===" << std::endl;
    std::cout << "Total elements: " << curElementCount << std::endl;
    std::cout << "Entry point: " << entrypointNode << std::endl;
    std::cout << "Max level: " << maxLevel << std::endl;
    std::cout << "M: " << M << ", M0: " << M0 << std::endl;
    
    // Count total edges
    size_t total_edges = debugCountGraphEdges();
    std::cout << "Total graph edges: " << total_edges << std::endl;
    
    // Print metadata for each element
    std::cout << "\nElement metadata:" << std::endl;
    for (size_t i = 0; i < std::min(curElementCount, idToMetaData.size()); ++i) {
        if (idToMetaData[i].label != INVALID_LABEL) {
            std::cout << "  Element " << i << ": label=" << idToMetaData[i].label 
                      << ", topLevel=" << idToMetaData[i].topLevel << std::endl;
        }
    }
    
    // Print graph keys and their neighbors
    debugPrintAllGraphKeys();
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::debugPrintNodeNeighbors(idType node_id) const {
    if (node_id >= curElementCount) {
        std::cout << "Node " << node_id << " is out of range (max: " << (curElementCount-1) << ")" << std::endl;
        return;
    }
    
    std::cout << "\n=== Neighbors for Node " << node_id << " ===" << std::endl;
    std::cout << "Label: " << getExternalLabel(node_id) << std::endl;
    std::cout << "Top level: " << idToMetaData[node_id].topLevel << std::endl;
    
    // Check each level
    for (size_t level = 0; level <= idToMetaData[node_id].topLevel; ++level) {
        GraphKey graphKey(node_id, level);
        std::string neighbors_data;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &neighbors_data);
        
        if (status.ok()) {
            const idType* neighbors = reinterpret_cast<const idType*>(neighbors_data.data());
            size_t num_neighbors = neighbors_data.size() / sizeof(idType);
            std::cout << "  Level " << level << " (" << num_neighbors << " neighbors): ";
            for (size_t i = 0; i < num_neighbors; i++) {
                std::cout << neighbors[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "  Level " << level << ": No neighbors found" << std::endl;
        }
    }
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::debugPrintAllGraphKeys() const {
    std::cout << "\n=== All Graph Keys in RocksDB ===" << std::endl;
    
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
            const GraphKey* graphKey = reinterpret_cast<const GraphKey*>(key.data() + 3);
            std::cout << "  Key " << key_count << ": node=" << graphKey->id 
                      << ", level=" << graphKey->level;
        } else {
            std::cout << "  Key " << key_count << ": invalid format (size=" << key.size() << ")";
        }
        
        // Get neighbors count
        auto neighborsSlice = it->value();
        size_t num_neighbors = neighborsSlice.size() / sizeof(idType);
        total_neighbors += num_neighbors;
        
        std::cout << " (" << num_neighbors << " neighbors)" << std::endl;
        
        // Print first few neighbors
        if (num_neighbors > 0) {
            const idType* neighbors = reinterpret_cast<const idType*>(neighborsSlice.data());
            std::cout << "    Neighbors: ";
            for (size_t i = 0; i < std::min(num_neighbors, size_t(5)); i++) {
                std::cout << neighbors[i] << " ";
            }
            if (num_neighbors > 5) {
                std::cout << "... (and " << (num_neighbors - 5) << " more)";
            }
            std::cout << std::endl;
        }
        
        key_count++;
    }
    
    std::cout << "Total graph keys: " << key_count << std::endl;
    std::cout << "Total neighbor connections: " << total_neighbors << std::endl;
    
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
    std::cout << "\n=== Graph Connectivity Validation ===" << std::endl;
    
    if (curElementCount == 0) {
        std::cout << "Index is empty, nothing to validate" << std::endl;
        return;
    }
    
    // Check if entry point exists and has neighbors
    if (entrypointNode != INVALID_ID) {
        std::cout << "Entry point " << entrypointNode << " exists" << std::endl;
        debugPrintNodeNeighbors(entrypointNode);
    } else {
        std::cout << "WARNING: No entry point set!" << std::endl;
    }
    
    // Check connectivity for first few elements
    size_t elements_to_check = std::min(curElementCount, size_t(5));
    std::cout << "\nChecking connectivity for first " << elements_to_check << " elements:" << std::endl;
    
    for (size_t i = 0; i < elements_to_check; ++i) {
        if (idToMetaData[i].label != INVALID_LABEL) {
            std::cout << "\nElement " << i << ":" << std::endl;
            debugPrintNodeNeighbors(i);
        }
    }
    
    // Check for isolated nodes (nodes with no neighbors at any level)
    std::cout << "\nChecking for isolated nodes..." << std::endl;
    size_t isolated_count = 0;
    
    for (size_t i = 0; i < curElementCount; ++i) {
        if (idToMetaData[i].label == INVALID_LABEL) continue;
        
        bool has_neighbors = false;
        for (size_t level = 0; level <= idToMetaData[i].topLevel; ++level) {
            GraphKey graphKey(i, level);
            std::string neighbors_data;
            rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf, graphKey.asSlice(), &neighbors_data);
            
            if (status.ok() && neighbors_data.size() > 0) {
                has_neighbors = true;
                break;
            }
        }
        
        if (!has_neighbors) {
            std::cout << "  WARNING: Element " << i << " (label " << idToMetaData[i].label 
                      << ") has no neighbors at any level!" << std::endl;
            isolated_count++;
        }
    }
    
    if (isolated_count == 0) {
        std::cout << "  All elements have at least some neighbors" << std::endl;
    } else {
        std::cout << "  Found " << isolated_count << " isolated elements" << std::endl;
    }
}

template <typename DataType, typename DistType>
candidatesLabelsMaxHeap<DistType> *
HNSWDiskIndex<DataType, DistType>::hierarchicalSearch(const void *data_point, idType ep_id, size_t ef, size_t k, void *timeoutCtx,
                                                      VecSimQueryReply_Code *rc) const {
    if (rc) *rc = VecSim_QueryReply_OK;
    
    std::cout << "  hierarchicalSearch: Starting hierarchical search from entry point " << ep_id << std::endl;
    std::cout << "  hierarchicalSearch: Target ef=" << ef << ", k=" << k << std::endl;
    
    // Get the current entry point state
    auto [curr_entry_point, max_level] = safeGetEntryPointState();
    if (curr_entry_point == INVALID_ID) {
        std::cout << "  hierarchicalSearch: No valid entry point found" << std::endl;
        if (rc) *rc = VecSim_QueryReply_OK; // Just return OK but no results
        return nullptr;
    }
    
    std::cout << "  hierarchicalSearch: Current entry point: " << curr_entry_point << ", max level: " << max_level << std::endl;
    
    // Initialize result containers
    candidatesLabelsMaxHeap<DistType> *top_candidates = getNewMaxPriorityQueue();
    candidatesMaxHeap<DistType> candidate_set(this->allocator);
    
    // Use a simple set for visited nodes tracking
    std::unordered_set<idType> visited_set;
    
    // Start from the provided entry point (not the global entry point)
    idType curr_element = ep_id;
    DistType curr_dist = this->calcDistance(data_point, getDataByInternalId(curr_element));
    
    std::cout << "  hierarchicalSearch: Entry point distance: " << curr_dist << std::endl;
    
    // Add entry point to results
    if (!isMarkedDeleted(curr_element)) {
        top_candidates->emplace(curr_dist, getExternalLabel(curr_element));
        visited_set.insert(curr_element);
    }
    
    // Phase 1: Search from top level down to level 1 (hierarchical traversal)
    std::cout << "  hierarchicalSearch: Phase 1 - Hierarchical traversal from level " << max_level << " to 1" << std::endl;
    for (size_t level = max_level; level > 0; --level) {
        std::cout << "  hierarchicalSearch: Searching at level " << level << std::endl;
        
        // Search at this level using the current element as entry point
        candidatesMaxHeap<DistType> level_candidates = searchLayer(curr_element, data_point, level, ef);
        
        if (!level_candidates.empty()) {
            // Find the closest element at this level to continue the search
            curr_element = level_candidates.top().second;
            curr_dist = level_candidates.top().first;
            std::cout << "  hierarchicalSearch: Level " << level << " - closest element: " << curr_element 
                      << " with distance: " << curr_dist << std::endl;
        } else {
            std::cout << "  hierarchicalSearch: Level " << level << " - no candidates found" << std::endl;
        }
        
        // Check timeout
        if (timeoutCtx && VECSIM_TIMEOUT(timeoutCtx)) {
            if (rc) *rc = VecSim_QueryReply_TimedOut;
            delete top_candidates;
            return nullptr;
        }
    }
    
    // Phase 2: Search at the bottom layer (level 0) with beam search
    std::cout << "  hierarchicalSearch: Phase 2 - Bottom layer search at level 0" << std::endl;
    std::cout << "  hierarchicalSearch: Starting from element " << curr_element << " with distance " << curr_dist << std::endl;
    
    // Reset visited set for bottom layer search
    visited_set.clear();
    visited_set.insert(curr_element);
    
    // Initialize candidate set with current element and its neighbors at level 0
    // Since candidatesMaxHeap doesn't have clear(), we'll create a new one
    candidate_set = candidatesMaxHeap<DistType>(this->allocator);
    candidate_set.emplace(-curr_dist, curr_element);
    
    // Add neighbors of the current element at level 0 to get started
    GraphKey startKey(curr_element, 0);
    std::string start_neighbors_data;
    rocksdb::Status start_status = db->Get(rocksdb::ReadOptions(), cf, startKey.asSlice(), &start_neighbors_data);
    
    if (start_status.ok()) {
        const idType* start_neighbors = reinterpret_cast<const idType*>(start_neighbors_data.data());
        size_t num_start_neighbors = start_neighbors_data.size() / sizeof(idType);
        std::cout << "  hierarchicalSearch: Adding " << num_start_neighbors << " initial neighbors from level 0" << std::endl;
        
        for (size_t i = 0; i < num_start_neighbors; i++) {
            idType neighbor_id = start_neighbors[i];
            if (neighbor_id < curElementCount && visited_set.find(neighbor_id) == visited_set.end()) {
                DistType neighbor_dist = this->calcDistance(data_point, getDataByInternalId(neighbor_id));
                candidate_set.emplace(-neighbor_dist, neighbor_id);
                std::cout << "  hierarchicalSearch: Added initial neighbor " << neighbor_id << " with distance " << neighbor_dist << std::endl;
            }
        }
    }
    
    // Beam search at bottom layer
    DistType lower_bound = curr_dist;
    
    std::cout << "  hierarchicalSearch: Starting beam search with initial lower bound: " << lower_bound << std::endl;
    std::cout << "  hierarchicalSearch: Initial candidate set size: " << candidate_set.size() << std::endl;
    
    while (!candidate_set.empty()) {
        // Check timeout
        if (timeoutCtx && VECSIM_TIMEOUT(timeoutCtx)) {
            if (rc) *rc = VecSim_QueryReply_TimedOut;
            break;
        }
        
        auto curr_pair = candidate_set.top();
        DistType curr_candidate_dist = -curr_pair.first;
        idType curr_candidate_id = curr_pair.second;
        candidate_set.pop();
        
        std::cout << "  hierarchicalSearch: Processing candidate " << curr_candidate_id 
                  << " with distance " << curr_candidate_dist << std::endl;
        
        // If we have enough candidates and current distance is worse, stop
        if (top_candidates->size() >= ef && curr_candidate_dist > lower_bound) {
            std::cout << "  hierarchicalSearch: Stopping search - have " << top_candidates->size() 
                      << " candidates and current distance " << curr_candidate_dist 
                      << " > lower bound " << lower_bound << std::endl;
            break;
        }
        
        // Process this candidate
        processCandidate(curr_candidate_id, data_point, 0, ef, 
                        reinterpret_cast<void*>(&visited_set), 0, 
                        *top_candidates, candidate_set, lower_bound);
        
        std::cout << "  hierarchicalSearch: After processing candidate " << curr_candidate_id 
                  << " - top candidates: " << top_candidates->size() 
                  << ", candidate set: " << candidate_set.size() << std::endl;
        
        // Update lower bound based on current top candidates
        if (top_candidates->size() >= ef) {
            lower_bound = top_candidates->top().first;
            std::cout << "  hierarchicalSearch: Updated lower bound to: " << lower_bound << std::endl;
        }
        
        // Continue searching until we have enough candidates or exhaust all possibilities
        if (top_candidates->size() >= ef && candidate_set.empty()) {
            std::cout << "  hierarchicalSearch: Stopping search - have " << top_candidates->size() 
                      << " candidates and no more candidates to explore" << std::endl;
            break;
        }
    }
    
    // Trim results to k
    while (top_candidates->size() > k) {
        top_candidates->pop();
    }
    
    std::cout << "  hierarchicalSearch: Search completed. Final results: " << top_candidates->size() << std::endl;
    if (rc) *rc = VecSim_QueryReply_OK;
    return top_candidates;
}
