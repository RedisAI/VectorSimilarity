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
// #include "../RediSearch/.install/boost/boost/unordered/concurrent_flat_map.hpp"

// Includes that should be in the inherited class
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/spaces/computer/preprocessors.h"
#include "VecSim/algorithms/hnsw/visited_nodes_handler.h"

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

static constexpr char GraphKeyPrefix[3] = "GK";
// #pragma pack(1)
struct GraphKey {
    char keyPrefix[3] = {'G', 'K', '\0'};
    // uint8_t version;
    uint16_t level;
    idType id;

    GraphKey(idType id, size_t level) : level(level), id(id) {}

    rocksdb::Slice asSlice() const {
        return rocksdb::Slice(reinterpret_cast<const char *>(this), sizeof(*this));
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
    vecsim_stl::vector<const void *> quantizedData; // quantized data for the elements
    vecsim_stl::unordered_map<labelType, idType> labelToIdMap;
    rocksdb::DB *db;                 // RocksDB database, not owned by the index
    rocksdb::ColumnFamilyHandle *cf; // RocksDB column family handle, not owned by the index

    mutable std::shared_mutex indexDataGuard;
    mutable VisitedNodesHandlerPool visitedNodesHandlerPool;

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
    void addVector(labelType label, const void *vector, size_t &curMaxLevel, idType &curEntryPoint,
                   vecsim_stl::vector<DiskElementMetaData> &new_elements_meta_data,
                    vecsim_stl::vector<const void*> &new_elements_quantized_data,
                   std::unordered_map<idType, std::vector<idType>> &delta_list);
    void patchDeltaList(std::unordered_map<idType, std::vector<idType>> &delta_list,
                        vecsim_stl::vector<DiskElementMetaData> &new_elements_meta_data,
                        std::unordered_map<idType, idType> &new_ids_mapping);

    // Missing method declarations - adding stub implementations
    const void* getDataByInternalId(idType id) const;
    candidatesMaxHeap<DistType> searchLayer(idType ep_id, const void* data_point, size_t level, size_t ef) const;
    void greedySearchLevel(const void* data_point, size_t level, idType& curr_element, DistType& cur_dist, void* timeoutCtx = nullptr, VecSimQueryReply_Code* rc = nullptr) const;
    std::pair<idType, size_t> safeGetEntryPointState() const;
    VisitedNodesHandler* getVisitedList() const;
    void returnVisitedList(VisitedNodesHandler* visited_nodes_handler) const;
    candidatesLabelsMaxHeap<DistType>* getNewMaxPriorityQueue() const;
    bool isMarkedDeleted(idType id) const;
    labelType getExternalLabel(idType id) const;
    void processCandidate(idType candidate_id, const void* data_point, size_t level, size_t ef,
                         void* visited_tags, size_t visited_tag, candidatesLabelsMaxHeap<DistType>& top_candidates,
                         candidatesMaxHeap<DistType>& candidate_set, DistType& lowerBound) const;

    size_t getRandomLevel() const {
        std::uniform_real_distribution<double> distribution;
        double r = -log(distribution(levelGenerator)) * mult;
        return static_cast<size_t>(r);
    }

    idType searchBottomLayerEP(const rocksdb::Snapshot *snp, const void *query_data, void *timeoutCtx,
                               VecSimQueryReply_Code *rc) const;

    candidatesLabelsMaxHeap<DistType> *
    searchBottomLayer_WithTimeout(const rocksdb::Snapshot *snp, idType ep_id, const void *data_point, size_t ef, size_t k,
                                  void *timeoutCtx, VecSimQueryReply_Code *rc) const;

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

    // Pure virtual methods from VecSimIndexInterface
    int addVector(const void *blob, labelType label) override;

private:
    // Internal helper that doesn't acquire locks
    int addVectorInternal(const void *blob, labelType label);
    int deleteVector(labelType label) override;
    double getDistanceFrom_Unsafe(labelType id, const void *blob) const override;
    size_t indexSize() const override;
    size_t indexCapacity() const override;
    size_t indexLabelCount() const override;
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
      idToMetaData(1000, this->allocator), quantizedData(this->allocator),
      labelToIdMap(this->allocator), db(db), cf(cf), indexDataGuard(),
      visitedNodesHandlerPool(1000, this->allocator) {

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
        deleted_ids.push_back(labelToIdMap[label]);
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
    vecsim_stl::vector<ElementMetaData> new_elements_meta_data(this->allocator);
    vecsim_stl::vector<const void*> new_elements_quantized_data(this->allocator);
    new_elements_meta_data.reserve(new_elements.size());
    new_elements_quantized_data.reserve(new_elements.size());
    // std::unordered_map<labelType, idType> new_labels_mapping;
    for (const auto &[label, vector] : new_elements) {
        addVector(label, vector, curMaxLevel, curEntryPoint, new_elements_meta_data, new_elements_quantized_data, delta_list);
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
    // this->lastMode = STANDARD_KNN;

    if (curElementCount == 0 || k == 0) {
        return rep;
    }

    auto processed_query_ptr = this->preprocessQuery(query_data);
    const void *processed_query = processed_query_ptr.get();
    void *timeoutCtx = nullptr;

    // Get original efRuntime and store it.
    size_t query_ef = this->ef;

    if (queryParams) {
        timeoutCtx = queryParams->timeoutCtx;
        if (queryParams->hnswRuntimeParams.efRuntime != 0) {
            query_ef = queryParams->hnswRuntimeParams.efRuntime;
        }
    }

    auto snapshot = this->db->GetSnapshot();

    idType bottom_layer_ep = searchBottomLayerEP(snapshot, processed_query, timeoutCtx, &rep->code);
    if (VecSim_OK != rep->code || bottom_layer_ep == INVALID_ID) {
        // Although we checked that the index is not empty (curElementCount == 0), it might be
        // that another thread deleted all the elements or didn't finish inserting the first element
        // yet. Anyway, we observed that the index is empty, so we return an empty result list.
        goto finish;
    }

    // We now oun the results heap, we need to free (delete) it when we done
    candidatesLabelsMaxHeap<DistType> *results;
    results = searchBottomLayer_WithTimeout(snapshot, bottom_layer_ep, processed_query, std::max(query_ef, k),
                                            k, timeoutCtx, &rep->code);

    if (VecSim_OK == rep->code) {
        rep->results.resize(results->size());
        for (auto result = rep->results.rbegin(); result != rep->results.rend(); result++) {
            std::tie(result->score, result->id) = results->top();
            results->pop();
        }
    }
    delete results;

finish:
    this->db->ReleaseSnapshot(snapshot);
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
void HNSWDiskIndex<DataType, DistType>::addVector(
    labelType label, const void *vector, size_t &curMaxLevel, idType &curEntryPoint,
    vecsim_stl::vector<DiskElementMetaData> &new_elements_meta_data,
    vecsim_stl::vector<const void*> &new_elements_quantized_data,
    std::unordered_map<idType, std::vector<idType>> &delta_list) {

    idType id = curElementCount + new_elements_meta_data.size();
    new_elements_meta_data.emplace_back(label, getRandomLevel());
    auto &new_element = new_elements_meta_data.back();

    auto writeOptions = rocksdb::WriteOptions();
    writeOptions.disableWAL = true;

    size_t commonLevel;
    idType curStartNode = curEntryPoint;
    if (new_element.topLevel > curMaxLevel) {
        // The new element is the first one in the new level. Add an empty list for all the
        // new levels.
        for (size_t i = curMaxLevel + 1; i <= new_element.topLevel; ++i) {
            auto newKey = GraphKey(id, i);
            auto emptySlice = rocksdb::Slice();
            this->db->Put(writeOptions, cf, newKey.asSlice(), emptySlice);
        }
        commonLevel = curMaxLevel;
        curMaxLevel = new_element.topLevel;
        curEntryPoint = id;
    } else {
        // The new element is not the first one in the new level. Look for the entry point in the
        // first common level.

        DistType cur_dist = this->calcDistance(vector, getDataByInternalId(curStartNode));
        for (auto level = curMaxLevel; level > new_element.topLevel; level--) {
            // this is done for the levels which are above the max level
            // to which we are going to insert the new element. We do
            // a greedy search in the graph starting from the entry point
            // at each level, and move on with the closest element we can find.
            // When there is no improvement to do, we take a step down.
            greedySearchLevel(vector, level, curStartNode, cur_dist);
        }
        commonLevel = new_element.topLevel;
    }
    // Insert the new element to all the levels from the common level to the bottom level
    for (int level = commonLevel; level >= 0; --level) {
        candidatesMaxHeap<DistType> top_candidates =
            searchLayer(curStartNode, vector, level, efConstruction);

        candidatesList<DistType> new_neighbors(this->allocator);
        new_neighbors.reserve(top_candidates.size());
        new_neighbors.insert(new_neighbors.end(), top_candidates.begin(), top_candidates.end());
        curStartNode = getNeighborsByHeuristic2(new_neighbors, level == 0 ? M0 : M);

        // Add the new element to the graph
        auto newKey = GraphKey(id, level);
        std::vector<idType> neighbor_ids(new_neighbors.size());
        for (size_t i = 0; i < new_neighbors.size(); ++i) {
            neighbor_ids[i] = new_neighbors[i].second;
        }
        size_t bytes = neighbor_ids.size() * sizeof(idType);
        auto neighbors_slice =
            rocksdb::Slice(reinterpret_cast<const char *>(neighbor_ids.data()), bytes);
        this->db->Put(writeOptions, cf, newKey.asSlice(), neighbors_slice);

        // Update the delta list with the new element
        for (const auto &candidate : top_candidates) {
            delta_list[candidate.second].push_back(id);
        }
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
idType HNSWDiskIndex<DataType, DistType>::searchBottomLayerEP(const rocksdb::Snapshot *snp, const void *query_data, void *timeoutCtx,
                                                          VecSimQueryReply_Code *rc) const {
    *rc = VecSim_QueryReply_OK;

    auto [curr_element, max_level] = safeGetEntryPointState();
    if (curr_element == INVALID_ID)
        return curr_element; // index is empty.

    DistType cur_dist = this->calcDistance(query_data, getDataByInternalId(curr_element));
    for (size_t level = max_level; level > 0 && curr_element != INVALID_ID; --level) {
        greedySearchLevel(query_data, level, curr_element, cur_dist, timeoutCtx, rc);
    }
    return curr_element;
}

template <typename DataType, typename DistType>
candidatesLabelsMaxHeap<DistType> *
HNSWDiskIndex<DataType, DistType>::searchBottomLayer_WithTimeout(const rocksdb::Snapshot *snp, idType ep_id, const void *data_point,
                                                             size_t ef, size_t k, void *timeoutCtx,
                                                             VecSimQueryReply_Code *rc) const {

    auto *visited_nodes_handler = getVisitedList();
    tag_t visited_tag = visited_nodes_handler->getFreshTag();

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

    visited_nodes_handler->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();

        if ((-curr_el_pair.first) > lowerBound && top_candidates->size() >= ef) {
            break;
        }
        if (VECSIM_TIMEOUT(timeoutCtx)) {
            returnVisitedList(visited_nodes_handler);
            *rc = VecSim_QueryReply_TimedOut;
            return top_candidates;
        }
        candidate_set.pop();

        processCandidate(curr_el_pair.second, data_point, 0, ef,
                         visited_nodes_handler->getElementsTags(), visited_tag, *top_candidates,
                         candidate_set, lowerBound);
    }
    returnVisitedList(visited_nodes_handler);
    while (top_candidates->size() > k) {
        top_candidates->pop();
    }
    *rc = VecSim_QueryReply_OK;
    return top_candidates;
}

/********************************** Stub Implementations **********************************/

template <typename DataType, typename DistType>
const void* HNSWDiskIndex<DataType, DistType>::getDataByInternalId(idType id) const {
    // Check if the id is valid
    if (id >= quantizedData.size()) {
        return nullptr;
    }

    // Return the quantized data for this internal id
    return quantizedData[id];
}

template <typename DataType, typename DistType>
candidatesMaxHeap<DistType> HNSWDiskIndex<DataType, DistType>::searchLayer(idType ep_id, const void* data_point, size_t level, size_t ef) const {
    // TODO: Implement proper layer search
    candidatesMaxHeap<DistType> candidates(this->allocator);
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
void HNSWDiskIndex<DataType, DistType>::greedySearchLevel(const void* data_point, size_t level, idType& curr_element, DistType& cur_dist, void* timeoutCtx, VecSimQueryReply_Code* rc) const {
    // TODO: Implement proper greedy search
    // For now, do nothing
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
void HNSWDiskIndex<DataType, DistType>::processCandidate(idType candidate_id, const void* data_point, size_t level, size_t ef,
                     void* visited_tags, size_t visited_tag, candidatesLabelsMaxHeap<DistType>& top_candidates,
                     candidatesMaxHeap<DistType>& candidate_set, DistType& lowerBound) const {
    // TODO: Implement proper candidate processing
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

// Implement missing pure virtual methods from VecSimIndexInterface
template <typename DataType, typename DistType>
int HNSWDiskIndex<DataType, DistType>::addVector(const void *blob, labelType label) {
    std::lock_guard<std::shared_mutex> guard(indexDataGuard);
    return addVectorInternal(blob, label);
}

template <typename DataType, typename DistType>
int HNSWDiskIndex<DataType, DistType>::addVectorInternal(const void *blob, labelType label) {
    // Check if label already exists
    auto it = labelToIdMap.find(label);
    if (it != labelToIdMap.end()) {
        // Label already exists, this is an update (return 0)
        idType existing_id = it->second;
        if (existing_id < quantizedData.size()) {
            // Update the existing data
            // For now, we'll just store the raw data (not actually quantized)
            // In a real implementation, this would go through quantization
            quantizedData[existing_id] = blob;
        }
        return 0;
    }

    // This is a new vector
    idType new_id = quantizedData.size();

    // Store the data (for now, just store the raw pointer - in real implementation this would be quantized and copied)
    quantizedData.push_back(blob);

    // Update the label to ID mapping
    labelToIdMap[label] = new_id;

    // Expand metadata if needed
    if (new_id >= idToMetaData.size()) {
        idToMetaData.resize(new_id + 1, DiskElementMetaData(label));
    } else {
        idToMetaData[new_id] = DiskElementMetaData(label);
    }

    // Update element count
    curElementCount++;

    return 1; // New insertion
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
    return this->curElementCount;
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::indexCapacity() const {
    return idToMetaData.size();
}

template <typename DataType, typename DistType>
size_t HNSWDiskIndex<DataType, DistType>::indexLabelCount() const {
    return this->curElementCount;
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
