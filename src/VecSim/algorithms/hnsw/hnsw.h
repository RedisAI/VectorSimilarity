/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "visited_nodes_handler.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/utils/data_block.h"
#include "VecSim/utils/vecsim_results_container.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/tombstone_interface.h"

#ifdef BUILD_TESTS
#include "hnsw_serialization_utils.h"
#include "VecSim/utils/serializer.h"
#endif

#include <deque>
#include <memory>
#include <cassert>
#include <climits>
#include <queue>
#include <random>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <sys/resource.h>
#include <fstream>
#include <shared_mutex>

using std::pair;

typedef uint16_t linkListSize;
typedef uint8_t elementFlags;

template <typename DistType>
using candidatesMaxHeap = vecsim_stl::max_priority_queue<DistType, idType>;
template <typename DistType>
using candidatesLabelsMaxHeap = vecsim_stl::abstract_priority_queue<DistType, labelType>;
using graphNodeType = pair<idType, ushort>; // represented as: (element_id, level)

////////////////////////////////////// Auxiliary HNSW structs //////////////////////////////////////

// Vectors flags (for marking a specific vector)
typedef enum {
    DELETE_MARK = 0x1, // element is logically deleted, but still exists in the graph
    IN_PROCESS = 0x2,  // element is being inserted into the graph
} Flags;

// The state of the index and the newly inserted vector to be passed into addVector API in case that
// the index global data structures are updated atomically from an external scope (such as in
// tiered index),
// TODO: this might need to be generalized for future usages of async indexing.
struct AddVectorCtx {
    idType newElementId;
    int elementMaxLevel;
    idType currEntryPoint;
    int currMaxLevel;
};

#pragma pack(1)
struct ElementMetaData {
    labelType label;
    elementFlags flags;

    ElementMetaData(labelType label = SIZE_MAX) noexcept : label(label), flags(IN_PROCESS) {}
};
#pragma pack() // restore default packing

struct LevelData {
    vecsim_stl::vector<idType> *incomingEdges;
    linkListSize numLinks;
    // Flexible array member - https://en.wikipedia.org/wiki/Flexible_array_member
    // Using this trick, we can have the links list as part of the LevelData struct, and avoid
    // the need to dereference a pointer to get to the links list.
    // We have to calculate the size of the struct manually, as `sizeof(LevelData)` will not include
    // this member. We do so in the constructor of the index, under the name `levelDataSize` (and
    // `elementGraphDataSize`). Notice that this member must be the last member of the struct and
    // all nesting structs.
    idType links[];

    LevelData(std::shared_ptr<VecSimAllocator> allocator)
        : incomingEdges(new (allocator) vecsim_stl::vector<idType>(allocator)), numLinks(0) {}
};

struct ElementGraphData {
    size_t toplevel;
    std::mutex neighborsGuard;
    LevelData *others;
    LevelData level0;

    ElementGraphData(size_t maxLevel, size_t high_level_size,
                     std::shared_ptr<VecSimAllocator> allocator)
        : toplevel(maxLevel), others(nullptr), level0(allocator) {
        if (toplevel > 0) {
            others = (LevelData *)allocator->callocate(high_level_size * toplevel);
            if (others == nullptr) {
                throw std::runtime_error("VecSim index low memory error");
            }
            for (size_t i = 0; i < maxLevel; i++) {
                new ((char *)others + i * high_level_size) LevelData(allocator);
            }
        }
    }
    ~ElementGraphData() = delete; // Should be destroyed using `destroyGraphData`
};

//////////////////////////////////// HNSW index implementation ////////////////////////////////////

template <typename DataType, typename DistType>
class HNSWIndex : public VecSimIndexAbstract<DistType>,
                  public VecSimIndexTombstone
#ifdef BUILD_TESTS
    ,
                  public Serializer
#endif
{
protected:
    // Index build parameters
    size_t maxElements;
    size_t M;
    size_t M0;
    size_t efConstruction;

    // Index search parameter
    size_t ef;
    double epsilon;

    // Index meta-data (based on the data dimensionality and index parameters)
    size_t elementGraphDataSize;
    size_t levelDataSize;
    double mult;

    // Index level generator of the top level for a new element
    std::default_random_engine levelGenerator;

    // Index global state - these should be guarded by the indexDataGuard lock in
    // multithreaded scenario.
    size_t curElementCount;
    idType entrypointNode;
    size_t maxLevel; // this is the top level of the entry point's element

    // Index data
    vecsim_stl::vector<DataBlock> vectorBlocks;
    vecsim_stl::vector<DataBlock> graphDataBlocks;
    vecsim_stl::vector<ElementMetaData> idToMetaData;

    // Used for marking the visited nodes in graph scans (the pool supports parallel graph scans).
    // This is mutable since the object changes upon search operations as well (which are const).
    mutable VisitedNodesHandlerPool visitedNodesHandlerPool;

    mutable std::shared_mutex indexDataGuard;

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_base_tests_friends.h"

#include "hnsw_serializer_declarations.h"
#endif

protected:
    HNSWIndex() = delete;                  // default constructor is disabled.
    HNSWIndex(const HNSWIndex &) = delete; // default (shallow) copy constructor is disabled.
    inline size_t getRandomLevel(double reverse_size);
    inline void removeExtraLinks(candidatesMaxHeap<DistType> candidates, size_t Mcurmax,
                                 LevelData &node_level, const vecsim_stl::vector<bool> &bitmap,
                                 idType *removed_links, size_t *removed_links_num);
    template <bool has_marked_deleted, typename Identifier> // Either idType or labelType
    inline void
    processCandidate(idType curNodeId, const void *data_point, size_t layer, size_t ef,
                     tag_t *elements_tags, tag_t visited_tag,
                     vecsim_stl::abstract_priority_queue<DistType, Identifier> &top_candidates,
                     candidatesMaxHeap<DistType> &candidates_set, DistType &lowerBound) const;
    template <bool has_marked_deleted>
    inline void processCandidate_RangeSearch(
        idType curNodeId, const void *data_point, size_t layer, double epsilon,
        tag_t *elements_tags, tag_t visited_tag,
        std::unique_ptr<vecsim_stl::abstract_results_container> &top_candidates,
        candidatesMaxHeap<DistType> &candidate_set, DistType lowerBound, DistType radius) const;
    template <bool has_marked_deleted>
    candidatesMaxHeap<DistType> searchLayer(idType ep_id, const void *data_point, size_t layer,
                                            size_t ef) const;
    template <bool has_marked_deleted>
    candidatesLabelsMaxHeap<DistType> *
    searchBottomLayer_WithTimeout(idType ep_id, const void *data_point, size_t ef, size_t k,
                                  void *timeoutCtx, VecSimQueryResult_Code *rc) const;
    template <bool has_marked_deleted>
    VecSimQueryResult *searchRangeBottomLayer_WithTimeout(idType ep_id, const void *data_point,
                                                          double epsilon, DistType radius,
                                                          void *timeoutCtx,
                                                          VecSimQueryResult_Code *rc) const;
    void getNeighborsByHeuristic2(candidatesMaxHeap<DistType> &top_candidates, size_t M);
    // Helper function for re-selecting node's neighbors which was selected as a neighbor for
    // a newly inserted node. Also, responsible for mutually connect the new node and the neighbor
    // (unidirectional or bidirectional connection).
    // *Note that node_lock and neighbor_lock should be locked upon calling this function*
    void revisitNeighborConnections(size_t level, idType new_node_id,
                                    const std::pair<DistType, idType> &neighbor_data,
                                    LevelData &new_node_level, LevelData &neighbor_level);
    inline idType mutuallyConnectNewElement(idType new_node_id,
                                            candidatesMaxHeap<DistType> &top_candidates,
                                            size_t level);
    void mutuallyUpdateForRepairedNode(idType node_id, size_t level,
                                       vecsim_stl::vector<idType> &neighbors_to_remove,
                                       vecsim_stl::vector<idType> &nodes_to_update,
                                       vecsim_stl::vector<idType> &chosen_neighbors,
                                       size_t max_M_cur);

    template <bool running_query>
    void greedySearchLevel(const void *vector_data, size_t level, idType &curObj, DistType &curDist,
                           void *timeoutCtx = nullptr, VecSimQueryResult_Code *rc = nullptr) const;
    void repairConnectionsForDeletion(idType element_internal_id, idType neighbour_id,
                                      LevelData &node_level, LevelData &neighbor_level,
                                      size_t level, vecsim_stl::vector<bool> &neighbours_bitmap);
    inline void destroyGraphData(ElementGraphData *em);
    inline void replaceEntryPoint();

    template <bool has_marked_deleted>
    inline void SwapLastIdWithDeletedId(idType element_internal_id, ElementGraphData *last_element,
                                        void *last_element_data);

    // Protected internal function that implements generic single vector insertion.
    void appendVector(const void *vector_data, labelType label,
                      AddVectorCtx *auxiliaryCtx = nullptr);

    // Protected internal functions for index resizing.
    inline void growByBlock();
    inline void shrinkByBlock();
    // DO NOT USE DIRECTLY. Use `[grow|shrink]ByBlock` instead.
    inline void resizeIndexCommon(size_t new_max_elements);

    // Protected internal function that implements generic single vector deletion.
    void removeVectorInPlace(idType id);

    inline void emplaceToHeap(vecsim_stl::abstract_priority_queue<DistType, idType> &heap,
                              DistType dist, idType id) const;
    inline void emplaceToHeap(vecsim_stl::abstract_priority_queue<DistType, labelType> &heap,
                              DistType dist, idType id) const;
    // Helper method that swaps the last element in the ids list with the given one (equivalent to
    // removing the given element id from the list).
    inline bool removeIdFromList(vecsim_stl::vector<idType> &element_ids_list, idType element_id);

    template <bool has_marked_deleted>
    void removeAndSwap(idType internalId);

    inline size_t getVectorRelativeIndex(idType id) const { return id % this->blockSize; }

    // Flagging API
    template <Flags FLAG>
    inline void markAs(idType internalId) {
        __atomic_fetch_or(&idToMetaData[internalId].flags, FLAG, 0);
    }
    template <Flags FLAG>
    inline void unmarkAs(idType internalId) {
        __atomic_fetch_and(&idToMetaData[internalId].flags, ~FLAG, 0);
    }
    template <Flags FLAG>
    inline bool isMarkedAs(idType internalId) const {
        return idToMetaData[internalId].flags & FLAG;
    }

public:
    HNSWIndex(const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
              size_t random_seed = 100, size_t initial_pool_size = 1);
    virtual ~HNSWIndex();

    inline void setEf(size_t ef);
    inline size_t getEf() const;
    inline void setEpsilon(double epsilon);
    inline double getEpsilon() const;
    inline size_t indexSize() const override;
    inline size_t indexCapacity() const override;
    inline size_t getEfConstruction() const;
    inline size_t getM() const;
    inline size_t getMaxLevel() const;
    inline labelType getEntryPointLabel() const;
    inline labelType getExternalLabel(idType internal_id) const {
        return idToMetaData[internal_id].label;
    }
    // Check if the given label exists in the labels lookup while holding the index data lock.
    // Optionally validate that the associated vector(s) are not in process and done indexing
    // (this option is used currently for tests).
    virtual inline bool safeCheckIfLabelExistsInIndex(labelType label,
                                                      bool also_done_processing = false) const = 0;
    inline auto safeGetEntryPointState() const;
    inline void lockIndexDataGuard() const;
    inline void unlockIndexDataGuard() const;
    inline void lockNodeLinks(idType node_id) const;
    inline void unlockNodeLinks(idType node_id) const;
    inline void lockNodeLinks(ElementGraphData *node_data) const;
    inline void unlockNodeLinks(ElementGraphData *node_data) const;
    inline VisitedNodesHandler *getVisitedList() const;
    inline void returnVisitedList(VisitedNodesHandler *visited_nodes_handler) const;
    VecSimIndexInfo info() const override;
    VecSimIndexBasicInfo basicInfo() const override;
    VecSimInfoIterator *infoIterator() const override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override;
    inline const char *getDataByInternalId(idType internal_id) const;
    inline ElementGraphData *getGraphDataByInternalId(idType internal_id) const;
    inline LevelData &getLevelData(idType internal_id, size_t level) const;
    inline LevelData &getLevelData(ElementGraphData *element, size_t level) const;
    inline idType searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                      VecSimQueryResult_Code *rc) const;

    VecSimQueryResult_List topKQuery(const void *query_data, size_t k,
                                     VecSimQueryParams *queryParams) const override;
    VecSimQueryResult_List rangeQuery(const void *query_data, double radius,
                                      VecSimQueryParams *queryParams) const override;

    inline void markDeletedInternal(idType internalId);
    inline bool isMarkedDeleted(idType internalId) const;
    inline bool isInProcess(idType internalId) const;
    inline void unmarkInProcess(idType internalId);
    AddVectorCtx storeNewElement(labelType label, const void *vector_data);
    void removeAndSwapDeletedElement(idType internalId);
    void repairNodeConnections(idType node_id, size_t level);
    // For prefetching only.
    inline const ElementMetaData *getMetaDataAddress(idType internal_id) const {
        return idToMetaData.data() + internal_id;
    }
    vecsim_stl::vector<graphNodeType> safeCollectAllNodeIncomingNeighbors(idType node_id) const;
    // Return all the labels in the index - this should be used for computing the number of distinct
    // labels in a tiered index, and caller should hold the index data guard.
    virtual inline vecsim_stl::set<labelType> getLabelsSet() const = 0;

    // Inline priority queue getter that need to be implemented by derived class.
    virtual inline candidatesLabelsMaxHeap<DistType> *getNewMaxPriorityQueue() const = 0;
    virtual double safeGetDistanceFrom(labelType label, const void *vector_data) const = 0;

#ifdef BUILD_TESTS
    /**
     * @brief Used for testing - store vector(s) data associated with a given label. This function
     * copies the vector(s)' data buffer(s) and place it in the output vector
     *
     * @param label
     * @param vectors_output empty vector to be modified, should store the blob(s) associated with
     * the label.
     */
    virtual void getDataByLabel(labelType label,
                                std::vector<std::vector<DataType>> &vectors_output) const = 0;
#endif

protected:
    // inline label to id setters that need to be implemented by derived class
    virtual inline std::unique_ptr<vecsim_stl::abstract_results_container>
    getNewResultsContainer(size_t cap) const = 0;
    virtual inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) = 0;
    virtual inline void setVectorId(labelType label, idType id) = 0;
    virtual inline void resizeLabelLookup(size_t new_max_elements) = 0;
};

/**
 * getters and setters of index data
 */

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::setEf(size_t ef) {
    this->ef = ef;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getEf() const {
    return this->ef;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::setEpsilon(double epsilon) {
    this->epsilon = epsilon;
}

template <typename DataType, typename DistType>
double HNSWIndex<DataType, DistType>::getEpsilon() const {
    return this->epsilon;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::indexSize() const {
    return this->curElementCount;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::indexCapacity() const {
    return this->maxElements;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getEfConstruction() const {
    return this->efConstruction;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getM() const {
    return this->M;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getMaxLevel() const {
    return this->maxLevel;
}

template <typename DataType, typename DistType>
labelType HNSWIndex<DataType, DistType>::getEntryPointLabel() const {
    if (entrypointNode != INVALID_ID)
        return getExternalLabel(entrypointNode);
    return SIZE_MAX;
}

template <typename DataType, typename DistType>
const char *HNSWIndex<DataType, DistType>::getDataByInternalId(idType internal_id) const {
    return vectorBlocks[internal_id / this->blockSize].getElement(internal_id % this->blockSize);
}

template <typename DataType, typename DistType>
ElementGraphData *
HNSWIndex<DataType, DistType>::getGraphDataByInternalId(idType internal_id) const {
    return (ElementGraphData *)graphDataBlocks[internal_id / this->blockSize].getElement(
        internal_id % this->blockSize);
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(levelGenerator)) * reverse_size;
    return (size_t)r;
}

template <typename DataType, typename DistType>
LevelData &HNSWIndex<DataType, DistType>::getLevelData(idType internal_id, size_t level) const {
    return getLevelData(getGraphDataByInternalId(internal_id), level);
}

template <typename DataType, typename DistType>
LevelData &HNSWIndex<DataType, DistType>::getLevelData(ElementGraphData *elem, size_t level) const {
    assert(level <= elem->toplevel);
    if (level == 0) {
        return elem->level0;
    } else {
        return *(LevelData *)((char *)elem->others + (level - 1) * this->levelDataSize);
    }
}

template <typename DataType, typename DistType>
VisitedNodesHandler *HNSWIndex<DataType, DistType>::getVisitedList() const {
    return visitedNodesHandlerPool.getAvailableVisitedNodesHandler();
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::returnVisitedList(
    VisitedNodesHandler *visited_nodes_handler) const {
    visitedNodesHandlerPool.returnVisitedNodesHandlerToPool(visited_nodes_handler);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::markDeletedInternal(idType internalId) {
    // Here we are holding the global index data guard (and the main index lock of the tiered index
    // for shared ownership).
    assert(internalId < this->curElementCount);
    if (!isMarkedDeleted(internalId)) {
        if (internalId == entrypointNode) {
            // Internally, we hold and release the entrypoint neighbors lock.
            replaceEntryPoint();
        }
        // Atomically set the deletion mark flag (note that other parallel threads may set the flags
        // at the same time (for changing the IN_PROCESS flag).
        markAs<DELETE_MARK>(internalId);
        this->numMarkedDeleted++;
    }
}

template <typename DataType, typename DistType>
bool HNSWIndex<DataType, DistType>::isMarkedDeleted(idType internalId) const {
    return isMarkedAs<DELETE_MARK>(internalId);
}

template <typename DataType, typename DistType>
bool HNSWIndex<DataType, DistType>::isInProcess(idType internalId) const {
    return isMarkedAs<IN_PROCESS>(internalId);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::unmarkInProcess(idType internalId) {
    // Atomically unset the IN_PROCESS mark flag (note that other parallel threads may set the flags
    // at the same time (for marking the element with MARK_DELETE flag).
    unmarkAs<IN_PROCESS>(internalId);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::lockIndexDataGuard() const {
    indexDataGuard.lock();
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::unlockIndexDataGuard() const {
    indexDataGuard.unlock();
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::lockNodeLinks(ElementGraphData *node_data) const {
    node_data->neighborsGuard.lock();
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::unlockNodeLinks(ElementGraphData *node_data) const {
    node_data->neighborsGuard.unlock();
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::lockNodeLinks(idType node_id) const {
    lockNodeLinks(getGraphDataByInternalId(node_id));
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::unlockNodeLinks(idType node_id) const {
    unlockNodeLinks(getGraphDataByInternalId(node_id));
}

/**
 * helper functions
 */
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::removeExtraLinks(
    candidatesMaxHeap<DistType> candidates, size_t Mcurmax, LevelData &node_level,
    const vecsim_stl::vector<bool> &neighbors_bitmap, idType *removed_links,
    size_t *removed_links_num) {

    auto orig_candidates = candidates;
    // candidates will store the newly selected neighbours (for the relevant node).
    getNeighborsByHeuristic2(candidates, Mcurmax);

    // check the diff in the link list, save the neighbours
    // that were chosen to be removed, and update the new neighbours
    size_t removed_idx = 0;
    size_t link_idx = 0;

    while (orig_candidates.size() > 0) {
        if (orig_candidates.top().second != candidates.top().second) {
            if (neighbors_bitmap[orig_candidates.top().second]) {
                removed_links[removed_idx++] = orig_candidates.top().second;
            }
            orig_candidates.pop();
        } else {
            node_level.links[link_idx++] = candidates.top().second;
            candidates.pop();
            orig_candidates.pop();
        }
    }
    node_level.numLinks = link_idx;
    *removed_links_num = removed_idx;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::emplaceToHeap(
    vecsim_stl::abstract_priority_queue<DistType, idType> &heap, DistType dist, idType id) const {
    heap.emplace(dist, id);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::emplaceToHeap(
    vecsim_stl::abstract_priority_queue<DistType, labelType> &heap, DistType dist,
    idType id) const {
    heap.emplace(dist, getExternalLabel(id));
}

// This function handles both label heaps and internal ids heaps. It uses the `emplaceToHeap`
// overloading to emplace correctly for both cases.
template <typename DataType, typename DistType>
template <bool has_marked_deleted, typename Identifier>
void HNSWIndex<DataType, DistType>::processCandidate(
    idType curNodeId, const void *query_data, size_t layer, size_t ef, tag_t *elements_tags,
    tag_t visited_tag, vecsim_stl::abstract_priority_queue<DistType, Identifier> &top_candidates,
    candidatesMaxHeap<DistType> &candidate_set, DistType &lowerBound) const {

    ElementGraphData *cur_element = getGraphDataByInternalId(curNodeId);
    lockNodeLinks(cur_element);
    LevelData &node_level = getLevelData(cur_element, layer);

    if (node_level.numLinks > 0) {

        const char *cur_data, *next_data;
        // Pre-fetch first candidate tag address.
        __builtin_prefetch(elements_tags + node_level.links[0]);
        // Pre-fetch first candidate data block address.
        next_data = getDataByInternalId(node_level.links[0]);
        __builtin_prefetch(next_data);

        for (linkListSize j = 0; j < node_level.numLinks - 1; j++) {
            idType candidate_id = node_level.links[j];
            cur_data = next_data;

            // Pre-fetch next candidate tag address.
            __builtin_prefetch(elements_tags + node_level.links[j + 1]);
            // Pre-fetch next candidate data block address.
            next_data = getDataByInternalId(node_level.links[j + 1]);
            __builtin_prefetch(next_data);

            if (elements_tags[candidate_id] == visited_tag || isInProcess(candidate_id))
                continue;

            elements_tags[candidate_id] = visited_tag;

            DistType cur_dist = this->distFunc(query_data, cur_data, this->dim);
            if (lowerBound > cur_dist || top_candidates.size() < ef) {

                candidate_set.emplace(-cur_dist, candidate_id);

                // Insert the candidate to the top candidates heap only if it is not marked as
                // deleted.
                if (!has_marked_deleted || !isMarkedDeleted(candidate_id))
                    emplaceToHeap(top_candidates, cur_dist, candidate_id);

                if (top_candidates.size() > ef)
                    top_candidates.pop();

                // If we have marked deleted elements, we need to verify that `top_candidates` is
                // not empty (since we might have not added any non-deleted element yet).
                if (!has_marked_deleted || !top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }

        // Running the last neighbor outside the loop to avoid prefetching invalid neighbor
        idType candidate_id = node_level.links[node_level.numLinks - 1];
        cur_data = next_data;

        if (elements_tags[candidate_id] != visited_tag && !isInProcess(candidate_id)) {

            elements_tags[candidate_id] = visited_tag;

            DistType cur_dist = this->distFunc(query_data, cur_data, this->dim);
            if (lowerBound > cur_dist || top_candidates.size() < ef) {
                candidate_set.emplace(-cur_dist, candidate_id);

                // Insert the candidate to the top candidates heap only if it is not marked as
                // deleted.
                if (!has_marked_deleted || !isMarkedDeleted(candidate_id))
                    emplaceToHeap(top_candidates, cur_dist, candidate_id);

                if (top_candidates.size() > ef)
                    top_candidates.pop();

                // If we have marked deleted elements, we need to verify that `top_candidates` is
                // not empty (since we might have not added any non-deleted element yet).
                if (!has_marked_deleted || !top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }
    unlockNodeLinks(cur_element);
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
void HNSWIndex<DataType, DistType>::processCandidate_RangeSearch(
    idType curNodeId, const void *query_data, size_t layer, double epsilon, tag_t *elements_tags,
    tag_t visited_tag, std::unique_ptr<vecsim_stl::abstract_results_container> &results,
    candidatesMaxHeap<DistType> &candidate_set, DistType dyn_range, DistType radius) const {

    auto *cur_element = getGraphDataByInternalId(curNodeId);
    lockNodeLinks(cur_element);
    LevelData &node_level = getLevelData(cur_element, layer);
    if (node_level.numLinks > 0) {

        const char *cur_data, *next_data;
        // Pre-fetch first candidate tag address.
        __builtin_prefetch(elements_tags + node_level.links[0]);
        // Pre-fetch first candidate data block address.
        next_data = getDataByInternalId(node_level.links[0]);
        __builtin_prefetch(next_data);

        for (linkListSize j = 0; j < node_level.numLinks - 1; j++) {
            idType candidate_id = node_level.links[j];
            cur_data = next_data;

            // Pre-fetch next candidate tag address.
            __builtin_prefetch(elements_tags + node_level.links[j + 1]);
            // Pre-fetch next candidate data block address.
            next_data = getDataByInternalId(node_level.links[j + 1]);
            __builtin_prefetch(next_data);

            if (elements_tags[candidate_id] == visited_tag || isInProcess(candidate_id))
                continue;

            elements_tags[candidate_id] = visited_tag;

            DistType cur_dist = this->distFunc(query_data, cur_data, this->dim);
            if (cur_dist < dyn_range) {
                candidate_set.emplace(-cur_dist, candidate_id);

                // If the new candidate is in the requested radius, add it to the results set.
                if (cur_dist <= radius && (!has_marked_deleted || !isMarkedDeleted(candidate_id))) {
                    results->emplace(getExternalLabel(candidate_id), cur_dist);
                }
            }
        }
        // Running the last candidate outside the loop to avoid prefetching invalid candidate
        idType candidate_id = node_level.links[node_level.numLinks - 1];
        cur_data = next_data;

        if (elements_tags[candidate_id] != visited_tag && !isInProcess(candidate_id)) {

            elements_tags[candidate_id] = visited_tag;

            DistType cur_dist = this->distFunc(query_data, cur_data, this->dim);
            if (cur_dist < dyn_range) {
                candidate_set.emplace(-cur_dist, candidate_id);

                // If the new candidate is in the requested radius, add it to the results set.
                if (cur_dist <= radius && (!has_marked_deleted || !isMarkedDeleted(candidate_id))) {
                    results->emplace(getExternalLabel(candidate_id), cur_dist);
                }
            }
        }
    }
    unlockNodeLinks(cur_element);
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
candidatesMaxHeap<DistType>
HNSWIndex<DataType, DistType>::searchLayer(idType ep_id, const void *data_point, size_t layer,
                                           size_t ef) const {

    auto *visited_nodes_handler = getVisitedList();
    tag_t visited_tag = visited_nodes_handler->getFreshTag();

    candidatesMaxHeap<DistType> top_candidates(this->allocator);
    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    DistType lowerBound;
    if (!has_marked_deleted || !isMarkedDeleted(ep_id)) {
        DistType dist = this->distFunc(data_point, getDataByInternalId(ep_id), this->dim);
        lowerBound = dist;
        top_candidates.emplace(dist, ep_id);
        candidate_set.emplace(-dist, ep_id);
    } else {
        lowerBound = std::numeric_limits<DistType>::max();
        candidate_set.emplace(-lowerBound, ep_id);
    }

    visited_nodes_handler->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();

        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() >= ef) {
            break;
        }
        candidate_set.pop();

        processCandidate<has_marked_deleted>(curr_el_pair.second, data_point, layer, ef,
                                             visited_nodes_handler->getElementsTags(), visited_tag,
                                             top_candidates, candidate_set, lowerBound);
    }

    returnVisitedList(visited_nodes_handler);
    return top_candidates;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::getNeighborsByHeuristic2(
    candidatesMaxHeap<DistType> &top_candidates, const size_t M) {
    if (top_candidates.size() < M) {
        return;
    }

    candidatesMaxHeap<DistType> queue_closest(this->allocator);
    vecsim_stl::vector<pair<DistType, idType>> return_list(this->allocator);
    vecsim_stl::vector<const void *> cached_vectors(this->allocator);
    return_list.reserve(M);
    cached_vectors.reserve(M);
    while (top_candidates.size() > 0) {
        // the distance is saved negatively to have the queue ordered such that first is closer
        // (higher).
        queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
        top_candidates.pop();
    }

    while (queue_closest.size() && return_list.size() < M) {
        pair<DistType, idType> current_pair = queue_closest.top();
        DistType candidate_to_query_dist = -current_pair.first;
        queue_closest.pop();
        bool good = true;
        const void *curr_vector = getDataByInternalId(current_pair.second);

        // a candidate is "good" to become a neighbour, unless we find
        // another item that was already selected to the neighbours set which is closer
        // to both q and the candidate than the distance between the candidate and q.
        for (size_t i = 0; i < return_list.size(); i++) {
            DistType candidate_to_selected_dist =
                this->distFunc(cached_vectors[i], curr_vector, this->dim);
            if (candidate_to_selected_dist < candidate_to_query_dist) {
                good = false;
                break;
            }
        }
        if (good) {
            cached_vectors.push_back(curr_vector);
            return_list.push_back(current_pair);
        }
    }

    for (pair<DistType, idType> current_pair : return_list) {
        top_candidates.emplace(-current_pair.first, current_pair.second);
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::revisitNeighborConnections(
    size_t level, idType new_node_id, const std::pair<DistType, idType> &neighbor_data,
    LevelData &new_node_level, LevelData &neighbor_level) {
    // Note - expect that node_lock and neighbor_lock are locked at that point.

    // Collect the existing neighbors and the new node as the neighbor's neighbors candidates.
    candidatesMaxHeap<DistType> candidates(this->allocator);
    // Add the new node along with the pre-calculated distance to the current neighbor,
    candidates.emplace(neighbor_data.first, new_node_id);

    idType selected_neighbor = neighbor_data.second;
    const void *selected_neighbor_data = getDataByInternalId(selected_neighbor);
    for (size_t j = 0; j < neighbor_level.numLinks; j++) {
        candidates.emplace(this->distFunc(getDataByInternalId(neighbor_level.links[j]),
                                          selected_neighbor_data, this->dim),
                           neighbor_level.links[j]);
    }

    std::vector<idType> nodes_to_update;
    auto orig_candidates = candidates;

    // Candidates will store the newly selected neighbours (for the neighbor).
    size_t max_M_cur = level ? M : M0;
    getNeighborsByHeuristic2(candidates, max_M_cur);

    // Go over the original candidates set, and save the ones chosen to be removed to update later
    // on.
    bool cur_node_chosen = false;
    while (orig_candidates.size() > 0) {
        idType orig_candidate = orig_candidates.top().second;
        // If the current original candidate was not selected as neighbor by the heuristics, it
        // should be updated and removed from the neighbor's neighbors.
        if (candidates.empty() || orig_candidate != candidates.top().second) {
            // Don't add the new_node_id to nodes_to_update, it will be inserted either way later.
            if (orig_candidate != new_node_id) {
                nodes_to_update.push_back(orig_candidate);
            }
            orig_candidates.pop();
            // Otherwise, the original candidate was selected to remain a neighbor - no need to
            // update.
        } else {
            candidates.pop();
            orig_candidates.pop();
            if (orig_candidate == new_node_id) {
                cur_node_chosen = true;
            }
        }
    }

    // Acquire all relevant locks for making the updates for the selected neighbor - all its removed
    // neighbors, along with the neighbors itself and the cur node.
    // but first, we release the node and neighbors lock to avoid deadlocks.
    unlockNodeLinks(new_node_id);
    unlockNodeLinks(selected_neighbor);

    nodes_to_update.push_back(selected_neighbor);
    nodes_to_update.push_back(new_node_id);

    std::sort(nodes_to_update.begin(), nodes_to_update.end());
    size_t nodes_to_update_count = nodes_to_update.size();
    for (size_t i = 0; i < nodes_to_update_count; i++) {
        lockNodeLinks(nodes_to_update[i]);
    }
    size_t neighbour_neighbours_idx = 0;
    bool update_cur_node_required = true;
    for (size_t i = 0; i < neighbor_level.numLinks; i++) {
        if (!std::binary_search(nodes_to_update.begin(), nodes_to_update.end(),
                                neighbor_level.links[i])) {
            // The neighbor is not in the "to_update" nodes list - leave it as is.
            neighbor_level.links[neighbour_neighbours_idx++] = neighbor_level.links[i];
            continue;
        } else if (neighbor_level.links[i] == new_node_id) {
            // The new node got into the neighbor's neighbours - this means there was an update in
            // another thread during between we released and reacquire the locks - leave it
            // as is.
            neighbor_level.links[neighbour_neighbours_idx++] = neighbor_level.links[i];
            update_cur_node_required = false;
            continue;
        }
        // Now we know that we are looking at a node to be removed from the neighbor's neighbors.
        auto removed_node = neighbor_level.links[i];
        LevelData &removed_node_level = getLevelData(removed_node, level);
        // Perform the mutual update:
        // if the removed node id (the neighbour's neighbour to be removed)
        // wasn't pointing to the neighbour (i.e., the edge was uni-directional),
        // we should remove the current neighbor from the node's incoming edges.
        // otherwise, the edge turned from bidirectional to uni-directional, so we insert it to the
        // neighbour's incoming edges set. Note: we assume that every update is performed atomically
        // mutually, so it should be sufficient to look at the removed node's incoming edges set
        // alone.
        if (!removeIdFromList(*removed_node_level.incomingEdges, selected_neighbor)) {
            neighbor_level.incomingEdges->push_back(removed_node);
        }
    }

    if (update_cur_node_required && new_node_level.numLinks < max_M_cur &&
        !isMarkedDeleted(new_node_id) && !isMarkedDeleted(selected_neighbor)) {
        // update the connection between the new node and the neighbor.
        new_node_level.links[new_node_level.numLinks++] = selected_neighbor;
        if (cur_node_chosen && neighbour_neighbours_idx < max_M_cur) {
            // connection is mutual - both new node and the selected neighbor in each other's list.
            neighbor_level.links[neighbour_neighbours_idx++] = new_node_id;
        } else {
            // unidirectional connection - put the new node in the neighbour's incoming edges.
            neighbor_level.incomingEdges->push_back(new_node_id);
        }
    }
    // Done updating the neighbor's neighbors.
    neighbor_level.numLinks = neighbour_neighbours_idx;
    for (size_t i = 0; i < nodes_to_update_count; i++) {
        unlockNodeLinks(nodes_to_update[i]);
    }
}

template <typename DataType, typename DistType>
idType HNSWIndex<DataType, DistType>::mutuallyConnectNewElement(
    idType new_node_id, candidatesMaxHeap<DistType> &top_candidates, size_t level) {

    // The maximum number of neighbors allowed for an existing neighbor (not new).
    size_t max_M_cur = level ? M : M0;

    // Filter the top candidates to the selected neighbors by the algorithm heuristics.
    getNeighborsByHeuristic2(top_candidates, M);
    assert(top_candidates.size() <= M &&
           "Should be not be more than M candidates returned by the heuristic");

    // Hold (distance_from_new_node_id, neighbor_id) pair for every selected neighbor.
    vecsim_stl::vector<std::pair<DistType, idType>> selected_neighbors(this->allocator);
    selected_neighbors.reserve(M);
    while (!top_candidates.empty()) {
        selected_neighbors.push_back(top_candidates.top());
        top_candidates.pop();
    }

    // The closest vector that has found to be returned (and start the scan from it in the next
    // level).
    idType next_closest_entry_point = selected_neighbors.back().second;
    auto *new_node_level = getGraphDataByInternalId(new_node_id);
    LevelData &new_node_level_data = getLevelData(new_node_level, level);
    assert(new_node_level_data.numLinks == 0 &&
           "The newly inserted element should have blank link list");

    for (auto &neighbor_data : selected_neighbors) {
        idType selected_neighbor = neighbor_data.second; // neighbor's id
        auto *neighbor_graph_data = getGraphDataByInternalId(selected_neighbor);
        if (new_node_id < selected_neighbor) {
            lockNodeLinks(new_node_level);
            lockNodeLinks(neighbor_graph_data);
        } else {
            lockNodeLinks(neighbor_graph_data);
            lockNodeLinks(new_node_level);
        }

        // validations...
        assert(new_node_level_data.numLinks <= max_M_cur && "Neighbors number exceeds limit");
        assert(selected_neighbor != new_node_id && "Trying to connect an element to itself");

        // Revalidate the updated count - this may change between iterations due to releasing the
        // lock.
        if (new_node_level_data.numLinks == max_M_cur) {
            // The new node cannot add more neighbors
            this->log("Couldn't add all chosen neighbors upon inserting a new node");
            unlockNodeLinks(new_node_level);
            unlockNodeLinks(neighbor_graph_data);
            break;
        }

        // If one of the two nodes has already deleted - skip the operation.
        if (isMarkedDeleted(new_node_id) || isMarkedDeleted(selected_neighbor)) {
            unlockNodeLinks(new_node_level);
            unlockNodeLinks(neighbor_graph_data);
            continue;
        }

        LevelData &neighbor_level_data = getLevelData(neighbor_graph_data, level);

        // if the neighbor's neighbors list has the capacity to add the new node, make the update
        // and finish.
        if (neighbor_level_data.numLinks < max_M_cur) {
            new_node_level_data.links[new_node_level_data.numLinks++] = selected_neighbor;
            neighbor_level_data.links[neighbor_level_data.numLinks++] = new_node_id;
            unlockNodeLinks(new_node_level);
            unlockNodeLinks(neighbor_graph_data);
            continue;
        }

        // Otherwise - we need to re-evaluate the neighbor's neighbors.
        // We collect all the existing neighbors and the new node as candidates, and mutually update
        // the neighbor's neighbors. We also release the acquired locks inside this call.
        revisitNeighborConnections(level, new_node_id, neighbor_data, new_node_level_data,
                                   neighbor_level_data);
    }
    return next_closest_entry_point;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::repairConnectionsForDeletion(
    idType element_internal_id, idType neighbour_id, LevelData &node_level,
    LevelData &neighbor_level, size_t level, vecsim_stl::vector<bool> &neighbours_bitmap) {

    // put the deleted element's neighbours in the candidates.
    candidatesMaxHeap<DistType> candidates(this->allocator);
    auto neighbours_data = getDataByInternalId(neighbour_id);
    for (size_t j = 0; j < node_level.numLinks; j++) {
        // Don't put the neighbor itself in his own candidates
        if (node_level.links[j] == neighbour_id) {
            continue;
        }
        candidates.emplace(
            this->distFunc(getDataByInternalId(node_level.links[j]), neighbours_data, this->dim),
            node_level.links[j]);
    }

    // add the deleted element's neighbour's original neighbors in the candidates.
    vecsim_stl::vector<bool> neighbour_orig_neighbours_set(curElementCount, false, this->allocator);

    for (size_t j = 0; j < neighbor_level.numLinks; j++) {
        neighbour_orig_neighbours_set[neighbor_level.links[j]] = true;
        // Don't add the removed element to the candidates, nor nodes that are already in the
        // candidates set.
        if (neighbours_bitmap[neighbor_level.links[j]] ||
            neighbor_level.links[j] == element_internal_id) {
            continue;
        }
        candidates.emplace(this->distFunc(neighbours_data,
                                          getDataByInternalId(neighbor_level.links[j]), this->dim),
                           neighbor_level.links[j]);
    }

    size_t Mcurmax = level ? M : M0;
    size_t removed_links_num;
    idType removed_links[neighbor_level.numLinks];
    removeExtraLinks(candidates, Mcurmax, neighbor_level, neighbour_orig_neighbours_set,
                     removed_links, &removed_links_num);

    // remove neighbour id from the incoming list of nodes for his
    // neighbours that were chosen to remove
    for (size_t i = 0; i < removed_links_num; i++) {
        idType node_id = removed_links[i];
        LevelData &node_level = getLevelData(node_id, level);

        // if the node id (the neighbour's neighbour to be removed)
        // wasn't pointing to the neighbour (edge was one directional),
        // we should remove it from the node's incoming edges.
        // otherwise, edge turned from bidirectional to one directional,
        // and it should be saved in the neighbor's incoming edges.
        if (!removeIdFromList(*node_level.incomingEdges, neighbour_id)) {
            neighbor_level.incomingEdges->push_back(node_id);
        }
    }

    // updates for the new edges created
    for (size_t i = 0; i < neighbor_level.numLinks; i++) {
        idType node_id = neighbor_level.links[i];
        if (!neighbour_orig_neighbours_set[node_id]) {
            LevelData &node_level = getLevelData(node_id, level);
            // if the node has an edge to the neighbour as well, remove it
            // from the incoming nodes of the neighbour
            // otherwise, need to update the edge as incoming.

            bool bidirectional_edge = false;
            for (size_t j = 0; j < node_level.numLinks; j++) {
                if (node_level.links[j] == neighbour_id) {
                    // Swap the last element with the current one (equivalent to removing the
                    // neighbor from the list) - this should always succeed and return true.
                    removeIdFromList(*neighbor_level.incomingEdges, node_id);
                    bidirectional_edge = true;
                    break;
                }
            }
            if (!bidirectional_edge) {
                node_level.incomingEdges->push_back(neighbour_id);
            }
        }
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::replaceEntryPoint() {
    idType old_entry_point_id = entrypointNode;
    auto *old_entry_point = getGraphDataByInternalId(old_entry_point_id);

    // Sets an (arbitrary) new entry point, after deleting the current entry point.
    while (old_entry_point_id == entrypointNode) {
        // Use volatile for this variable, so that in case we would have to busy wait for this
        // element to finish its indexing, the compiler will not use optimizations. Otherwise,
        // the compiler might evaluate 'isInProcess(candidate_in_process)' once instead of calling
        // it multiple times in a busy wait manner, and we'll run into an infinite loop if the
        // candidate is in process when we reach the loop.
        volatile idType candidate_in_process = INVALID_ID;

        // Go over the entry point's neighbors at the top level.
        lockNodeLinks(old_entry_point);
        LevelData &old_ep_level = getLevelData(old_entry_point, maxLevel);
        // Tries to set the (arbitrary) first neighbor as the entry point which is not deleted,
        // if exists.
        for (size_t i = 0; i < old_ep_level.numLinks; i++) {
            if (!isMarkedDeleted(old_ep_level.links[i])) {
                if (!isInProcess(old_ep_level.links[i])) {
                    entrypointNode = old_ep_level.links[i];
                    unlockNodeLinks(old_entry_point);
                    return;
                } else {
                    // Store this candidate which is currently being inserted into the graph in
                    // case we won't find other candidate at the top level.
                    candidate_in_process = old_ep_level.links[i];
                }
            }
        }
        unlockNodeLinks(old_entry_point);

        // If there is no neighbors in the current level, check for any vector at
        // this level to be the new entry point.
        idType cur_id = 0;
        for (DataBlock &graph_data_block : graphDataBlocks) {
            size_t size = graph_data_block.getLength();
            for (size_t i = 0; i < size; i++) {
                auto cur_element = (ElementGraphData *)graph_data_block.getElement(i);
                if (cur_element->toplevel == maxLevel && cur_id != old_entry_point_id &&
                    !isMarkedDeleted(cur_id)) {
                    // Found a non element in the current max level.
                    if (!isInProcess(cur_id)) {
                        entrypointNode = cur_id;
                        return;
                    } else if (candidate_in_process == INVALID_ID) {
                        // This element is still in process, and there hasn't been another candidate
                        // in process that has found in this level.
                        candidate_in_process = cur_id;
                    }
                }
                cur_id++;
            }
        }
        // If we only found candidates which are in process at this level, do busy wait until they
        // are done being processed (this should happen in very rare cases...). Since
        // candidate_in_process was declared volatile, we can be sure that isInProcess is called in
        // every iteration.
        if (candidate_in_process != INVALID_ID) {
            while (isInProcess(candidate_in_process))
                ;
            entrypointNode = candidate_in_process;
            return;
        }
        // If we didn't find any vector at the top level, decrease the maxLevel and try again,
        // until we find a new entry point, or the index is empty.
        assert(old_entry_point_id == entrypointNode);
        maxLevel--;
        if ((int)maxLevel < 0) {
            maxLevel = HNSW_INVALID_LEVEL;
            entrypointNode = INVALID_ID;
        }
    }
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
void HNSWIndex<DataType, DistType>::SwapLastIdWithDeletedId(idType element_internal_id,
                                                            ElementGraphData *last_element,
                                                            void *last_element_data) {
    // Swap label - this is relevant when the last element's label exists (it is not marked as
    // deleted). For inplace delete, this is always the case.
    if (!has_marked_deleted || !isMarkedDeleted(curElementCount)) {
        replaceIdOfLabel(getExternalLabel(curElementCount), element_internal_id, curElementCount);
    }

    // Swap neighbours
    for (size_t level = 0; level <= last_element->toplevel; level++) {
        auto &cur_level = getLevelData(last_element, level);

        // Go over the neighbours that also points back to the last element whose is going to
        // change, and update the id.
        for (size_t i = 0; i < cur_level.numLinks; i++) {
            idType neighbour_id = cur_level.links[i];
            LevelData &neighbor_level = getLevelData(neighbour_id, level);

            bool bidirectional_edge = false;
            for (size_t j = 0; j < neighbor_level.numLinks; j++) {
                // if the edge is bidirectional, update for this neighbor
                if (neighbor_level.links[j] == curElementCount) {
                    bidirectional_edge = true;
                    neighbor_level.links[j] = element_internal_id;
                    break;
                }
            }

            // If this edge is uni-directional, we should update the id in the neighbor's
            // incoming edges.
            if (!bidirectional_edge) {
                auto it = std::find(neighbor_level.incomingEdges->begin(),
                                    neighbor_level.incomingEdges->end(), curElementCount);
                // This should always succeed
                assert(it != neighbor_level.incomingEdges->end());
                *it = element_internal_id;
            }
        }

        // Next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // updates.
        for (auto incoming_edge : *cur_level.incomingEdges) {
            LevelData &incoming_neighbor_level = getLevelData(incoming_edge, level);
            for (size_t j = 0; j < incoming_neighbor_level.numLinks; j++) {
                if (incoming_neighbor_level.links[j] == curElementCount) {
                    incoming_neighbor_level.links[j] = element_internal_id;
                    break;
                }
            }
        }
    }

    // Move the last element's data to the deleted element's place
    auto element = getGraphDataByInternalId(element_internal_id);
    memcpy((void *)element, last_element, this->elementGraphDataSize);

    auto data = getDataByInternalId(element_internal_id);
    memcpy((void *)data, last_element_data, this->dataSize);

    this->idToMetaData[element_internal_id] = this->idToMetaData[curElementCount];

    if (curElementCount == this->entrypointNode) {
        this->entrypointNode = element_internal_id;
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::destroyGraphData(ElementGraphData *egd) {
    delete egd->level0.incomingEdges;
    LevelData *cur_ld = egd->others;
    for (size_t i = 0; i < egd->toplevel; i++) {
        delete cur_ld->incomingEdges;
        cur_ld = (LevelData *)((char *)cur_ld + this->levelDataSize);
    }
    this->allocator->free_allocation(egd->others);
}

// This function is greedily searching for the closest candidate to the given data point at the
// given level, starting at the given node. It sets `curObj` to the closest node found, and
// `curDist` to the distance to this node. If `running_query` is true, the search will check for
// timeout and return if it has occurred. `timeoutCtx` and `rc` must be valid if `running_query` is
// true. *Note that we assume that level is higher than 0*. Also, if we're not running a query (we
// are searching neighbors for a new vector), then bestCand should be a non-deleted element!
template <typename DataType, typename DistType>
template <bool running_query>
void HNSWIndex<DataType, DistType>::greedySearchLevel(const void *vector_data, size_t level,
                                                      idType &bestCand, DistType &curDist,
                                                      void *timeoutCtx,
                                                      VecSimQueryResult_Code *rc) const {
    bool changed;
    // Don't allow choosing a deleted node as an entry point upon searching for neighbors
    // candidates (that is, we're NOT running a query, but inserting a new vector).
    idType bestNonDeletedCand = bestCand;

    do {
        if (running_query && VECSIM_TIMEOUT(timeoutCtx)) {
            *rc = VecSim_QueryResult_TimedOut;
            bestCand = INVALID_ID;
            return;
        }

        changed = false;
        auto *element = getGraphDataByInternalId(bestCand);
        lockNodeLinks(element);
        LevelData &node_level_data = getLevelData(element, level);

        for (int i = 0; i < node_level_data.numLinks; i++) {
            idType candidate = node_level_data.links[i];
            assert(candidate < this->curElementCount && "candidate error: out of index range");
            if (isInProcess(candidate)) {
                continue;
            }
            DistType d = this->distFunc(vector_data, getDataByInternalId(candidate), this->dim);
            if (d < curDist) {
                curDist = d;
                bestCand = candidate;
                changed = true;
                // Run this code only for non-query code - update the best non deleted cand as well.
                // Upon running a query, we don't mind having a deleted element as an entry point
                // for the next level, as eventually we return non-deleted elements in level 0.
                if (!running_query && !isMarkedDeleted(candidate)) {
                    bestNonDeletedCand = bestCand;
                }
            }
        }
        unlockNodeLinks(element);
    } while (changed);
    if (!running_query) {
        bestCand = bestNonDeletedCand;
    }
}

template <typename DataType, typename DistType>
vecsim_stl::vector<graphNodeType>
HNSWIndex<DataType, DistType>::safeCollectAllNodeIncomingNeighbors(idType node_id) const {
    vecsim_stl::vector<graphNodeType> incoming_neighbors(this->allocator);

    auto element = getGraphDataByInternalId(node_id);
    for (size_t level = 0; level <= element->toplevel; level++) {
        // Save the node neighbor's in the current level while holding its neighbors lock.
        std::vector<idType> neighbors_copy;
        lockNodeLinks(element);
        auto &node_level_data = getLevelData(element, level);
        // Store the deleted element's neighbours.
        neighbors_copy.assign(node_level_data.links,
                              node_level_data.links + node_level_data.numLinks);
        unlockNodeLinks(element);

        // Go over the neighbours and collect tho ones that also points back to the removed node.
        for (auto neighbour_id : neighbors_copy) {
            // Hold the neighbor's lock while we are going over its neighbors.
            auto *neighbor = getGraphDataByInternalId(neighbour_id);
            lockNodeLinks(neighbor);
            LevelData &neighbour_level_data = getLevelData(neighbor, level);

            for (size_t j = 0; j < neighbour_level_data.numLinks; j++) {
                // A bidirectional edge was found - this connection should be repaired.
                if (neighbour_level_data.links[j] == node_id) {
                    incoming_neighbors.emplace_back(neighbour_id, (ushort)level);
                    break;
                }
            }
            unlockNodeLinks(neighbor);
        }

        // Next, collect the rest of incoming edges (the ones that are not bidirectional) in the
        // current level to repair them.
        lockNodeLinks(element);
        for (auto incoming_edge : *node_level_data.incomingEdges) {
            incoming_neighbors.emplace_back(incoming_edge, (ushort)level);
        }
        unlockNodeLinks(element);
    }
    return incoming_neighbors;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::resizeIndexCommon(size_t new_max_elements) {
    assert(new_max_elements % this->blockSize == 0 &&
           "new_max_elements must be a multiple of blockSize");
    resizeLabelLookup(new_max_elements);
    visitedNodesHandlerPool.resize(new_max_elements);
    idToMetaData.resize(new_max_elements);
    idToMetaData.shrink_to_fit();

    maxElements = new_max_elements;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::growByBlock() {
    size_t new_max_elements = maxElements + this->blockSize;

    // Validations
    assert(vectorBlocks.size() == graphDataBlocks.size());
    assert(vectorBlocks.size() == 0 || vectorBlocks.back().getLength() == this->blockSize);

    vectorBlocks.emplace_back(this->blockSize, this->dataSize, this->allocator, this->alignment);
    graphDataBlocks.emplace_back(this->blockSize, this->elementGraphDataSize, this->allocator);

    resizeIndexCommon(new_max_elements);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::shrinkByBlock() {
    assert(maxElements >= this->blockSize);
    size_t new_max_elements = maxElements - this->blockSize;

    // Validations
    assert(vectorBlocks.size() == graphDataBlocks.size());
    assert(vectorBlocks.size() > 0);
    assert(vectorBlocks.back().getLength() == 0);

    vectorBlocks.pop_back();
    graphDataBlocks.pop_back();

    resizeIndexCommon(new_max_elements);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::mutuallyUpdateForRepairedNode(
    idType node_id, size_t level, vecsim_stl::vector<idType> &neighbors_to_remove,
    vecsim_stl::vector<idType> &nodes_to_update, vecsim_stl::vector<idType> &chosen_neighbors,
    size_t max_M_cur) {
    // Sort the nodes to remove set for fast lookup.
    std::sort(neighbors_to_remove.begin(), neighbors_to_remove.end());

    // Acquire the required locks for the updates, after sorting the nodes to update
    // (to avoid deadlocks)
    nodes_to_update.push_back(node_id);
    std::sort(nodes_to_update.begin(), nodes_to_update.end());
    size_t nodes_to_update_count = nodes_to_update.size();
    for (size_t i = 0; i < nodes_to_update_count; i++) {
        lockNodeLinks(nodes_to_update[i]);
    }

    LevelData &node_level = getLevelData(node_id, level);

    // Perform mutual updates: go over the node's neighbors and overwrite the neighbors to remove
    // that are still exist.
    size_t node_neighbors_idx = 0;
    for (size_t i = 0; i < node_level.numLinks; i++) {
        if (!std::binary_search(nodes_to_update.begin(), nodes_to_update.end(),
                                node_level.links[i])) {
            // The repaired node added a new neighbor that we didn't account for before in the
            // meantime - leave it as is.
            node_level.links[node_neighbors_idx++] = node_level.links[i];
            continue;
        }
        // Check if the current neighbor is in the chosen neighbors list, and remove it from there
        // if so.
        if (removeIdFromList(chosen_neighbors, node_level.links[i])) {
            // A chosen neighbor is already connected to the node - leave it as is.
            node_level.links[node_neighbors_idx++] = node_level.links[i];
            continue;
        }
        // Now we know that we are looking at a neighbor that needs to be removed.
        auto removed_node = node_level.links[i];
        LevelData &removed_node_level = getLevelData(removed_node, level);
        // Perform the mutual update:
        // if the removed node id (the node's neighbour to be removed)
        // wasn't pointing to the node (i.e., the edge was uni-directional),
        // we should remove the current neighbor from the node's incoming edges.
        // otherwise, the edge turned from bidirectional to uni-directional, so we insert it to the
        // neighbour's incoming edges set. Note: we assume that every update is performed atomically
        // mutually, so it should be sufficient to look at the removed node's incoming edges set
        // alone.
        if (!removeIdFromList(*removed_node_level.incomingEdges, node_id)) {
            node_level.incomingEdges->push_back(removed_node);
        }
    }

    // Go over the chosen new neighbors that are not connected yet and perform updates.
    for (auto chosen_id : chosen_neighbors) {
        if (node_neighbors_idx == max_M_cur) {
            // Cannot add more new neighbors, we reached the capacity.
            this->log("Couldn't add all the chosen new nodes upon updating %u, as we reached the"
                      " maximum number of neighbors per node",
                      node_id);
            break;
        }
        // We don't add new neighbors for deleted nodes - if node_id is deleted we can finish.
        // Also, don't add new neighbors to a node who is currently being indexed in parallel, as it
        // may choose the same element as its neighbor right after the repair is done and connect it
        // to it, and have a duplicate neighbor as a result.
        if (isMarkedDeleted(node_id) || isInProcess(node_id)) {
            break;
        }
        // If this specific new neighbor is deleted, we don't add this connection and continue.
        // Also, don't add a new node whose being indexed in parallel, as it may choose this node
        // as its neighbor and create a double connection (then this node will have a duplicate
        // neighbor).
        if (isMarkedDeleted(chosen_id) || isInProcess(chosen_id)) {
            continue;
        }
        node_level.links[node_neighbors_idx++] = chosen_id;
        // If the node is in the chosen new node incoming edges, there is a unidirectional
        // connection from the chosen node to the repaired node that turns into bidirectional. Then,
        // remove it from the incoming edges set. Otherwise, the edge is created unidirectional, so
        // we add it to the unidirectional edges set. Note: we assume that all updates occur
        // mutually and atomically, then can rely on this assumption.
        if (!removeIdFromList(*node_level.incomingEdges, chosen_id)) {
            getLevelData(chosen_id, level).incomingEdges->push_back(node_id);
        }
    }
    // Done updating the node's neighbors.
    node_level.numLinks = node_neighbors_idx;
    for (size_t i = 0; i < nodes_to_update_count; i++) {
        unlockNodeLinks(nodes_to_update[i]);
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::repairNodeConnections(idType node_id, size_t level) {

    candidatesMaxHeap<DistType> neighbors_candidates(this->allocator);
    // Use bitmaps for fast accesses:
    // node_orig_neighbours_set is used to differentiate between the neighbors that will *not* be
    // selected by the heuristics - only the ones that were originally neighbors should be removed.
    vecsim_stl::vector<bool> node_orig_neighbours_set(maxElements, false, this->allocator);
    // neighbors_candidates_set is used to store the nodes that were already collected as
    // candidates, so we will not collect them again as candidates if we run into them from another
    // path.
    vecsim_stl::vector<bool> neighbors_candidates_set(maxElements, false, this->allocator);
    vecsim_stl::vector<idType> deleted_neighbors(this->allocator);

    // Go over the repaired node neighbors, collect the non-deleted ones to be neighbors candidates
    // after the repair as well.
    const void *node_data = getDataByInternalId(node_id);
    auto *element = getGraphDataByInternalId(node_id);
    lockNodeLinks(element);
    LevelData &node_level_data = getLevelData(element, level);
    for (size_t j = 0; j < node_level_data.numLinks; j++) {
        node_orig_neighbours_set[node_level_data.links[j]] = true;
        // Don't add the removed element to the candidates.
        if (isMarkedDeleted(node_level_data.links[j])) {
            deleted_neighbors.push_back(node_level_data.links[j]);
            continue;
        }
        neighbors_candidates_set[node_level_data.links[j]] = true;
        neighbors_candidates.emplace(
            this->distFunc(node_data, getDataByInternalId(node_level_data.links[j]), this->dim),
            node_level_data.links[j]);
    }
    unlockNodeLinks(element);

    // If there are not deleted neighbors at that point the repair job has already been made by
    // another parallel job, and there is no need to repair the node anymore.
    if (deleted_neighbors.empty()) {
        return;
    }

    // Hold 3 sets of nodes - all the original neighbors at that point to later (potentially)
    // update, subset of these which are the chosen neighbors nodes, and a subset of the original
    // neighbors that are going to be removed.
    vecsim_stl::vector<idType> nodes_to_update(this->allocator);
    vecsim_stl::vector<idType> chosen_neighbors(this->allocator);
    vecsim_stl::vector<idType> neighbors_to_remove(this->allocator);

    // Go over the deleted nodes and collect their neighbors to the candidates set.
    for (idType deleted_neighbor_id : deleted_neighbors) {
        nodes_to_update.push_back(deleted_neighbor_id);
        neighbors_to_remove.push_back(deleted_neighbor_id);

        auto *neighbor = getGraphDataByInternalId(deleted_neighbor_id);
        lockNodeLinks(neighbor);
        LevelData &neighbor_level_data = getLevelData(neighbor, level);

        for (size_t j = 0; j < neighbor_level_data.numLinks; j++) {
            // Don't add removed elements to the candidates, nor nodes that are already in the
            // candidates set, nor the original node to repair itself.
            if (isMarkedDeleted(neighbor_level_data.links[j]) ||
                neighbors_candidates_set[neighbor_level_data.links[j]] ||
                neighbor_level_data.links[j] == node_id) {
                continue;
            }
            neighbors_candidates_set[neighbor_level_data.links[j]] = true;
            neighbors_candidates.emplace(
                this->distFunc(node_data, getDataByInternalId(neighbor_level_data.links[j]),
                               this->dim),
                neighbor_level_data.links[j]);
        }
        unlockNodeLinks(neighbor);
    }

    // Copy the original candidates, and run the heuristics. Afterwards, neighbors_candidates will
    // store the newly selected neighbours (for the node), while candidates which were originally
    // neighbors and are not going to be selected, are going to be removed.
    auto orig_candidates = neighbors_candidates;
    size_t max_M_cur = level ? M : M0;
    getNeighborsByHeuristic2(neighbors_candidates, max_M_cur);

    while (!orig_candidates.empty()) {
        idType orig_candidate = orig_candidates.top().second;
        if (neighbors_candidates.empty() || orig_candidate != neighbors_candidates.top().second) {
            if (node_orig_neighbours_set[orig_candidate]) {
                neighbors_to_remove.push_back(orig_candidate);
                nodes_to_update.push_back(orig_candidate);
            }
            orig_candidates.pop();
        } else {
            chosen_neighbors.push_back(orig_candidate);
            nodes_to_update.push_back(orig_candidate);
            neighbors_candidates.pop();
            orig_candidates.pop();
        }
    }

    // Perform the actual updates for the node and the impacted neighbors while holding the nodes'
    // locks.
    mutuallyUpdateForRepairedNode(node_id, level, neighbors_to_remove, nodes_to_update,
                                  chosen_neighbors, max_M_cur);
}

template <typename DataType, typename DistType>
inline bool
HNSWIndex<DataType, DistType>::removeIdFromList(vecsim_stl::vector<idType> &element_ids_list,
                                                idType element_id) {
    auto it = std::find(element_ids_list.begin(), element_ids_list.end(), element_id);
    if (it != element_ids_list.end()) {
        // Swap the last element with the current one (equivalent to removing the element id from
        // the list).
        *it = element_ids_list.back();
        element_ids_list.pop_back();
        return true;
    }
    return false;
}

/**
 * Ctor / Dtor
 */
/* typedef struct {
    VecSimType type;     // Datatype to index.
    size_t dim;          // Vector's dimension.
    VecSimMetric metric; // Distance metric to use in the index.
    size_t initialCapacity;
    size_t blockSize;
    size_t M;
    size_t efConstruction;
    size_t efRuntime;
    double epsilon;
} HNSWParams; */
template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::HNSWIndex(const HNSWParams *params,
                                         const AbstractIndexInitParams &abstractInitParams,
                                         size_t random_seed, size_t pool_initial_size)
    : VecSimIndexAbstract<DistType>(abstractInitParams), VecSimIndexTombstone(),
      maxElements(RoundUpInitialCapacity(params->initialCapacity, this->blockSize)),
      vectorBlocks(this->allocator), graphDataBlocks(this->allocator),
      idToMetaData(maxElements, this->allocator),
      visitedNodesHandlerPool(pool_initial_size, maxElements, this->allocator) {

    M = params->M ? params->M : HNSW_DEFAULT_M;
    M0 = M * 2;
    if (M0 > UINT16_MAX)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");

    efConstruction = params->efConstruction ? params->efConstruction : HNSW_DEFAULT_EF_C;
    efConstruction = std::max(efConstruction, M);
    ef = params->efRuntime ? params->efRuntime : HNSW_DEFAULT_EF_RT;
    epsilon = params->epsilon > 0.0 ? params->epsilon : HNSW_DEFAULT_EPSILON;

    curElementCount = 0;
    numMarkedDeleted = 0;

    // initializations for special treatment of the first node
    entrypointNode = INVALID_ID;
    maxLevel = HNSW_INVALID_LEVEL;

    if (M <= 1)
        throw std::runtime_error("HNSW index parameter M cannot be 1");
    mult = 1 / log(1.0 * M);
    levelGenerator.seed(random_seed);

    elementGraphDataSize = sizeof(ElementGraphData) + sizeof(idType) * M0;
    levelDataSize = sizeof(LevelData) + sizeof(idType) * M;

    size_t initial_vector_size = this->maxElements / this->blockSize;
    vectorBlocks.reserve(initial_vector_size);
    graphDataBlocks.reserve(initial_vector_size);
}

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::~HNSWIndex() {
    for (idType id = 0; id < curElementCount; id++) {
        destroyGraphData(getGraphDataByInternalId(id));
    }
}

/**
 * Index API functions
 */

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
void HNSWIndex<DataType, DistType>::removeAndSwap(idType internalId) {

    auto element = getGraphDataByInternalId(internalId);

    if (has_marked_deleted) {
        // If the index allows marking vectors as deleted (as in tiered HNSW), the id to remove
        // cannot be the entry point, as it should have been replaced upon marking it as deleted.
        assert(entrypointNode != internalId);
    } else if (entrypointNode == internalId) {
        // For inplace delete, we replace entry point now.
        assert(element->toplevel == maxLevel);
        replaceEntryPoint();
    }

    // Remove the deleted id form the relevant incoming edges sets in which it appears.
    for (size_t level = 0; level <= element->toplevel; level++) {
        LevelData &cur_level = getLevelData(element, level);
        for (size_t i = 0; i < cur_level.numLinks; i++) {
            LevelData &neighbour = getLevelData(cur_level.links[i], level);
            // This should always succeed, since every outgoing edge should be unidirectional at
            // this point (after all the repair jobs are done).
            removeIdFromList(*neighbour.incomingEdges, internalId);
        }
    }

    // Free the element's resources
    destroyGraphData(element);

    // We can say now that the element has removed completely from index.
    --curElementCount;
    if (has_marked_deleted) {
        --numMarkedDeleted;
    }

    // Get the last element's metadata and data.
    // If we are deleting the last element, we already destroyed it's metadata.
    DataBlock &last_vector_block = vectorBlocks.back();
    auto last_element_data = last_vector_block.removeAndFetchLastElement();
    DataBlock &last_gd_block = graphDataBlocks.back();
    auto last_element = (ElementGraphData *)last_gd_block.removeAndFetchLastElement();

    // Swap the last id with the deleted one, and invalidate the last id data.
    if (curElementCount != internalId) {
        SwapLastIdWithDeletedId<has_marked_deleted>(internalId, last_element, last_element_data);
    }

    // If we need to free a complete block and there is at least one block between the
    // capacity and the size.
    if (curElementCount % this->blockSize == 0) {
        shrinkByBlock();
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::removeAndSwapDeletedElement(idType internalId) {
    removeAndSwap<true>(internalId);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::removeVectorInPlace(const idType element_internal_id) {

    vecsim_stl::vector<bool> neighbours_bitmap(this->allocator);

    // Go over the element's nodes at every level and repair the effected connections.
    auto element = getGraphDataByInternalId(element_internal_id);
    for (size_t level = 0; level <= element->toplevel; level++) {
        LevelData &cur_level = getLevelData(element, level);
        // Reset the neighbours' bitmap for the current level.
        neighbours_bitmap.assign(curElementCount, false);
        // Store the deleted element's neighbours set in a bitmap for fast access.
        for (size_t j = 0; j < cur_level.numLinks; j++) {
            neighbours_bitmap[cur_level.links[j]] = true;
        }
        // Go over the neighbours that also points back to the removed point and make a local
        // repair.
        for (size_t i = 0; i < cur_level.numLinks; i++) {
            idType neighbour_id = cur_level.links[i];
            LevelData &neighbor_level = getLevelData(neighbour_id, level);

            bool bidirectional_edge = false;
            for (size_t j = 0; j < neighbor_level.numLinks; j++) {
                // If the edge is bidirectional, do repair for this neighbor.
                if (neighbor_level.links[j] == element_internal_id) {
                    bidirectional_edge = true;
                    repairConnectionsForDeletion(element_internal_id, neighbour_id, cur_level,
                                                 neighbor_level, level, neighbours_bitmap);
                    break;
                }
            }

            // If this edge is uni-directional, we should remove the element from the neighbor's
            // incoming edges.
            if (!bidirectional_edge) {
                // This should always return true (remove should succeed).
                removeIdFromList(*neighbor_level.incomingEdges, element_internal_id);
            }
        }

        // Next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // repairs.
        for (auto incoming_edge : *cur_level.incomingEdges) {
            repairConnectionsForDeletion(element_internal_id, incoming_edge, cur_level,
                                         getLevelData(incoming_edge, level), level,
                                         neighbours_bitmap);
        }
    }
    // Finally, remove the element from the index and make a swap with the last internal id to
    // avoid fragmentation and reclaim memory when needed.
    removeAndSwap<false>(element_internal_id);
}

// Store the new element in the global data structures and keep the new state. In multithreaded
// scenario, the index data guard should be held by the caller (exclusive lock).
template <typename DataType, typename DistType>
AddVectorCtx HNSWIndex<DataType, DistType>::storeNewElement(labelType label,
                                                            const void *vector_data) {
    AddVectorCtx state{};

    // Choose randomly the maximum level in which the new element will be in the index.
    state.elementMaxLevel = getRandomLevel(mult);

    // Access and update the index global data structures with the new element meta-data.
    state.newElementId = curElementCount++;

    // Create the new element's graph metadata.
    // We must assign manually enough memory on the stack and not just declare an `ElementGraphData`
    // variable, since it has a flexible array member.
    char tmpData[this->elementGraphDataSize];
    memset(tmpData, 0, this->elementGraphDataSize);
    ElementGraphData *cur_egd = (ElementGraphData *)tmpData;
    // Allocate memory (inside `ElementGraphData` constructor) for the links in higher levels and
    // initialize this memory to zeros. The reason for doing it here is that we might mark this
    // vector as deleted BEFORE we finish its indexing. In that case, we will collect the incoming
    // edges to this element in every level, and try to access its link lists in higher levels.
    // Therefore, we allocate it here and initialize it with zeros, (otherwise we might crash...)
    try {
        new (cur_egd) ElementGraphData(state.elementMaxLevel, levelDataSize, this->allocator);
    } catch (std::runtime_error &e) {
        this->log("Error - allocating memory for new element failed due to low memory");
        throw e;
    }

    if (indexSize() > indexCapacity()) {
        growByBlock();
    } else if (state.newElementId % this->blockSize == 0) {
        // If we had an initial capacity, we might have to allocate new blocks for the data and
        // meta-data.
        this->vectorBlocks.emplace_back(this->blockSize, this->dataSize, this->allocator,
                                        this->alignment);
        this->graphDataBlocks.emplace_back(this->blockSize, this->elementGraphDataSize,
                                           this->allocator);
    }

    // Insert the new element to the data block
    this->vectorBlocks.back().addElement(vector_data);
    this->graphDataBlocks.back().addElement(cur_egd);
    // We mark id as in process *before* we set it in the label lookup, otherwise we might check
    // that the label exist with safeCheckIfLabelExistsInIndex and see that IN_PROCESS flag is
    // clear.
    this->idToMetaData[state.newElementId] = ElementMetaData(label);
    setVectorId(label, state.newElementId);

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
    return state;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::appendVector(const void *vector_data, const labelType label,
                                                 AddVectorCtx *auxiliaryCtx) {

    // If auxiliaryCtx is not NULL, the index state has already been updated from outside (such as
    // in tiered index). Also, the synchronization responsibility in this case is on the caller,
    // otherwise, this function should acquire and release the lock to ensure proper parallelism.
    AddVectorCtx state{};
    if (auxiliaryCtx == nullptr) {
        this->lockIndexDataGuard();
        state = storeNewElement(label, vector_data);
        if (state.currMaxLevel >= state.elementMaxLevel) {
            this->unlockIndexDataGuard();
        }
    } else {
        state = *auxiliaryCtx;
    }
    // Deconstruct the state variables from the auxiliaryCtx. prev_entry_point and prev_max_level
    // are the entry point and index max level at the point of time when the element was stored, and
    // they may (or may not) have changed due to the insertion.
    auto [new_element_id, element_max_level, prev_entry_point, prev_max_level] = state;

    // Start scanning the graph from the current entry point.
    idType curr_element = prev_entry_point;
    // This condition only means that we are not inserting the first (non-deleted) element.
    if (curr_element != INVALID_ID) {
        DistType cur_dist = std::numeric_limits<DistType>::max();
        int max_common_level;
        if (element_max_level < prev_max_level) {
            max_common_level = element_max_level;
            cur_dist = this->distFunc(vector_data, getDataByInternalId(curr_element), this->dim);
            for (int level = prev_max_level; level > element_max_level; level--) {
                // this is done for the levels which are above the max level
                // to which we are going to insert the new element. We do
                // a greedy search in the graph starting from the entry point
                // at each level, and move on with the closest element we can find.
                // When there is no improvement to do, we take a step down.
                greedySearchLevel<false>(vector_data, level, curr_element, cur_dist);
            }
        } else {
            max_common_level = prev_max_level;
        }

        for (int level = max_common_level; (int)level >= 0; level--) {
            candidatesMaxHeap<DistType> top_candidates =
                searchLayer<false>(curr_element, vector_data, level, efConstruction);
            curr_element = mutuallyConnectNewElement(new_element_id, top_candidates, level);
        }
    } else {
        // Inserting the first (non-deleted) element to the graph - do nothing.
    }
    unmarkInProcess(new_element_id);
    if (auxiliaryCtx == nullptr && state.currMaxLevel < state.elementMaxLevel) {
        // No external auxiliaryCtx, so it's this function responsibility to release the lock.
        this->unlockIndexDataGuard();
    }
}

template <typename DataType, typename DistType>
auto HNSWIndex<DataType, DistType>::safeGetEntryPointState() const {
    std::shared_lock<std::shared_mutex> lock(indexDataGuard);
    return std::make_pair(entrypointNode, maxLevel);
}

template <typename DataType, typename DistType>
idType HNSWIndex<DataType, DistType>::searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                                          VecSimQueryResult_Code *rc) const {
    *rc = VecSim_QueryResult_OK;

    auto [curr_element, max_level] = safeGetEntryPointState();
    if (curr_element == INVALID_ID)
        return curr_element; // index is empty.

    DistType cur_dist = this->distFunc(query_data, getDataByInternalId(curr_element), this->dim);
    for (size_t level = max_level; level > 0 && curr_element != INVALID_ID; level--) {
        greedySearchLevel<true>(query_data, level, curr_element, cur_dist, timeoutCtx, rc);
    }
    return curr_element;
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
candidatesLabelsMaxHeap<DistType> *
HNSWIndex<DataType, DistType>::searchBottomLayer_WithTimeout(idType ep_id, const void *data_point,
                                                             size_t ef, size_t k, void *timeoutCtx,
                                                             VecSimQueryResult_Code *rc) const {

    auto *visited_nodes_handler = getVisitedList();
    tag_t visited_tag = visited_nodes_handler->getFreshTag();

    candidatesLabelsMaxHeap<DistType> *top_candidates = getNewMaxPriorityQueue();
    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    DistType lowerBound;
    if (!has_marked_deleted || !isMarkedDeleted(ep_id)) {
        // If ep is not marked as deleted, get its distance and set lower bound and heaps
        // accordingly
        DistType dist = this->distFunc(data_point, getDataByInternalId(ep_id), this->dim);
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
            *rc = VecSim_QueryResult_TimedOut;
            return top_candidates;
        }
        candidate_set.pop();

        processCandidate<has_marked_deleted>(curr_el_pair.second, data_point, 0, ef,
                                             visited_nodes_handler->getElementsTags(), visited_tag,
                                             *top_candidates, candidate_set, lowerBound);
    }
    returnVisitedList(visited_nodes_handler);
    while (top_candidates->size() > k) {
        top_candidates->pop();
    }
    *rc = VecSim_QueryResult_OK;
    return top_candidates;
}

template <typename DataType, typename DistType>
VecSimQueryResult_List
HNSWIndex<DataType, DistType>::topKQuery(const void *query_data, size_t k,
                                         VecSimQueryParams *queryParams) const {

    VecSimQueryResult_List rl = {0};
    this->lastMode = STANDARD_KNN;

    if (curElementCount == 0 || k == 0) {
        rl.code = VecSim_QueryResult_OK;
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    void *timeoutCtx = nullptr;

    // Get original efRuntime and store it.
    size_t query_ef = this->ef;

    if (queryParams) {
        timeoutCtx = queryParams->timeoutCtx;
        if (queryParams->hnswRuntimeParams.efRuntime != 0) {
            query_ef = queryParams->hnswRuntimeParams.efRuntime;
        }
    }

    idType bottom_layer_ep = searchBottomLayerEP(query_data, timeoutCtx, &rl.code);
    if (VecSim_OK != rl.code) {
        return rl;
    } else if (bottom_layer_ep == INVALID_ID) {
        // Although we checked that the index is not empty (curElementCount == 0), it might be
        // that another thread deleted all the elements or didn't finish inserting the first element
        // yet. Anyway, we observed that the index is empty, so we return an empty result list.
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    // We now oun the results heap, we need to free (delete) it when we done
    candidatesLabelsMaxHeap<DistType> *results;
    if (this->numMarkedDeleted) {
        results = searchBottomLayer_WithTimeout<true>(
            bottom_layer_ep, query_data, std::max(query_ef, k), k, timeoutCtx, &rl.code);
    } else {
        results = searchBottomLayer_WithTimeout<false>(
            bottom_layer_ep, query_data, std::max(query_ef, k), k, timeoutCtx, &rl.code);
    }

    if (VecSim_OK == rl.code) {
        rl.results = array_new_len<VecSimQueryResult>(results->size(), results->size());
        for (int i = (int)results->size() - 1; i >= 0; --i) {
            VecSimQueryResult_SetId(rl.results[i], results->top().second);
            VecSimQueryResult_SetScore(rl.results[i], results->top().first);
            results->pop();
        }
    }
    delete results;
    return rl;
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
VecSimQueryResult *HNSWIndex<DataType, DistType>::searchRangeBottomLayer_WithTimeout(
    idType ep_id, const void *data_point, double epsilon, DistType radius, void *timeoutCtx,
    VecSimQueryResult_Code *rc) const {

    *rc = VecSim_QueryResult_OK;
    auto res_container = getNewResultsContainer(10); // arbitrary initial cap.

    auto *visited_nodes_handler = getVisitedList();
    tag_t visited_tag = visited_nodes_handler->getFreshTag();

    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    // Set the initial effective-range to be at least the distance from the entry-point.
    DistType ep_dist, dynamic_range, dynamic_range_search_boundaries;
    if (has_marked_deleted && isMarkedDeleted(ep_id)) {
        // If ep is marked as deleted, set initial ranges to max
        ep_dist = std::numeric_limits<DistType>::max();
        dynamic_range_search_boundaries = dynamic_range = ep_dist;
    } else {
        // If ep is not marked as deleted, get its distance and set ranges accordingly
        ep_dist = this->distFunc(data_point, getDataByInternalId(ep_id), this->dim);
        dynamic_range = ep_dist;
        if (ep_dist <= radius) {
            // Entry-point is within the radius - add it to the results.
            res_container->emplace(getExternalLabel(ep_id), ep_dist);
            dynamic_range = radius; // to ensure that dyn_range >= radius.
        }
        dynamic_range_search_boundaries = dynamic_range * (1.0 + epsilon);
    }

    candidate_set.emplace(-ep_dist, ep_id);
    visited_nodes_handler->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();
        // If the best candidate is outside the dynamic range in more than epsilon (relatively) - we
        // finish the search.

        if ((-curr_el_pair.first) > dynamic_range_search_boundaries) {
            break;
        }
        if (VECSIM_TIMEOUT(timeoutCtx)) {
            *rc = VecSim_QueryResult_TimedOut;
            break;
        }
        candidate_set.pop();

        // Decrease the effective range, but keep dyn_range >= radius.
        if (-curr_el_pair.first < dynamic_range && -curr_el_pair.first >= radius) {
            dynamic_range = -curr_el_pair.first;
            dynamic_range_search_boundaries = dynamic_range * (1.0 + epsilon);
        }

        // Go over the candidate neighbours, add them to the candidates list if they are within the
        // epsilon environment of the dynamic range, and add them to the results if they are in the
        // requested radius.
        // Here we send the radius as double to match the function arguments type.
        processCandidate_RangeSearch<has_marked_deleted>(
            curr_el_pair.second, data_point, 0, epsilon, visited_nodes_handler->getElementsTags(),
            visited_tag, res_container, candidate_set, dynamic_range_search_boundaries, radius);
    }
    returnVisitedList(visited_nodes_handler);
    return res_container->get_results();
}

template <typename DataType, typename DistType>
VecSimQueryResult_List
HNSWIndex<DataType, DistType>::rangeQuery(const void *query_data, double radius,
                                          VecSimQueryParams *queryParams) const {

    VecSimQueryResult_List rl = {0};
    this->lastMode = RANGE_QUERY;

    if (curElementCount == 0) {
        rl.code = VecSim_QueryResult_OK;
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }
    void *timeoutCtx = nullptr;

    double query_epsilon = epsilon;
    if (queryParams) {
        timeoutCtx = queryParams->timeoutCtx;
        if (queryParams->hnswRuntimeParams.epsilon != 0.0) {
            query_epsilon = queryParams->hnswRuntimeParams.epsilon;
        }
    }

    idType bottom_layer_ep = searchBottomLayerEP(query_data, timeoutCtx, &rl.code);
    // Although we checked that the index is not empty (curElementCount == 0), it might be
    // that another thread deleted all the elements or didn't finish inserting the first element
    // yet. Anyway, we observed that the index is empty, so we return an empty result list.
    if (VecSim_OK != rl.code || bottom_layer_ep == INVALID_ID) {
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    // search bottom layer
    // Here we send the radius as double to match the function arguments type.
    if (this->numMarkedDeleted)
        rl.results = searchRangeBottomLayer_WithTimeout<true>(
            bottom_layer_ep, query_data, query_epsilon, radius, timeoutCtx, &rl.code);
    else
        rl.results = searchRangeBottomLayer_WithTimeout<false>(
            bottom_layer_ep, query_data, query_epsilon, radius, timeoutCtx, &rl.code);
    return rl;
}

template <typename DataType, typename DistType>
VecSimIndexInfo HNSWIndex<DataType, DistType>::info() const {

    VecSimIndexInfo info;
    info.commonInfo = this->getCommonInfo();

    info.commonInfo.basicInfo.algo = VecSimAlgo_HNSWLIB;
    info.hnswInfo.M = this->getM();
    info.hnswInfo.efConstruction = this->getEfConstruction();
    info.hnswInfo.efRuntime = this->getEf();
    info.hnswInfo.epsilon = this->epsilon;
    info.hnswInfo.max_level = this->getMaxLevel();
    info.hnswInfo.entrypoint = this->getEntryPointLabel();
    info.hnswInfo.visitedNodesPoolSize = this->visitedNodesHandlerPool.getPoolSize();
    info.hnswInfo.numberOfMarkedDeletedNodes = this->getNumMarkedDeleted();
    return info;
}

template <typename DataType, typename DistType>
VecSimIndexBasicInfo HNSWIndex<DataType, DistType>::basicInfo() const {
    VecSimIndexBasicInfo info = this->getBasicInfo();
    info.algo = VecSimAlgo_HNSWLIB;
    info.isTiered = false;
    return info;
}

template <typename DataType, typename DistType>
VecSimInfoIterator *HNSWIndex<DataType, DistType>::infoIterator() const {
    VecSimIndexInfo info = this->info();
    // For readability. Update this number when needed.
    size_t numberOfInfoFields = 17;
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields);

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::ALGORITHM_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .fieldValue = {FieldValue{
                             .stringValue = VecSimAlgo_ToString(info.commonInfo.basicInfo.algo)}}});

    this->addCommonInfoToIterator(infoIterator, info.commonInfo);

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.commonInfo.basicInfo.blockSize}}});

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_M_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.M}}});

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::HNSW_EF_CONSTRUCTION_STRING,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.efConstruction}}});

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_EF_RUNTIME_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.efRuntime}}});

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_MAX_LEVEL,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.max_level}}});

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_ENTRYPOINT,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.entrypoint}}});

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_EPSILON_STRING,
                         .fieldType = INFOFIELD_FLOAT64,
                         .fieldValue = {FieldValue{.floatingPointValue = info.hnswInfo.epsilon}}});

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::HNSW_NUM_MARKED_DELETED,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.numberOfMarkedDeletedNodes}}});

    return infoIterator;
}

template <typename DataType, typename DistType>
bool HNSWIndex<DataType, DistType>::preferAdHocSearch(size_t subsetSize, size_t k,
                                                      bool initial_check) const {
    // This heuristic is based on sklearn decision tree classifier (with 20 leaves nodes) -
    // see scripts/HNSW_batches_clf.py
    size_t index_size = this->indexSize();
    // Referring to too large subset size as if it was the maximum possible size.
    subsetSize = std::min(subsetSize, index_size);

    size_t d = this->dim;
    size_t M = this->getM();
    float r = (index_size == 0) ? 0.0f : (float)(subsetSize) / (float)this->indexLabelCount();
    bool res;

    // node 0
    if (index_size <= 30000) {
        // node 1
        if (index_size <= 5500) {
            // node 5
            res = true;
        } else {
            // node 6
            if (r <= 0.17) {
                // node 11
                res = true;
            } else {
                // node 12
                if (k <= 12) {
                    // node 13
                    if (d <= 55) {
                        // node 17
                        res = false;
                    } else {
                        // node 18
                        if (M <= 10) {
                            // node 19
                            res = false;
                        } else {
                            // node 20
                            res = true;
                        }
                    }
                } else {
                    // node 14
                    res = true;
                }
            }
        }
    } else {
        // node 2
        if (r < 0.07) {
            // node 3
            if (index_size <= 750000) {
                // node 15
                res = true;
            } else {
                // node 16
                if (k <= 7) {
                    // node 21
                    res = false;
                } else {
                    // node 22
                    if (r <= 0.03) {
                        // node 23
                        res = true;
                    } else {
                        // node 24
                        res = false;
                    }
                }
            }
        } else {
            // node 4
            if (d <= 75) {
                // node 7
                res = false;
            } else {
                // node 8
                if (k <= 12) {
                    // node 9
                    if (r <= 0.21) {
                        // node 27
                        if (M <= 57) {
                            // node 29
                            if (index_size <= 75000) {
                                // node 31
                                res = true;
                            } else {
                                // node 32
                                res = false;
                            }
                        } else {
                            // node 30
                            res = true;
                        }
                    } else {
                        // node 28
                        res = false;
                    }
                } else {
                    // node 10
                    if (M <= 10) {
                        // node 25
                        if (r <= 0.17) {
                            // node 33
                            res = true;
                        } else {
                            // node 34
                            res = false;
                        }
                    } else {
                        // node 26
                        if (index_size <= 300000) {
                            // node 35
                            res = true;
                        } else {
                            // node 36
                            if (r <= 0.17) {
                                // node 37
                                res = true;
                            } else {
                                // node 38
                                res = false;
                            }
                        }
                    }
                }
            }
        }
    }
    // Set the mode - if this isn't the initial check, we switched mode form batches to ad-hoc.
    this->lastMode =
        res ? (initial_check ? HYBRID_ADHOC_BF : HYBRID_BATCHES_TO_ADHOC_BF) : HYBRID_BATCHES;
    return res;
}

#ifdef BUILD_TESTS
#include "hnsw_serializer.h"
#endif
