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
typedef uint16_t elementFlags;

template <typename DistType>
using candidatesMaxHeap = vecsim_stl::max_priority_queue<DistType, idType>;
template <typename DistType>
using candidatesLabelsMaxHeap = vecsim_stl::abstract_priority_queue<DistType, labelType>;
using graphNodeType = pair<idType, ushort>; // represented as: (element_id, level)

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
    size_t max_elements_;
    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;

    // Index search parameter
    size_t ef_;
    double epsilon_;

    // Index meta-data (based on the data dimensionality and index parameters)
    size_t size_data_per_element_;
    size_t size_links_per_element_;
    size_t size_links_level0_;
    size_t label_offset_;
    size_t offsetData_, offsetLevel0_;
    size_t incoming_links_offset0;
    size_t incoming_links_offset;
    double mult_;

    // Index level generator of the top level for a new element
    std::default_random_engine level_generator_;

    // Index global state - these should be guarded by the index_data_guard_ lock in
    // multithreaded scenario.
    size_t cur_element_count;
    vecsim_stl::vector<size_t> element_levels_;
    idType entrypoint_node_;
    size_t max_level_; // this is the top level of the entry point's element

    // Index data
    char *data_level0_memory_; // neighbors in level 0, element label, flags and data (vector)
    char **linkLists_;         // neighbors in level higher than 0

    // Used for marking the visited nodes in graph scans (the pool supports parallel graph scans).
    // This is mutable since the object changes upon search operations as well (which are const).
    mutable VisitedNodesHandlerPool visited_nodes_handler_pool;

    mutable std::shared_mutex index_data_guard_;
    mutable vecsim_stl::vector<std::mutex> element_neighbors_locks_;

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_base_tests_friends.h"

#include "hnsw_serializer_declarations.h"
#endif

protected:
    HNSWIndex() = delete;                  // default constructor is disabled.
    HNSWIndex(const HNSWIndex &) = delete; // default (shallow) copy constructor is disabled.
    inline void setExternalLabel(idType internal_id, labelType label);
    inline labelType *getExternalLabelPtr(idType internal_id) const;
    inline size_t getRandomLevel(double reverse_size);
    inline vecsim_stl::vector<idType> *getIncomingEdgesPtr(idType internal_id, size_t level) const;
    inline void setIncomingEdgesPtr(idType internal_id, size_t level, void *edges_ptr);
    inline elementFlags *getElementFlags(idType internal_id) const;
    inline idType *getNodeNeighborsAtBaseLevel(idType internal_id) const;
    inline idType *getNodeNeighborsAtNonBaseLevel(idType internal_id, size_t level) const;
    inline void setNodeNeighborsCount(idType *list, linkListSize size);
    inline void removeExtraLinks(candidatesMaxHeap<DistType> candidates, size_t Mcurmax,
                                 idType *node_neighbors, const vecsim_stl::vector<bool> &bitmap,
                                 idType *removed_links, size_t *removed_links_num);
    template <bool has_marked_deleted, typename Identifier> // Either idType or labelType
    inline DistType
    processCandidate(idType curNodeId, const void *data_point, size_t layer, size_t ef,
                     tag_t visited_tag, tag_t *elements_tags,
                     vecsim_stl::abstract_priority_queue<DistType, Identifier> &top_candidates,
                     candidatesMaxHeap<DistType> &candidates_set, DistType lowerBound) const;
    template <bool has_marked_deleted>
    inline void processCandidate_RangeSearch(
        idType curNodeId, const void *data_point, size_t layer, double epsilon, tag_t visited_tag,
        tag_t *elements_tags,
        std::unique_ptr<vecsim_stl::abstract_results_container> &top_candidates,
        candidatesMaxHeap<DistType> &candidate_set, DistType lowerBound, double radius) const;
    template <bool has_marked_deleted>
    candidatesMaxHeap<DistType> searchLayer(idType ep_id, const void *data_point, size_t layer,
                                            size_t ef) const;
    template <bool has_marked_deleted>
    candidatesLabelsMaxHeap<DistType> *
    searchBottomLayer_WithTimeout(idType ep_id, const void *data_point, size_t ef, size_t k,
                                  void *timeoutCtx, VecSimQueryResult_Code *rc) const;
    template <bool has_marked_deleted>
    VecSimQueryResult *searchRangeBottomLayer_WithTimeout(idType ep_id, const void *data_point,
                                                          double epsilon, double radius,
                                                          void *timeoutCtx,
                                                          VecSimQueryResult_Code *rc) const;
    void getNeighborsByHeuristic2(candidatesMaxHeap<DistType> &top_candidates, size_t M);
    // Helper function for re-selecting node's neighbors which was selected as a neighbor for
    // a newly inserted node. Also, responsible for mutually connect the new node and the neighbor
    // (unidirectional or bidirectional connection).
    // *Note that node_lock and neighbor_lock should be locked upon calling this function*
    void revisitNeighborConnections(size_t level, idType new_node_id,
                                    const std::pair<DistType, idType> &neighbor_data,
                                    idType *new_node_neighbors_list,
                                    idType *neighbor_neighbors_list,
                                    std::unique_lock<std::mutex> &node_lock,
                                    std::unique_lock<std::mutex> &neighbor_lock);
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
                                      idType *neighbours_list, idType *neighbour_neighbours_list,
                                      size_t level, vecsim_stl::vector<bool> &neighbours_bitmap);
    inline void replaceEntryPoint();
    inline void resizeIndexInternal(size_t new_max_elements);

    template <bool has_marked_deleted>
    inline void SwapLastIdWithDeletedId(idType element_internal_id);

    // Protected internal function that implements generic single vector insertion.
    void appendVector(const void *vector_data, labelType label,
                      AddVectorCtx *auxiliaryCtx = nullptr);

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

    // Do nothing - relevant for tiered index only.
    void runGC() {};

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
    inline labelType getExternalLabel(idType internal_id) const;
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
    inline VisitedNodesHandler *getVisitedList() const;
    inline void returnVisitedList(VisitedNodesHandler *visited_nodes_handler) const;
    VecSimIndexInfo info() const override;
    VecSimIndexBasicInfo basicInfo() const override;
    VecSimInfoIterator *infoIterator() const override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override;
    char *getDataByInternalId(idType internal_id) const;
    inline idType *getNodeNeighborsAtLevel(idType internal_id, size_t level) const;
    inline linkListSize getNodeNeighborsCount(const idType *list) const;
    inline idType searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                      VecSimQueryResult_Code *rc) const;

    VecSimQueryResult_List topKQuery(const void *query_data, size_t k,
                                     VecSimQueryParams *queryParams) const override;
    VecSimQueryResult_List rangeQuery(const void *query_data, double radius,
                                      VecSimQueryParams *queryParams) const override;

    inline void markDeletedInternal(idType internalId);
    inline bool isMarkedDeleted(idType internalId) const;
    inline bool isInProcess(idType internalId) const;
    inline void markInProcess(idType internalId);
    inline void unmarkInProcess(idType internalId);
    void increaseCapacity() override;
    AddVectorCtx storeNewElement(labelType label);
    void removeAndSwapDeletedElement(idType internalId);
    void repairNodeConnections(idType node_id, size_t level);
    inline size_t getElementTopLevel(idType internalId);
    vecsim_stl::vector<graphNodeType> safeCollectAllNodeIncomingNeighbors(idType node_id,
                                                                          size_t node_top_level);
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
    ef_ = ef;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getEf() const {
    return ef_;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::setEpsilon(double epsilon) {
    epsilon_ = epsilon;
}

template <typename DataType, typename DistType>
double HNSWIndex<DataType, DistType>::getEpsilon() const {
    return epsilon_;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::indexSize() const {
    return cur_element_count;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::indexCapacity() const {
    return max_elements_;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getEfConstruction() const {
    return ef_construction_;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getM() const {
    return M_;
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getMaxLevel() const {
    return max_level_;
}

template <typename DataType, typename DistType>
labelType HNSWIndex<DataType, DistType>::getEntryPointLabel() const {
    if (entrypoint_node_ != INVALID_ID)
        return getExternalLabel(entrypoint_node_);
    return SIZE_MAX;
}

template <typename DataType, typename DistType>
labelType HNSWIndex<DataType, DistType>::getExternalLabel(idType internal_id) const {
    labelType return_label;
    memcpy(&return_label,
           (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
           sizeof(labelType));
    return return_label;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::setExternalLabel(idType internal_id, labelType label) {
    memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label,
           sizeof(labelType));
}

template <typename DataType, typename DistType>
labelType *HNSWIndex<DataType, DistType>::getExternalLabelPtr(idType internal_id) const {
    return (labelType *)(data_level0_memory_ + internal_id * size_data_per_element_ +
                         label_offset_);
}

template <typename DataType, typename DistType>
char *HNSWIndex<DataType, DistType>::getDataByInternalId(idType internal_id) const {
    return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (size_t)r;
}

template <typename DataType, typename DistType>
vecsim_stl::vector<idType> *HNSWIndex<DataType, DistType>::getIncomingEdgesPtr(idType internal_id,
                                                                               size_t level) const {
    if (level == 0) {
        return reinterpret_cast<vecsim_stl::vector<idType> *>(
            *(void **)(data_level0_memory_ + internal_id * size_data_per_element_ +
                       incoming_links_offset0));
    }
    return reinterpret_cast<vecsim_stl::vector<idType> *>(
        *(void **)(linkLists_[internal_id] + (level - 1) * size_links_per_element_ +
                   incoming_links_offset));
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::setIncomingEdgesPtr(idType internal_id, size_t level,
                                                        void *edges_ptr) {
    if (level == 0) {
        memcpy(data_level0_memory_ + internal_id * size_data_per_element_ + incoming_links_offset0,
               &edges_ptr, sizeof(void *));
    } else {
        memcpy(linkLists_[internal_id] + (level - 1) * size_links_per_element_ +
                   incoming_links_offset,
               &edges_ptr, sizeof(void *));
    }
}

template <typename DataType, typename DistType>
elementFlags *HNSWIndex<DataType, DistType>::getElementFlags(idType internal_id) const {
    // elementFlags offset is 0 from the start of the element metadata
    return (elementFlags *)(data_level0_memory_ + internal_id * size_data_per_element_ +
                            offsetLevel0_);
}

template <typename DataType, typename DistType>
idType *HNSWIndex<DataType, DistType>::getNodeNeighborsAtBaseLevel(idType internal_id) const {
    // links offset at level 0 is `sizeof(elementFlags) + sizeof(linkListSize)` from the start of
    // the element metadata
    return (idType *)(data_level0_memory_ + internal_id * size_data_per_element_ +
                      sizeof(elementFlags) + sizeof(linkListSize) + offsetLevel0_);
}

template <typename DataType, typename DistType>
idType *HNSWIndex<DataType, DistType>::getNodeNeighborsAtNonBaseLevel(idType internal_id,
                                                                      size_t level) const {
    // links offset at level >0 is `sizeof(linkListSize)` from the start of the element metadata
    return (idType *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_ +
                      sizeof(linkListSize));
}

template <typename DataType, typename DistType>
idType *HNSWIndex<DataType, DistType>::getNodeNeighborsAtLevel(idType internal_id,
                                                               size_t level) const {
    return level == 0 ? getNodeNeighborsAtBaseLevel(internal_id)
                      : getNodeNeighborsAtNonBaseLevel(internal_id, level);
}

template <typename DataType, typename DistType>
linkListSize HNSWIndex<DataType, DistType>::getNodeNeighborsCount(const idType *list) const {
    return *(((linkListSize *)list) - 1);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::setNodeNeighborsCount(idType *list, const linkListSize size) {
    *(((linkListSize *)list) - 1) = size;
}

template <typename DataType, typename DistType>
VisitedNodesHandler *HNSWIndex<DataType, DistType>::getVisitedList() const {
    return visited_nodes_handler_pool.getAvailableVisitedNodesHandler();
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::returnVisitedList(
    VisitedNodesHandler *visited_nodes_handler) const {
    visited_nodes_handler_pool.returnVisitedNodesHandlerToPool(visited_nodes_handler);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::markDeletedInternal(idType internalId) {
    // Here we are holding the global index data guard (and the main index lock of the tiered index
    // for shared ownership).
    assert(internalId < this->cur_element_count);
    if (!isMarkedDeleted(internalId)) {
        if (internalId == entrypoint_node_) {
            // Internally, we hold and release the entrypoint neighbors lock.
            replaceEntryPoint();
        }
        // Atomically set the deletion mark flag (note that other parallel threads may set the flags
        // at the same time (for changing the IN_PROCESS flag).
        __atomic_fetch_or(getElementFlags(internalId), DELETE_MARK, 0);
        this->num_marked_deleted++;
    }
}

template <typename DataType, typename DistType>
bool HNSWIndex<DataType, DistType>::isMarkedDeleted(idType internalId) const {
    elementFlags *flags = getElementFlags(internalId);
    return *flags & DELETE_MARK;
}

template <typename DataType, typename DistType>
bool HNSWIndex<DataType, DistType>::isInProcess(idType internalId) const {
    elementFlags *flags = getElementFlags(internalId);
    return *flags & IN_PROCESS;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::markInProcess(idType internalId) {
    // Atomically set the IN_PROCESS mark flag. Even though other threads shouldn't modify the flags
    // at that time (we're holding index global data guard, so this element cannot be marked as
    // deleted in parallel), we do it for safety.
    __atomic_fetch_or(getElementFlags(internalId), IN_PROCESS, 0);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::unmarkInProcess(idType internalId) {
    // Atomically unset the IN_PROCESS mark flag (note that other parallel threads may set the flags
    // at the same time (for marking the element with MARK_DELETE flag).
    __atomic_fetch_and(getElementFlags(internalId), ~IN_PROCESS, 0);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::lockIndexDataGuard() const {
    index_data_guard_.lock();
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::unlockIndexDataGuard() const {
    index_data_guard_.unlock();
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::lockNodeLinks(idType node_id) const {
    element_neighbors_locks_[node_id].lock();
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::unlockNodeLinks(idType node_id) const {
    element_neighbors_locks_[node_id].unlock();
}

template <typename DataType, typename DistType>
inline size_t HNSWIndex<DataType, DistType>::getElementTopLevel(idType internalId) {
    return element_levels_[internalId];
}

/**
 * helper functions
 */
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::removeExtraLinks(
    candidatesMaxHeap<DistType> candidates, size_t Mcurmax, idType *node_neighbors,
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
            node_neighbors[link_idx++] = candidates.top().second;
            candidates.pop();
            orig_candidates.pop();
        }
    }
    setNodeNeighborsCount(node_neighbors, link_idx);
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
DistType HNSWIndex<DataType, DistType>::processCandidate(
    idType curNodeId, const void *data_point, size_t layer, size_t ef, tag_t visited_tag,
    tag_t *elements_tags, vecsim_stl::abstract_priority_queue<DistType, Identifier> &top_candidates,
    candidatesMaxHeap<DistType> &candidate_set, DistType lowerBound) const {

    std::unique_lock<std::mutex> lock(element_neighbors_locks_[curNodeId]);
    idType *node_links = getNodeNeighborsAtLevel(curNodeId, layer);
    linkListSize links_num = getNodeNeighborsCount(node_links);

    __builtin_prefetch(elements_tags + *node_links);
    __builtin_prefetch(getDataByInternalId(*node_links));

    for (size_t j = 0; j < links_num; j++) {
        idType *candidate_pos = node_links + j;
        idType candidate_id = *candidate_pos;
        idType *next_candidate_pos = node_links + j + 1;

        __builtin_prefetch(elements_tags + *next_candidate_pos);
        __builtin_prefetch(getDataByInternalId(*next_candidate_pos));

        if (elements_tags[candidate_id] == visited_tag || isInProcess(candidate_id))
            continue;

        elements_tags[candidate_id] = visited_tag;
        char *currObj1 = (getDataByInternalId(candidate_id));

        DistType dist1 = this->dist_func(data_point, currObj1, this->dim);
        if (lowerBound > dist1 || top_candidates.size() < ef) {
            candidate_set.emplace(-dist1, candidate_id);

            // Insert the candidate to the top candidates heap only if it is not marked as deleted.
            if (!has_marked_deleted || !isMarkedDeleted(candidate_id))
                emplaceToHeap(top_candidates, dist1, candidate_id);

            if (top_candidates.size() > ef)
                top_candidates.pop();

            // If we have marked deleted elements, we need to verify that `top_candidates` is not
            // empty (since we might have not added any non-deleted element yet).
            if (!has_marked_deleted || !top_candidates.empty())
                lowerBound = top_candidates.top().first;
        }
    }
    // Pre-fetch the neighbours list of the top candidate (the one that is going
    // to be processed in the next iteration) into memory cache, to improve performance.
    __builtin_prefetch(getNodeNeighborsAtLevel(candidate_set.top().second, layer));

    return lowerBound;
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
void HNSWIndex<DataType, DistType>::processCandidate_RangeSearch(
    idType curNodeId, const void *query_data, size_t layer, double epsilon, tag_t visited_tag,
    tag_t *elements_tags, std::unique_ptr<vecsim_stl::abstract_results_container> &results,
    candidatesMaxHeap<DistType> &candidate_set, DistType dyn_range, double radius) const {

    std::unique_lock<std::mutex> lock(element_neighbors_locks_[curNodeId]);
    idType *node_links = getNodeNeighborsAtLevel(curNodeId, layer);
    linkListSize links_num = getNodeNeighborsCount(node_links);

    __builtin_prefetch(elements_tags + *node_links);
    __builtin_prefetch(getDataByInternalId(*node_links));

    // Cast radius once instead of each time we check that candidate_dist <= radius_
    DistType radius_ = DistType(radius);
    for (size_t j = 0; j < links_num; j++) {
        idType *candidate_pos = node_links + j;
        idType candidate_id = *candidate_pos;

        // Pre-fetch the next candidate data into memory cache, to improve performance.
        idType *next_candidate_pos = node_links + j + 1;
        __builtin_prefetch(elements_tags + *next_candidate_pos);
        __builtin_prefetch(getDataByInternalId(*next_candidate_pos));

        if (elements_tags[candidate_id] == visited_tag || isInProcess(candidate_id))
            continue;
        elements_tags[candidate_id] = visited_tag;
        char *candidate_data = getDataByInternalId(candidate_id);

        DistType candidate_dist = this->dist_func(query_data, candidate_data, this->dim);
        if (candidate_dist < dyn_range) {
            candidate_set.emplace(-candidate_dist, candidate_id);

            // If the new candidate is in the requested radius, add it to the results set.
            if (candidate_dist <= radius_ &&
                (!has_marked_deleted || !isMarkedDeleted(candidate_id))) {
                results->emplace(getExternalLabel(candidate_id), candidate_dist);
            }
        }
    }
    // Pre-fetch the neighbours list of the top candidate (the one that is going
    // to be processed in the next iteration) into memory cache, to improve performance.
    __builtin_prefetch(getNodeNeighborsAtLevel(candidate_set.top().second, layer));
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
        DistType dist = this->dist_func(data_point, getDataByInternalId(ep_id), this->dim);
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

        lowerBound = processCandidate<has_marked_deleted>(
            curr_el_pair.second, data_point, layer, ef, visited_tag,
            visited_nodes_handler->getElementsTags(), top_candidates, candidate_set, lowerBound);
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
    while (top_candidates.size() > 0) {
        // the distance is saved negatively to have the queue ordered such that first is closer
        // (higher).
        queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
        top_candidates.pop();
    }

    while (queue_closest.size()) {
        if (return_list.size() >= M)
            break;
        pair<DistType, idType> current_pair = queue_closest.top();
        DistType candidate_to_query_dist = -current_pair.first;
        queue_closest.pop();
        bool good = true;

        // a candidate is "good" to become a neighbour, unless we find
        // another item that was already selected to the neighbours set which is closer
        // to both q and the candidate than the distance between the candidate and q.
        for (pair<DistType, idType> second_pair : return_list) {
            DistType candidate_to_selected_dist =
                this->dist_func(getDataByInternalId(second_pair.second),
                                getDataByInternalId(current_pair.second), this->dim);
            if (candidate_to_selected_dist < candidate_to_query_dist) {
                good = false;
                break;
            }
        }
        if (good) {
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
    idType *new_node_neighbors_list, idType *neighbor_neighbors_list,
    std::unique_lock<std::mutex> &node_lock, std::unique_lock<std::mutex> &neighbor_lock) {
    // Note - expect that node_lock and neighbor_lock are locked at that point.

    // Collect the existing neighbors and the new node as the neighbor's neighbors candidates.
    candidatesMaxHeap<DistType> candidates(this->allocator);
    // Add the new node along with the pre-calculated distance to the current neighbor,
    candidates.emplace(neighbor_data.first, new_node_id);

    idType selected_neighbor = neighbor_data.second;
    for (size_t j = 0; j < getNodeNeighborsCount(neighbor_neighbors_list); j++) {
        candidates.emplace(this->dist_func(getDataByInternalId(neighbor_neighbors_list[j]),
                                           getDataByInternalId(selected_neighbor), this->dim),
                           neighbor_neighbors_list[j]);
    }

    std::vector<idType> nodes_to_update;
    auto orig_candidates = candidates;

    // Candidates will store the newly selected neighbours (for the neighbor).
    size_t max_M_cur = level ? maxM_ : maxM0_;
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
    node_lock.unlock();
    neighbor_lock.unlock();

    nodes_to_update.push_back(selected_neighbor);
    nodes_to_update.push_back(new_node_id);

    std::sort(nodes_to_update.begin(), nodes_to_update.end());
    size_t nodes_to_update_count = nodes_to_update.size();
    std::unique_lock<std::mutex> locks[nodes_to_update_count];
    for (size_t i = 0; i < nodes_to_update_count; i++) {
        locks[i] = std::unique_lock<std::mutex>(element_neighbors_locks_[nodes_to_update[i]]);
    }

    auto *neighbour_incoming_edges = getIncomingEdgesPtr(selected_neighbor, level);
    size_t neighbor_neighbors_count = getNodeNeighborsCount(neighbor_neighbors_list);

    size_t neighbour_neighbours_idx = 0;
    bool update_cur_node_required = true;
    for (size_t i = 0; i < neighbor_neighbors_count; i++) {
        if (!std::binary_search(nodes_to_update.begin(), nodes_to_update.end(),
                                neighbor_neighbors_list[i])) {
            // The neighbor is not in the "to_update" nodes list - leave it as is.
            neighbor_neighbors_list[neighbour_neighbours_idx++] = neighbor_neighbors_list[i];
            continue;
        } else if (neighbor_neighbors_list[i] == new_node_id) {
            // The new node got into the neighbor's neighbours - this means there was an update in
            // another thread during between we released and reacquire the locks - leave it
            // as is.
            neighbor_neighbors_list[neighbour_neighbours_idx++] = neighbor_neighbors_list[i];
            update_cur_node_required = false;
            continue;
        }
        // Now we know that we are looking at a node to be removed from the neighbor's neighbors.
        auto removed_node = neighbor_neighbors_list[i];
        auto *removed_node_incoming_edges = getIncomingEdgesPtr(removed_node, level);
        // Perform the mutual update:
        // if the removed node id (the neighbour's neighbour to be removed)
        // wasn't pointing to the neighbour (i.e., the edge was uni-directional),
        // we should remove the current neighbor from the node's incoming edges.
        // otherwise, the edge turned from bidirectional to uni-directional, so we insert it to the
        // neighbour's incoming edges set. Note: we assume that every update is performed atomically
        // mutually, so it should be sufficient to look at the removed node's incoming edges set
        // alone.
        if (!removeIdFromList(*removed_node_incoming_edges, selected_neighbor)) {
            neighbour_incoming_edges->push_back(removed_node);
        }
    }

    size_t cur_node_neighbors_count = getNodeNeighborsCount(new_node_neighbors_list);
    if (update_cur_node_required && cur_node_neighbors_count < max_M_cur &&
        !isMarkedDeleted(new_node_id) && !isMarkedDeleted(selected_neighbor)) {
        // update the connection between the new node and the neighbor.
        new_node_neighbors_list[cur_node_neighbors_count++] = selected_neighbor;
        setNodeNeighborsCount(new_node_neighbors_list, cur_node_neighbors_count);
        if (cur_node_chosen && neighbour_neighbours_idx < max_M_cur) {
            // connection is mutual - both new node and the selected neighbor in each other's list.
            neighbor_neighbors_list[neighbour_neighbours_idx++] = new_node_id;
        } else {
            // unidirectional connection - put the new node in the neighbour's incoming edges.
            neighbour_incoming_edges->push_back(new_node_id);
        }
    }
    // Done updating the neighbor's neighbors.
    setNodeNeighborsCount(neighbor_neighbors_list, neighbour_neighbours_idx);
}

template <typename DataType, typename DistType>
idType HNSWIndex<DataType, DistType>::mutuallyConnectNewElement(
    idType new_node_id, candidatesMaxHeap<DistType> &top_candidates, size_t level) {

    // The maximum number of neighbors allowed for an existing neighbor (not new).
    size_t max_M_cur = level ? maxM_ : maxM0_;

    // Filter the top candidates to the selected neighbors by the algorithm heuristics.
    getNeighborsByHeuristic2(top_candidates, M_);
    assert(top_candidates.size() <= M_ &&
           "Should be not be more than M_ candidates returned by the heuristic");

    // Hold (distance_from_new_node_id, neighbor_id) pair for every selected neighbor.
    vecsim_stl::vector<std::pair<DistType, idType>> selected_neighbors(this->allocator);
    selected_neighbors.reserve(M_);
    while (!top_candidates.empty()) {
        selected_neighbors.push_back(top_candidates.top());
        top_candidates.pop();
    }

    // The closest vector that has found to be returned (and start the scan from it in the next
    // level).
    idType next_closest_entry_point = selected_neighbors.back().second;
    idType *new_node_neighbors_list = getNodeNeighborsAtLevel(new_node_id, level);
    assert(getNodeNeighborsCount(new_node_neighbors_list) == 0 &&
           "The newly inserted element should have blank link list");

    // Create the incoming edges for the new node in the current level.
    auto *incoming_edges = new (this->allocator) vecsim_stl::vector<idType>(this->allocator);
    setIncomingEdgesPtr(new_node_id, level, (void *)incoming_edges);

    for (auto &neighbor_data : selected_neighbors) {
        idType selected_neighbor = neighbor_data.second; // neighbor's id
        std::unique_lock<std::mutex> node_lock;
        std::unique_lock<std::mutex> neighbor_lock;
        idType lower_id = (new_node_id < selected_neighbor) ? new_node_id : selected_neighbor;
        if (lower_id == new_node_id) {
            node_lock = std::unique_lock<std::mutex>(element_neighbors_locks_[new_node_id]);
            neighbor_lock =
                std::unique_lock<std::mutex>(element_neighbors_locks_[selected_neighbor]);
        } else {
            neighbor_lock =
                std::unique_lock<std::mutex>(element_neighbors_locks_[selected_neighbor]);
            node_lock = std::unique_lock<std::mutex>(element_neighbors_locks_[new_node_id]);
        }

        // get the updated count - this may change between iterations due to releasing the lock.
        linkListSize cur_node_neighbors_count = getNodeNeighborsCount(new_node_neighbors_list);
        idType *neighbor_neighbors_list = getNodeNeighborsAtLevel(selected_neighbor, level);
        linkListSize neighbor_neighbors_count = getNodeNeighborsCount(neighbor_neighbors_list);

        // validations...
        assert(cur_node_neighbors_count <= max_M_cur && "Neighbors number exceeds limit");
        assert(selected_neighbor != new_node_id && "Trying to connect an element to itself");

        if (cur_node_neighbors_count == max_M_cur) {
            // The new node cannot add more neighbors
            break;
        }

        // If one of the two nodes has already deleted - skip the operation.
        if (isMarkedDeleted(new_node_id) || isMarkedDeleted(selected_neighbor)) {
            continue;
        }

        // if the neighbor's neighbors list has the capacity to add the new node, make the update
        // and finish.
        if (neighbor_neighbors_count < max_M_cur) {
            new_node_neighbors_list[cur_node_neighbors_count] = selected_neighbor;
            setNodeNeighborsCount(new_node_neighbors_list, cur_node_neighbors_count + 1);
            neighbor_neighbors_list[neighbor_neighbors_count] = new_node_id;
            setNodeNeighborsCount(neighbor_neighbors_list, neighbor_neighbors_count + 1);
            continue;
        }

        // Otherwise - we need to re-evaluate the neighbor's neighbors.
        // We collect all the existing neighbors and the new node as candidates, and mutually update
        // the neighbor's neighbors.
        revisitNeighborConnections(level, new_node_id, neighbor_data, new_node_neighbors_list,
                                   neighbor_neighbors_list, node_lock, neighbor_lock);
    }
    return next_closest_entry_point;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::repairConnectionsForDeletion(
    idType element_internal_id, idType neighbour_id, idType *neighbours,
    idType *neighbour_neighbours, size_t level, vecsim_stl::vector<bool> &neighbours_bitmap) {

    // put the deleted element's neighbours in the candidates.
    candidatesMaxHeap<DistType> candidates(this->allocator);
    linkListSize neighbours_count = getNodeNeighborsCount(neighbours);
    for (size_t j = 0; j < neighbours_count; j++) {
        // Don't put the neighbor itself in his own candidates
        if (neighbours[j] == neighbour_id) {
            continue;
        }
        candidates.emplace(this->dist_func(getDataByInternalId(neighbours[j]),
                                           getDataByInternalId(neighbour_id), this->dim),
                           neighbours[j]);
    }

    // add the deleted element's neighbour's original neighbors in the candidates.
    vecsim_stl::vector<bool> neighbour_orig_neighbours_set(cur_element_count, false,
                                                           this->allocator);
    linkListSize neighbour_neighbours_count = getNodeNeighborsCount(neighbour_neighbours);

    for (size_t j = 0; j < neighbour_neighbours_count; j++) {
        neighbour_orig_neighbours_set[neighbour_neighbours[j]] = true;
        // Don't add the removed element to the candidates, nor nodes that are already in the
        // candidates set.
        if (neighbours_bitmap[neighbour_neighbours[j]] ||
            neighbour_neighbours[j] == element_internal_id) {
            continue;
        }
        candidates.emplace(this->dist_func(getDataByInternalId(neighbour_id),
                                           getDataByInternalId(neighbour_neighbours[j]), this->dim),
                           neighbour_neighbours[j]);
    }

    size_t Mcurmax = level ? maxM_ : maxM0_;
    size_t removed_links_num;
    idType removed_links[neighbour_neighbours_count];
    removeExtraLinks(candidates, Mcurmax, neighbour_neighbours, neighbour_orig_neighbours_set,
                     removed_links, &removed_links_num);

    // remove neighbour id from the incoming list of nodes for his
    // neighbours that were chosen to remove
    auto *neighbour_incoming_edges = getIncomingEdgesPtr(neighbour_id, level);

    for (size_t i = 0; i < removed_links_num; i++) {
        idType node_id = removed_links[i];
        auto *node_incoming_edges = getIncomingEdgesPtr(node_id, level);

        // if the node id (the neighbour's neighbour to be removed)
        // wasn't pointing to the neighbour (edge was one directional),
        // we should remove it from the node's incoming edges.
        // otherwise, edge turned from bidirectional to one directional,
        // and it should be saved in the neighbor's incoming edges.
        if (!removeIdFromList(*node_incoming_edges, neighbour_id)) {
            neighbour_incoming_edges->push_back(node_id);
        }
    }

    // updates for the new edges created
    linkListSize updated_links_num = getNodeNeighborsCount(neighbour_neighbours);
    for (size_t i = 0; i < updated_links_num; i++) {
        idType node_id = neighbour_neighbours[i];
        if (!neighbour_orig_neighbours_set[node_id]) {
            auto *node_incoming_edges = getIncomingEdgesPtr(node_id, level);
            // if the node has an edge to the neighbour as well, remove it
            // from the incoming nodes of the neighbour
            // otherwise, need to update the edge as incoming.
            idType *node_links = getNodeNeighborsAtLevel(node_id, level);
            unsigned short node_links_size = getNodeNeighborsCount(node_links);

            bool bidirectional_edge = false;
            for (size_t j = 0; j < node_links_size; j++) {
                if (node_links[j] == neighbour_id) {
                    // Swap the last element with the current one (equivalent to removing the
                    // neighbor from the list) - this should always succeed and return true.
                    removeIdFromList(*neighbour_incoming_edges, node_id);
                    bidirectional_edge = true;
                    break;
                }
            }
            if (!bidirectional_edge) {
                node_incoming_edges->push_back(neighbour_id);
            }
        }
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::replaceEntryPoint() {
    idType old_entry = entrypoint_node_;
    // Sets an (arbitrary) new entry point, after deleting the current entry point.
    while (old_entry == entrypoint_node_) {
        // Use volatile for this variable, so that in case we would have to busy wait for this
        // element to finish its indexing, the compiler will not use optimizations. Otherwise,
        // the compiler might evaluate 'isInProcess(candidate_in_process)' once instead of calling
        // it multiple times in a busy wait manner, and we'll run into an infinite loop if the
        // candidate is in process when we reach the loop.
        volatile idType candidate_in_process = INVALID_ID;
        {
            // Go over the entry point's neighbors at the top level.
            std::unique_lock<std::mutex> lock(this->element_neighbors_locks_[entrypoint_node_]);
            idType *top_level_list = getNodeNeighborsAtLevel(old_entry, max_level_);
            auto neighbors_count = getNodeNeighborsCount(top_level_list);
            // Tries to set the (arbitrary) first neighbor as the entry point which is not deleted,
            // if exists.
            for (size_t i = 0; i < neighbors_count; i++) {
                if (!isMarkedDeleted(top_level_list[i])) {
                    if (!isInProcess(top_level_list[i])) {
                        entrypoint_node_ = top_level_list[i];
                        return;
                    } else {
                        // Store this candidate which is currently being inserted into the graph in
                        // case we won't find other candidate at the top level.
                        candidate_in_process = top_level_list[i];
                    }
                }
            }
        }
        // If there is no neighbors in the current level, check for any vector at
        // this level to be the new entry point.
        for (idType cur_id = 0; cur_id < cur_element_count; cur_id++) {
            if (element_levels_[cur_id] == max_level_ && cur_id != old_entry &&
                !isMarkedDeleted(cur_id)) {
                // Found a non element in the current max level.
                if (!isInProcess(cur_id)) {
                    entrypoint_node_ = cur_id;
                    return;
                } else if (candidate_in_process == INVALID_ID) {
                    // This element is still in process, and there hasn't been another candidate in
                    // process that has found in this level.
                    candidate_in_process = cur_id;
                }
            }
        }
        // If we only found candidates which are in process at this level, do busy wait until they
        // are done being processed (this should happen in very rare cases...). Since
        // candidate_in_process was declared volatile, we can be sure that isInProcess is called in
        // every iteration.
        if (candidate_in_process != INVALID_ID) {
            while (isInProcess(candidate_in_process))
                ;
            entrypoint_node_ = candidate_in_process;
            return;
        }
        // If we didn't find any vector at the top level, decrease the max_level_ and try again,
        // until we find a new entry point, or the index is empty.
        assert(old_entry == entrypoint_node_);
        max_level_--;
        if ((int)max_level_ < 0) {
            max_level_ = HNSW_INVALID_LEVEL;
            entrypoint_node_ = INVALID_ID;
        }
    }
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
void HNSWIndex<DataType, DistType>::SwapLastIdWithDeletedId(idType element_internal_id) {
    // Swap label - this is relevant when the last element's label exists (it is not marked as
    // deleted). For inplace delete, this is always the case.
    if (!has_marked_deleted || !isMarkedDeleted(cur_element_count)) {
        replaceIdOfLabel(getExternalLabel(cur_element_count), element_internal_id,
                         cur_element_count);
    }

    // Swap neighbours
    size_t last_element_top_level = element_levels_[cur_element_count];
    for (size_t level = 0; level <= last_element_top_level; level++) {
        idType *neighbours = getNodeNeighborsAtLevel(cur_element_count, level);
        linkListSize neighbours_count = getNodeNeighborsCount(neighbours);

        // Go over the neighbours that also points back to the last element whose is going to
        // change, and update the id.
        for (size_t i = 0; i < neighbours_count; i++) {
            idType neighbour_id = neighbours[i];
            idType *neighbour_neighbours = getNodeNeighborsAtLevel(neighbour_id, level);
            linkListSize neighbour_neighbours_count = getNodeNeighborsCount(neighbour_neighbours);

            bool bidirectional_edge = false;
            for (size_t j = 0; j < neighbour_neighbours_count; j++) {
                // if the edge is bidirectional, update for this neighbor
                if (neighbour_neighbours[j] == cur_element_count) {
                    bidirectional_edge = true;
                    neighbour_neighbours[j] = element_internal_id;
                    break;
                }
            }

            // If this edge is uni-directional, we should update the id in the neighbor's
            // incoming edges.
            if (!bidirectional_edge) {
                auto *neighbour_incoming_edges = getIncomingEdgesPtr(neighbour_id, level);
                // This should always succeed and return true.
                removeIdFromList(*neighbour_incoming_edges, cur_element_count);
                neighbour_incoming_edges->push_back(element_internal_id);
            }
        }

        // Next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // updates.
        auto *incoming_edges = getIncomingEdgesPtr(cur_element_count, level);
        for (auto incoming_edge : *incoming_edges) {
            idType *incoming_neighbour_neighbours = getNodeNeighborsAtLevel(incoming_edge, level);
            linkListSize incoming_neighbour_neighbours_count =
                getNodeNeighborsCount(incoming_neighbour_neighbours);
            for (size_t j = 0; j < incoming_neighbour_neighbours_count; j++) {
                if (incoming_neighbour_neighbours[j] == cur_element_count) {
                    incoming_neighbour_neighbours[j] = element_internal_id;
                    break;
                }
            }
        }
    }

    // Swap the last_id level 0 data, and invalidate the deleted id's data.
    memcpy(data_level0_memory_ + element_internal_id * size_data_per_element_ + offsetLevel0_,
           data_level0_memory_ + cur_element_count * size_data_per_element_ + offsetLevel0_,
           size_data_per_element_);
    memset(data_level0_memory_ + cur_element_count * size_data_per_element_ + offsetLevel0_, 0,
           size_data_per_element_);

    // Swap pointer of higher levels links.
    linkLists_[element_internal_id] = linkLists_[cur_element_count];
    linkLists_[cur_element_count] = nullptr;

    // Swap top element level.
    element_levels_[element_internal_id] = element_levels_[cur_element_count];
    element_levels_[cur_element_count] = HNSW_INVALID_LEVEL;

    if (cur_element_count == this->entrypoint_node_) {
        this->entrypoint_node_ = element_internal_id;
    }
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
        std::unique_lock<std::mutex> lock(element_neighbors_locks_[bestCand]);
        idType *node_links = getNodeNeighborsAtNonBaseLevel(bestCand, level);
        linkListSize links_count = getNodeNeighborsCount(node_links);

        for (int i = 0; i < links_count; i++) {
            idType candidate = node_links[i];
            assert(candidate < this->cur_element_count && "candidate error: out of index range");
            if (isInProcess(candidate)) {
                continue;
            }
            DistType d = this->dist_func(vector_data, getDataByInternalId(candidate), this->dim);
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
    } while (changed);
    if (!running_query) {
        bestCand = bestNonDeletedCand;
    }
}

template <typename DataType, typename DistType>
vecsim_stl::vector<graphNodeType>
HNSWIndex<DataType, DistType>::safeCollectAllNodeIncomingNeighbors(idType node_id,
                                                                   size_t node_top_level) {
    vecsim_stl::vector<graphNodeType> incoming_neighbors(this->allocator);

    for (size_t level = 0; level <= node_top_level; level++) {
        // Save the node neighbor's in the current level while holding its neighbors lock.
        std::vector<idType> neighbors_copy;
        std::unique_lock<std::mutex> element_lock(element_neighbors_locks_[node_id]);
        auto *neighbours = getNodeNeighborsAtLevel(node_id, level);
        unsigned short neighbours_count = getNodeNeighborsCount(neighbours);
        // Store the deleted element's neighbours.
        neighbors_copy.assign(neighbours, neighbours + neighbours_count);
        element_lock.unlock();

        // Go over the neighbours and collect tho ones that also points back to the removed node.
        for (auto neighbour_id : neighbors_copy) {
            // Hold the neighbor's lock while we are going over its neighbors.
            std::unique_lock<std::mutex> neighbor_lock(element_neighbors_locks_[neighbour_id]);
            auto *neighbour_neighbours = getNodeNeighborsAtLevel(neighbour_id, level);
            unsigned short neighbour_neighbours_count = getNodeNeighborsCount(neighbour_neighbours);
            for (size_t j = 0; j < neighbour_neighbours_count; j++) {
                // A bidirectional edge was found - this connection should be repaired.
                if (neighbour_neighbours[j] == node_id) {
                    incoming_neighbors.emplace_back(neighbour_id, (ushort)level);
                    break;
                }
            }
        }

        // Next, collect the rest of incoming edges (the ones that are not bidirectional) in the
        // current level to repair them.
        element_lock.lock();
        auto *incoming_edges = getIncomingEdgesPtr(node_id, level);
        // Note that the deleted element might be in the process of indexing into the graph in the
        // meantime (in async mode). Since the incoming_edges lists in every level are allocated
        // while the element is being indexed into that level (in lazy mode), we may find ourselves
        // in a situation where the incoming edges was not allocated yet in this level (but we do
        // guarantee that the pointer is NULL in that case). In which case, we just continue. We
        // also validate that we won't add new edges to a deleted node later on.
        if (!incoming_edges)
            continue;
        for (auto incoming_edge : *incoming_edges) {
            incoming_neighbors.emplace_back(incoming_edge, (ushort)level);
        }
    }
    return incoming_neighbors;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::resizeIndexInternal(size_t new_max_elements) {
    element_levels_.resize(new_max_elements);
    element_levels_.shrink_to_fit();
    resizeLabelLookup(new_max_elements);
    visited_nodes_handler_pool.resize(new_max_elements);
    vecsim_stl::vector<std::mutex>(new_max_elements, this->allocator)
        .swap(element_neighbors_locks_);
    // Reallocate base layer
    char *data_level0_memory_new = (char *)this->allocator->reallocate(
        data_level0_memory_, new_max_elements * size_data_per_element_);
    if (data_level0_memory_new == nullptr)
        throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
    data_level0_memory_ = data_level0_memory_new;

    // Reallocate all other layers
    char **linkLists_new =
        (char **)this->allocator->reallocate(linkLists_, sizeof(void *) * new_max_elements);
    if (linkLists_new == nullptr)
        throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
    linkLists_ = linkLists_new;

    max_elements_ = new_max_elements;
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
    std::unique_lock<std::mutex> locks[nodes_to_update_count];
    for (size_t i = 0; i < nodes_to_update_count; i++) {
        locks[i] = std::unique_lock<std::mutex>(element_neighbors_locks_[nodes_to_update[i]]);
    }

    idType *node_neighbors = getNodeNeighborsAtLevel(node_id, level);
    linkListSize node_neighbors_count = getNodeNeighborsCount(node_neighbors);
    auto *node_incoming_edges = getIncomingEdgesPtr(node_id, level);

    // Perform mutual updates: go over the node's neighbors and overwrite the neighbors to remove
    // that are still exist.
    size_t node_neighbors_idx = 0;
    for (size_t i = 0; i < node_neighbors_count; i++) {
        if (!std::binary_search(nodes_to_update.begin(), nodes_to_update.end(),
                                node_neighbors[i])) {
            // The repaired node added a new neighbor that we didn't account for before in the
            // meantime - leave it as is.
            node_neighbors[node_neighbors_idx++] = node_neighbors[i];
            continue;
        }
        // Check if the current neighbor is in the chosen neighbors list, and remove it from there
        // if so.
        if (removeIdFromList(chosen_neighbors, node_neighbors[i])) {
            // A chosen neighbor is already connected to the node - leave it as is.
            node_neighbors[node_neighbors_idx++] = node_neighbors[i];
            continue;
        }
        // Now we know that we are looking at a neighbor that needs to be removed.
        auto removed_node = node_neighbors[i];
        auto *removed_node_incoming_edges = getIncomingEdgesPtr(removed_node, level);
        // Perform the mutual update:
        // if the removed node id (the node's neighbour to be removed)
        // wasn't pointing to the node (i.e., the edge was uni-directional),
        // we should remove the current neighbor from the node's incoming edges.
        // otherwise, the edge turned from bidirectional to uni-directional, so we insert it to the
        // neighbour's incoming edges set. Note: we assume that every update is performed atomically
        // mutually, so it should be sufficient to look at the removed node's incoming edges set
        // alone.
        if (!removeIdFromList(*removed_node_incoming_edges, node_id)) {
            node_incoming_edges->push_back(removed_node);
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
        auto *new_neighbor_incoming_edges = getIncomingEdgesPtr(chosen_id, level);
        node_neighbors[node_neighbors_idx++] = chosen_id;
        // If the node is in the chosen new node incoming edges, there is a unidirectional
        // connection from the chosen node to the repaired node that turns into bidirectional. Then,
        // remove it from the incoming edges set. Otherwise, the edge is created unidirectional, so
        // we add it to the unidirectional edges set. Note: we assume that all updates occur
        // mutually and atomically, then can rely on this assumption.
        if (!removeIdFromList(*node_incoming_edges, chosen_id)) {
            new_neighbor_incoming_edges->push_back(node_id);
        }
    }
    // Done updating the node's neighbors.
    setNodeNeighborsCount(node_neighbors, node_neighbors_idx);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::repairNodeConnections(idType node_id, size_t level) {

    candidatesMaxHeap<DistType> neighbors_candidates(this->allocator);
    // Use bitmaps for fast accesses:
    // node_orig_neighbours_set is used to diffrentiate between the neighboes that will *not* be
    // selected by the heuritics - only the ones that were originally neighbors should be removed.
    vecsim_stl::vector<bool> node_orig_neighbours_set(max_elements_, false, this->allocator);
    // neighbors_candidates_set is used to store the nodes that were already collected as
    // candidates, so we will not collect them again as candidates if we run into them from another
    // path.
    vecsim_stl::vector<bool> neighbors_candidates_set(max_elements_, false, this->allocator);
    vecsim_stl::vector<idType> deleted_neighbors(this->allocator);

    // Go over the repaired node neighbors, collect the non-deleted ones to be neighbors candidates
    // after the repair as well.
    {
        std::unique_lock<std::mutex> node_lock(element_neighbors_locks_[node_id]);
        idType *node_neighbors = getNodeNeighborsAtLevel(node_id, level);
        linkListSize node_neighbors_count = getNodeNeighborsCount(node_neighbors);
        for (size_t j = 0; j < node_neighbors_count; j++) {
            node_orig_neighbours_set[node_neighbors[j]] = true;
            // Don't add the removed element to the candidates.
            if (isMarkedDeleted(node_neighbors[j])) {
                deleted_neighbors.push_back(node_neighbors[j]);
                continue;
            }
            neighbors_candidates_set[node_neighbors[j]] = true;
            neighbors_candidates.emplace(this->dist_func(getDataByInternalId(node_id),
                                                         getDataByInternalId(node_neighbors[j]),
                                                         this->dim),
                                         node_neighbors[j]);
        }
    }
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

        std::unique_lock<std::mutex> neighbor_lock(
            this->element_neighbors_locks_[deleted_neighbor_id]);
        idType *neighbor_neighbours = getNodeNeighborsAtLevel(deleted_neighbor_id, level);
        linkListSize neighbor_neighbours_count = getNodeNeighborsCount(neighbor_neighbours);

        for (size_t j = 0; j < neighbor_neighbours_count; j++) {
            // Don't add removed elements to the candidates, nor nodes that are already in the
            // candidates set, nor the original node to repair itself.
            if (isMarkedDeleted(neighbor_neighbours[j]) ||
                neighbors_candidates_set[neighbor_neighbours[j]] ||
                neighbor_neighbours[j] == node_id) {
                continue;
            }
            neighbors_candidates_set[neighbor_neighbours[j]] = true;
            neighbors_candidates.emplace(
                this->dist_func(getDataByInternalId(node_id),
                                getDataByInternalId(neighbor_neighbours[j]), this->dim),
                neighbor_neighbours[j]);
        }
    }

    // Copy the original candidates, and run the heuristics. Afterwards, neighbors_candidates will
    // store the newly selected neighbours (for the node), while candidates which were originally
    // neighbors and are not going to be selected, are going to be removed.
    auto orig_candidates = neighbors_candidates;
    size_t max_M_cur = level ? maxM_ : maxM0_;
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
      max_elements_(params->initialCapacity), element_levels_(max_elements_, this->allocator),
      visited_nodes_handler_pool(pool_initial_size, max_elements_, this->allocator),
      element_neighbors_locks_(max_elements_, this->allocator) {
    size_t M = params->M ? params->M : HNSW_DEFAULT_M;
    if (M > UINT16_MAX / 2)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;

    size_t ef_construction = params->efConstruction ? params->efConstruction : HNSW_DEFAULT_EF_C;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = params->efRuntime ? params->efRuntime : HNSW_DEFAULT_EF_RT;
    epsilon_ = params->epsilon > 0.0 ? params->epsilon : HNSW_DEFAULT_EPSILON;

    cur_element_count = 0;
    num_marked_deleted = 0;

    // initializations for special treatment of the first node
    entrypoint_node_ = INVALID_ID;
    max_level_ = HNSW_INVALID_LEVEL;

    if (M <= 1)
        throw std::runtime_error("HNSW index parameter M cannot be 1");
    mult_ = 1 / log(1.0 * M_);
    level_generator_.seed(random_seed);

    // data_level0_memory will look like this:
    // | ---2--- | -----2----- | -----4*M0----------- | ---------8-------- |-data_size-| ---8--- |
    // | <flags> | <links_len> | <link_1> <link_2>... |<incoming_links_ptr>|   <data>   | <label> |

    size_links_level0_ =
        sizeof(linkListSize) + sizeof(elementFlags) + maxM0_ * sizeof(idType) + sizeof(void *);
    size_data_per_element_ = size_links_level0_ + this->data_size + sizeof(labelType);

    // No need to test for overflow because we passed the test for size_links_level0_ and this is
    // less.
    incoming_links_offset0 = maxM0_ * sizeof(idType) + sizeof(linkListSize) + sizeof(elementFlags);
    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + this->data_size;
    offsetLevel0_ = 0;

    data_level0_memory_ =
        (char *)this->allocator->callocate(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory");

    linkLists_ = (char **)this->allocator->callocate(sizeof(void *) * max_elements_);
    if (linkLists_ == nullptr)
        throw std::runtime_error("Not enough memory: HNSWIndex failed to allocate linklists");

    // The i-th entry in linkLists array points to max_level[i] (continuous)
    // chunks of memory, each one will look like this:
    // | -----2----- | -----4*M-------------- | ----------8--------- |
    // | <links_len> | <link_1> <link_2> ...  | <incoming_links_ptr> |
    size_links_per_element_ = sizeof(linkListSize) + maxM_ * sizeof(idType) + sizeof(void *);
    // No need to test for overflow because we passed the test for incoming_links_offset0 and this
    // is less.
    incoming_links_offset = maxM_ * sizeof(idType) + sizeof(linkListSize);
}

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::~HNSWIndex() {
    for (idType id = 0; id < cur_element_count; id++) {
        for (size_t level = 0; level <= element_levels_[id]; level++) {
            delete getIncomingEdgesPtr(id, level);
        }
        if (element_levels_[id] > 0)
            this->allocator->free_allocation(linkLists_[id]);
    }

    this->allocator->free_allocation(linkLists_);
    this->allocator->free_allocation(data_level0_memory_);
}

/**
 * Index API functions
 */
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::increaseCapacity() {
    size_t vectors_to_add = this->blockSize - max_elements_ % this->blockSize;
    resizeIndexInternal(max_elements_ + vectors_to_add);
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
void HNSWIndex<DataType, DistType>::removeAndSwap(idType internalId) {

    // Delete the incoming edges sets for this element at every level.
    size_t element_top_level = element_levels_[internalId];
    for (size_t level = 0; level <= element_top_level; level++) {
        auto *incoming_edges = getIncomingEdgesPtr(internalId, level);
        assert(!has_marked_deleted || incoming_edges->size() == 0);
        delete incoming_edges;
    }

    if (has_marked_deleted) {
        // If the index allows marking vectors as deleted (as in tiered HNSW), the id to remove
        // cannot be the entry point, as it should have been replaced upon marking it as deleted.
        assert(entrypoint_node_ != internalId);
    } else if (internalId == entrypoint_node_) {
        // For inplace delete, we replace entry point now.
        assert(element_top_level == max_level_);
        replaceEntryPoint();
    }

    // We can say now that the element has removed completely from index.
    --cur_element_count;
    if (has_marked_deleted) {
        --num_marked_deleted;
    }

    // Remove the deleted id form the relevant incoming edges sets in which it appears.
    for (size_t level = 0; level <= element_top_level; level++) {
        auto *neighbours = getNodeNeighborsAtLevel(internalId, level);
        auto neighbours_count = getNodeNeighborsCount(neighbours);
        for (size_t i = 0; i < neighbours_count; i++) {
            idType neighbour_id = neighbours[i];
            // This should always succeed, since every outgoing edge should be unidirectional at
            // this point (after all the repair jobs are done).
            auto *neighbour_incoming_edges = getIncomingEdgesPtr(neighbour_id, level);
            removeIdFromList(*neighbour_incoming_edges, internalId);
        }
    }

    // Swap the last id with the deleted one, and invalidate the last id data.
    if (element_levels_[internalId] > 0) {
        this->allocator->free_allocation(linkLists_[internalId]);
        linkLists_[internalId] = nullptr;
    }
    if (cur_element_count == internalId) {
        // We're deleting the last internal id, just invalidate data without swapping.
        memset(data_level0_memory_ + cur_element_count * size_data_per_element_ + offsetLevel0_, 0,
               size_data_per_element_);
    } else {
        SwapLastIdWithDeletedId<has_marked_deleted>(internalId);
    }

    // If we need to free a complete block and there is at least one block between the
    // capacity and the size.
    if (cur_element_count % this->blockSize == 0 &&
        cur_element_count + this->blockSize <= max_elements_) {

        // Check if the capacity is aligned to block size.
        size_t extra_space_to_free = max_elements_ % this->blockSize;

        // Remove one block from the capacity.
        this->resizeIndexInternal(max_elements_ - this->blockSize - extra_space_to_free);
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
    size_t element_top_level = element_levels_[element_internal_id];
    for (size_t level = 0; level <= element_top_level; level++) {
        idType *neighbours = getNodeNeighborsAtLevel(element_internal_id, level);
        linkListSize neighbours_count = getNodeNeighborsCount(neighbours);
        // Reset the neighbours' bitmap for the current level.
        neighbours_bitmap.assign(cur_element_count, false);
        // Store the deleted element's neighbours set in a bitmap for fast access.
        for (size_t j = 0; j < neighbours_count; j++) {
            neighbours_bitmap[neighbours[j]] = true;
        }
        // Go over the neighbours that also points back to the removed point and make a local
        // repair.
        for (size_t i = 0; i < neighbours_count; i++) {
            idType neighbour_id = neighbours[i];
            idType *neighbour_neighbours = getNodeNeighborsAtLevel(neighbour_id, level);
            linkListSize neighbour_neighbours_count = getNodeNeighborsCount(neighbour_neighbours);

            bool bidirectional_edge = false;
            for (size_t j = 0; j < neighbour_neighbours_count; j++) {
                // If the edge is bidirectional, do repair for this neighbor.
                if (neighbour_neighbours[j] == element_internal_id) {
                    bidirectional_edge = true;
                    repairConnectionsForDeletion(element_internal_id, neighbour_id, neighbours,
                                                 neighbour_neighbours, level, neighbours_bitmap);
                    break;
                }
            }

            // If this edge is uni-directional, we should remove the element from the neighbor's
            // incoming edges.
            if (!bidirectional_edge) {
                auto *neighbour_incoming_edges = getIncomingEdgesPtr(neighbour_id, level);
                // This should always return true (remove should succeed).
                removeIdFromList(*neighbour_incoming_edges, element_internal_id);
            }
        }

        // Next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // repairs.
        auto *incoming_edges = getIncomingEdgesPtr(element_internal_id, level);
        for (auto incoming_edge : *incoming_edges) {
            idType *incoming_node_neighbours = getNodeNeighborsAtLevel(incoming_edge, level);
            repairConnectionsForDeletion(element_internal_id, incoming_edge, neighbours,
                                         incoming_node_neighbours, level, neighbours_bitmap);
        }
    }
    // Finally, remove the element from the index and make a swap with the last internal id to
    // avoid fragmentation and reclaim memory when needed.
    removeAndSwap<false>(element_internal_id);
}

// Store the new element in the global data structures and keep the new state. In multithreaded
// scenario, the index data guard should be held by the caller (exclusive lock).
template <typename DataType, typename DistType>
AddVectorCtx HNSWIndex<DataType, DistType>::storeNewElement(labelType label) {
    AddVectorCtx state{};

    // Choose randomly the maximum level in which the new element will be in the index.
    state.elementMaxLevel = getRandomLevel(mult_);

    // Access and update the index global data structures with the new element meta-data.
    state.newElementId = cur_element_count++;
    assert(indexCapacity() >= indexSize());
    // Reset the data (and meta-data) for id=state.newElementId in the index.
    memset(data_level0_memory_ + state.newElementId * size_data_per_element_ + offsetLevel0_, 0,
           size_data_per_element_);
    // We mark id as in process *before* we set it in the label lookup, otherwise we might check
    // that the label exist with safeCheckIfLabelExistsInIndex and see that IN_PROCESS flag is
    // clear.
    markInProcess(state.newElementId);
    setVectorId(label, state.newElementId);
    element_levels_[state.newElementId] = state.elementMaxLevel;
    // Allocate memory for the links in higher levels and initialize this memory to zeros. The
    // reason for doing it here is that we might mark this vector as deleted BEFORE we finish its
    // indexing. In that case, we will collect the incoming edges to this element in every level,
    // and try to access its link lists in higher levels. Therefore, we allocate it here and
    // initialize it with zeros, (otherwise we might crash...)
    if (state.elementMaxLevel > 0) {
        linkLists_[state.newElementId] =
            (char *)this->allocator->callocate(size_links_per_element_ * state.elementMaxLevel);
        if (linkLists_[state.newElementId] == nullptr) {
            this->log(
                "Error - allocating memory for links in higher level failed due to low memory");
            throw std::runtime_error("VecSim index low memory error");
        }
    }

    state.currMaxLevel = (int)max_level_;
    state.currEntryPoint = entrypoint_node_;
    if (state.elementMaxLevel > state.currMaxLevel) {
        if (entrypoint_node_ == INVALID_ID && max_level_ != HNSW_INVALID_LEVEL) {
            throw std::runtime_error("Internal error - inserting the first element to the graph,"
                                     " but the current max level is not INVALID");
        }
        // If the new elements max level is higher than the maximum level the currently exists in
        // the graph, update the max level and set the new element as entry point.
        entrypoint_node_ = state.newElementId;
        max_level_ = state.elementMaxLevel;
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
        if (indexSize() == indexCapacity()) {
            increaseCapacity();
        }
        state = storeNewElement(label);
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
    // Initialisation of the vector data and its label.
    setExternalLabel(new_element_id, label);
    memcpy(getDataByInternalId(new_element_id), vector_data, this->data_size);

    // Start scanning the graph from the current entry point.
    idType curr_element = prev_entry_point;
    // This condition only means that we are not inserting the first (non-deleted) element.
    if (curr_element != INVALID_ID) {
        DistType cur_dist = std::numeric_limits<DistType>::max();
        if (element_max_level < prev_max_level) {
            cur_dist = this->dist_func(vector_data, getDataByInternalId(curr_element), this->dim);
            for (int level = prev_max_level; level > element_max_level; level--) {
                // this is done for the levels which are above the max level
                // to which we are going to insert the new element. We do
                // a greedy search in the graph starting from the entry point
                // at each level, and move on with the closest element we can find.
                // When there is no improvement to do, we take a step down.
                greedySearchLevel<false>(vector_data, level, curr_element, cur_dist);
            }
        }

        auto max_common_level = std::min(element_max_level, prev_max_level);
        for (int level = max_common_level; (int)level >= 0; level--) {
            candidatesMaxHeap<DistType> top_candidates =
                searchLayer<false>(curr_element, vector_data, level, ef_construction_);
            curr_element = mutuallyConnectNewElement(new_element_id, top_candidates, level);
        }

        if (element_max_level > prev_max_level) {
            // Create the incoming edges sets for the new levels.
            for (int level_idx = prev_max_level + 1; level_idx <= element_max_level; level_idx++) {
                auto *incoming_edges =
                    new (this->allocator) vecsim_stl::vector<idType>(this->allocator);
                setIncomingEdgesPtr(new_element_id, level_idx, incoming_edges);
            }
        }
    } else {
        // Inserting the first (non-deleted) element to the graph - only need to allocate incoming
        // neighbors sets without creating any connections.
        for (int level_idx = 0; level_idx <= element_max_level; level_idx++) {
            auto *incoming_edges =
                new (this->allocator) vecsim_stl::vector<idType>(this->allocator);
            setIncomingEdgesPtr(new_element_id, level_idx, incoming_edges);
        }
    }
    unmarkInProcess(new_element_id);
    if (auxiliaryCtx == nullptr && state.currMaxLevel < state.elementMaxLevel) {
        // No external auxiliaryCtx, so it's this function responsibility to release the lock.
        this->unlockIndexDataGuard();
    }
}

template <typename DataType, typename DistType>
auto HNSWIndex<DataType, DistType>::safeGetEntryPointState() const {
    std::shared_lock<std::shared_mutex> lock(index_data_guard_);
    return std::make_pair(entrypoint_node_, max_level_);
}

template <typename DataType, typename DistType>
idType HNSWIndex<DataType, DistType>::searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                                          VecSimQueryResult_Code *rc) const {
    *rc = VecSim_QueryResult_OK;

    auto [curr_element, max_level] = safeGetEntryPointState();
    if (curr_element == INVALID_ID)
        return curr_element; // index is empty.

    DistType cur_dist = this->dist_func(query_data, getDataByInternalId(curr_element), this->dim);
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
        DistType dist = this->dist_func(data_point, getDataByInternalId(ep_id), this->dim);
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

        lowerBound = processCandidate<has_marked_deleted>(
            curr_el_pair.second, data_point, 0, ef, visited_tag,
            visited_nodes_handler->getElementsTags(), *top_candidates, candidate_set, lowerBound);
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
    this->last_mode = STANDARD_KNN;

    if (cur_element_count == 0 || k == 0) {
        rl.code = VecSim_QueryResult_OK;
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    void *timeoutCtx = nullptr;

    // Get original efRuntime and store it.
    size_t ef = ef_;

    if (queryParams) {
        timeoutCtx = queryParams->timeoutCtx;
        if (queryParams->hnswRuntimeParams.efRuntime != 0) {
            ef = queryParams->hnswRuntimeParams.efRuntime;
        }
    }

    idType bottom_layer_ep = searchBottomLayerEP(query_data, timeoutCtx, &rl.code);
    if (VecSim_OK != rl.code) {
        return rl;
    } else if (bottom_layer_ep == INVALID_ID) {
        // Although we checked that the index is not empty (cur_element_count == 0), it might be
        // that another thread deleted all the elements or didn't finish inserting the first element
        // yet. Anyway, we observed that the index is empty, so we return an empty result list.
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    // We now oun the results heap, we need to free (delete) it when we done
    candidatesLabelsMaxHeap<DistType> *results;
    if (this->num_marked_deleted) {
        results = searchBottomLayer_WithTimeout<true>(bottom_layer_ep, query_data, std::max(ef, k),
                                                      k, timeoutCtx, &rl.code);
    } else {
        results = searchBottomLayer_WithTimeout<false>(bottom_layer_ep, query_data, std::max(ef, k),
                                                       k, timeoutCtx, &rl.code);
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
    idType ep_id, const void *data_point, double epsilon, double radius, void *timeoutCtx,
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
        ep_dist = this->dist_func(data_point, getDataByInternalId(ep_id), this->dim);
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

    // Cast radius once instead of each time we check that -curr_el_pair.first >= radius_.
    DistType radius_ = DistType(radius);
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
        if (-curr_el_pair.first < dynamic_range && -curr_el_pair.first >= radius_) {
            dynamic_range = -curr_el_pair.first;
            dynamic_range_search_boundaries = dynamic_range * (1.0 + epsilon);
        }

        // Go over the candidate neighbours, add them to the candidates list if they are within the
        // epsilon environment of the dynamic range, and add them to the results if they are in the
        // requested radius.
        // Here we send the radius as double to match the function arguments type.
        processCandidate_RangeSearch<has_marked_deleted>(
            curr_el_pair.second, data_point, 0, epsilon, visited_tag,
            visited_nodes_handler->getElementsTags(), res_container, candidate_set,
            dynamic_range_search_boundaries, radius);
    }
    returnVisitedList(visited_nodes_handler);
    return res_container->get_results();
}

template <typename DataType, typename DistType>
VecSimQueryResult_List
HNSWIndex<DataType, DistType>::rangeQuery(const void *query_data, double radius,
                                          VecSimQueryParams *queryParams) const {

    VecSimQueryResult_List rl = {0};
    this->last_mode = RANGE_QUERY;

    if (cur_element_count == 0) {
        rl.code = VecSim_QueryResult_OK;
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }
    void *timeoutCtx = nullptr;

    double epsilon = epsilon_;
    if (queryParams) {
        timeoutCtx = queryParams->timeoutCtx;
        if (queryParams->hnswRuntimeParams.epsilon != 0.0) {
            epsilon = queryParams->hnswRuntimeParams.epsilon;
        }
    }

    idType bottom_layer_ep = searchBottomLayerEP(query_data, timeoutCtx, &rl.code);
    // Although we checked that the index is not empty (cur_element_count == 0), it might be
    // that another thread deleted all the elements or didn't finish inserting the first element
    // yet. Anyway, we observed that the index is empty, so we return an empty result list.
    if (VecSim_OK != rl.code || bottom_layer_ep == INVALID_ID) {
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    // search bottom layer
    // Here we send the radius as double to match the function arguments type.
    if (this->num_marked_deleted)
        rl.results = searchRangeBottomLayer_WithTimeout<true>(bottom_layer_ep, query_data, epsilon,
                                                              radius, timeoutCtx, &rl.code);
    else
        rl.results = searchRangeBottomLayer_WithTimeout<false>(bottom_layer_ep, query_data, epsilon,
                                                               radius, timeoutCtx, &rl.code);
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
    info.hnswInfo.epsilon = this->epsilon_;
    info.hnswInfo.max_level = this->getMaxLevel();
    info.hnswInfo.entrypoint = this->getEntryPointLabel();
    info.hnswInfo.visitedNodesPoolSize = this->visited_nodes_handler_pool.getPoolSize();
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
    this->last_mode =
        res ? (initial_check ? HYBRID_ADHOC_BF : HYBRID_BATCHES_TO_ADHOC_BF) : HYBRID_BATCHES;
    return res;
}

#ifdef BUILD_TESTS
#include "hnsw_serializer.h"
#endif
