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

using std::pair;

#define HNSW_INVALID_ID    UINT_MAX
#define HNSW_INVALID_LEVEL SIZE_MAX

// This type is strongly bounded to `idType` because of the way we get the link list:
//
// linklistsizeint *neighbours_list = get_linklist_at_level(element_internal_id, level);
// unsigned short neighbours_count = getListCount(neighbours_list);
// auto *neighbours = (idType *)(neighbours_list + 1);
//
// TODO: reduce the type to smaller type when possible
typedef idType linklistsizeint;

template <typename DistType>
using candidatesMaxHeap = vecsim_stl::max_priority_queue<DistType, idType>;
template <typename DistType>
using candidatesLabelsMaxHeap = vecsim_stl::abstract_priority_queue<DistType, labelType>;

template <typename DataType, typename DistType>
#ifdef BUILD_TESTS
class HNSWIndex : public VecSimIndexAbstract<DistType>,
                  public Serializer
#else
class HNSWIndex : public VecSimIndexAbstract<DistType>
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
    size_t data_size_;
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

    // Index state
    size_t cur_element_count;
    // TODO: after introducing the memory reclaim upon delete, max_id is redundant since the valid
    // internal ids are being kept as a continuous sequence [0, 1, ..,, cur_element_count-1].
    // We can remove this field completely if we change the serialization version, as the decoding
    // relies on this field.
    idType max_id;
    size_t maxlevel_;

    // Index data structures
    idType entrypoint_node_;
    char *data_level0_memory_;
    char **linkLists_;
    vecsim_stl::vector<size_t> element_levels_;
    std::shared_ptr<VisitedNodesHandler> visited_nodes_handler;

    // used for synchronization only when parallel indexing / searching is enabled.
#ifdef ENABLE_PARALLELIZATION
    std::unique_ptr<VisitedNodesHandlerPool> visited_nodes_handler_pool;
    size_t pool_initial_size;
    std::mutex global;
    std::mutex cur_element_count_guard_;
    std::vector<std::mutex> link_list_locks_;
#endif

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
    inline linklistsizeint *get_linklist0(idType internal_id) const;
    inline linklistsizeint *get_linklist(idType internal_id, size_t level) const;
    inline void setListCount(linklistsizeint *ptr, unsigned short int size);
    inline void removeExtraLinks(linklistsizeint *node_ll, candidatesMaxHeap<DistType> candidates,
                                 size_t Mcurmax, idType *node_neighbors,
                                 const vecsim_stl::vector<bool> &bitmap, idType *removed_links,
                                 size_t *removed_links_num);
    template <typename Identifier> // Either idType or labelType
    inline DistType
    processCandidate(idType curNodeId, const void *data_point, size_t layer, size_t ef,
                     tag_t visited_tag,
                     vecsim_stl::abstract_priority_queue<DistType, Identifier> &top_candidates,
                     candidatesMaxHeap<DistType> &candidates_set, DistType lowerBound) const;
    inline void processCandidate_RangeSearch(
        idType curNodeId, const void *data_point, size_t layer, double epsilon, tag_t visited_tag,
        std::unique_ptr<vecsim_stl::abstract_results_container> &top_candidates,
        candidatesMaxHeap<DistType> &candidate_set, DistType lowerBound, double radius) const;
    candidatesMaxHeap<DistType> searchLayer(idType ep_id, const void *data_point, size_t layer,
                                            size_t ef) const;
    candidatesLabelsMaxHeap<DistType> *
    searchBottomLayer_WithTimeout(idType ep_id, const void *data_point, size_t ef, size_t k,
                                  void *timeoutCtx, VecSimQueryResult_Code *rc) const;
    VecSimQueryResult *searchRangeBottomLayer_WithTimeout(idType ep_id, const void *data_point,
                                                          double epsilon, double radius,
                                                          void *timeoutCtx,
                                                          VecSimQueryResult_Code *rc) const;
    void getNeighborsByHeuristic2(candidatesMaxHeap<DistType> &top_candidates, size_t M);
    inline idType mutuallyConnectNewElement(idType cur_c,
                                            candidatesMaxHeap<DistType> &top_candidates,
                                            size_t level);
    void repairConnectionsForDeletion(idType element_internal_id, idType neighbour_id,
                                      idType *neighbours_list, idType *neighbour_neighbours_list,
                                      size_t level, vecsim_stl::vector<bool> &neighbours_bitmap);
    inline void replaceEntryPoint();
    inline void resizeIndex(size_t new_max_elements);
    inline void SwapLastIdWithDeletedId(idType element_internal_id);

    // Protected internal function that implements generic single vector insertion.
    int appendVector(const void *vector_data, labelType label);

    // Protected internal function that implements generic single vector deletion.
    int removeVector(idType id);

    inline void emplaceToHeap(vecsim_stl::abstract_priority_queue<DistType, idType> &heap,
                              DistType dist, idType id) const;
    inline void emplaceToHeap(vecsim_stl::abstract_priority_queue<DistType, labelType> &heap,
                              DistType dist, idType id) const;

public:
    HNSWIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator,
              size_t random_seed = 100, size_t initial_pool_size = 1);
    virtual ~HNSWIndex();

    inline void setEf(size_t ef);
    inline size_t getEf() const;
    inline void setEpsilon(double epsilon);
    inline double getEpsilon() const;
    inline size_t indexSize() const override;
    inline size_t getIndexCapacity() const;
    inline size_t getEfConstruction() const;
    inline size_t getM() const;
    inline size_t getMaxLevel() const;
    inline idType getEntryPointId() const;
    inline labelType getEntryPointLabel() const;
    inline labelType getExternalLabel(idType internal_id) const;
    inline VisitedNodesHandler *getVisitedList() const;
    VecSimIndexDebugInfo debugInfo() const override;
    VecSimDebugInfoIterator *debugInfoIterator() const override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;
    char *getDataByInternalId(idType internal_id) const;
    inline linklistsizeint *get_linklist_at_level(idType internal_id, size_t level) const;
    inline unsigned short int getListCount(const linklistsizeint *ptr) const;
    inline idType searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                      VecSimQueryResult_Code *rc) const;

    VecSimQueryResult_List topKQuery(const void *query_data, size_t k,
                                     VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *query_data, double radius,
                                      VecSimQueryParams *queryParams) override;

    // inline priority queue getter that need to be implemented by derived class
    virtual inline candidatesLabelsMaxHeap<DistType> *getNewMaxPriorityQueue() const = 0;

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
size_t HNSWIndex<DataType, DistType>::getIndexCapacity() const {
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
    return maxlevel_;
}

template <typename DataType, typename DistType>
labelType HNSWIndex<DataType, DistType>::getEntryPointLabel() const {
    if (entrypoint_node_ != HNSW_INVALID_ID)
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
linklistsizeint *HNSWIndex<DataType, DistType>::get_linklist0(idType internal_id) const {
    return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ +
                               offsetLevel0_);
}

template <typename DataType, typename DistType>
linklistsizeint *HNSWIndex<DataType, DistType>::get_linklist(idType internal_id,
                                                             size_t level) const {
    return (linklistsizeint *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
}

template <typename DataType, typename DistType>
linklistsizeint *HNSWIndex<DataType, DistType>::get_linklist_at_level(idType internal_id,
                                                                      size_t level) const {
    return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
}

template <typename DataType, typename DistType>
unsigned short int HNSWIndex<DataType, DistType>::getListCount(const linklistsizeint *ptr) const {
    return *((unsigned short int *)ptr);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::setListCount(linklistsizeint *ptr, unsigned short int size) {
    *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
}

template <typename DataType, typename DistType>
idType HNSWIndex<DataType, DistType>::getEntryPointId() const {
    return entrypoint_node_;
}

template <typename DataType, typename DistType>
VisitedNodesHandler *HNSWIndex<DataType, DistType>::getVisitedList() const {
    return visited_nodes_handler.get();
}

/**
 * helper functions
 */
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::removeExtraLinks(
    linklistsizeint *node_ll, candidatesMaxHeap<DistType> candidates, size_t Mcurmax,
    idType *node_neighbors, const vecsim_stl::vector<bool> &neighbors_bitmap, idType *removed_links,
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
    setListCount(node_ll, link_idx);
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
template <typename Identifier>
DistType HNSWIndex<DataType, DistType>::processCandidate(
    idType curNodeId, const void *data_point, size_t layer, size_t ef, tag_t visited_tag,
    vecsim_stl::abstract_priority_queue<DistType, Identifier> &top_candidates,
    candidatesMaxHeap<DistType> &candidate_set, DistType lowerBound) const {

#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> lock(link_list_locks_[curNodeId]);
#endif
    linklistsizeint *node_ll = get_linklist_at_level(curNodeId, layer);
    size_t links_num = getListCount(node_ll);
    auto *node_links = (idType *)(node_ll + 1);

    __builtin_prefetch(visited_nodes_handler->getElementsTags() + *node_links);
    __builtin_prefetch(getDataByInternalId(*node_links));

    for (size_t j = 0; j < links_num; j++) {
        idType *candidate_pos = node_links + j;
        idType candidate_id = *candidate_pos;

        // Pre-fetch the next candidate data into memory cache, to improve performance.
        idType *next_candidate_pos = node_links + j + 1;
        __builtin_prefetch(visited_nodes_handler->getElementsTags() + *next_candidate_pos);
        __builtin_prefetch(getDataByInternalId(*next_candidate_pos));

        if (this->visited_nodes_handler->getNodeTag(candidate_id) == visited_tag)
            continue;

        this->visited_nodes_handler->tagNode(candidate_id, visited_tag);
        char *currObj1 = (getDataByInternalId(candidate_id));

        DistType dist1 = this->dist_func(data_point, currObj1, this->dim);
        if (lowerBound > dist1 || top_candidates.size() < ef) {
            candidate_set.emplace(-dist1, candidate_id);

            emplaceToHeap(top_candidates, dist1, candidate_id);

            if (top_candidates.size() > ef)
                top_candidates.pop();

            lowerBound = top_candidates.top().first;
        }
    }
    // Pre-fetch the neighbours list of the top candidate (the one that is going
    // to be processed in the next iteration) into memory cache, to improve performance.
    __builtin_prefetch(get_linklist_at_level(candidate_set.top().second, layer));

    return lowerBound;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::processCandidate_RangeSearch(
    idType curNodeId, const void *query_data, size_t layer, double epsilon, tag_t visited_tag,
    std::unique_ptr<vecsim_stl::abstract_results_container> &results,
    candidatesMaxHeap<DistType> &candidate_set, DistType dyn_range, double radius) const {

#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> lock(link_list_locks_[curNodeId]);
#endif
    linklistsizeint *node_ll = get_linklist_at_level(curNodeId, layer);
    size_t links_num = getListCount(node_ll);
    auto *node_links = (idType *)(node_ll + 1);

    __builtin_prefetch(visited_nodes_handler->getElementsTags() + *(node_ll + 1));
    __builtin_prefetch(getDataByInternalId(*node_links));

    // Cast radius once instead of each time we check that candidate_dist <= radius_
    DistType radius_ = DistType(radius);
    for (size_t j = 0; j < links_num; j++) {
        idType *candidate_pos = node_links + j;
        idType candidate_id = *candidate_pos;

        // Pre-fetch the next candidate data into memory cache, to improve performance.
        idType *next_candidate_pos = node_links + j + 1;
        __builtin_prefetch(visited_nodes_handler->getElementsTags() + *next_candidate_pos);
        __builtin_prefetch(getDataByInternalId(*next_candidate_pos));

        if (this->visited_nodes_handler->getNodeTag(candidate_id) == visited_tag)
            continue;
        this->visited_nodes_handler->tagNode(candidate_id, visited_tag);
        char *candidate_data = getDataByInternalId(candidate_id);

        DistType candidate_dist = this->dist_func(query_data, candidate_data, this->dim);
        if (candidate_dist < dyn_range) {
            candidate_set.emplace(-candidate_dist, candidate_id);

            // If the new candidate is in the requested radius, add it to the results set.
            if (candidate_dist <= radius_) {
                results->emplace(getExternalLabel(candidate_id), candidate_dist);
            }
        }
    }
    // Pre-fetch the neighbours list of the top candidate (the one that is going
    // to be processed in the next iteration) into memory cache, to improve performance.
    __builtin_prefetch(get_linklist_at_level(candidate_set.top().second, layer));
}

template <typename DataType, typename DistType>
candidatesMaxHeap<DistType>
HNSWIndex<DataType, DistType>::searchLayer(idType ep_id, const void *data_point, size_t layer,
                                           size_t ef) const {

#ifdef ENABLE_PARALLELIZATION
    this->visited_nodes_handler =
        this->visited_nodes_handler_pool->getAvailableVisitedNodesHandler();
#endif

    tag_t visited_tag = this->visited_nodes_handler->getFreshTag();

    candidatesMaxHeap<DistType> top_candidates(this->allocator);
    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    DistType dist = this->dist_func(data_point, getDataByInternalId(ep_id), this->dim);
    DistType lowerBound = dist;
    top_candidates.emplace(dist, ep_id);
    candidate_set.emplace(-dist, ep_id);

    this->visited_nodes_handler->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() >= ef) {
            break;
        }
        candidate_set.pop();

        lowerBound = processCandidate(curr_el_pair.second, data_point, layer, ef, visited_tag,
                                      top_candidates, candidate_set, lowerBound);
    }

#ifdef ENABLE_PARALLELIZATION
    visited_nodes_handler_pool->returnVisitedNodesHandlerToPool(this->visited_nodes_handler);
#endif
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
idType HNSWIndex<DataType, DistType>::mutuallyConnectNewElement(
    idType cur_c, candidatesMaxHeap<DistType> &top_candidates, size_t level) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, M_);
    if (top_candidates.size() > M_)
        throw std::runtime_error(
            "Should be not be more than M_ candidates returned by the heuristic");

    vecsim_stl::vector<idType> selectedNeighbors(this->allocator);
    selectedNeighbors.reserve(M_);
    while (top_candidates.size() > 0) {
        selectedNeighbors.push_back(top_candidates.top().second);
        top_candidates.pop();
    }

    idType next_closest_entry_point = selectedNeighbors.back();
    {
        linklistsizeint *ll_cur = get_linklist_at_level(cur_c, level);
        if (*ll_cur) {
            throw std::runtime_error("The newly inserted element should have blank link list");
        }
        setListCount(ll_cur, selectedNeighbors.size());
        auto *data = (idType *)(ll_cur + 1);
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            if (data[idx])
                throw std::runtime_error("Possible memory corruption");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");
            data[idx] = selectedNeighbors[idx];
        }
        auto *incoming_edges = new (this->allocator) vecsim_stl::vector<idType>(this->allocator);
        setIncomingEdgesPtr(cur_c, level, (void *)incoming_edges);
    }

    // go over the selected neighbours - selectedNeighbor is the neighbour id
    vecsim_stl::vector<bool> neighbors_bitmap(this->allocator);
    for (idType selectedNeighbor : selectedNeighbors) {
#ifdef ENABLE_PARALLELIZATION
        std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbor]);
#endif
        linklistsizeint *ll_other = get_linklist_at_level(selectedNeighbor, level);
        size_t sz_link_list_other = getListCount(ll_other);

        if (sz_link_list_other > Mcurmax)
            throw std::runtime_error("Bad value of sz_link_list_other");
        if (selectedNeighbor == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");
        if (level > element_levels_[selectedNeighbor])
            throw std::runtime_error("Trying to make a link on a non-existent level");

        // get the array of neighbours - for the current neighbour
        auto *neighbor_neighbors = (idType *)(ll_other + 1);

        // If the selected neighbor can add another link (hasn't reached the max) - add it.
        if (sz_link_list_other < Mcurmax) {
            neighbor_neighbors[sz_link_list_other] = cur_c;
            setListCount(ll_other, sz_link_list_other + 1);
        } else {
            // try finding "weak" elements to replace it with the new one with the heuristic:
            candidatesMaxHeap<DistType> candidates(this->allocator);
            // (re)use the bitmap to represent the set of the original neighbours for the current
            // selected neighbour.
            neighbors_bitmap.assign(max_id + 1, false);
            DistType d_max = this->dist_func(getDataByInternalId(cur_c),
                                             getDataByInternalId(selectedNeighbor), this->dim);
            candidates.emplace(d_max, cur_c);
            // consider cur_c as if it was a link of the selected neighbor
            neighbors_bitmap[cur_c] = true;
            for (size_t j = 0; j < sz_link_list_other; j++) {
                candidates.emplace(this->dist_func(getDataByInternalId(neighbor_neighbors[j]),
                                                   getDataByInternalId(selectedNeighbor),
                                                   this->dim),
                                   neighbor_neighbors[j]);
                neighbors_bitmap[neighbor_neighbors[j]] = true;
            }

            auto removed_links_alloc =
                this->getAllocator()->allocate_unique((sz_link_list_other + 1) * sizeof(idType));
            auto removed_links = static_cast<idType *>(removed_links_alloc.get());
            size_t removed_links_num;
            removeExtraLinks(ll_other, candidates, Mcurmax, neighbor_neighbors, neighbors_bitmap,
                             removed_links, &removed_links_num);

            // remove the current neighbor from the incoming list of nodes for the
            // neighbours that were chosen to remove (if edge wasn't bidirectional)
            auto *neighbour_incoming_edges = getIncomingEdgesPtr(selectedNeighbor, level);
            for (size_t i = 0; i < removed_links_num; i++) {
                idType node_id = removed_links[i];
                auto *node_incoming_edges = getIncomingEdgesPtr(node_id, level);
                // if we removed cur_c (the node just inserted), then it points to the current
                // neighbour, but not vise versa.
                if (node_id == cur_c) {
                    neighbour_incoming_edges->push_back(cur_c);
                    continue;
                }

                // if the node id (the neighbour's neighbour to be removed)
                // wasn't pointing to the neighbour (i.e., the edge was uni-directional),
                // we should remove the current neighbor from the node's incoming edges.
                // otherwise, the edge turned from bidirectional to
                // uni-directional, so we insert it to the neighbour's
                // incoming edges set.
                auto it = std::find(node_incoming_edges->begin(), node_incoming_edges->end(),
                                    selectedNeighbor);
                if (it != node_incoming_edges->end()) {
                    node_incoming_edges->erase(it);
                } else {
                    neighbour_incoming_edges->push_back(node_id);
                }
            }
        }
    }
    return next_closest_entry_point;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::repairConnectionsForDeletion(
    idType element_internal_id, idType neighbour_id, idType *neighbours_list,
    idType *neighbour_neighbours_list, size_t level, vecsim_stl::vector<bool> &neighbours_bitmap) {

    // put the deleted element's neighbours in the candidates.
    candidatesMaxHeap<DistType> candidates(this->allocator);
    unsigned short neighbours_count = getListCount(neighbours_list);
    auto *neighbours = (idType *)(neighbours_list + 1);
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
    vecsim_stl::vector<bool> neighbour_orig_neighbours_set(max_id + 1, false, this->allocator);
    unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);
    auto *neighbour_neighbours = (idType *)(neighbour_neighbours_list + 1);
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
    auto removed_links_alloc =
        this->getAllocator()->allocate_unique(neighbour_neighbours_count * sizeof(idType));
    idType *removed_links = static_cast<idType *>(removed_links_alloc.get());
    removeExtraLinks(neighbour_neighbours_list, candidates, Mcurmax, neighbour_neighbours,
                     neighbour_orig_neighbours_set, removed_links, &removed_links_num);

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
        auto it = std::find(node_incoming_edges->begin(), node_incoming_edges->end(), neighbour_id);
        if (it != node_incoming_edges->end()) {
            node_incoming_edges->erase(it);
        } else {
            neighbour_incoming_edges->push_back(node_id);
        }
    }

    // updates for the new edges created
    unsigned short updated_links_num = getListCount(neighbour_neighbours_list);
    for (size_t i = 0; i < updated_links_num; i++) {
        idType node_id = neighbour_neighbours[i];
        if (!neighbour_orig_neighbours_set[node_id]) {
            auto *node_incoming_edges = getIncomingEdgesPtr(node_id, level);
            // if the node has an edge to the neighbour as well, remove it
            // from the incoming nodes of the neighbour
            // otherwise, need to update the edge as incoming.
            linklistsizeint *node_links_list = get_linklist_at_level(node_id, level);
            unsigned short node_links_size = getListCount(node_links_list);
            auto *node_links = (idType *)(node_links_list + 1);
            bool bidirectional_edge = false;
            for (size_t j = 0; j < node_links_size; j++) {
                if (node_links[j] == neighbour_id) {
                    neighbour_incoming_edges->erase(std::find(neighbour_incoming_edges->begin(),
                                                              neighbour_incoming_edges->end(),
                                                              node_id));
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
        linklistsizeint *top_level_list = get_linklist_at_level(old_entry, maxlevel_);
        if (getListCount(top_level_list) > 0) {
            // Tries to set the (arbitrary) first neighbor as the entry point, if exists.
            entrypoint_node_ = ((idType *)(top_level_list + 1))[0];
        } else {
            // If there is no neighbors in the current level, check for any vector at
            // this level to be the new entry point.
            for (idType cur_id = 0; cur_id < cur_element_count; cur_id++) {
                if (element_levels_[cur_id] == maxlevel_ && cur_id != old_entry) {
                    entrypoint_node_ = cur_id;
                    break;
                }
            }
        }
        // If we didn't find any vector at the top level, decrease the maxlevel_ and try again,
        // until we find a new entry point, or the index is empty.
        if (old_entry == entrypoint_node_) {
            maxlevel_--;
            if ((int)maxlevel_ < 0) {
                maxlevel_ = HNSW_INVALID_LEVEL;
                entrypoint_node_ = HNSW_INVALID_ID;
            }
        }
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::SwapLastIdWithDeletedId(idType element_internal_id) {
    // swap label
    replaceIdOfLabel(getExternalLabel(max_id), element_internal_id, max_id);

    // swap neighbours
    size_t last_element_top_level = element_levels_[max_id];
    for (size_t level = 0; level <= last_element_top_level; level++) {
        linklistsizeint *neighbours_list = get_linklist_at_level(max_id, level);
        unsigned short neighbours_count = getListCount(neighbours_list);
        auto *neighbours = (idType *)(neighbours_list + 1);

        // go over the neighbours that also points back to the last element whose is going to
        // change, and update the id.
        for (size_t i = 0; i < neighbours_count; i++) {
            idType neighbour_id = neighbours[i];
            linklistsizeint *neighbour_neighbours_list = get_linklist_at_level(neighbour_id, level);
            unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);

            auto *neighbour_neighbours = (idType *)(neighbour_neighbours_list + 1);
            bool bidirectional_edge = false;
            for (size_t j = 0; j < neighbour_neighbours_count; j++) {
                // if the edge is bidirectional, update for this neighbor
                if (neighbour_neighbours[j] == max_id) {
                    bidirectional_edge = true;
                    neighbour_neighbours[j] = element_internal_id;
                    break;
                }
            }

            // if this edge is uni-directional, we should update the id in the neighbor's
            // incoming edges.
            if (!bidirectional_edge) {
                auto *neighbour_incoming_edges = getIncomingEdgesPtr(neighbour_id, level);
                auto it = std::find(neighbour_incoming_edges->begin(),
                                    neighbour_incoming_edges->end(), max_id);
                assert(it != neighbour_incoming_edges->end());
                neighbour_incoming_edges->erase(it);
                neighbour_incoming_edges->push_back(element_internal_id);
            }
        }

        // next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // updates.
        auto *incoming_edges = getIncomingEdgesPtr(max_id, level);
        for (auto incoming_edge : *incoming_edges) {
            linklistsizeint *incoming_neighbour_neighbours_list =
                get_linklist_at_level(incoming_edge, level);
            unsigned short incoming_neighbour_neighbours_count =
                getListCount(incoming_neighbour_neighbours_list);
            auto *incoming_neighbour_neighbours =
                (idType *)(incoming_neighbour_neighbours_list + 1);
            for (size_t j = 0; j < incoming_neighbour_neighbours_count; j++) {
                if (incoming_neighbour_neighbours[j] == max_id) {
                    incoming_neighbour_neighbours[j] = element_internal_id;
                    break;
                }
            }
        }
    }

    // swap the last_id level 0 data, and invalidate the deleted id's data
    memcpy(data_level0_memory_ + element_internal_id * size_data_per_element_ + offsetLevel0_,
           data_level0_memory_ + max_id * size_data_per_element_ + offsetLevel0_,
           size_data_per_element_);
    memset(data_level0_memory_ + max_id * size_data_per_element_ + offsetLevel0_, 0,
           size_data_per_element_);

    // swap pointer of higher levels links
    linkLists_[element_internal_id] = linkLists_[max_id];
    linkLists_[max_id] = nullptr;

    // swap top element level
    element_levels_[element_internal_id] = element_levels_[max_id];
    element_levels_[max_id] = HNSW_INVALID_LEVEL;

    if (max_id == this->entrypoint_node_) {
        this->entrypoint_node_ = element_internal_id;
    }
}

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
                                         std::shared_ptr<VecSimAllocator> allocator,
                                         size_t random_seed, size_t pool_initial_size)
    : VecSimIndexAbstract<DistType>(allocator, params->dim, params->type, params->metric,
                                    params->blockSize, params->multi),
      max_elements_(params->initialCapacity),
      data_size_(VecSimType_sizeof(params->type) * this->dim),
      element_levels_(max_elements_, allocator)

#ifdef ENABLE_PARALLELIZATION
      ,
      link_list_locks_(max_elements_)
#endif
{
    size_t M = params->M ? params->M : HNSW_DEFAULT_M;
    if (M > SIZE_MAX / 2)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;

    size_t ef_construction = params->efConstruction ? params->efConstruction : HNSW_DEFAULT_EF_C;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = params->efRuntime ? params->efRuntime : HNSW_DEFAULT_EF_RT;
    epsilon_ = params->epsilon > 0.0 ? params->epsilon : HNSW_DEFAULT_EPSILON;

    cur_element_count = 0;
    max_id = HNSW_INVALID_ID;
#ifdef ENABLE_PARALLELIZATION
    pool_initial_size = pool_initial_size;
    visited_nodes_handler_pool = std::unique_ptr<VisitedNodesHandlerPool>(
        new (this->allocator)
            VisitedNodesHandlerPool(pool_initial_size, max_elements_, this->allocator));
#else
    visited_nodes_handler = std::shared_ptr<VisitedNodesHandler>(
        new (this->allocator) VisitedNodesHandler(max_elements_, this->allocator));
#endif

    // initializations for special treatment of the first node
    entrypoint_node_ = HNSW_INVALID_ID;
    maxlevel_ = HNSW_INVALID_LEVEL;

    if (M <= 1)
        throw std::runtime_error("HNSW index parameter M cannot be 1");
    mult_ = 1 / log(1.0 * M_);
    level_generator_.seed(random_seed);

    // data_level0_memory will look like this:
    // | -----4------ | -----4*M0----------- | ----------8----------| --data_size_-- | ----8---- |
    // | <links_len>  | <link_1> <link_2>... | <incoming_links_ptr> |     <data>     |  <label>  |
    if (maxM0_ > ((SIZE_MAX - sizeof(void *) - sizeof(linklistsizeint)) / sizeof(idType)) + 1)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    size_links_level0_ = sizeof(linklistsizeint) + maxM0_ * sizeof(idType) + sizeof(void *);

    if (size_links_level0_ > SIZE_MAX - data_size_ - sizeof(labelType))
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labelType);

    // No need to test for overflow because we passed the test for size_links_level0_ and this is
    // less.
    incoming_links_offset0 = maxM0_ * sizeof(idType) + sizeof(linklistsizeint);
    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
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
    // | -----4----- | -----4*M-------------- | ----------8--------- |
    // | <links_len> | <link_1> <link_2> ...  | <incoming_links_ptr> |
    size_links_per_element_ = sizeof(linklistsizeint) + maxM_ * sizeof(idType) + sizeof(void *);
    // No need to test for overflow because we passed the test for incoming_links_offset0 and this
    // is less.
    incoming_links_offset = maxM_ * sizeof(idType) + sizeof(linklistsizeint);
}

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::~HNSWIndex() {
    if (max_id != HNSW_INVALID_ID) {
        for (idType id = 0; id <= max_id; id++) {
            for (size_t level = 0; level <= element_levels_[id]; level++) {
                delete getIncomingEdgesPtr(id, level);
            }
            if (element_levels_[id] > 0)
                this->allocator->free_allocation(linkLists_[id]);
        }
    }

    this->allocator->free_allocation(linkLists_);
    this->allocator->free_allocation(data_level0_memory_);
}

/**
 * Index API functions
 */
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::resizeIndex(size_t new_max_elements) {
    element_levels_.resize(new_max_elements);
    element_levels_.shrink_to_fit();
    resizeLabelLookup(new_max_elements);
#ifdef ENABLE_PARALLELIZATION
    visited_nodes_handler_pool = std::unique_ptr<VisitedNodesHandlerPool>(
        new (this->allocator)
            VisitedNodesHandlerPool(this->pool_initial_size, new_max_elements, this->allocator));
    std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);
#else
    visited_nodes_handler = std::shared_ptr<VisitedNodesHandler>(
        new (this->allocator) VisitedNodesHandler(new_max_elements, this->allocator));
#endif
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
int HNSWIndex<DataType, DistType>::removeVector(const idType element_internal_id) {

    vecsim_stl::vector<bool> neighbours_bitmap(this->allocator);

    // go over levels and repair connections
    size_t element_top_level = element_levels_[element_internal_id];
    for (size_t level = 0; level <= element_top_level; level++) {
        linklistsizeint *neighbours_list = get_linklist_at_level(element_internal_id, level);
        unsigned short neighbours_count = getListCount(neighbours_list);
        auto *neighbours = (idType *)(neighbours_list + 1);
        // reset the neighbours' bitmap for the current level.
        neighbours_bitmap.assign(max_id + 1, false);
        // store the deleted element's neighbours set in a bitmap for fast access.
        for (size_t j = 0; j < neighbours_count; j++) {
            neighbours_bitmap[neighbours[j]] = true;
        }
        // go over the neighbours that also points back to the removed point and make a local
        // repair.
        for (size_t i = 0; i < neighbours_count; i++) {
            idType neighbour_id = neighbours[i];
            linklistsizeint *neighbour_neighbours_list = get_linklist_at_level(neighbour_id, level);
            unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);

            auto *neighbour_neighbours = (idType *)(neighbour_neighbours_list + 1);
            bool bidirectional_edge = false;
            for (size_t j = 0; j < neighbour_neighbours_count; j++) {
                // if the edge is bidirectional, do repair for this neighbor
                if (neighbour_neighbours[j] == element_internal_id) {
                    bidirectional_edge = true;
                    repairConnectionsForDeletion(element_internal_id, neighbour_id, neighbours_list,
                                                 neighbour_neighbours_list, level,
                                                 neighbours_bitmap);
                    break;
                }
            }

            // if this edge is uni-directional, we should remove the element from the neighbor's
            // incoming edges.
            if (!bidirectional_edge) {
                auto *neighbour_incoming_edges = getIncomingEdgesPtr(neighbour_id, level);
                neighbour_incoming_edges->erase(std::find(neighbour_incoming_edges->begin(),
                                                          neighbour_incoming_edges->end(),
                                                          element_internal_id));
            }
        }

        // next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // repairs.
        auto *incoming_edges = getIncomingEdgesPtr(element_internal_id, level);
        for (auto incoming_edge : *incoming_edges) {
            linklistsizeint *incoming_node_neighbours_list =
                get_linklist_at_level(incoming_edge, level);
            repairConnectionsForDeletion(element_internal_id, incoming_edge, neighbours_list,
                                         incoming_node_neighbours_list, level, neighbours_bitmap);
        }
        delete incoming_edges;
    }

    // replace the entry point with another one, if we are deleting the current entry point.
    if (element_internal_id == entrypoint_node_) {
        assert(element_top_level == maxlevel_);
        replaceEntryPoint();
    }

    // Swap the last id with the deleted one, and invalidate the last id data.
    if (element_levels_[element_internal_id] > 0) {
        this->allocator->free_allocation(linkLists_[element_internal_id]);
        linkLists_[element_internal_id] = nullptr;
    }
    if (max_id == element_internal_id) {
        // we're deleting the last internal id, just invalidate data without swapping.
        memset(data_level0_memory_ + max_id * size_data_per_element_ + offsetLevel0_, 0,
               size_data_per_element_);
    } else {
        SwapLastIdWithDeletedId(element_internal_id);
    }
    --cur_element_count;
    --max_id;

    // If we need to free a complete block & there is a least one block between the
    // capacity and the size.
    if (cur_element_count % this->blockSize == 0 &&
        cur_element_count + this->blockSize <= max_elements_) {

        // Check if the capacity is aligned to block size.
        size_t extra_space_to_free = max_elements_ % this->blockSize;

        // Remove one block from the capacity.
        this->resizeIndex(max_elements_ - this->blockSize - extra_space_to_free);
    }
    return true;
}

template <typename DataType, typename DistType>
int HNSWIndex<DataType, DistType>::appendVector(const void *vector_data, const labelType label) {

    idType cur_c;

    auto normalized_blob = this->getAllocator()->allocate_unique(this->dim * sizeof(DataType));
    if (this->metric == VecSimMetric_Cosine) {
        memcpy(normalized_blob.get(), vector_data, this->dim * sizeof(DataType));
        normalizeVector(static_cast<DataType *>(normalized_blob.get()), this->dim);
        vector_data = normalized_blob.get();
    }

    {
#ifdef ENABLE_PARALLELIZATION
        std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);
#endif

        if (cur_element_count >= max_elements_) {
            size_t vectors_to_add = this->blockSize - max_elements_ % this->blockSize;
            resizeIndex(max_elements_ + vectors_to_add);
        }
        cur_c = max_id = cur_element_count++;
        setVectorId(label, cur_c);
    }
#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
#endif
    // choose randomly the maximum level in which the new element will be in the index.
    size_t element_max_level = getRandomLevel(mult_);
    element_levels_[cur_c] = element_max_level;

#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> entry_point_lock(global);
#endif
    size_t maxlevelcopy = maxlevel_;

#ifdef ENABLE_PARALLELIZATION
    if (element_max_level <= maxlevelcopy)
        entry_point_lock.unlock();
#endif
    size_t currObj = entrypoint_node_;

    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0,
           size_data_per_element_);

    // Initialisation of the data and label
    setExternalLabel(cur_c, label);
    memcpy(getDataByInternalId(cur_c), vector_data, data_size_);

    if (element_max_level > 0) {
        linkLists_[cur_c] =
            (char *)this->allocator->allocate(size_links_per_element_ * element_max_level);
        if (linkLists_[cur_c] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        memset(linkLists_[cur_c], 0, size_links_per_element_ * element_max_level);
    }

    // this condition only means that we are not inserting the first element.
    if (entrypoint_node_ != HNSW_INVALID_ID) {
        if (element_max_level < maxlevelcopy) {
            DistType cur_dist =
                this->dist_func(vector_data, getDataByInternalId(currObj), this->dim);
            for (size_t level = maxlevelcopy; level > element_max_level; level--) {
                // this is done for the levels which are above the max level
                // to which we are going to insert the new element. We do
                // a greedy search in the graph starting from the entry point
                // at each level, and move on with the closest element we can find.
                // When there is no improvement to do, we take a step down.
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
#ifdef ENABLE_PARALLELIZATION
                    std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
#endif
                    data = get_linklist(currObj, level);
                    int size = getListCount(data);

                    auto *datal = (idType *)(data + 1);
                    for (int i = 0; i < size; i++) {
                        idType cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error(
                                "candidate error: candidate id is out of index range");

                        DistType d =
                            this->dist_func(vector_data, getDataByInternalId(cand), this->dim);
                        if (d < cur_dist) {
                            cur_dist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        for (size_t level = std::min(element_max_level, maxlevelcopy); (int)level >= 0; level--) {
            if (level > maxlevelcopy || level < 0) // possible?
                throw std::runtime_error("Level error");

            candidatesMaxHeap<DistType> top_candidates =
                searchLayer(currObj, vector_data, level, ef_construction_);
            currObj = mutuallyConnectNewElement(cur_c, top_candidates, level);
        }

        // updating the maximum level (holding a global lock)
        if (element_max_level > maxlevelcopy) {
            entrypoint_node_ = cur_c;
            maxlevel_ = element_max_level;
            // create the incoming edges set for the new levels.
            for (size_t level_idx = maxlevelcopy + 1; level_idx <= element_max_level; level_idx++) {
                auto *incoming_edges =
                    new (this->allocator) vecsim_stl::vector<idType>(this->allocator);
                setIncomingEdgesPtr(cur_c, level_idx, incoming_edges);
            }
        }
    } else {
        // Do nothing for the first element
        entrypoint_node_ = 0;
        for (size_t level_idx = maxlevel_ + 1; level_idx <= element_max_level; level_idx++) {
            auto *incoming_edges =
                new (this->allocator) vecsim_stl::vector<idType>(this->allocator);
            setIncomingEdgesPtr(cur_c, level_idx, incoming_edges);
        }
        maxlevel_ = element_max_level;
    }
    return true;
}

template <typename DataType, typename DistType>
idType HNSWIndex<DataType, DistType>::searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                                          VecSimQueryResult_Code *rc) const {

    if (cur_element_count == 0) {
        return entrypoint_node_;
    }
    idType currObj = entrypoint_node_;
    DistType cur_dist =
        this->dist_func(query_data, getDataByInternalId(entrypoint_node_), this->dim);
    for (size_t level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            if (__builtin_expect(VecSimIndexAbstract<DistType>::timeoutCallback(timeoutCtx), 0)) {
                *rc = VecSim_QueryResult_TimedOut;
                return HNSW_INVALID_ID;
            }
            changed = false;
            linklistsizeint *node_ll = get_linklist(currObj, level);
            unsigned short links_count = getListCount(node_ll);
            auto *node_links = (idType *)(node_ll + 1);
            for (int i = 0; i < links_count; i++) {
                idType candidate = node_links[i];
                if (candidate > max_elements_)
                    throw std::runtime_error("candidate error: out of index range");

                DistType d = this->dist_func(query_data, getDataByInternalId(candidate), this->dim);
                if (d < cur_dist) {
                    cur_dist = d;
                    currObj = candidate;
                    changed = true;
                }
            }
        }
    }
    *rc = VecSim_QueryResult_OK;
    return currObj;
}

template <typename DataType, typename DistType>
candidatesLabelsMaxHeap<DistType> *
HNSWIndex<DataType, DistType>::searchBottomLayer_WithTimeout(idType ep_id, const void *data_point,
                                                             size_t ef, size_t k, void *timeoutCtx,
                                                             VecSimQueryResult_Code *rc) const {

#ifdef ENABLE_PARALLELIZATION
    this->visited_nodes_handler =
        this->visited_nodes_handler_pool->getAvailableVisitedNodesHandler();
#endif

    tag_t visited_tag = this->visited_nodes_handler->getFreshTag();

    candidatesLabelsMaxHeap<DistType> *top_candidates = getNewMaxPriorityQueue();
    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    DistType dist = this->dist_func(data_point, getDataByInternalId(ep_id), this->dim);
    DistType lowerBound = dist;
    top_candidates->emplace(dist, getExternalLabel(ep_id));
    candidate_set.emplace(-dist, ep_id);

    this->visited_nodes_handler->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates->size() >= ef) {
            break;
        }
        if (__builtin_expect(VecSimIndexAbstract<DistType>::timeoutCallback(timeoutCtx), 0)) {
            *rc = VecSim_QueryResult_TimedOut;
            return top_candidates;
        }
        candidate_set.pop();

        lowerBound = processCandidate(curr_el_pair.second, data_point, 0, ef, visited_tag,
                                      *top_candidates, candidate_set, lowerBound);
    }
#ifdef ENABLE_PARALLELIZATION
    visited_nodes_handler_pool->returnVisitedNodesHandlerToPool(this->visited_nodes_handler);
#endif
    while (top_candidates->size() > k) {
        top_candidates->pop();
    }
    *rc = VecSim_QueryResult_OK;
    return top_candidates;
}

template <typename DataType, typename DistType>
VecSimQueryResult_List HNSWIndex<DataType, DistType>::topKQuery(const void *query_data, size_t k,
                                                                VecSimQueryParams *queryParams) {

    VecSimQueryResult_List rl = {0};
    this->last_mode = STANDARD_KNN;

    if (cur_element_count == 0 || k == 0) {
        rl.code = VecSim_QueryResult_OK;
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    void *timeoutCtx = nullptr;

    auto normalized_blob = this->getAllocator()->allocate_unique(this->dim * sizeof(DataType));
    if (this->metric == VecSimMetric_Cosine) {
        memcpy(normalized_blob.get(), query_data, this->dim * sizeof(DataType));
        normalizeVector(static_cast<DataType *>(normalized_blob.get()), this->dim);
        query_data = normalized_blob.get();
    }
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
    }

    // We now oun the results heap, we need to free (delete) it when we done
    candidatesLabelsMaxHeap<DistType> *results = searchBottomLayer_WithTimeout(
        bottom_layer_ep, query_data, std::max(ef, k), k, timeoutCtx, &rl.code);

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
VecSimQueryResult *HNSWIndex<DataType, DistType>::searchRangeBottomLayer_WithTimeout(
    idType ep_id, const void *data_point, double epsilon, double radius, void *timeoutCtx,
    VecSimQueryResult_Code *rc) const {

    *rc = VecSim_QueryResult_OK;
    auto res_container = getNewResultsContainer(10); // arbitrary initial cap.

#ifdef ENABLE_PARALLELIZATION
    this->visited_nodes_handler =
        this->visited_nodes_handler_pool->getAvailableVisitedNodesHandler();
#endif

    tag_t visited_tag = this->visited_nodes_handler->getFreshTag();
    candidatesMaxHeap<DistType> candidate_set(this->allocator);

    // Set the initial effective-range to be at least the distance from the entry-point.
    DistType ep_dist = this->dist_func(data_point, getDataByInternalId(ep_id), this->dim);
    DistType dynamic_range = ep_dist;

    if (ep_dist <= radius) {
        // Entry-point is within the radius - add it to the results.
        res_container->emplace(getExternalLabel(ep_id), ep_dist);
        dynamic_range = radius; // to ensure that dyn_range >= radius.
    }

    DistType dynamic_range_search_boundaries = dynamic_range * (1.0 + epsilon);
    candidate_set.emplace(-ep_dist, ep_id);
    this->visited_nodes_handler->tagNode(ep_id, visited_tag);

    // Cast radius once instead of each time we check that -curr_el_pair.first >= radius_.
    DistType radius_ = DistType(radius);
    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();
        // If the best candidate is outside the dynamic range in more than epsilon (relatively) - we
        // finish the search.
        if ((-curr_el_pair.first) > dynamic_range_search_boundaries) {
            break;
        }
        if (__builtin_expect(VecSimIndexAbstract<DistType>::timeoutCallback(timeoutCtx), 0)) {
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
        processCandidate_RangeSearch(curr_el_pair.second, data_point, 0, epsilon, visited_tag,
                                     res_container, candidate_set, dynamic_range_search_boundaries,
                                     radius);
    }

#ifdef ENABLE_PARALLELIZATION
    visited_nodes_handler_pool->returnVisitedNodesHandlerToPool(this->visited_nodes_handler);
#endif
    return res_container->get_results();
}

template <typename DataType, typename DistType>
VecSimQueryResult_List HNSWIndex<DataType, DistType>::rangeQuery(const void *query_data,
                                                                 double radius,
                                                                 VecSimQueryParams *queryParams) {
    VecSimQueryResult_List rl = {0};
    this->last_mode = RANGE_QUERY;

    if (cur_element_count == 0) {
        rl.code = VecSim_QueryResult_OK;
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }
    void *timeoutCtx = nullptr;

    auto normalized_blob = this->getAllocator()->allocate_unique(this->dim * sizeof(DataType));
    if (this->metric == VecSimMetric_Cosine) {
        memcpy(normalized_blob.get(), query_data, this->dim * sizeof(DataType));
        normalizeVector(static_cast<DataType *>(normalized_blob.get()), this->dim);
        query_data = normalized_blob.get();
    }

    double epsilon = epsilon_;
    if (queryParams) {
        timeoutCtx = queryParams->timeoutCtx;
        if (queryParams->hnswRuntimeParams.epsilon != 0.0) {
            epsilon = queryParams->hnswRuntimeParams.epsilon;
        }
    }

    idType bottom_layer_ep = searchBottomLayerEP(query_data, timeoutCtx, &rl.code);
    if (VecSim_OK != rl.code) {
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    // search bottom layer
    // Here we send the radius as double to match the function arguments type.
    rl.results = searchRangeBottomLayer_WithTimeout(bottom_layer_ep, query_data, epsilon, radius,
                                                    timeoutCtx, &rl.code);

    return rl;
}

template <typename DataType, typename DistType>
VecSimIndexDebugInfo HNSWIndex<DataType, DistType>::debugInfo() const {

    VecSimIndexDebugInfo info;
    info.algo = VecSimAlgo_HNSWLIB;
    info.hnswInfo.dim = this->dim;
    info.hnswInfo.type = this->vecType;
    info.hnswInfo.isMulti = this->isMulti;
    info.hnswInfo.metric = this->metric;
    info.hnswInfo.blockSize = this->blockSize;
    info.hnswInfo.M = this->getM();
    info.hnswInfo.efConstruction = this->getEfConstruction();
    info.hnswInfo.efRuntime = this->getEf();
    info.hnswInfo.epsilon = this->epsilon_;
    info.hnswInfo.indexSize = this->indexSize();
    info.hnswInfo.indexLabelCount = this->indexLabelCount();
    info.hnswInfo.max_level = this->getMaxLevel();
    info.hnswInfo.entrypoint = this->getEntryPointLabel();
    info.hnswInfo.memory = this->getAllocationSize();
    info.hnswInfo.last_mode = this->last_mode;
    return info;
}

template <typename DataType, typename DistType>
VecSimDebugInfoIterator *HNSWIndex<DataType, DistType>::debugInfoIterator() const {
    VecSimIndexDebugInfo info = this->debugInfo();
    // For readability. Update this number when needed.
    size_t numberOfInfoFields = 12;
    auto *infoIterator = new VecSimDebugInfoIterator(numberOfInfoFields);

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::ALGORITHM_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimAlgo_ToString(info.algo)}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::TYPE_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimType_ToString(info.hnswInfo.type)}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::DIMENSION_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.dim}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::METRIC_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimMetric_ToString(info.hnswInfo.metric)}}});

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::IS_MULTI_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.isMulti}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.indexSize}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::INDEX_LABEL_COUNT_STRING,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.indexLabelCount}}});
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
        VecSim_InfoField{.fieldName = VecSimCommonStrings::MEMORY_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.memory}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::SEARCH_MODE_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .fieldValue = {FieldValue{
                             .stringValue = VecSimSearchMode_ToString(info.hnswInfo.last_mode)}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_EPSILON_STRING,
                         .fieldType = INFOFIELD_FLOAT64,
                         .fieldValue = {FieldValue{.floatingPointValue = info.hnswInfo.epsilon}}});

    return infoIterator;
}

template <typename DataType, typename DistType>
bool HNSWIndex<DataType, DistType>::preferAdHocSearch(size_t subsetSize, size_t k,
                                                      bool initial_check) {
    // This heuristic is based on sklearn decision tree classifier (with 20 leaves nodes) -
    // see scripts/HNSW_batches_clf.py
    size_t index_size = this->indexSize();
    if (subsetSize > index_size) {
        throw std::runtime_error("internal error: subset size cannot be larger than index size");
    }
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
