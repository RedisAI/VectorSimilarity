#pragma once

#include "visited_nodes_handler.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/space_interface.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/query_result_struct.h"

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

namespace hnswlib {
using namespace std;

#define HNSW_INVALID_ID    UINT_MAX
#define HNSW_INVALID_LEVEL SIZE_MAX

typedef size_t labeltype;
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
using candidatesMaxHeap = vecsim_stl::max_priority_queue<pair<dist_t, tableint>>;
template <typename dist_t>
using candidatesLabelsMaxHeap = vecsim_stl::max_priority_queue<pair<dist_t, labeltype>>;

template <typename dist_t>
class HierarchicalNSW : public VecsimBaseObject {
private:
    // Index build parameters
    size_t max_elements_;
    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;

    // Index search parameter
    size_t ef_;

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
    tableint max_id;
    size_t maxlevel_;

    // Index data structures
    tableint entrypoint_node_;
    char *data_level0_memory_;
    char **linkLists_;
    vecsim_stl::vector<size_t> element_levels_;
    vecsim_stl::unordered_map<labeltype, tableint> label_lookup_;
    std::shared_ptr<VisitedNodesHandler> visited_nodes_handler;

    // used for synchronization only when parallel indexing / searching is enabled.
#ifdef ENABLE_PARALLELIZATION
    std::unique_ptr<VisitedNodesHandlerPool> visited_nodes_handler_pool;
    size_t pool_initial_size;
    std::mutex global;
    std::mutex cur_element_count_guard_;
    std::vector<std::mutex> link_list_locks_;
#endif

    // callback for computing distance between two points in the underline space.
    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_;
#ifdef BUILD_TESTS
    friend class HNSWIndexSerializer;
    // Allow the following test to access the index size private member.
    friend class HNSWLibTest_preferAdHocOptimization_Test;
    friend class HNSWLibTest_test_dynamic_hnsw_info_iterator_Test;
    friend class AllocatorTest_testIncomingEdgesSet_Test;
    friend class AllocatorTest_test_hnsw_reclaim_memory_Test;
    friend class HNSWLibTest_testSizeEstimation_Test;
#endif

    HierarchicalNSW() {}                                // default constructor
    HierarchicalNSW(const HierarchicalNSW &) = default; // default (shallow) copy constructor
    void setExternalLabel(tableint internal_id, labeltype label);
    labeltype *getExternalLabelPtr(tableint internal_id) const;
    size_t getRandomLevel(double reverse_size);
    vecsim_stl::vector<tableint> *getIncomingEdgesPtr(tableint internal_id, size_t level) const;
    void setIncomingEdgesPtr(tableint internal_id, size_t level, void *set_ptr);
    linklistsizeint *get_linklist0(tableint internal_id) const;
    linklistsizeint *get_linklist(tableint internal_id, size_t level) const;
    void setListCount(linklistsizeint *ptr, unsigned short int size);
    void removeExtraLinks(linklistsizeint *node_ll, candidatesMaxHeap<dist_t> candidates,
                          size_t Mcurmax, tableint *node_neighbors,
                          const vecsim_stl::vector<bool> &bitmap, tableint *removed_links,
                          size_t *removed_links_num);
    inline dist_t processCandidate(tableint curNodeId, const void *data_point, size_t layer,
                                   size_t ef, tag_t visited_tag,
                                   candidatesMaxHeap<dist_t> &top_candidates,
                                   candidatesMaxHeap<dist_t> &candidates_set,
                                   dist_t lowerBound) const;
    candidatesMaxHeap<dist_t> searchLayer(tableint ep_id, const void *data_point, size_t layer,
                                          size_t ef) const;
    candidatesLabelsMaxHeap<dist_t> searchBottomLayer_WithTimeout(tableint ep_id,
                                                                  const void *data_point, size_t ef,
                                                                  size_t k, void *timeoutCtx,
                                                                  VecSimQueryResult_Code *rc) const;
    void getNeighborsByHeuristic2(candidatesMaxHeap<dist_t> &top_candidates, size_t M);
    tableint mutuallyConnectNewElement(tableint cur_c, candidatesMaxHeap<dist_t> &top_candidates,
                                       size_t level);
    void repairConnectionsForDeletion(tableint element_internal_id, tableint neighbour_id,
                                      tableint *neighbours_list,
                                      tableint *neighbour_neighbours_list, size_t level,
                                      vecsim_stl::vector<bool> &neighbours_bitmap);
    void replaceEntryPoint(tableint element_internal_id);
    void SwapLastIdWithDeletedId(tableint element_internal_id, tableint last_element_internal_id);

public:
    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements,
                    std::shared_ptr<VecSimAllocator> allocator, size_t M = 16,
                    size_t ef_construction = 200, size_t ef = 10, size_t random_seed = 100,
                    size_t initial_pool_size = 1);
    virtual ~HierarchicalNSW();

    void setEf(size_t ef);
    size_t getEf() const;
    size_t getIndexSize() const;
    size_t getIndexCapacity() const;
    size_t getEfConstruction() const;
    size_t getM() const;
    size_t getMaxLevel() const;
    size_t getEntryPointLabel() const;
    tableint getEntryPointId() const;
    labeltype getExternalLabel(tableint internal_id) const;
    VisitedNodesHandler *getVisitedList() const;
    char *getDataByInternalId(tableint internal_id) const;
    linklistsizeint *get_linklist_at_level(tableint internal_id, size_t level) const;
    unsigned short int getListCount(const linklistsizeint *ptr) const;
    void resizeIndex(size_t new_max_elements);
    bool removePoint(labeltype label);
    void addPoint(const void *data_point, labeltype label);
    dist_t getDistanceByLabelFromPoint(labeltype label, const void *data_point);
    tableint searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                 VecSimQueryResult_Code *rc) const;
    vecsim_stl::max_priority_queue<pair<dist_t, labeltype>>
    searchKnn(const void *query_data, size_t k, void *timeoutCtx, VecSimQueryResult_Code *rc) const;
};

/**
 * getters and setters of index data
 */

template <typename dist_t>
void HierarchicalNSW<dist_t>::setEf(size_t ef) {
    ef_ = ef;
}

template <typename dist_t>
size_t HierarchicalNSW<dist_t>::getEf() const {
    return ef_;
}

template <typename dist_t>
size_t HierarchicalNSW<dist_t>::getIndexSize() const {
    return cur_element_count;
}

template <typename dist_t>
size_t HierarchicalNSW<dist_t>::getIndexCapacity() const {
    return max_elements_;
}

template <typename dist_t>
size_t HierarchicalNSW<dist_t>::getEfConstruction() const {
    return ef_construction_;
}

template <typename dist_t>
size_t HierarchicalNSW<dist_t>::getM() const {
    return M_;
}

template <typename dist_t>
size_t HierarchicalNSW<dist_t>::getMaxLevel() const {
    return maxlevel_;
}

template <typename dist_t>
size_t HierarchicalNSW<dist_t>::getEntryPointLabel() const {
    if (entrypoint_node_ != HNSW_INVALID_ID)
        return (size_t)getExternalLabel(entrypoint_node_);
    return SIZE_MAX;
}

template <typename dist_t>
labeltype HierarchicalNSW<dist_t>::getExternalLabel(tableint internal_id) const {
    labeltype return_label;
    memcpy(&return_label,
           (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
           sizeof(labeltype));
    return return_label;
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::setExternalLabel(tableint internal_id, labeltype label) {
    memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label,
           sizeof(labeltype));
}

template <typename dist_t>
labeltype *HierarchicalNSW<dist_t>::getExternalLabelPtr(tableint internal_id) const {
    return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ +
                         label_offset_);
}

template <typename dist_t>
char *HierarchicalNSW<dist_t>::getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
}

template <typename dist_t>
size_t HierarchicalNSW<dist_t>::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (size_t)r;
}

template <typename dist_t>
dist_t HierarchicalNSW<dist_t>::getDistanceByLabelFromPoint(labeltype label,
                                                            const void *data_point) {
    if (label_lookup_.find(label) == label_lookup_.end()) {
        return INVALID_SCORE;
    }
    dist_t t =
        fstdistfunc_(data_point, getDataByInternalId(label_lookup_[label]), dist_func_param_);
    return t;
}

template <typename dist_t>
vecsim_stl::vector<tableint> *HierarchicalNSW<dist_t>::getIncomingEdgesPtr(tableint internal_id,
                                                                           size_t level) const {
    if (level == 0) {
        return reinterpret_cast<vecsim_stl::vector<tableint> *>(
            *(void **)(data_level0_memory_ + internal_id * size_data_per_element_ +
                       incoming_links_offset0));
    }
    return reinterpret_cast<vecsim_stl::vector<tableint> *>(
        *(void **)(linkLists_[internal_id] + (level - 1) * size_links_per_element_ +
                   incoming_links_offset));
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::setIncomingEdgesPtr(tableint internal_id, size_t level,
                                                  void *set_ptr) {
    if (level == 0) {
        memcpy(data_level0_memory_ + internal_id * size_data_per_element_ + incoming_links_offset0,
               &set_ptr, sizeof(void *));
    } else {
        memcpy(linkLists_[internal_id] + (level - 1) * size_links_per_element_ +
                   incoming_links_offset,
               &set_ptr, sizeof(void *));
    }
}

template <typename dist_t>
linklistsizeint *HierarchicalNSW<dist_t>::get_linklist0(tableint internal_id) const {
    return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ +
                               offsetLevel0_);
}

template <typename dist_t>
linklistsizeint *HierarchicalNSW<dist_t>::get_linklist(tableint internal_id, size_t level) const {
    return (linklistsizeint *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
}

template <typename dist_t>
linklistsizeint *HierarchicalNSW<dist_t>::get_linklist_at_level(tableint internal_id,
                                                                size_t level) const {
    return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
}

template <typename dist_t>
unsigned short int HierarchicalNSW<dist_t>::getListCount(const linklistsizeint *ptr) const {
    return *((unsigned short int *)ptr);
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::setListCount(linklistsizeint *ptr, unsigned short int size) {
    *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
}

template <typename dist_t>
tableint HierarchicalNSW<dist_t>::getEntryPointId() const {
    return entrypoint_node_;
}

template <typename dist_t>
VisitedNodesHandler *HierarchicalNSW<dist_t>::getVisitedList() const {
    return visited_nodes_handler.get();
}

/**
 * helper functions
 */
template <typename dist_t>
void HierarchicalNSW<dist_t>::removeExtraLinks(linklistsizeint *node_ll,
                                               candidatesMaxHeap<dist_t> candidates, size_t Mcurmax,
                                               tableint *node_neighbors,
                                               const vecsim_stl::vector<bool> &neighbors_bitmap,
                                               tableint *removed_links, size_t *removed_links_num) {

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

template <typename dist_t>
dist_t HierarchicalNSW<dist_t>::processCandidate(tableint curNodeId, const void *data_point,
                                                 size_t layer, size_t ef, tag_t visited_tag,
                                                 candidatesMaxHeap<dist_t> &top_candidates,
                                                 candidatesMaxHeap<dist_t> &candidate_set,
                                                 dist_t lowerBound) const {

#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> lock(link_list_locks_[curNodeId]);
#endif
    linklistsizeint *node_ll = get_linklist_at_level(curNodeId, layer);
    size_t links_num = getListCount(node_ll);
    auto *node_links = (tableint *)(node_ll + 1);

    __builtin_prefetch(visited_nodes_handler->getElementsTags() + *(node_ll + 1));
    __builtin_prefetch(visited_nodes_handler->getElementsTags() + *(node_ll + 1) + 64);
    __builtin_prefetch(getDataByInternalId(*node_links));
    __builtin_prefetch(getDataByInternalId(*(node_links + 1)));

    for (size_t j = 0; j < links_num; j++) {
        tableint candidate_id = *(node_links + j);

        __builtin_prefetch(visited_nodes_handler->getElementsTags() + *(node_links + j + 1));
        __builtin_prefetch(getDataByInternalId(*(node_links + j + 1)));

        if (this->visited_nodes_handler->getNodeTag(candidate_id) == visited_tag)
            continue;
        this->visited_nodes_handler->tagNode(candidate_id, visited_tag);
        char *currObj1 = (getDataByInternalId(candidate_id));

        dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
        if (top_candidates.size() < ef || lowerBound > dist1) {
            candidate_set.emplace(-dist1, candidate_id);

            __builtin_prefetch(getDataByInternalId(candidate_set.top().second));

            top_candidates.emplace(dist1, candidate_id);

            if (top_candidates.size() > ef)
                top_candidates.pop();

            if (!top_candidates.empty())
                lowerBound = top_candidates.top().first;
        }
    }
    return lowerBound;
}

template <typename dist_t>
candidatesMaxHeap<dist_t> HierarchicalNSW<dist_t>::searchLayer(tableint ep_id,
                                                               const void *data_point, size_t layer,
                                                               size_t ef) const {

#ifdef ENABLE_PARALLELIZATION
    this->visited_nodes_handler =
        this->visited_nodes_handler_pool->getAvailableVisitedNodesHandler();
#endif

    tag_t visited_tag = this->visited_nodes_handler->getFreshTag();

    candidatesMaxHeap<dist_t> top_candidates(this->allocator);
    candidatesMaxHeap<dist_t> candidate_set(this->allocator);

    dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
    dist_t lowerBound = dist;
    top_candidates.emplace(dist, ep_id);
    candidate_set.emplace(-dist, ep_id);

    this->visited_nodes_handler->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidate_set.top();
        if ((-curr_el_pair.first) > lowerBound) {
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

template <typename dist_t>
void HierarchicalNSW<dist_t>::getNeighborsByHeuristic2(candidatesMaxHeap<dist_t> &top_candidates,
                                                       const size_t M) {
    if (top_candidates.size() < M) {
        return;
    }

    candidatesMaxHeap<dist_t> queue_closest(this->allocator);
    vecsim_stl::vector<std::pair<dist_t, tableint>> return_list(this->allocator);
    while (top_candidates.size() > 0) {
        // the distance is saved negatively to have the queue ordered such that first is closer
        // (higher).
        queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
        top_candidates.pop();
    }

    while (queue_closest.size()) {
        if (return_list.size() >= M)
            break;
        std::pair<dist_t, tableint> current_pair = queue_closest.top();
        dist_t dist_to_query = -current_pair.first;
        queue_closest.pop();
        bool good = true;

        // a candidate is "good" to become a neighbour, unless we find
        // another item that was already selected to the neighbours set which is closer
        // to both q and the candidate than the distance between the candidate and q.
        for (std::pair<dist_t, tableint> second_pair : return_list) {
            dist_t curdist =
                fstdistfunc_(getDataByInternalId(second_pair.second),
                             getDataByInternalId(current_pair.second), dist_func_param_);
            ;
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.push_back(current_pair);
        }
    }

    for (std::pair<dist_t, tableint> current_pair : return_list) {
        top_candidates.emplace(-current_pair.first, current_pair.second);
    }
}

template <typename dist_t>
tableint HierarchicalNSW<dist_t>::mutuallyConnectNewElement(
    tableint cur_c, candidatesMaxHeap<dist_t> &top_candidates, size_t level) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, M_);
    if (top_candidates.size() > M_)
        throw std::runtime_error(
            "Should be not be more than M_ candidates returned by the heuristic");

    vecsim_stl::vector<tableint> selectedNeighbors(this->allocator);
    selectedNeighbors.reserve(M_);
    while (top_candidates.size() > 0) {
        selectedNeighbors.push_back(top_candidates.top().second);
        top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();
    {
        linklistsizeint *ll_cur = get_linklist_at_level(cur_c, level);
        if (*ll_cur) {
            throw std::runtime_error("The newly inserted element should have blank link list");
        }
        setListCount(ll_cur, selectedNeighbors.size());
        auto *data = (tableint *)(ll_cur + 1);
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            if (data[idx])
                throw std::runtime_error("Possible memory corruption");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");
            data[idx] = selectedNeighbors[idx];
        }
        auto *incoming_edges = new (this->allocator) vecsim_stl::vector<tableint>(this->allocator);
        setIncomingEdgesPtr(cur_c, level, (void *)incoming_edges);
    }

    // go over the selected neighbours - selectedNeighbor is the neighbour id
    vecsim_stl::vector<bool> neighbors_bitmap(allocator);
    for (tableint selectedNeighbor : selectedNeighbors) {
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
        auto *neighbor_neighbors = (tableint *)(ll_other + 1);

        // If the selected neighbor can add another link (hasn't reached the max) - add it.
        if (sz_link_list_other < Mcurmax) {
            neighbor_neighbors[sz_link_list_other] = cur_c;
            setListCount(ll_other, sz_link_list_other + 1);
        } else {
            // try finding "weak" elements to replace it with the new one with the heuristic:
            candidatesMaxHeap<dist_t> candidates(this->allocator);
            // (re)use the bitmap to represent the set of the original neighbours for the current
            // selected neighbour.
            neighbors_bitmap.assign(max_id + 1, false);
            dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c),
                                        getDataByInternalId(selectedNeighbor), dist_func_param_);
            candidates.emplace(d_max, cur_c);
            // consider cur_c as if it was a link of the selected neighbor
            neighbors_bitmap[cur_c] = true;
            for (size_t j = 0; j < sz_link_list_other; j++) {
                candidates.emplace(fstdistfunc_(getDataByInternalId(neighbor_neighbors[j]),
                                                getDataByInternalId(selectedNeighbor),
                                                dist_func_param_),
                                   neighbor_neighbors[j]);
                neighbors_bitmap[neighbor_neighbors[j]] = true;
            }

            tableint removed_links[sz_link_list_other + 1];
            size_t removed_links_num;
            removeExtraLinks(ll_other, candidates, Mcurmax, neighbor_neighbors, neighbors_bitmap,
                             removed_links, &removed_links_num);

            // remove the current neighbor from the incoming list of nodes for the
            // neighbours that were chosen to remove (if edge wasn't bidirectional)
            auto *neighbour_incoming_edges = getIncomingEdgesPtr(selectedNeighbor, level);
            for (size_t i = 0; i < removed_links_num; i++) {
                tableint node_id = removed_links[i];
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

template <typename dist_t>
void HierarchicalNSW<dist_t>::repairConnectionsForDeletion(
    tableint element_internal_id, tableint neighbour_id, tableint *neighbours_list,
    tableint *neighbour_neighbours_list, size_t level,
    vecsim_stl::vector<bool> &neighbours_bitmap) {

    // put the deleted element's neighbours in the candidates.
    candidatesMaxHeap<dist_t> candidates(this->allocator);
    unsigned short neighbours_count = getListCount(neighbours_list);
    auto *neighbours = (tableint *)(neighbours_list + 1);
    for (size_t j = 0; j < neighbours_count; j++) {
        // Don't put the neighbor itself in his own candidates
        if (neighbours[j] == neighbour_id) {
            continue;
        }
        candidates.emplace(fstdistfunc_(getDataByInternalId(neighbours[j]),
                                        getDataByInternalId(neighbour_id), dist_func_param_),
                           neighbours[j]);
    }

    // add the deleted element's neighbour's original neighbors in the candidates.
    vecsim_stl::vector<bool> neighbour_orig_neighbours_set(max_id + 1, false, this->allocator);
    unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);
    auto *neighbour_neighbours = (tableint *)(neighbour_neighbours_list + 1);
    for (size_t j = 0; j < neighbour_neighbours_count; j++) {
        neighbour_orig_neighbours_set[neighbour_neighbours[j]] = true;
        // Don't add the removed element to the candidates, nor nodes that are already in the
        // candidates set.
        if (neighbours_bitmap[neighbour_neighbours[j]] ||
            neighbour_neighbours[j] == element_internal_id) {
            continue;
        }
        candidates.emplace(fstdistfunc_(getDataByInternalId(neighbour_id),
                                        getDataByInternalId(neighbour_neighbours[j]),
                                        dist_func_param_),
                           neighbour_neighbours[j]);
    }

    size_t Mcurmax = level ? maxM_ : maxM0_;
    size_t removed_links_num;
    tableint removed_links[neighbour_neighbours_count];
    removeExtraLinks(neighbour_neighbours_list, candidates, Mcurmax, neighbour_neighbours,
                     neighbour_orig_neighbours_set, removed_links, &removed_links_num);

    // remove neighbour id from the incoming list of nodes for his
    // neighbours that were chosen to remove
    auto *neighbour_incoming_edges = getIncomingEdgesPtr(neighbour_id, level);

    for (size_t i = 0; i < removed_links_num; i++) {
        tableint node_id = removed_links[i];
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
        tableint node_id = neighbour_neighbours[i];
        if (!neighbour_orig_neighbours_set[node_id]) {
            auto *node_incoming_edges = getIncomingEdgesPtr(node_id, level);
            // if the node has an edge to the neighbour as well, remove it
            // from the incoming nodes of the neighbour
            // otherwise, need to update the edge as incoming.
            linklistsizeint *node_links_list = get_linklist_at_level(node_id, level);
            unsigned short node_links_size = getListCount(node_links_list);
            auto *node_links = (tableint *)(node_links_list + 1);
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

template <typename dist_t>
void HierarchicalNSW<dist_t>::replaceEntryPoint(tableint element_internal_id) {
    // Sets an (arbitrary) new entry point, after deleting the current entry point.
    while (element_internal_id == entrypoint_node_) {
        linklistsizeint *top_level_list = get_linklist_at_level(element_internal_id, maxlevel_);
        if (getListCount(top_level_list) > 0) {
            // Tries to set the (arbitrary) first neighbor as the entry point, if exists.
            entrypoint_node_ = ((tableint *)(top_level_list + 1))[0];
        } else {
            // If there is no neighbors in the current level, check for any vector at
            // this level to be the new entry point.
            for (tableint cur_id = 0; cur_id < cur_element_count; cur_id++) {
                if (element_levels_[cur_id] == maxlevel_ && cur_id != element_internal_id) {
                    entrypoint_node_ = cur_id;
                    break;
                }
            }
        }
        // If we didn't find any vector at the top level, decrease the maxlevel_ and try again,
        // until we find a new entry point, or the index is empty.
        if (element_internal_id == entrypoint_node_) {
            maxlevel_--;
            if ((int)maxlevel_ < 0) {
                maxlevel_ = HNSW_INVALID_LEVEL;
                entrypoint_node_ = HNSW_INVALID_ID;
            }
        }
    }
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::SwapLastIdWithDeletedId(tableint element_internal_id,
                                                      tableint last_element_internal_id) {
    // swap label
    labeltype last_element_label = getExternalLabel(last_element_internal_id);
    label_lookup_[last_element_label] = element_internal_id;

    // swap neighbours
    size_t last_element_top_level = element_levels_[last_element_internal_id];
    for (size_t level = 0; level <= last_element_top_level; level++) {
        linklistsizeint *neighbours_list = get_linklist_at_level(last_element_internal_id, level);
        unsigned short neighbours_count = getListCount(neighbours_list);
        auto *neighbours = (tableint *)(neighbours_list + 1);

        // go over the neighbours that also points back to the last element whose is going to
        // change, and update the id.
        for (size_t i = 0; i < neighbours_count; i++) {
            tableint neighbour_id = neighbours[i];
            linklistsizeint *neighbour_neighbours_list = get_linklist_at_level(neighbour_id, level);
            unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);

            auto *neighbour_neighbours = (tableint *)(neighbour_neighbours_list + 1);
            bool bidirectional_edge = false;
            for (size_t j = 0; j < neighbour_neighbours_count; j++) {
                // if the edge is bidirectional, update for this neighbor
                if (neighbour_neighbours[j] == last_element_internal_id) {
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
                                    neighbour_incoming_edges->end(), last_element_internal_id);
                neighbour_incoming_edges->erase(it);
                neighbour_incoming_edges->push_back(element_internal_id);
            }
        }

        // next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // updates.
        auto *incoming_edges = getIncomingEdgesPtr(last_element_internal_id, level);
        for (auto incoming_edge : *incoming_edges) {
            linklistsizeint *incoming_neighbour_neighbours_list =
                get_linklist_at_level(incoming_edge, level);
            unsigned short incoming_neighbour_neighbours_count =
                getListCount(incoming_neighbour_neighbours_list);
            auto *incoming_neighbour_neighbours =
                (tableint *)(incoming_neighbour_neighbours_list + 1);
            for (size_t j = 0; j < incoming_neighbour_neighbours_count; j++) {
                if (incoming_neighbour_neighbours[j] == last_element_internal_id) {
                    incoming_neighbour_neighbours[j] = element_internal_id;
                    break;
                }
            }
        }
    }

    // swap the last_id level 0 data, and invalidate the deleted id's data
    memcpy(data_level0_memory_ + element_internal_id * size_data_per_element_ + offsetLevel0_,
           data_level0_memory_ + last_element_internal_id * size_data_per_element_ + offsetLevel0_,
           size_data_per_element_);
    memset(data_level0_memory_ + last_element_internal_id * size_data_per_element_ + offsetLevel0_,
           0, size_data_per_element_);

    // swap pointer of higher levels links
    linkLists_[element_internal_id] = linkLists_[last_element_internal_id];
    linkLists_[last_element_internal_id] = nullptr;

    // swap top element level
    element_levels_[element_internal_id] = element_levels_[last_element_internal_id];
    element_levels_[last_element_internal_id] = HNSW_INVALID_LEVEL;

    if (last_element_internal_id == this->entrypoint_node_) {
        this->entrypoint_node_ = element_internal_id;
    }
}

template <typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements,
                                         std::shared_ptr<VecSimAllocator> allocator, size_t M,
                                         size_t ef_construction, size_t ef, size_t random_seed,
                                         size_t pool_initial_size)
    : VecsimBaseObject(allocator), element_levels_(max_elements, allocator),
      label_lookup_(max_elements, allocator)

#ifdef ENABLE_PARALLELIZATION
          link_list_locks_(max_elements),
#endif
{

    max_elements_ = max_elements;
    if (M > SIZE_MAX / 2)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = ef;

    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_data_dim();

    cur_element_count = 0;
    max_id = HNSW_INVALID_ID;
#ifdef ENABLE_PARALLELIZATION
    pool_initial_size = pool_initial_size;
    visited_nodes_handler_pool = std::unique_ptr<VisitedNodesHandlerPool>(new (
        this->allocator) VisitedNodesHandlerPool(pool_initial_size, max_elements, this->allocator));
#else
    visited_nodes_handler = std::shared_ptr<VisitedNodesHandler>(
        new (this->allocator) VisitedNodesHandler(max_elements, this->allocator));
#endif

    // initializations for special treatment of the first node
    entrypoint_node_ = HNSW_INVALID_ID;
    maxlevel_ = HNSW_INVALID_LEVEL;

    if (M <= 1)
        throw std::runtime_error("HNSW index parameter M cannot be 1");
    mult_ = 1 / log(1.0 * M_);
    level_generator_.seed(random_seed);

    // data_level0_memory will look like this:
    // -----4------ | -----4*M0----------- | ----8------------------| ------32------- | ----8---- |
    // <links_len>  | <link_1> <link_2>... | <incoming_links_set> |   <data>        |  <label>
    if (maxM0_ > ((SIZE_MAX - sizeof(void *) - sizeof(linklistsizeint)) / sizeof(tableint)) + 1)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    size_links_level0_ = sizeof(linklistsizeint) + maxM0_ * sizeof(tableint) + sizeof(void *);

    if (size_links_level0_ > SIZE_MAX - data_size_ - sizeof(labeltype))
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);

    // No need to test for overflow because we passed the test for size_links_level0_ and this is
    // less.
    incoming_links_offset0 = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ =
        (char *)this->allocator->callocate(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory");

    linkLists_ = (char **)this->allocator->callocate(sizeof(void *) * max_elements_);
    if (linkLists_ == nullptr)
        throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");

    // The i-th entry in linkLists array points to max_level[i] (continuous)
    // chunks of memory, each one will look like this:
    // -----4------ | -----4*M-------------- | ----8------------------|
    // <links_len>  | <link_1> <link_2> ...  | <incoming_links_set>
    size_links_per_element_ = sizeof(linklistsizeint) + maxM_ * sizeof(tableint) + sizeof(void *);
    // No need to test for overflow because we passed the test for incoming_links_offset0 and this
    // is less.
    incoming_links_offset = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
}

template <typename dist_t>
HierarchicalNSW<dist_t>::~HierarchicalNSW() {
    if (max_id != HNSW_INVALID_ID) {
        for (tableint id = 0; id <= max_id; id++) {
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
template <typename dist_t>
void HierarchicalNSW<dist_t>::resizeIndex(size_t new_max_elements) {
    element_levels_.resize(new_max_elements);
    element_levels_.shrink_to_fit();
    label_lookup_.reserve(new_max_elements);
#ifdef ENABLE_PARALLELIZATION
    visited_nodes_handler_pool = std::unique_ptr<VisitedNodesHandlerPool>(
        new (this->allocator)
            VisitedNodesHandlerPool(this->pool_initial_size, new_max_elements, this->allocator));
    std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);
#else
    visited_nodes_handler = std::unique_ptr<VisitedNodesHandler>(
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

template <typename dist_t>
bool HierarchicalNSW<dist_t>::removePoint(const labeltype label) {
    // check that the label actually exists in the graph, and update the number of elements.
    if (label_lookup_.find(label) == label_lookup_.end()) {
        return true;
    }
    tableint element_internal_id = label_lookup_[label];
    vecsim_stl::vector<bool> neighbours_bitmap(allocator);

    // go over levels and repair connections
    size_t element_top_level = element_levels_[element_internal_id];
    for (size_t level = 0; level <= element_top_level; level++) {
        linklistsizeint *neighbours_list = get_linklist_at_level(element_internal_id, level);
        unsigned short neighbours_count = getListCount(neighbours_list);
        auto *neighbours = (tableint *)(neighbours_list + 1);
        // reset the neighbours' bitmap for the current level.
        neighbours_bitmap.assign(max_id + 1, false);
        // store the deleted element's neighbours set in a bitmap for fast access.
        for (size_t j = 0; j < neighbours_count; j++) {
            neighbours_bitmap[neighbours[j]] = true;
        }
        // go over the neighbours that also points back to the removed point and make a local
        // repair.
        for (size_t i = 0; i < neighbours_count; i++) {
            tableint neighbour_id = neighbours[i];
            linklistsizeint *neighbour_neighbours_list = get_linklist_at_level(neighbour_id, level);
            unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);

            auto *neighbour_neighbours = (tableint *)(neighbour_neighbours_list + 1);
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
        replaceEntryPoint(element_internal_id);
    }

    // Swap the last id with the deleted one, and invalidate the last id data.
    tableint last_element_internal_id = --cur_element_count;
    --max_id;
    label_lookup_.erase(label);
    if (element_levels_[element_internal_id] > 0) {
        this->allocator->free_allocation(linkLists_[element_internal_id]);
        linkLists_[element_internal_id] = nullptr;
    }
    if (last_element_internal_id == element_internal_id) {
        // we're deleting the last internal id, just invalidate data without swapping.
        memset(data_level0_memory_ + last_element_internal_id * size_data_per_element_ +
                   offsetLevel0_,
               0, size_data_per_element_);
    } else {
        SwapLastIdWithDeletedId(element_internal_id, last_element_internal_id);
    }
    return true;
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::addPoint(const void *data_point, const labeltype label) {

    tableint cur_c;

    {
#ifdef ENABLE_PARALLELIZATION
        std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);
#endif
        // Checking if an element with the given label already exists. if so, remove it.
        if (label_lookup_.find(label) != label_lookup_.end()) {
            removePoint(label);
        }
        if (cur_element_count >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }
        cur_c = max_id = cur_element_count++;
        label_lookup_[label] = cur_c;
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
    memcpy(getExternalLabelPtr(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);

    if (element_max_level > 0) {
        linkLists_[cur_c] =
            (char *)this->allocator->allocate(size_links_per_element_ * element_max_level + 1);
        if (linkLists_[cur_c] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        memset(linkLists_[cur_c], 0, size_links_per_element_ * element_max_level + 1);
    }

    // this condition only means that we are not inserting the first element.
    if (entrypoint_node_ != HNSW_INVALID_ID) {
        if (element_max_level < maxlevelcopy) {
            dist_t cur_dist =
                fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
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

                    auto *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error(
                                "candidate error: candidate id is out of index range");

                        dist_t d =
                            fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
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

            candidatesMaxHeap<dist_t> top_candidates =
                searchLayer(currObj, data_point, level, ef_construction_);
            currObj = mutuallyConnectNewElement(cur_c, top_candidates, level);
        }

        // updating the maximum level (holding a global lock)
        if (element_max_level > maxlevelcopy) {
            entrypoint_node_ = cur_c;
            maxlevel_ = element_max_level;
            // create the incoming edges set for the new levels.
            for (size_t level_idx = maxlevelcopy + 1; level_idx <= element_max_level; level_idx++) {
                auto *incoming_edges =
                    new (this->allocator) vecsim_stl::vector<tableint>(this->allocator);
                setIncomingEdgesPtr(cur_c, level_idx, incoming_edges);
            }
        }
    } else {
        // Do nothing for the first element
        entrypoint_node_ = 0;
        for (size_t level_idx = maxlevel_ + 1; level_idx <= element_max_level; level_idx++) {
            auto *incoming_edges =
                new (this->allocator) vecsim_stl::vector<tableint>(this->allocator);
            setIncomingEdgesPtr(cur_c, level_idx, incoming_edges);
        }
        maxlevel_ = element_max_level;
    }
}

template <typename dist_t>
tableint HierarchicalNSW<dist_t>::searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                                      VecSimQueryResult_Code *rc) const {

    if (cur_element_count == 0) {
        return entrypoint_node_;
    }
    tableint currObj = entrypoint_node_;
    dist_t cur_dist =
        fstdistfunc_(query_data, getDataByInternalId(entrypoint_node_), dist_func_param_);
    for (size_t level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            if (__builtin_expect(VecSimIndex::timeoutCallback(timeoutCtx), 0)) {
                *rc = VecSim_QueryResult_TimedOut;
                return HNSW_INVALID_ID;
            }
            changed = false;
            linklistsizeint *node_ll = get_linklist(currObj, level);
            unsigned short links_count = getListCount(node_ll);
            auto *node_links = (tableint *)(node_ll + 1);
            for (int i = 0; i < links_count; i++) {
                tableint candidate = node_links[i];
                if (candidate > max_elements_)
                    throw std::runtime_error("candidate error: out of index range");

                dist_t d =
                    fstdistfunc_(query_data, getDataByInternalId(candidate), dist_func_param_);
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

template <typename dist_t>
candidatesLabelsMaxHeap<dist_t>
HierarchicalNSW<dist_t>::searchBottomLayer_WithTimeout(tableint ep_id, const void *data_point,
                                                       size_t ef, size_t k, void *timeoutCtx,
                                                       VecSimQueryResult_Code *rc) const {
    candidatesLabelsMaxHeap<dist_t> results(this->allocator);

#ifdef ENABLE_PARALLELIZATION
    this->visited_nodes_handler =
        this->visited_nodes_handler_pool->getAvailableVisitedNodesHandler();
#endif

    tag_t visited_tag = this->visited_nodes_handler->getFreshTag();

    candidatesMaxHeap<dist_t> top_candidates(this->allocator);
    candidatesMaxHeap<dist_t> candidate_set(this->allocator);

    dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
    dist_t lowerBound = dist;
    top_candidates.emplace(dist, ep_id);
    candidate_set.emplace(-dist, ep_id);

    this->visited_nodes_handler->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidate_set.top();
        if ((-curr_el_pair.first) > lowerBound) {
            break;
        }
        if (__builtin_expect(VecSimIndex::timeoutCallback(timeoutCtx), 0)) {
            *rc = VecSim_QueryResult_TimedOut;
            return results;
        }
        candidate_set.pop();

        lowerBound = processCandidate(curr_el_pair.second, data_point, 0, ef, visited_tag,
                                      top_candidates, candidate_set, lowerBound);
    }
#ifdef ENABLE_PARALLELIZATION
    visited_nodes_handler_pool->returnVisitedNodesHandlerToPool(this->visited_nodes_handler);
#endif
    while (top_candidates.size() > k) {
        top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
        auto &res = top_candidates.top();
        results.emplace(res.first, getExternalLabel(res.second));
        top_candidates.pop();
    }
    *rc = VecSim_QueryResult_OK;
    return results;
}

template <typename dist_t>
candidatesLabelsMaxHeap<dist_t>
HierarchicalNSW<dist_t>::searchKnn(const void *query_data, size_t k, void *timeoutCtx,
                                   VecSimQueryResult_Code *rc) const {

    if (cur_element_count == 0) {
        *rc = VecSim_QueryResult_OK;
        return candidatesLabelsMaxHeap<dist_t>(this->allocator);
    }

    tableint bottom_layer_ep = searchBottomLayerEP(query_data, timeoutCtx, rc);
    if (VecSim_OK != *rc) {
        return candidatesLabelsMaxHeap<dist_t>(this->allocator);
    }
    candidatesLabelsMaxHeap<dist_t> results = searchBottomLayer_WithTimeout(
        bottom_layer_ep, query_data, std::max(ef_, k), k, timeoutCtx, rc);

    return results;
}

} // namespace hnswlib
