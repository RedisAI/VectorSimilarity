#pragma once

#include "visited_list_pool.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim//spaces/space_interface.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"

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

namespace hnswlib {

using namespace std;

typedef size_t labeltype;
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
struct CompareByFirst {
    constexpr bool operator()(pair<dist_t, tableint> const &a,
                              pair<dist_t, tableint> const &b) const noexcept {
        return a.first < b.first;
    }
};

template <typename dist_t>
using CandidatesQueue =
    vecsim_stl::priority_queue<pair<dist_t, tableint>, vecsim_stl::vector<pair<dist_t, tableint>>,
                               CompareByFirst<dist_t>>;

template <typename dist_t>
class HierarchicalNSW : VecsimBaseObject {

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
    int max_id;
    int maxlevel_;

    // Index data structures
    int enterpoint_node_;
    char *data_level0_memory_;
    char **linkLists_;
    vecsim_stl::vector<int> element_levels_;
    vecsim_stl::set<tableint> available_ids;
    vecsim_stl::unordered_map<labeltype, tableint> label_lookup_;
    VisitedListPool *visited_list_pool_;

    // used for synchronization only when parallel indexing / searching is enabled.
#ifdef ENABLE_PARALLELIZATION
    std::mutex global;
    std::mutex cur_element_count_guard_;
    std::vector<std::mutex> link_list_locks_;
#endif

    // callback for computing distance between two points in the underline space.
    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_;

    labeltype getExternalLabel(tableint internal_id) const;
    void setExternalLabel(tableint internal_id, labeltype label);
    labeltype *getExternalLabelPtr(tableint internal_id) const;
    char *getDataByInternalId(tableint internal_id) const;
    int getRandomLevel(double reverse_size);
    vecsim_stl::set<tableint> *getIncomingEdgesPtr(tableint internal_id, int level) const;
    void setIncomingEdgesPtr(tableint internal_id, int level, void *set_ptr);
    linklistsizeint *get_linklist0(tableint internal_id) const;
    linklistsizeint *get_linklist(tableint internal_id, int level) const;
    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const;
    unsigned short int getListCount(const linklistsizeint *ptr) const;
    void setListCount(linklistsizeint *ptr, unsigned short int size);
    void removeExtraLinks(linklistsizeint *node_ll, CandidatesQueue<dist_t> candidates,
                          size_t Mcurmax, tableint *node_neighbors,
                          const vecsim_stl::set<tableint> &orig_neighbors, tableint *removed_links,
                          size_t *removed_links_num);
    CandidatesQueue<dist_t> searchLayer(tableint ep_id, const void *data_point, int layer,
                                        size_t ef) const;
    void getNeighborsByHeuristic2(CandidatesQueue<dist_t> &top_candidates, size_t M);
    tableint mutuallyConnectNewElement(tableint cur_c, CandidatesQueue<dist_t> &top_candidates,
                                       int level);
    void repairConnectionsForDeletion(tableint element_internal_id, tableint neighbour_id,
                                      tableint *neighbours_list,
                                      tableint *neighbour_neighbours_list, int level);

public:
    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements,
                    std::shared_ptr<VecSimAllocator> allocator, size_t M = 16,
                    size_t ef_construction = 200, size_t ef = 10, size_t random_seed = 100);
    ~HierarchicalNSW();

    void setEf(size_t ef);
    size_t getEf() const;
    size_t getIndexSize() const;
    size_t getIndexCapacity() const;
    size_t getEfConstruction() const;
    size_t getM() const;
    size_t getMaxLevel() const;
    void resizeIndex(size_t new_max_elements);
    bool removePoint(labeltype label);
    void addPoint(const void *data_point, labeltype label);
    vecsim_stl::priority_queue<pair<dist_t, labeltype>> searchKnn(const void *query_data,
                                                                  size_t k) const;
    void checkIntegrity();
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
int HierarchicalNSW<dist_t>::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
}

template <typename dist_t>
vecsim_stl::set<tableint> *HierarchicalNSW<dist_t>::getIncomingEdgesPtr(tableint internal_id,
                                                                        int level) const {
    if (level == 0) {
        return reinterpret_cast<vecsim_stl::set<tableint> *>(
            *(void **)(data_level0_memory_ + internal_id * size_data_per_element_ +
                       incoming_links_offset0));
    }
    return reinterpret_cast<vecsim_stl::set<tableint> *>(
        *(void **)(linkLists_[internal_id] + (level - 1) * size_links_per_element_ +
                   incoming_links_offset));
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::setIncomingEdgesPtr(tableint internal_id, int level, void *set_ptr) {
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
linklistsizeint *HierarchicalNSW<dist_t>::get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
}

template <typename dist_t>
linklistsizeint *HierarchicalNSW<dist_t>::get_linklist_at_level(tableint internal_id,
                                                                int level) const {
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

/**
 * helper functions
 */
template <typename dist_t>
void HierarchicalNSW<dist_t>::removeExtraLinks(linklistsizeint *node_ll,
                                               CandidatesQueue<dist_t> candidates, size_t Mcurmax,
                                               tableint *node_neighbors,
                                               const vecsim_stl::set<tableint> &orig_neighbors,
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
            if (orig_neighbors.find(orig_candidates.top().second) != orig_neighbors.end()) {
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
CandidatesQueue<dist_t> HierarchicalNSW<dist_t>::searchLayer(tableint ep_id, const void *data_point,
                                                             int layer, size_t ef) const {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    CandidatesQueue<dist_t> top_candidates(this->allocator);
    CandidatesQueue<dist_t> candidate_set(this->allocator);

    dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
    dist_t lowerBound = dist;
    top_candidates.emplace(dist, ep_id);
    candidate_set.emplace(-dist, ep_id);

    visited_array[ep_id] = visited_array_tag;

    while (!candidate_set.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidate_set.top();
        if ((-curr_el_pair.first) > lowerBound) {
            break;
        }
        candidate_set.pop();

        tableint curNodeNum = curr_el_pair.second;
#ifdef ENABLE_PARALLELIZATION
        std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);
#endif
        linklistsizeint *node_ll = get_linklist_at_level(curNodeNum, layer);
        size_t links_num = getListCount(node_ll);
        auto *node_links = (tableint *)(node_ll + 1);
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(node_ll + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(visited_array + *(node_ll + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*node_links), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(node_links + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < links_num; j++) {
            tableint candidate_id = *(node_links + j);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(node_links + j + 1)), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(node_links + j + 1)), _MM_HINT_T0);
#endif
            if (visited_array[candidate_id] == visited_array_tag)
                continue;
            visited_array[candidate_id] = visited_array_tag;
            char *currObj1 = (getDataByInternalId(candidate_id));

            dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
            if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                candidate_set.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
                top_candidates.emplace(dist1, candidate_id);

                if (top_candidates.size() > ef)
                    top_candidates.pop();

                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }
#ifdef ENABLE_PARALLELIZATION
    visited_list_pool_->returnVisitedListToPool(vl);
#endif
    return top_candidates;
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::getNeighborsByHeuristic2(CandidatesQueue<dist_t> &top_candidates,
                                                       const size_t M) {
    if (top_candidates.size() < M) {
        return;
    }

    vecsim_stl::priority_queue<std::pair<dist_t, tableint>> queue_closest(this->allocator);
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
tableint HierarchicalNSW<dist_t>::mutuallyConnectNewElement(tableint cur_c,
                                                            CandidatesQueue<dist_t> &top_candidates,
                                                            int level) {
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
        auto *incoming_edges = new vecsim_stl::set<tableint>(this->allocator);
        setIncomingEdgesPtr(cur_c, level, (void *)incoming_edges);
    }

    // go over the selected neighbours - selectedNeighbor is the neighbour id
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
            CandidatesQueue<dist_t> candidates(this->allocator);
            vecsim_stl::set<tableint> orig_neighbors_set(this->allocator);
            dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c),
                                        getDataByInternalId(selectedNeighbor), dist_func_param_);
            candidates.emplace(d_max, cur_c);
            // consider cur_c as if it was a link of the selected neighbor
            orig_neighbors_set.insert(cur_c);

            for (size_t j = 0; j < sz_link_list_other; j++) {
                candidates.emplace(fstdistfunc_(getDataByInternalId(neighbor_neighbors[j]),
                                                getDataByInternalId(selectedNeighbor),
                                                dist_func_param_),
                                   neighbor_neighbors[j]);
                orig_neighbors_set.insert(neighbor_neighbors[j]);
            }

            tableint removed_links[sz_link_list_other + 1];
            size_t removed_links_num;
            removeExtraLinks(ll_other, candidates, Mcurmax, neighbor_neighbors, orig_neighbors_set,
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
                    neighbour_incoming_edges->insert(cur_c);
                    continue;
                }

                // if the node id (the neighbour's neighbour to be removed)
                // wasn't pointing to the neighbour (i.e., the edge was uni-directional),
                // we should remove the current neighbor from the node's incoming edges.
                // otherwise, the edge turned from bidirectional to
                // uni-directional, so we insert it to the neighbour's
                // incoming edges set.
                if (node_incoming_edges->find(selectedNeighbor) != node_incoming_edges->end()) {
                    node_incoming_edges->erase(selectedNeighbor);
                } else {
                    neighbour_incoming_edges->insert(node_id);
                }
            }
        }
    }
    return next_closest_entry_point;
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::repairConnectionsForDeletion(tableint element_internal_id,
                                                           tableint neighbour_id,
                                                           tableint *neighbours_list,
                                                           tableint *neighbour_neighbours_list,
                                                           int level) {

    // put the deleted element's neighbours in the candidates.
    CandidatesQueue<dist_t> candidates(this->allocator);
    vecsim_stl::set<tableint> candidates_set(this->allocator);
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
        candidates_set.insert(neighbours[j]);
    }

    // add the deleted element's neighbour's original neighbors in the candidates.
    vecsim_stl::set<tableint> neighbour_orig_neighbours_set(this->allocator);
    unsigned short neighbour_neighbours_count = getListCount(neighbour_neighbours_list);
    auto *neighbour_neighbours = (tableint *)(neighbour_neighbours_list + 1);
    for (size_t j = 0; j < neighbour_neighbours_count; j++) {
        neighbour_orig_neighbours_set.insert(neighbour_neighbours[j]);
        // Don't add the removed element to the candidates, nor nodes that are already in the
        // candidates set.
        if (candidates_set.find(neighbour_neighbours[j]) != candidates_set.end() ||
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
        if (node_incoming_edges->find(neighbour_id) != node_incoming_edges->end()) {
            node_incoming_edges->erase(neighbour_id);
        } else {
            neighbour_incoming_edges->insert(node_id);
        }
    }

    // updates for the new edges created
    unsigned short updated_links_num = getListCount(neighbour_neighbours_list);
    for (size_t i = 0; i < updated_links_num; i++) {
        tableint node_id = neighbour_neighbours[i];
        if (neighbour_orig_neighbours_set.find(node_id) == neighbour_orig_neighbours_set.end()) {
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
                    neighbour_incoming_edges->erase(node_id);
                    bidirectional_edge = true;
                    break;
                }
            }
            if (!bidirectional_edge) {
                node_incoming_edges->insert(neighbour_id);
            }
        }
    }
}

template <typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements,
                                         std::shared_ptr<VecSimAllocator> allocator, size_t M,
                                         size_t ef_construction, size_t ef, size_t random_seed)
    : VecsimBaseObject(allocator), element_levels_(max_elements, allocator),
      available_ids(allocator), label_lookup_(allocator)

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
    max_id = -1;
    visited_list_pool_ =
        new (this->allocator) VisitedListPool(1, (int)max_elements, this->allocator);

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

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

    data_level0_memory_ = (char *)this->allocator->allocate(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory");

    linkLists_ = (char **)this->allocator->allocate(sizeof(void *) * max_elements_);
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
    for (int id = 0; id <= max_id; id++) {
        if (available_ids.find(id) != available_ids.end()) {
            continue;
        }
        for (size_t level = 0; level <= element_levels_[id]; level++) {
            delete getIncomingEdgesPtr(id, level);
        }
        if (element_levels_[id] > 0)
            this->allocator->free_allocation(linkLists_[id]);
    }
    this->allocator->free_allocation(linkLists_);
    this->allocator->free_allocation(data_level0_memory_);
    delete visited_list_pool_;
}

/**
 * Index API functions
 */
template <typename dist_t>
void HierarchicalNSW<dist_t>::resizeIndex(size_t new_max_elements) {
    if (new_max_elements < cur_element_count)
        throw std::runtime_error(
            "Cannot resize, max element is less than the current number of elements");
    delete visited_list_pool_;
    visited_list_pool_ =
        new (this->allocator) VisitedListPool(1, (int)new_max_elements, this->allocator);
    element_levels_.resize(new_max_elements);
#ifdef ENABLE_PARALLELIZATION
    std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);
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
    tableint element_internal_id;
    if (label_lookup_.find(label) == label_lookup_.end()) {
        return true;
    }
    // add the element id to the available ids for future reuse.
    element_internal_id = label_lookup_[label];
    cur_element_count--;
    label_lookup_.erase(label);
    available_ids.insert(element_internal_id);

    // go over levels from the top and repair connections
    int element_top_level = element_levels_[element_internal_id];
    for (int level = element_top_level; level >= 0; level--) {
        linklistsizeint *neighbours_list = get_linklist_at_level(element_internal_id, level);
        unsigned short neighbours_count = getListCount(neighbours_list);
        auto *neighbours = (tableint *)(neighbours_list + 1);

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
                                                 neighbour_neighbours_list, level);
                    break;
                }
            }

            // if this edge is uni-directional, we should remove the element from the neighbor's
            // incoming edges.
            if (!bidirectional_edge) {
                auto *neighbour_incoming_edges = getIncomingEdgesPtr(neighbour_id, level);
                neighbour_incoming_edges->erase(element_internal_id);
            }
        }

        // next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // repairs.
        auto *incoming_edges = getIncomingEdgesPtr(element_internal_id, level);
        for (auto incoming_edge : *incoming_edges) {
            linklistsizeint *incoming_node_neighbours_list =
                get_linklist_at_level(incoming_edge, level);
            repairConnectionsForDeletion(element_internal_id, incoming_edge, neighbours_list,
                                         incoming_node_neighbours_list, level);
        }
        delete incoming_edges;
    }

    // replace the entry point with another one, if we are deleting the current entry point.
    if (element_internal_id == enterpoint_node_) {
        assert(element_top_level == maxlevel_);
        linklistsizeint *top_level_list = get_linklist_at_level(element_internal_id, maxlevel_);
        unsigned short list_len = getListCount(top_level_list);
        while (list_len == 0) {
            maxlevel_--;
            if (maxlevel_ < 0) {
                enterpoint_node_ = -1;
                break;
            }
            top_level_list = get_linklist_at_level(element_internal_id, maxlevel_);
            list_len = getListCount(top_level_list);
        }
        // set the (arbitrary) first neighbor as the entry point (if there is some element in the
        // index).
        if (enterpoint_node_ >= 0) {
            enterpoint_node_ = ((tableint *)(top_level_list + 1))[0];
        }
    }

    if (element_levels_[element_internal_id] > 0) {
        this->allocator->free_allocation(linkLists_[element_internal_id]);
    }
    memset(data_level0_memory_ + element_internal_id * size_data_per_element_ + offsetLevel0_, 0,
           size_data_per_element_);
    return true;
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::addPoint(const void *data_point, const labeltype label) {

    tableint cur_c = 0;

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
        if (available_ids.empty()) {
            cur_c = cur_element_count;
            max_id = cur_element_count;
        } else {
            cur_c = *available_ids.begin();
            available_ids.erase(available_ids.begin());
        }
        cur_element_count++;
        label_lookup_[label] = cur_c;
    }
#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
#endif
    // choose randomly the maximum level in which the new element will be in the index.
    int element_max_level = getRandomLevel(mult_);
    element_levels_[cur_c] = element_max_level;

#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> entry_point_lock(global);
#endif
    int maxlevelcopy = maxlevel_;

#ifdef ENABLE_PARALLELIZATION
    if (element_max_level <= maxlevelcopy)
        entry_point_lock.unlock();
#endif
    int currObj = enterpoint_node_;

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
    if (enterpoint_node_ != -1) {
        if (element_max_level < maxlevelcopy) {
            dist_t cur_dist =
                fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxlevelcopy; level > element_max_level; level--) {
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

        for (int level = std::min(element_max_level, maxlevelcopy); level >= 0; level--) {
            if (level > maxlevelcopy || level < 0) // possible?
                throw std::runtime_error("Level error");

            CandidatesQueue<dist_t> top_candidates =
                searchLayer(currObj, data_point, level, ef_construction_);
            currObj = mutuallyConnectNewElement(cur_c, top_candidates, level);
        }
    } else {
        // Do nothing for the first element
        enterpoint_node_ = 0;
        maxlevel_ = element_max_level;
    }

    // updating the maximum level (holding a global lock)
    if (element_max_level > maxlevelcopy) {
        enterpoint_node_ = cur_c;
        maxlevel_ = element_max_level;
        // create the incoming edges set for the new levels.
        for (size_t level_idx = maxlevelcopy + 1; level_idx <= element_max_level; level_idx++) {
            auto *incoming_edges = new vecsim_stl::set<tableint>(this->allocator);
            setIncomingEdgesPtr(cur_c, level_idx, incoming_edges);
        }
    }
}

template <typename dist_t>
vecsim_stl::priority_queue<pair<dist_t, labeltype>>
HierarchicalNSW<dist_t>::searchKnn(const void *query_data, size_t k) const {
    vecsim_stl::priority_queue<std::pair<dist_t, labeltype>> result(this->allocator);
    if (cur_element_count == 0)
        return result;

    tableint currObj = enterpoint_node_;
    dist_t cur_dist =
        fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
    for (int level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            linklistsizeint *node_ll = get_linklist(currObj, level);
            unsigned short links_count = getListCount(node_ll);
            auto *node_links = (tableint *)(node_ll + 1);
            for (int i = 0; i < links_count; i++) {
                tableint candidate = node_links[i];
                if (candidate < 0 || candidate > max_elements_)
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

    CandidatesQueue<dist_t> top_candidates = searchLayer(currObj, query_data, 0, std::max(ef_, k));

    while (top_candidates.size() > k) {
        top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
        std::pair<dist_t, tableint> rez = top_candidates.top();
        result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
        top_candidates.pop();
    }
    return result;
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::checkIntegrity() {

    struct rusage self_ru {};
    getrusage(RUSAGE_SELF, &self_ru);
    std::cerr << "memory usage is : " << self_ru.ru_maxrss << std::endl;

    int connections_checked = 0;
    int double_connections = 0;
    vecsim_stl::vector<int> inbound_connections_num(max_id, 0, this->allocator);
    size_t incoming_edges_sets_sizes = 0;

    for (int i = 0; i <= max_id; i++) {
        if (available_ids.find(i) != available_ids.end()) {
            continue;
        }
        for (int l = 0; l <= element_levels_[i]; l++) {
            linklistsizeint *ll_cur = get_linklist_at_level(i, l);
            int size = getListCount(ll_cur);
            auto *data = (tableint *)(ll_cur + 1);
            vecsim_stl::set<tableint> s(this->allocator);
            for (int j = 0; j < size; j++) {
                assert(data[j] >= 0);
                assert(data[j] <= cur_element_count);
                assert(data[j] != i);
                inbound_connections_num[data[j]]++;
                s.insert(data[j]);
                connections_checked++;
                // check if this is bidirectional
                linklistsizeint *ll_other = get_linklist_at_level(data[j], l);
                int size_other = getListCount(ll_other);
                auto *data_other = (tableint *)(ll_other + 1);
                for (int r = 0; r < size_other; r++) {
                    if (data_other[r] == (tableint)i) {
                        double_connections++;
                        break;
                    }
                }
            }
            assert(s.size() == size);
            incoming_edges_sets_sizes += getIncomingEdgesPtr(i, l)->size();
        }
    }
    assert(incoming_edges_sets_sizes + double_connections == connections_checked);
    std::cout << "uni-directional connections: " << incoming_edges_sets_sizes << std::endl;
    std::cout << "connections: " << connections_checked << std::endl;
    std::cout << "double connections: " << double_connections << std::endl;
    std::cout << "min in-degree: "
              << *std::min_element(inbound_connections_num.begin(), inbound_connections_num.end())
              << std::endl;
    std::cout << "max in-degree: "
              << *std::max_element(inbound_connections_num.begin(), inbound_connections_num.end())
              << std::endl;
    std::cout << "integrity ok\n";
}

} // namespace hnswlib
