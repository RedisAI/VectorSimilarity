#pragma once

#include "visited_nodes_handler.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/space_interface.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/utils/data_block.h"
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

struct level_data /*: public VecsimBaseObject*/ {
    vecsim_stl::set<tableint> incoming_edges;
    tableint numLinks;
    tableint links[];

    level_data(std::shared_ptr<VecSimAllocator> allocator)
        : incoming_edges(allocator), numLinks(0) {}
};

struct element_meta /*: public VecsimBaseObject*/ {
    size_t toplevel;
    level_data *others;
    level_data level0;

    element_meta(size_t maxLevel, std::shared_ptr<VecSimAllocator> allocator)
        : toplevel(maxLevel), others(nullptr), level0(allocator) {}
};

template <typename dist_t>
using candidatesMaxHeap = vecsim_stl::max_priority_queue<pair<dist_t, tableint>>;

template <typename dist_t, typename T>
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
    size_t element_meta_size_;
    size_t level_data_size_;
    size_t element_data_size_;
    size_t block_size_;
    double mult_;

    // Index level generator of the top level for a new element
    std::default_random_engine level_generator_;

    // Index state
    size_t cur_element_count;
    tableint max_id;
    size_t maxlevel_;

    // Index data structures
    tableint entrypoint_node_;
    vecsim_stl::vector<DataBlockMember *> idToMetaBlockMemberMapping;
    vecsim_stl::vector<DataBlock *> vectorBlocks;
    vecsim_stl::vector<DataBlock *> metaBlocks;
    vecsim_stl::set<tableint> available_ids;
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
#endif

    HierarchicalNSW() {}                                // default constructor
    HierarchicalNSW(const HierarchicalNSW &) = default; // default (shallow) copy constructor
    // void setExternalLabel(tableint internal_id, labeltype label);
    size_t getRandomLevel(double reverse_size);
    size_t getTopLevel(tableint internal_id) const;
    size_t removeExtraLinks(level_data *node_meta, candidatesMaxHeap<dist_t> candidates,
                            size_t Mcurmax, const vecsim_stl::set<tableint> &orig_neighbors,
                            tableint *removed_links);
    candidatesMaxHeap<dist_t> searchLayer(tableint ep_id, const void *data_point, size_t layer,
                                          size_t ef) const;
    void getNeighborsByHeuristic2(candidatesMaxHeap<dist_t> &top_candidates, size_t M);
    tableint mutuallyConnectNewElement(tableint cur_c, candidatesMaxHeap<dist_t> &top_candidates,
                                       size_t level);
    void repairConnectionsForDeletion(tableint element_internal_id, tableint neighbour_id,
                                      level_data *element_meta, level_data *neighbour_meta,
                                      size_t level);
    void destroyAllLevels(element_meta *em);

public:
    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements,
                    std::shared_ptr<VecSimAllocator> allocator, size_t M = 16,
                    size_t ef_construction = 200, size_t ef = 10, size_t random_seed = 100,
                    size_t initial_pool_size = 1, size_t elementBlock_size = 5);
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
    T *getDataByInternalId(tableint internal_id) const;
    level_data *getMetadata(tableint internal_id, size_t level) const;
    void resizeIndex(size_t new_max_elements);
    bool removePoint(labeltype label);
    void addPoint(const void *data_point, const labeltype label);
    dist_t getDistanceByLabelFromPoint(labeltype label, const void *data_point);
    tableint searchBottomLayerEP(const void *query_data) const;
    vecsim_stl::max_priority_queue<pair<dist_t, labeltype>> searchKnn(const void *query_data,
                                                                      size_t k) const;
};

/**
 * getters and setters of index data
 */

template <typename dist_t, typename T>
void HierarchicalNSW<dist_t, T>::setEf(size_t ef) {
    ef_ = ef;
}

template <typename dist_t, typename T>
size_t HierarchicalNSW<dist_t, T>::getEf() const {
    return ef_;
}

template <typename dist_t, typename T>
size_t HierarchicalNSW<dist_t, T>::getIndexSize() const {
    return cur_element_count;
}

template <typename dist_t, typename T>
size_t HierarchicalNSW<dist_t, T>::getIndexCapacity() const {
    return max_elements_;
}

template <typename dist_t, typename T>
size_t HierarchicalNSW<dist_t, T>::getEfConstruction() const {
    return ef_construction_;
}

template <typename dist_t, typename T>
size_t HierarchicalNSW<dist_t, T>::getM() const {
    return M_;
}

template <typename dist_t, typename T>
size_t HierarchicalNSW<dist_t, T>::getMaxLevel() const {
    return maxlevel_;
}

template <typename dist_t, typename T>
size_t HierarchicalNSW<dist_t, T>::getEntryPointLabel() const {
    if (entrypoint_node_ != HNSW_INVALID_ID)
        return (size_t)getExternalLabel(entrypoint_node_);
    return SIZE_MAX;
}

template <typename dist_t, typename T>
labeltype HierarchicalNSW<dist_t, T>::getExternalLabel(tableint internal_id) const {
    return idToMetaBlockMemberMapping[internal_id]->label;
}

// template <typename dist_t, typename T>
// void HierarchicalNSW<dist_t, T>::setExternalLabel(tableint internal_id, labeltype label) {
//     idToMetaBlockMemberMapping[internal_id]->label = label;
// }

template <typename dist_t, typename T>
T *HierarchicalNSW<dist_t, T>::getDataByInternalId(tableint internal_id) const {
    DataBlockMember *bm = idToMetaBlockMemberMapping[internal_id];
    return (T *)(this->vectorBlocks[bm->block->getIndex()]->getData(bm->index));
}

template <typename dist_t, typename T>
size_t HierarchicalNSW<dist_t, T>::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (size_t)r;
}

template <typename dist_t, typename T>
dist_t HierarchicalNSW<dist_t, T>::getDistanceByLabelFromPoint(labeltype label,
                                                               const void *data_point) {
    if (label_lookup_.find(label) == label_lookup_.end()) {
        return INVALID_SCORE;
    }
    dist_t t =
        fstdistfunc_(data_point, getDataByInternalId(label_lookup_[label]), dist_func_param_);
    return t;
}

template <typename dist_t, typename T>
level_data *HierarchicalNSW<dist_t, T>::getMetadata(tableint internal_id, size_t level) const {
    DataBlockMember *bm = idToMetaBlockMemberMapping[internal_id];
    if (level) {
        return (level_data *)((char *)((element_meta *)(bm->block->getData(bm->index)))->others +
                              level_data_size_ * (level - 1));
    } else {
        return &((element_meta *)(bm->block->getData(bm->index)))->level0;
    }
}

template <typename dist_t, typename T>
size_t HierarchicalNSW<dist_t, T>::getTopLevel(tableint internal_id) const {
    DataBlockMember *bm = idToMetaBlockMemberMapping[internal_id];
    return bm ? ((element_meta *)(bm->block->getData(bm->index)))->toplevel : HNSW_INVALID_LEVEL;
}

template <typename dist_t, typename T>
tableint HierarchicalNSW<dist_t, T>::getEntryPointId() const {
    return entrypoint_node_;
}

template <typename dist_t, typename T>
VisitedNodesHandler *HierarchicalNSW<dist_t, T>::getVisitedList() const {
    return visited_nodes_handler.get();
}

/**
 * helper functions
 */
template <typename dist_t, typename T>
size_t HierarchicalNSW<dist_t, T>::removeExtraLinks(level_data *node_meta,
                                                    candidatesMaxHeap<dist_t> candidates,
                                                    size_t Mcurmax,
                                                    const vecsim_stl::set<tableint> &orig_neighbors,
                                                    tableint *removed_links) {

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
            node_meta->links[link_idx++] = candidates.top().second;
            candidates.pop();
            orig_candidates.pop();
        }
    }
    node_meta->numLinks = link_idx;
    return removed_idx;
}

template <typename dist_t, typename T>
candidatesMaxHeap<dist_t> HierarchicalNSW<dist_t, T>::searchLayer(tableint ep_id,
                                                                  const void *data_point,
                                                                  size_t layer, size_t ef) const {

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

        tableint curNodeNum = curr_el_pair.second;
#ifdef ENABLE_PARALLELIZATION
        std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);
#endif
        level_data *node_meta = getMetadata(curNodeNum, layer);
        size_t links_num = node_meta->numLinks;
        auto *node_links = node_meta->links;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_nodes_handler->getElementsTags() + *(node_meta->links)),
                     _MM_HINT_T0);
        _mm_prefetch((char *)(visited_nodes_handler->getElementsTags() + *(node_meta->links) + 64),
                     _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(node_meta->links)), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(node_meta->links + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < links_num; j++) {
            tableint candidate_id = node_links[j];
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_nodes_handler->getElementsTags() + *(node_links + j + 1)),
                         _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(node_links + j + 1)), _MM_HINT_T0);
#endif
            if (this->visited_nodes_handler->getNodeTag(candidate_id) == visited_tag)
                continue;
            this->visited_nodes_handler->tagNode(candidate_id, visited_tag);
            T *currObj1 = (getDataByInternalId(candidate_id));

            dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
            if (top_candidates.size() < ef || lowerBound > dist1) {
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
    visited_nodes_handler_pool->returnVisitedNodesHandlerToPool(this->visited_nodes_handler);
#endif
    return top_candidates;
}

template <typename dist_t, typename T>
void HierarchicalNSW<dist_t, T>::getNeighborsByHeuristic2(candidatesMaxHeap<dist_t> &top_candidates,
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

template <typename dist_t, typename T>
tableint HierarchicalNSW<dist_t, T>::mutuallyConnectNewElement(
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
        level_data *meta = getMetadata(cur_c, level);
        if (meta->numLinks) {
            throw std::runtime_error("The newly inserted element should have blank link list");
        }
        meta->numLinks =
            selectedNeighbors.size(); // setListCount(ll_cur, selectedNeighbors.size());
        auto *data = meta->links;
        for (size_t idx = 0; idx < meta->numLinks; idx++) {
            if (data[idx])
                throw std::runtime_error("Possible memory corruption");
            if (level > getTopLevel(selectedNeighbors[idx]))
                throw std::runtime_error("Trying to make a link on a non-existent level");
            data[idx] = selectedNeighbors[idx];
        }
    }

    // go over the selected neighbours - selectedNeighbor is the neighbour id
    for (tableint selectedNeighbor : selectedNeighbors) {
#ifdef ENABLE_PARALLELIZATION
        std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbor]);
#endif
        level_data *other_meta = getMetadata(selectedNeighbor, level);

        if (other_meta->numLinks > Mcurmax)
            throw std::runtime_error("Bad value of other_meta->numLinks");
        if (selectedNeighbor == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");
        if (level > getTopLevel(selectedNeighbor))
            throw std::runtime_error("Trying to make a link on a non-existent level");

        // If the selected neighbor can add another link (hasn't reached the max) - add it.
        if (other_meta->numLinks < Mcurmax) {
            other_meta->links[other_meta->numLinks] = cur_c;
            other_meta->numLinks++;
        } else {
            // try finding "weak" elements to replace it with the new one with the heuristic:
            candidatesMaxHeap<dist_t> candidates(this->allocator);
            vecsim_stl::set<tableint> orig_neighbors_set(this->allocator);
            dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c),
                                        getDataByInternalId(selectedNeighbor), dist_func_param_);
            candidates.emplace(d_max, cur_c);
            // consider cur_c as if it was a link of the selected neighbor
            orig_neighbors_set.insert(cur_c);

            for (size_t j = 0; j < other_meta->numLinks; j++) {
                candidates.emplace(fstdistfunc_(getDataByInternalId(other_meta->links[j]),
                                                getDataByInternalId(selectedNeighbor),
                                                dist_func_param_),
                                   other_meta->links[j]);
                orig_neighbors_set.insert(other_meta->links[j]);
            }

            tableint removed_links[other_meta->numLinks + 1];
            size_t removed_links_num = removeExtraLinks(other_meta, candidates, Mcurmax,
                                                        orig_neighbors_set, removed_links);

            // remove the current neighbor from the incoming list of nodes for the
            // neighbours that were chosen to remove (if edge wasn't bidirectional)
            for (size_t i = 0; i < removed_links_num; i++) {
                tableint node_id = removed_links[i];
                level_data *node_meta = getMetadata(node_id, level);
                // if we removed cur_c (the node just inserted), then it points to the current
                // neighbour, but not vise versa.
                if (node_id == cur_c) {
                    other_meta->incoming_edges.insert(cur_c);
                    continue;
                }

                // if the node id (the neighbour's neighbour to be removed)
                // wasn't pointing to the neighbour (i.e., the edge was uni-directional),
                // we should remove the current neighbor from the node's incoming edges.
                // otherwise, the edge turned from bidirectional to
                // uni-directional, so we insert it to the neighbour's
                // incoming edges set.
                if (node_meta->incoming_edges.find(selectedNeighbor) !=
                    node_meta->incoming_edges.end()) {
                    node_meta->incoming_edges.erase(selectedNeighbor);
                } else {
                    other_meta->incoming_edges.insert(node_id);
                }
            }
        }
    }
    return next_closest_entry_point;
}

template <typename dist_t, typename T>
void HierarchicalNSW<dist_t, T>::repairConnectionsForDeletion(tableint element_internal_id,
                                                              tableint neighbour_id,
                                                              level_data *element_meta,
                                                              level_data *neighbour_meta,
                                                              size_t level) {

    // put the deleted element's neighbours in the candidates.
    candidatesMaxHeap<dist_t> candidates(this->allocator);
    vecsim_stl::set<tableint> candidates_set(this->allocator);
    unsigned short neighbours_count = element_meta->numLinks;
    auto *neighbours = element_meta->links;
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
    unsigned short neighbour_neighbours_count = neighbour_meta->numLinks;
    auto *neighbour_neighbours = neighbour_meta->links;
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
    tableint removed_links[neighbour_neighbours_count];
    size_t removed_links_num = removeExtraLinks(neighbour_meta, candidates, Mcurmax,
                                                neighbour_orig_neighbours_set, removed_links);

    // remove neighbour id from the incoming list of nodes for his
    // neighbours that were chosen to remove
    for (size_t i = 0; i < removed_links_num; i++) {
        tableint node_id = removed_links[i];
        level_data *node_meta = getMetadata(node_id, level);

        // if the node id (the neighbour's neighbour to be removed)
        // wasn't pointing to the neighbour (edge was one directional),
        // we should remove it from the node's incoming edges.
        // otherwise, edge turned from bidirectional to one directional,
        // and it should be saved in the neighbor's incoming edges.
        if (node_meta->incoming_edges.find(neighbour_id) != node_meta->incoming_edges.end()) {
            node_meta->incoming_edges.erase(neighbour_id);
        } else {
            neighbour_meta->incoming_edges.insert(node_id);
        }
    }

    // updates for the new edges created
    unsigned short updated_links_num = neighbour_meta->numLinks;
    for (size_t i = 0; i < updated_links_num; i++) {
        tableint node_id = neighbour_neighbours[i];
        if (neighbour_orig_neighbours_set.find(node_id) == neighbour_orig_neighbours_set.end()) {
            // if the node has an edge to the neighbour as well, remove it
            // from the incoming nodes of the neighbour
            // otherwise, need to update the edge as incoming.
            level_data *node_meta = getMetadata(node_id, level);
            unsigned short node_links_size = node_meta->numLinks;
            auto *node_links = node_meta->links;
            bool bidirectional_edge = false;
            for (size_t j = 0; j < node_links_size; j++) {
                if (node_links[j] == neighbour_id) {
                    neighbour_meta->incoming_edges.erase(node_id);
                    bidirectional_edge = true;
                    break;
                }
            }
            if (!bidirectional_edge) {
                node_meta->incoming_edges.insert(neighbour_id);
            }
        }
    }
}

template <typename dist_t, typename T>
HierarchicalNSW<dist_t, T>::HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements,
                                            std::shared_ptr<VecSimAllocator> allocator, size_t M,
                                            size_t ef_construction, size_t ef, size_t random_seed,
                                            size_t pool_initial_size, size_t block_size)
    : VecsimBaseObject(allocator), idToMetaBlockMemberMapping(allocator), vectorBlocks(allocator),
      metaBlocks(allocator), available_ids(allocator), label_lookup_(allocator)

#ifdef ENABLE_PARALLELIZATION
      ,
      link_list_locks_(max_elements)
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

    element_data_size_ = s->get_data_size();
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

    idToMetaBlockMemberMapping.resize(max_elements);
    block_size_ = block_size;

    if (maxM0_ >= (SIZE_MAX - sizeof(element_meta)) / sizeof(tableint))
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    element_meta_size_ = sizeof(element_meta) + maxM0_ * sizeof(tableint);
    level_data_size_ = sizeof(level_data) + maxM_ * sizeof(tableint);

    // data_level0_memory will look like this:
    // -----4------ | -----4*M0----------- | ----8--------------- | -dim*sizeof(type)- | ---8--- |
    // <links_len>  | <link_1> <link_2>... | <incoming_links_set> |       <data>       | <label> |

    // if (maxM0_ > ((SIZE_MAX - sizeof(void *) - sizeof(linklistsizeint)) / sizeof(tableint)) + 1)
    //     throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    // size_links_level0_ = sizeof(linklistsizeint) + maxM0_ * sizeof(tableint) + sizeof(void *);

    // if (size_links_level0_ > SIZE_MAX - data_size_ - sizeof(labeltype))
    //     throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    // size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);

    // No need to test for overflow because we passed the test for size_links_level0_ and this is
    // less.

    // incoming_links_offset0 = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    // offsetData_ = size_links_level0_;
    // label_offset_ = size_links_level0_ + data_size_;
    // offsetLevel0_ = 0;

    // data_level0_memory_ =
    //     (char *)this->allocator->callocate(max_elements_ * size_data_per_element_);
    // if (data_level0_memory_ == nullptr)
    //     throw std::runtime_error("Not enough memory");

    // linkLists_ = (char **)this->allocator->callocate(sizeof(void *) * max_elements_);
    // if (linkLists_ == nullptr)
    //     throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate
    //     linklists");

    // The i-th entry in linkLists array points to max_level[i] (continuous)
    // chunks of memory, each one will look like this:
    // -----4------ | -----4*M-------------- | ----8------------------|
    // <links_len>  | <link_1> <link_2> ...  | <incoming_links_set>   |

    // size_links_per_element_ = sizeof(linklistsizeint) + maxM_ * sizeof(tableint) + sizeof(void
    // *);

    // No need to test for overflow because we passed the test for incoming_links_offset0 and this
    // is less.

    // incoming_links_offset = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
}

template <typename dist_t, typename T>
void HierarchicalNSW<dist_t, T>::destroyAllLevels(element_meta *em) {
    // em->level0.~level_data();
    level_data *cur_ld = em->others;
    for (size_t i = 0; i < em->toplevel; i++) {
        cur_ld->~level_data();
        cur_ld = (level_data *)((char *)cur_ld + this->level_data_size_);
    }
}

template <typename dist_t, typename T>
HierarchicalNSW<dist_t, T>::~HierarchicalNSW() {
    for (auto &metaBlock : this->metaBlocks) {
        for (size_t i = 0; i < metaBlock->getLength(); i++) {
            element_meta *em = (element_meta *)metaBlock->getData(i);
            em->level0.~level_data();
            destroyAllLevels(em);
            this->allocator->free_allocation(em->others);
        }
        delete metaBlock;
    }
    for (auto &vectorBlock : this->vectorBlocks) {
        delete vectorBlock;
    }
}

/**
 * Index API functions
 */
template <typename dist_t, typename T>
void HierarchicalNSW<dist_t, T>::resizeIndex(size_t new_max_elements) {
    if (new_max_elements < cur_element_count)
        throw std::runtime_error(
            "Cannot resize, max element is less than the current number of elements");
#ifdef ENABLE_PARALLELIZATION
    visited_nodes_handler_pool = std::unique_ptr<VisitedNodesHandlerPool>(
        new (this->allocator)
            VisitedNodesHandlerPool(this->pool_initial_size, new_max_elements, this->allocator));
    std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);
#else
    visited_nodes_handler = std::unique_ptr<VisitedNodesHandler>(
        new (this->allocator) VisitedNodesHandler(new_max_elements, this->allocator));
#endif

    max_elements_ = new_max_elements;
    idToMetaBlockMemberMapping.resize(max_elements_);
}

template <typename dist_t, typename T>
bool HierarchicalNSW<dist_t, T>::removePoint(const labeltype label) {
    // check that the label actually exists in the graph, and update the number of elements.
    tableint element_internal_id;
    if (label_lookup_.find(label) == label_lookup_.end()) {
        return true;
    }
    element_internal_id = label_lookup_[label];

    // go over levels and repair connections
    size_t element_top_level = getTopLevel(element_internal_id);
    for (size_t level = 0; level <= element_top_level; level++) {
        level_data *elm_meta = getMetadata(element_internal_id, level);
        unsigned short neighbours_count = elm_meta->numLinks;
        auto *neighbours = elm_meta->links;

        // go over the neighbours that also points back to the removed point and make a local
        // repair.
        for (size_t i = 0; i < neighbours_count; i++) {
            tableint neighbour_id = neighbours[i];
            level_data *neighbour_meta = getMetadata(neighbour_id, level);
            unsigned short neighbour_neighbours_count = neighbour_meta->numLinks;
            auto *neighbour_neighbours = neighbour_meta->links;
            bool bidirectional_edge = false;
            for (size_t j = 0; j < neighbour_neighbours_count; j++) {
                // if the edge is bidirectional, do repair for this neighbor
                if (neighbour_neighbours[j] == element_internal_id) {
                    bidirectional_edge = true;
                    repairConnectionsForDeletion(element_internal_id, neighbour_id, elm_meta,
                                                 neighbour_meta, level);
                    break;
                }
            }

            // if this edge is uni-directional, we should remove the element from the neighbor's
            // incoming edges.
            if (!bidirectional_edge) {
                level_data *neighbour_meta = getMetadata(neighbour_id, level);
                neighbour_meta->incoming_edges.erase(element_internal_id);
            }
        }

        // next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // repairs.
        for (tableint incoming_edge : elm_meta->incoming_edges) {
            level_data *incoming_node_meta = getMetadata(incoming_edge, level);
            repairConnectionsForDeletion(element_internal_id, incoming_edge, elm_meta,
                                         incoming_node_meta, level);
        }
    }

    // replace the entry point with another one, if we are deleting the current entry point.
    if (element_internal_id == entrypoint_node_) {
        assert(element_top_level == maxlevel_);
        // Sets the (arbitrary) new entry point.
        while (element_internal_id == entrypoint_node_) {
            level_data *top_level_meta = getMetadata(element_internal_id, maxlevel_);

            if (top_level_meta->numLinks > 0) {
                // Tries to set the (arbitrary) first neighbor as the entry point.
                entrypoint_node_ = top_level_meta->links[0];
            } else {
                // If there is no neighbors in the current level, check for any vector at
                // this level to be the new entry point.
                for (tableint cur_id = 0; cur_id <= max_id; cur_id++) {
                    if (getTopLevel(cur_id) == maxlevel_ && cur_id != element_internal_id) {
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

    // add the element id to the available ids for future reuse.
    cur_element_count--;
    label_lookup_.erase(label);
    available_ids.insert(element_internal_id);

    DataBlockMember *metaBlockMember = this->idToMetaBlockMemberMapping[element_internal_id];
    DataBlock *metaBlock = metaBlockMember->block;
    size_t elementIndex = metaBlockMember->index;
    DataBlock *vectorBlock = this->vectorBlocks[metaBlock->getIndex()];

    DataBlock *lastMetaBlock = this->metaBlocks.back();     //[this->metaBlocks.size() - 1];
    DataBlock *lastVectorBlock = this->vectorBlocks.back(); //[this->vectorBlocks.size() - 1];
    DataBlockMember *lastBlockMember = lastMetaBlock->getMember(lastMetaBlock->getLength() - 1);

    vectorBlock->setMember(elementIndex, lastBlockMember);
    metaBlock->setMember(elementIndex, lastBlockMember);

    element_meta *meta_dest = (element_meta *)metaBlock->getData(elementIndex);
    if (meta_dest->others) {
        destroyAllLevels(meta_dest);
        this->allocator->free_allocation(meta_dest->others);
    }
    element_meta *meta_origin = (element_meta *)lastMetaBlock->removeAndFetchData();
    meta_dest->toplevel = meta_origin->toplevel;
    meta_dest->others = meta_origin->others;
    meta_dest->level0.incoming_edges = std::move(meta_origin->level0.incoming_edges);
    meta_origin->level0.~level_data();
    meta_dest->level0.numLinks = meta_origin->level0.numLinks;
    memmove((void *)meta_dest->level0.links, (void *)meta_origin->level0.links,
            sizeof(tableint) * this->maxM0_);

    void *vec_dest = vectorBlock->getData(elementIndex);
    void *vec_origin = lastVectorBlock->removeAndFetchData();
    memmove(vec_dest, vec_origin, this->element_data_size_);

    // Delete the element block membership
    delete metaBlockMember;
    this->idToMetaBlockMemberMapping[element_internal_id] = NULL;

    // If the last element block is emtpy;
    if (lastMetaBlock->getLength() == 0) {
        this->metaBlocks.pop_back();
        delete lastMetaBlock;
        this->vectorBlocks.pop_back();
        delete lastVectorBlock;
    }

    return true;
}

template <typename dist_t, typename T>
void HierarchicalNSW<dist_t, T>::addPoint(const void *data_point, const labeltype label) {

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
            resizeIndex(cur_element_count * 1.1 + 1);
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
    size_t element_max_level = getRandomLevel(mult_);
    void *dummy = this->allocator->callocate(element_meta_size_);
    // new(new_elm) element_meta(element_max_level, this->allocator);
    // *new_elm = element_meta(element_max_level, this->allocator);

#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> insertion(global);
#endif
    // Get vector and metadata blocks to store the new element in.
    DataBlock *metaBlock;
    DataBlock *vectorBlock;

    if (this->metaBlocks.size() == 0) {
        // No element blocks, create new one.
        metaBlock = new (this->allocator)
            DataBlock(this->block_size_, element_meta_size_, this->allocator, 0);
        this->metaBlocks.push_back(metaBlock);
        vectorBlock = new (this->allocator)
            DataBlock(this->block_size_, element_data_size_, this->allocator, 0, false);
        this->vectorBlocks.push_back(vectorBlock);
    } else {
        // Get the last element block.
        metaBlock = this->metaBlocks.back();
        vectorBlock = this->vectorBlocks.back();
        if (metaBlock->getLength() == this->block_size_) {
            // Last element block is full, create a new one.
            metaBlock = new (this->allocator) DataBlock(this->block_size_, element_meta_size_,
                                                        this->allocator, this->metaBlocks.size());
            this->metaBlocks.push_back(metaBlock);
            vectorBlock =
                new (this->allocator) DataBlock(this->block_size_, element_data_size_,
                                                this->allocator, this->metaBlocks.size(), false);
            this->vectorBlocks.push_back(vectorBlock);
        }
    }

    DataBlockMember *BlockMember = new (this->allocator) DataBlockMember(this->allocator);
    this->idToMetaBlockMemberMapping[cur_c] = BlockMember;
    BlockMember->label = label;
    vectorBlock->addData(BlockMember, data_point);
    metaBlock->addData(BlockMember, dummy);
    this->allocator->free_allocation(dummy);
    element_meta *new_elm = (element_meta *)metaBlock->getData(BlockMember->index);
    new (new_elm) element_meta(element_max_level, this->allocator);

#ifdef ENABLE_PARALLELIZATION
    insertion.unlock();
#endif

#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> entry_point_lock(global);
#endif
    size_t maxlevelcopy = maxlevel_;

#ifdef ENABLE_PARALLELIZATION
    if (element_max_level <= maxlevelcopy)
        entry_point_lock.unlock();
#endif
    size_t currObj = entrypoint_node_;

    if (element_max_level > 0) {
        new_elm->others =
            (level_data *)this->allocator->callocate(level_data_size_ * element_max_level);
        if (new_elm->others == nullptr)
            throw std::runtime_error(
                "Not enough memory: addPoint failed to allocate levels metadata");
        // for (size_t i = 0; i < element_max_level; i++) {
        //     new(this->allocator)(&(((level_data *)((char *)new_elm->others + i *
        //     level_data_size_))->incoming_edges)) vecsim_stl::set<tableint>(this->allocator);
        // }
        for (size_t i = 0; i < element_max_level; i++) {
            level_data *ld = (level_data *)((char *)new_elm->others + i * level_data_size_);
            new (ld) level_data(this->allocator);
            // *ld = level_data(this->allocator);
        }
        // for (size_t i = 0; i < element_max_level; i++) {
        //     level_data *l = reinterpret_cast<level_data *> (((char *)new_elm->others + i *
        //     element_meta_size_)); *l = level_data(this->allocator);
        // }
        // for (size_t i = 0; i < element_max_level; i++) {
        //     (level_data *)((char *)new_elm->others + i * element_meta_size_) = level_data();
        // }
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
                    level_data *meta;
#ifdef ENABLE_PARALLELIZATION
                    std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
#endif
                    meta = getMetadata(currObj, level);
                    int size = meta->numLinks;

                    auto *datal = meta->links;
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
        }
    } else {
        // Do nothing for the first element
        entrypoint_node_ = 0;
        maxlevel_ = element_max_level;
    }
}

template <typename dist_t, typename T>
tableint HierarchicalNSW<dist_t, T>::searchBottomLayerEP(const void *query_data) const {

    if (cur_element_count == 0) {
        return entrypoint_node_;
    }
    tableint currObj = entrypoint_node_;
    dist_t cur_dist =
        fstdistfunc_(query_data, getDataByInternalId(entrypoint_node_), dist_func_param_);
    for (size_t level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            level_data *meta = getMetadata(currObj, level);
            unsigned short links_count = meta->numLinks;
            auto *node_links = meta->links;
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
    return currObj;
}

template <typename dist_t, typename T>
vecsim_stl::max_priority_queue<pair<dist_t, labeltype>>
HierarchicalNSW<dist_t, T>::searchKnn(const void *query_data, size_t k) const {

    vecsim_stl::max_priority_queue<std::pair<dist_t, labeltype>> result(this->allocator);
    if (cur_element_count == 0)
        return result;

    tableint bottom_layer_ep = searchBottomLayerEP(query_data);
    vecsim_stl::max_priority_queue<pair<dist_t, tableint>> top_candidates =
        searchLayer(bottom_layer_ep, query_data, 0, std::max(ef_, k));

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

} // namespace hnswlib
