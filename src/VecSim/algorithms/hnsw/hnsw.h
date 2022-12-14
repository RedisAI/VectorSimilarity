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

using std::pair;

#define HNSW_INVALID_ID    UINT_MAX
#define HNSW_INVALID_LEVEL SIZE_MAX

typedef uint16_t linkListSize;
typedef uint16_t elementFlags;

template <typename DistType>
using candidatesMaxHeap = vecsim_stl::max_priority_queue<DistType, idType>;
template <typename DistType>
using candidatesLabelsMaxHeap = vecsim_stl::abstract_priority_queue<DistType, labelType>;

#pragma pack(2)

struct level_data {
    linkListSize numLinks;
    vecsim_stl::vector<idType> *incoming_edges;
    idType links[];

    level_data(std::shared_ptr<VecSimAllocator> allocator)
        : numLinks(0), incoming_edges(new (allocator) vecsim_stl::vector<idType>(allocator)) {}
};

struct element_meta_data {
    labelType label;
    elementFlags flags;

    element_meta_data(labelType label = SIZE_MAX) noexcept : label(label), flags(0) {}
};

struct element_graph_data {
    size_t toplevel;
    level_data *others;
    level_data level0;

    element_graph_data(size_t maxLevel, size_t high_level_size,
                       std::shared_ptr<VecSimAllocator> allocator)
        : toplevel(maxLevel), others(nullptr), level0(allocator) {
        if (toplevel > 0) {
            others = (level_data *)allocator->callocate(high_level_size * toplevel);
            if (others == nullptr) {
                throw std::runtime_error(
                    "Not enough memory: appendVector failed to allocate new element resources.");
            }
            for (size_t i = 0; i < maxLevel; i++) {
                new ((char *)others + i * high_level_size) level_data(allocator);
            }
        }
    }
};

#pragma pack() // restore default packing

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
    size_t element_graph_data_size_;
    size_t level_data_size_;
    size_t element_data_size_;
    double mult_;

    // Index level generator of the top level for a new element
    std::default_random_engine level_generator_;

    // Index state
    size_t cur_element_count;
    size_t maxlevel_;

    // Index data structures
    idType entrypoint_node_;
    vecsim_stl::vector<DataBlock> vector_blocks;
    vecsim_stl::vector<DataBlock> meta_blocks;
    vecsim_stl::vector<element_meta_data> idToMetaData;
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
    inline size_t getRandomLevel(double reverse_size);
    inline void removeExtraLinks(candidatesMaxHeap<DistType> candidates, size_t Mcurmax,
                                 level_data &node_meta, const vecsim_stl::vector<bool> &bitmap,
                                 idType *removed_links, size_t *removed_links_num);
    template <bool has_marked_deleted, typename Identifier> // Either idType or labelType
    inline const void
    processCandidate(idType curNodeId, const void *data_point, size_t layer, size_t ef,
                     VisitedNodesHandler *visited_nodes, tag_t visited_tag,
                     vecsim_stl::abstract_priority_queue<DistType, Identifier> &top_candidates,
                     candidatesMaxHeap<DistType> &candidates_set, DistType &lowerBound) const;
    template <bool has_marked_deleted>
    inline void processCandidate_RangeSearch(
        idType curNodeId, const void *data_point, size_t layer, double epsilon,
        VisitedNodesHandler *visited_nodes, tag_t visited_tag,
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
    inline idType mutuallyConnectNewElement(idType cur_c,
                                            candidatesMaxHeap<DistType> &top_candidates,
                                            size_t level);
    template <bool with_timeout>
    void greedySearchLevel(const void *vector_data, size_t level, idType &curObj, DistType &curDist,
                           void *timeoutCtx = nullptr, VecSimQueryResult_Code *rc = nullptr) const;
    void repairConnectionsForDeletion(idType element_internal_id, idType neighbour_id,
                                      level_data &element_meta, level_data &neighbour_meta,
                                      size_t level, vecsim_stl::vector<bool> &neighbours_bitmap);
    inline void destroyMetadata(element_graph_data *em);
    inline void replaceEntryPoint();
    inline void resizeIndex(size_t new_max_elements);
    inline void SwapLastIdWithDeletedId(idType element_internal_id,
                                        element_graph_data *last_element_meta,
                                        void *last_element_data);

    // Protected internal function that implements generic single vector insertion.
    int appendVector(const void *vector_data, labelType label);

    // Protected internal function that implements generic single vector deletion.
    int removeVector(idType id);

    inline void emplaceToHeap(vecsim_stl::abstract_priority_queue<DistType, idType> &heap,
                              DistType dist, idType id) const;
    inline void emplaceToHeap(vecsim_stl::abstract_priority_queue<DistType, labelType> &heap,
                              DistType dist, idType id) const;

    inline const DataBlock &getVectorVectorBlock(idType id) const;
    inline const DataBlock &getVectorMetaBlock(idType id) const;
    inline size_t getVectorRelativeIndex(idType id) const { return id % this->blockSize; }
    inline element_graph_data *getMetaDataByInternalId(idType internal_id) const;

public:
    HNSWIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator,
              size_t random_seed = 100, size_t initial_pool_size = 1);
    virtual ~HNSWIndex();

    inline void setEf(size_t ef) { ef_ = ef; }
    inline size_t getEf() const { return ef_; }
    inline void setEpsilon(double epsilon) { epsilon_ = epsilon; }
    inline double getEpsilon() const { return epsilon_; }
    inline size_t indexSize() const override { return cur_element_count - num_marked_deleted; }
    inline size_t getIndexCapacity() const { return max_elements_; }
    inline size_t getEfConstruction() const { return ef_construction_; }
    inline size_t getM() const { return M_; }
    inline size_t getMaxLevel() const { return maxlevel_; }
    inline VisitedNodesHandler *getVisitedList() const;
    VecSimIndexInfo info() const override;
    VecSimInfoIterator *infoIterator() const override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;
    inline const char *getDataByInternalId(idType internal_id) const;
    inline level_data &getLevelData(idType internal_id, size_t level) const;
    inline idType getEntryPointId() const { return entrypoint_node_; }
    inline labelType getEntryPointLabel() const;
    inline labelType getExternalLabel(idType internal_id) const {
        return idToMetaData[internal_id].label;
    }
    inline idType searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                      VecSimQueryResult_Code *rc) const;

    VecSimQueryResult_List topKQuery(const void *query_data, size_t k,
                                     VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *query_data, double radius,
                                      VecSimQueryParams *queryParams) override;

    inline void markDeletedInternal(idType internalId);
    inline void unmarkDeletedInternal(idType internalId);
    inline bool isMarkedDeleted(idType internalId) const;

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
labelType HNSWIndex<DataType, DistType>::getEntryPointLabel() const {
    if (entrypoint_node_ != HNSW_INVALID_ID)
        return getExternalLabel(entrypoint_node_);
    return SIZE_MAX;
}

template <typename DataType, typename DistType>
const char *HNSWIndex<DataType, DistType>::getDataByInternalId(idType internal_id) const {
    return vector_blocks[internal_id / this->blockSize].getElement(internal_id % this->blockSize);
}

template <typename DataType, typename DistType>
element_graph_data *
HNSWIndex<DataType, DistType>::getMetaDataByInternalId(idType internal_id) const {
    return (element_graph_data *)meta_blocks[internal_id / this->blockSize].getElement(
        internal_id % this->blockSize);
}

template <typename DataType, typename DistType>
const DataBlock &HNSWIndex<DataType, DistType>::getVectorVectorBlock(idType internal_id) const {
    return vector_blocks.at(internal_id / this->blockSize);
}

template <typename DataType, typename DistType>
const DataBlock &HNSWIndex<DataType, DistType>::getVectorMetaBlock(idType internal_id) const {
    return meta_blocks.at(internal_id / this->blockSize);
}

template <typename DataType, typename DistType>
size_t HNSWIndex<DataType, DistType>::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (size_t)r;
}

template <typename DataType, typename DistType>
level_data &HNSWIndex<DataType, DistType>::getLevelData(idType internal_id, size_t level) const {
    auto meta = getMetaDataByInternalId(internal_id);
    assert(level <= meta->toplevel);
    if (level == 0) {
        return meta->level0;
    } else {
        return *(level_data *)((char *)meta->others + (level - 1) * this->level_data_size_);
    }
}

template <typename DataType, typename DistType>
VisitedNodesHandler *HNSWIndex<DataType, DistType>::getVisitedList() const {
    return visited_nodes_handler.get();
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::markDeletedInternal(idType internalId) {
    assert(internalId < this->cur_element_count);
    if (!isMarkedDeleted(internalId)) {
        idToMetaData[internalId].flags |= DELETE_MARK;
        this->num_marked_deleted++;
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::unmarkDeletedInternal(idType internalId) {
    assert(internalId < this->cur_element_count);
    if (isMarkedDeleted(internalId)) {
        idToMetaData[internalId].flags &= ~DELETE_MARK;
        this->num_marked_deleted--;
    }
}

template <typename DataType, typename DistType>
bool HNSWIndex<DataType, DistType>::isMarkedDeleted(idType internalId) const {
    return idToMetaData[internalId].flags & DELETE_MARK;
}

/**
 * helper functions
 */
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::removeExtraLinks(
    candidatesMaxHeap<DistType> candidates, size_t Mcurmax, level_data &node_meta,
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
            node_meta.links[link_idx++] = candidates.top().second;
            candidates.pop();
            orig_candidates.pop();
        }
    }
    node_meta.numLinks = link_idx;
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
const void HNSWIndex<DataType, DistType>::processCandidate(
    idType curNodeId, const void *data_point, size_t layer, size_t ef,
    VisitedNodesHandler *visited_nodes, tag_t visited_tag,
    vecsim_stl::abstract_priority_queue<DistType, Identifier> &top_candidates,
    candidatesMaxHeap<DistType> &candidate_set, DistType &lowerBound) const {

#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> lock(link_list_locks_[curNodeId]);
#endif

    level_data &node_meta = getLevelData(curNodeId, layer);
    if (node_meta.numLinks > 0) {

        // Pre-fetch first candidate tag address.
        __builtin_prefetch(visited_nodes->getElementsTags() + node_meta.links[0]);
        // // Pre-fetch first candidate data block address.
        __builtin_prefetch(getDataByInternalId(node_meta.links[0]));

        for (linkListSize j = 0; j < node_meta.numLinks - 1; j++) {
            idType candidate_id = node_meta.links[j];

            // Pre-fetch next candidate tag address.
            __builtin_prefetch(visited_nodes->getElementsTags() + node_meta.links[j + 1]);
            // Pre-fetch next candidate data block address.
            __builtin_prefetch(getDataByInternalId(node_meta.links[j + 1]));

            if (visited_nodes->getNodeTag(candidate_id) == visited_tag)
                continue;

            visited_nodes->tagNode(candidate_id, visited_tag);
            const char *currObj1 = getDataByInternalId(candidate_id);

            DistType dist1 = this->dist_func(data_point, currObj1, this->dim);
            if (lowerBound > dist1 || top_candidates.size() < ef) {
                // Pre-fetch current candidate meta data
                __builtin_prefetch(this->idToMetaData.data() + candidate_id);

                candidate_set.emplace(-dist1, candidate_id);

                // Insert the candidate to the top candidates heap only if it is not marked as
                // deleted.
                if (!has_marked_deleted || !isMarkedDeleted(candidate_id))
                    emplaceToHeap(top_candidates, dist1, candidate_id);

                if (top_candidates.size() > ef)
                    top_candidates.pop();

                // If we have marked deleted elements, we need to verify that `top_candidates` is
                // not empty (since we might have not added any non-deleted element yet).
                if (!has_marked_deleted || !top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }

        // Running the last candidate outside the loop to avoid prefetching invalid candidate
        idType candidate_id = node_meta.links[node_meta.numLinks - 1];

        if (visited_nodes->getNodeTag(candidate_id) != visited_tag) {

            visited_nodes->tagNode(candidate_id, visited_tag);
            const char *currObj1 = getDataByInternalId(candidate_id);

            DistType dist1 = this->dist_func(data_point, currObj1, this->dim);
            if (lowerBound > dist1 || top_candidates.size() < ef) {
                // Pre-fetch current candidate meta data
                __builtin_prefetch(this->idToMetaData.data() + candidate_id);
                candidate_set.emplace(-dist1, candidate_id);

                // Insert the candidate to the top candidates heap only if it is not marked as
                // deleted.
                if (!has_marked_deleted || !isMarkedDeleted(candidate_id))
                    emplaceToHeap(top_candidates, dist1, candidate_id);

                if (top_candidates.size() > ef)
                    top_candidates.pop();

                // If we have marked deleted elements, we need to verify that `top_candidates` is
                // not empty (since we might have not added any non-deleted element yet).
                if (!has_marked_deleted || !top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }
    __builtin_prefetch(this->meta_blocks.data() + (candidate_set.top().second / this->blockSize));
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
void HNSWIndex<DataType, DistType>::processCandidate_RangeSearch(
    idType curNodeId, const void *query_data, size_t layer, double epsilon,
    VisitedNodesHandler *visited_nodes, tag_t visited_tag,
    std::unique_ptr<vecsim_stl::abstract_results_container> &results,
    candidatesMaxHeap<DistType> &candidate_set, DistType dyn_range, DistType radius) const {

#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> lock(link_list_locks_[curNodeId]);
#endif
    level_data &node_meta = getLevelData(curNodeId, layer);
    if (node_meta.numLinks > 0) {

        // Pre-fetch first candidate tag address.
        __builtin_prefetch(visited_nodes->getElementsTags() + node_meta.links[0]);
        // Pre-fetch first candidate data block address.
        __builtin_prefetch(getDataByInternalId(node_meta.links[0]));

        for (linkListSize j = 0; j < node_meta.numLinks - 1; j++) {
            idType candidate_id = node_meta.links[j];

            // Pre-fetch next candidate tag address.
            __builtin_prefetch(visited_nodes->getElementsTags() + node_meta.links[j + 1]);
            // Pre-fetch next candidate data block address.
            __builtin_prefetch(getDataByInternalId(node_meta.links[j + 1]));

            if (visited_nodes->getNodeTag(candidate_id) == visited_tag)
                continue;
            visited_nodes->tagNode(candidate_id, visited_tag);
            const char *candidate_data = getDataByInternalId(candidate_id);

            DistType candidate_dist = this->dist_func(query_data, candidate_data, this->dim);
            if (candidate_dist < dyn_range) {
                // Pre-fetch current candidate meta data
                __builtin_prefetch(this->idToMetaData.data() + candidate_id);
                candidate_set.emplace(-candidate_dist, candidate_id);

                // If the new candidate is in the requested radius, add it to the results set.
                if (candidate_dist <= radius &&
                    (!has_marked_deleted || !isMarkedDeleted(candidate_id))) {
                    results->emplace(getExternalLabel(candidate_id), candidate_dist);
                }
            }
        }
        // Running the last candidate outside the loop to avoid prefetching invalid candidate
        idType candidate_id = node_meta.links[node_meta.numLinks - 1];

        if (visited_nodes->getNodeTag(candidate_id) != visited_tag) {
            visited_nodes->tagNode(candidate_id, visited_tag);
            const char *candidate_data = getDataByInternalId(candidate_id);

            DistType candidate_dist = this->dist_func(query_data, candidate_data, this->dim);
            if (candidate_dist < dyn_range) {
                // Pre-fetch current candidate meta data
                __builtin_prefetch(this->idToMetaData.data() + candidate_id);
                candidate_set.emplace(-candidate_dist, candidate_id);

                // If the new candidate is in the requested radius, add it to the results set.
                if (candidate_dist <= radius &&
                    (!has_marked_deleted || !isMarkedDeleted(candidate_id))) {
                    results->emplace(getExternalLabel(candidate_id), candidate_dist);
                }
            }
        }
    }
    __builtin_prefetch(this->meta_blocks.data() + (candidate_set.top().second / this->blockSize));
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
candidatesMaxHeap<DistType>
HNSWIndex<DataType, DistType>::searchLayer(idType ep_id, const void *data_point, size_t layer,
                                           size_t ef) const {

    auto visited_nodes = getVisitedList();

    tag_t visited_tag = visited_nodes->getFreshTag();

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

    visited_nodes->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();
        // Pre-fetch the neighbours list of the top candidate (the one that is going
        // to be processed in the next iteration) into memory cache, to improve performance.
        __builtin_prefetch(getMetaDataByInternalId(curr_el_pair.second));

        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() >= ef) {
            break;
        }
        candidate_set.pop();

        processCandidate<has_marked_deleted>(curr_el_pair.second, data_point, layer, ef,
                                             visited_nodes, visited_tag, top_candidates,
                                             candidate_set, lowerBound);
    }

#ifdef ENABLE_PARALLELIZATION
    visited_nodes_handler_pool->returnVisitedNodesHandlerToPool(visited_nodes);
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
        level_data &cur_meta = getLevelData(cur_c, level);
        assert(cur_meta.numLinks == 0 && "The newly inserted element should have blank link list");
        cur_meta.numLinks = selectedNeighbors.size();

        idType *ll_cur = cur_meta.links;

        for (auto cur_neighbor = selectedNeighbors.rbegin();
             cur_neighbor != selectedNeighbors.rend(); ++cur_neighbor) {

            assert(*ll_cur == 0 && "Possible memory corruption");
            assert(level <= getMetaDataByInternalId(*cur_neighbor)->toplevel &&
                   "Trying to make a link on a non-existent level");

            *ll_cur = *cur_neighbor;
            ll_cur++;
        }
    }

    // go over the selected neighbours - selectedNeighbor is the neighbour id
    vecsim_stl::vector<bool> neighbors_bitmap(this->allocator);
    for (idType selectedNeighbor : selectedNeighbors) {
#ifdef ENABLE_PARALLELIZATION
        std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbor]);
#endif
        level_data &neighbor_meta = getLevelData(selectedNeighbor, level);

        if (neighbor_meta.numLinks > Mcurmax)
            throw std::runtime_error("Bad value of sz_link_list_other");
        if (selectedNeighbor == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");

        // If the selected neighbor can add another link (hasn't reached the max) - add it.
        if (neighbor_meta.numLinks < Mcurmax) {
            neighbor_meta.links[neighbor_meta.numLinks] = cur_c;
            neighbor_meta.numLinks++;
        } else {
            // try finding "weak" elements to replace it with the new one with the heuristic:
            candidatesMaxHeap<DistType> candidates(this->allocator);
            // (re)use the bitmap to represent the set of the original neighbours for the current
            // selected neighbour.
            neighbors_bitmap.assign(cur_element_count, false);
            DistType d_max = this->dist_func(getDataByInternalId(cur_c),
                                             getDataByInternalId(selectedNeighbor), this->dim);
            candidates.emplace(d_max, cur_c);
            // consider cur_c as if it was a link of the selected neighbor
            neighbors_bitmap[cur_c] = true;
            for (size_t j = 0; j < neighbor_meta.numLinks; j++) {
                candidates.emplace(this->dist_func(getDataByInternalId(neighbor_meta.links[j]),
                                                   getDataByInternalId(selectedNeighbor),
                                                   this->dim),
                                   neighbor_meta.links[j]);
                neighbors_bitmap[neighbor_meta.links[j]] = true;
            }

            idType removed_links[neighbor_meta.numLinks + 1];
            size_t removed_links_num;
            removeExtraLinks(candidates, Mcurmax, neighbor_meta, neighbors_bitmap, removed_links,
                             &removed_links_num);

            // remove the current neighbor from the incoming list of nodes for the
            // neighbours that were chosen to remove (if edge wasn't bidirectional)
            for (size_t i = 0; i < removed_links_num; i++) {
                idType node_id = removed_links[i];
                level_data &node_meta = getLevelData(node_id, level);
                // if we removed cur_c (the node just inserted), then it points to the current
                // neighbour, but not vise versa.
                if (node_id == cur_c) {
                    neighbor_meta.incoming_edges->push_back(cur_c);
                    continue;
                }

                // if the node id (the neighbour's neighbour to be removed)
                // wasn't pointing to the neighbour (i.e., the edge was uni-directional),
                // we should remove the current neighbor from the node's incoming edges.
                // otherwise, the edge turned from bidirectional to
                // uni-directional, so we insert it to the neighbour's
                // incoming edges set.
                auto it = std::find(node_meta.incoming_edges->begin(),
                                    node_meta.incoming_edges->end(), selectedNeighbor);
                if (it != node_meta.incoming_edges->end()) {
                    *it = node_meta.incoming_edges->back();
                    node_meta.incoming_edges->pop_back();
                } else {
                    neighbor_meta.incoming_edges->push_back(node_id);
                }
            }
        }
    }
    return next_closest_entry_point;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::repairConnectionsForDeletion(
    idType element_internal_id, idType neighbour_id, level_data &element_meta,
    level_data &neighbour_meta, size_t level, vecsim_stl::vector<bool> &neighbours_bitmap) {

    // put the deleted element's neighbours in the candidates.
    candidatesMaxHeap<DistType> candidates(this->allocator);
    for (size_t j = 0; j < element_meta.numLinks; j++) {
        // Don't put the neighbor itself in his own candidates
        if (element_meta.links[j] == neighbour_id) {
            continue;
        }
        candidates.emplace(this->dist_func(getDataByInternalId(element_meta.links[j]),
                                           getDataByInternalId(neighbour_id), this->dim),
                           element_meta.links[j]);
    }

    // add the deleted element's neighbour's original neighbors in the candidates.
    vecsim_stl::vector<bool> neighbour_orig_neighbours_set(cur_element_count, false,
                                                           this->allocator);

    for (size_t j = 0; j < neighbour_meta.numLinks; j++) {
        neighbour_orig_neighbours_set[neighbour_meta.links[j]] = true;
        // Don't add the removed element to the candidates, nor nodes that are already in the
        // candidates set.
        if (neighbours_bitmap[neighbour_meta.links[j]] ||
            neighbour_meta.links[j] == element_internal_id) {
            continue;
        }
        candidates.emplace(this->dist_func(getDataByInternalId(neighbour_id),
                                           getDataByInternalId(neighbour_meta.links[j]), this->dim),
                           neighbour_meta.links[j]);
    }

    size_t Mcurmax = level ? maxM_ : maxM0_;
    size_t removed_links_num;
    idType removed_links[neighbour_meta.numLinks];
    removeExtraLinks(candidates, Mcurmax, neighbour_meta, neighbour_orig_neighbours_set,
                     removed_links, &removed_links_num);

    // remove neighbour id from the incoming list of nodes for his
    // neighbours that were chosen to remove
    for (size_t i = 0; i < removed_links_num; i++) {
        idType node_id = removed_links[i];
        level_data &node_meta = getLevelData(node_id, level);

        // if the node id (the neighbour's neighbour to be removed)
        // wasn't pointing to the neighbour (edge was one directional),
        // we should remove it from the node's incoming edges.
        // otherwise, edge turned from bidirectional to one directional,
        // and it should be saved in the neighbor's incoming edges.
        auto it = std::find(node_meta.incoming_edges->begin(), node_meta.incoming_edges->end(),
                            neighbour_id);
        if (it != node_meta.incoming_edges->end()) {
            *it = node_meta.incoming_edges->back();
            node_meta.incoming_edges->pop_back();
        } else {
            neighbour_meta.incoming_edges->push_back(node_id);
        }
    }

    // updates for the new edges created
    for (size_t i = 0; i < neighbour_meta.numLinks; i++) {
        idType node_id = neighbour_meta.links[i];
        if (!neighbour_orig_neighbours_set[node_id]) {
            level_data &node_meta = getLevelData(node_id, level);
            // if the node has an edge to the neighbour as well, remove it
            // from the incoming nodes of the neighbour
            // otherwise, need to update the edge as incoming.

            bool bidirectional_edge = false;
            for (size_t j = 0; j < node_meta.numLinks; j++) {
                if (node_meta.links[j] == neighbour_id) {
                    auto it = std::find(neighbour_meta.incoming_edges->begin(),
                                        neighbour_meta.incoming_edges->end(), node_id);
                    assert(it != neighbour_meta.incoming_edges->end());
                    *it = neighbour_meta.incoming_edges->back();
                    neighbour_meta.incoming_edges->pop_back();
                    bidirectional_edge = true;
                    break;
                }
            }
            if (!bidirectional_edge) {
                node_meta.incoming_edges->push_back(neighbour_id);
            }
        }
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::replaceEntryPoint() {
    idType old_entry = entrypoint_node_;
    // Sets an (arbitrary) new entry point, after deleting the current entry point.
    while (old_entry == entrypoint_node_) {
        level_data &old_entry_meta = getLevelData(old_entry, maxlevel_);
        if (old_entry_meta.numLinks > 0) {
            // Tries to set the (arbitrary) first neighbor as the entry point, if exists.
            entrypoint_node_ = old_entry_meta.links[0];
            return;
        } else {
            // If there is no neighbors in the current level, check for any vector at
            // this level to be the new entry point.
            idType cur_id = 0;
            for (DataBlock &meta_block : meta_blocks) {
                size_t size = meta_block.getLength();
                for (size_t i = 0; i < size; i++) {
                    auto meta = (element_graph_data *)meta_block.getElement(i);
                    if (meta->toplevel == maxlevel_ && cur_id != old_entry) {
                        entrypoint_node_ = cur_id;
                        return;
                    }
                    cur_id++;
                }
            }
        }
        // If we didn't find any vector at the top level, decrease the maxlevel_ and try again,
        // until we find a new entry point, or the index is empty.
        maxlevel_--;
        if ((int)maxlevel_ < 0) {
            maxlevel_ = HNSW_INVALID_LEVEL;
            entrypoint_node_ = HNSW_INVALID_ID;
        }
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::SwapLastIdWithDeletedId(idType element_internal_id,
                                                            element_graph_data *last_element_meta,
                                                            void *last_element_data) {
    // swap label
    replaceIdOfLabel(getExternalLabel(cur_element_count), element_internal_id, cur_element_count);

    // swap neighbours
    auto *cur_meta = &last_element_meta->level0;
    for (size_t level = 0; level <= last_element_meta->toplevel; level++) {

        // go over the neighbours that also points back to the last element whose is going to
        // change, and update the id.
        for (size_t i = 0; i < cur_meta->numLinks; i++) {
            idType neighbour_id = cur_meta->links[i];
            level_data &neighbour_meta = getLevelData(neighbour_id, level);

            bool bidirectional_edge = false;
            for (size_t j = 0; j < neighbour_meta.numLinks; j++) {
                // if the edge is bidirectional, update for this neighbor
                if (neighbour_meta.links[j] == cur_element_count) {
                    bidirectional_edge = true;
                    neighbour_meta.links[j] = element_internal_id;
                    break;
                }
            }

            // if this edge is uni-directional, we should update the id in the neighbor's
            // incoming edges.
            if (!bidirectional_edge) {
                auto it = std::find(neighbour_meta.incoming_edges->begin(),
                                    neighbour_meta.incoming_edges->end(), cur_element_count);
                assert(it != neighbour_meta.incoming_edges->end());
                *it = element_internal_id;
            }
        }

        // next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // updates.
        for (auto incoming_edge : *cur_meta->incoming_edges) {
            level_data &incoming_neighbour_meta = getLevelData(incoming_edge, level);
            for (size_t j = 0; j < incoming_neighbour_meta.numLinks; j++) {
                if (incoming_neighbour_meta.links[j] == cur_element_count) {
                    incoming_neighbour_meta.links[j] = element_internal_id;
                    break;
                }
            }
        }
        // Set element level's meta for the next level (1 and above)
        cur_meta =
            (level_data *)((char *)last_element_meta->others + level * this->level_data_size_);
    }

    // Move the last element's data to the deleted element's place
    auto metadata = getMetaDataByInternalId(element_internal_id);
    memcpy((void *)metadata, last_element_meta, this->element_graph_data_size_);

    auto data = getDataByInternalId(element_internal_id);
    memcpy((void *)data, last_element_data, this->element_data_size_);

    this->idToMetaData[element_internal_id] = this->idToMetaData[cur_element_count];

    if (cur_element_count == this->entrypoint_node_) {
        this->entrypoint_node_ = element_internal_id;
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::destroyMetadata(element_graph_data *egd) {
    delete egd->level0.incoming_edges;
    level_data *cur_ld = egd->others;
    for (size_t i = 0; i < egd->toplevel; i++) {
        delete cur_ld->incoming_edges;
        cur_ld = (level_data *)((char *)cur_ld + this->level_data_size_);
    }
    this->allocator->free_allocation(egd->others);
}

// This function is greedily searching for the closest candidate to the given data point at the
// given level, starting at the given node. It sets `curObj` to the closest node found, and
// `curDist` to the distance to this node. If `with_timeout` is true, the search will check for
// timeout and return if it has occurred. `timeoutCtx` and `rc` must be valid if `with_timeout` is
// true.
template <typename DataType, typename DistType>
template <bool with_timeout>
void HNSWIndex<DataType, DistType>::greedySearchLevel(const void *vector_data, size_t level,
                                                      idType &curObj, DistType &curDist,
                                                      void *timeoutCtx,
                                                      VecSimQueryResult_Code *rc) const {
    bool changed;
    do {
        if (with_timeout && VECSIM_TIMEOUT(timeoutCtx)) {
            *rc = VecSim_QueryResult_TimedOut;
            curObj = HNSW_INVALID_ID;
            return;
        }
        changed = false;
#ifdef ENABLE_PARALLELIZATION
        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
#endif
        level_data &node_meta = getLevelData(curObj, level);

        for (int i = 0; i < node_meta.numLinks; i++) {
            idType candidate = node_meta.links[i];
            assert(candidate < this->cur_element_count && "candidate error: out of index range");

            DistType d = this->dist_func(vector_data, getDataByInternalId(candidate), this->dim);
            if (d < curDist) {
                curDist = d;
                curObj = candidate;
                changed = true;
            }
        }
    } while (changed);
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
      VecSimIndexTombstone(), max_elements_(params->initialCapacity),
      element_data_size_(VecSimType_sizeof(params->type) * this->dim), vector_blocks(allocator),
      meta_blocks(allocator), idToMetaData(max_elements_, allocator)

#ifdef ENABLE_PARALLELIZATION
      ,
      link_list_locks_(max_elements_)
#endif
{
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

    element_graph_data_size_ = sizeof(element_graph_data) + sizeof(idType) * maxM0_;
    level_data_size_ = sizeof(level_data) + sizeof(idType) * maxM_;

    size_t initial_vector_size = this->max_elements_ / this->blockSize;
    if (this->max_elements_ % this->blockSize != 0) {
        initial_vector_size++;
    }
    vector_blocks.reserve(initial_vector_size);
    meta_blocks.reserve(initial_vector_size);
}

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::~HNSWIndex() {
    for (idType id = 0; id < cur_element_count; id++) {
        destroyMetadata(getMetaDataByInternalId(id));
    }
}

/**
 * Index API functions
 */
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::resizeIndex(size_t new_max_elements) {
    idToMetaData.resize(new_max_elements);
    idToMetaData.shrink_to_fit();
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

    max_elements_ = new_max_elements;
}

template <typename DataType, typename DistType>
int HNSWIndex<DataType, DistType>::removeVector(const idType element_internal_id) {

    vecsim_stl::vector<bool> neighbours_bitmap(this->allocator);

    // go over levels and repair connections
    auto element_metadata = getMetaDataByInternalId(element_internal_id);
    auto cur_meta = &element_metadata->level0;
    for (size_t level = 0; level <= element_metadata->toplevel; level++) {
        // reset the neighbours' bitmap for the current level.
        neighbours_bitmap.assign(cur_element_count, false);
        // store the deleted element's neighbours set in a bitmap for fast access.
        for (size_t j = 0; j < cur_meta->numLinks; j++) {
            neighbours_bitmap[cur_meta->links[j]] = true;
        }
        // go over the neighbours that also points back to the removed point and make a local
        // repair.
        for (size_t i = 0; i < cur_meta->numLinks; i++) {
            idType neighbour_id = cur_meta->links[i];
            level_data &neighbour_metadata = getLevelData(neighbour_id, level);

            bool bidirectional_edge = false;
            for (size_t j = 0; j < neighbour_metadata.numLinks; j++) {
                // if the edge is bidirectional, do repair for this neighbor
                if (neighbour_metadata.links[j] == element_internal_id) {
                    bidirectional_edge = true;
                    repairConnectionsForDeletion(element_internal_id, neighbour_id, *cur_meta,
                                                 neighbour_metadata, level, neighbours_bitmap);
                    break;
                }
            }

            // if this edge is uni-directional, we should remove the element from the neighbor's
            // incoming edges.
            if (!bidirectional_edge) {
                auto it = std::find(neighbour_metadata.incoming_edges->begin(),
                                    neighbour_metadata.incoming_edges->end(), element_internal_id);
                assert(it != neighbour_metadata.incoming_edges->end());
                *it = neighbour_metadata.incoming_edges->back();
                neighbour_metadata.incoming_edges->pop_back();
            }
        }

        // next, go over the rest of incoming edges (the ones that are not bidirectional) and make
        // repairs.
        for (auto incoming_edge : *cur_meta->incoming_edges) {
            repairConnectionsForDeletion(element_internal_id, incoming_edge, *cur_meta,
                                         getLevelData(incoming_edge, level), level,
                                         neighbours_bitmap);
        }
        // Set element level's meta for the next level (1 and above)
        cur_meta =
            (level_data *)((char *)element_metadata->others + level * this->level_data_size_);
    }

    // replace the entry point with another one, if we are deleting the current entry point.
    if (element_internal_id == entrypoint_node_) {
        assert(element_metadata->toplevel == maxlevel_);
        replaceEntryPoint();
    }

    // Free the element's resources
    destroyMetadata(element_metadata);

    // We can say now that the element was deleted
    --cur_element_count;

    // Get the last element's metadata and data.
    // If we are deleting the last element, we already destroyed it's metadata.
    DataBlock &last_vector_block = vector_blocks.back();
    auto last_element_data = last_vector_block.removeAndFetchLastElement();
    DataBlock &last_meta_block = meta_blocks.back();
    auto last_element_meta = (element_graph_data *)last_meta_block.removeAndFetchLastElement();

    // Swap the last id with the deleted one, and invalidate the last id data.
    if (cur_element_count != element_internal_id) {
        SwapLastIdWithDeletedId(element_internal_id, last_element_meta, last_element_data);
    }
    if (last_meta_block.getLength() == 0) {
        meta_blocks.pop_back();
        vector_blocks.pop_back();
    }

    // If we need to free a complete block & there is a least one block between the
    // capacity and the size.
    // TODO: have initial capacity aligned to block size from the beginning, and resize when last
    // block is empty (checked above).
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

    DataType normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        memcpy(normalized_blob, vector_data, this->dim * sizeof(DataType));
        normalizeVector(normalized_blob, this->dim);
        vector_data = normalized_blob;
    }

    {
#ifdef ENABLE_PARALLELIZATION
        std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);
#endif

        if (cur_element_count >= max_elements_) {
            size_t vectors_to_add = this->blockSize - max_elements_ % this->blockSize;
            resizeIndex(max_elements_ + vectors_to_add);
        }
        cur_c = cur_element_count++;
        setVectorId(label, cur_c);
        this->idToMetaData[cur_c] = element_meta_data(label);
    }
#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
#endif
    // choose randomly the maximum level in which the new element will be in the index.
    size_t element_max_level = getRandomLevel(mult_);

#ifdef ENABLE_PARALLELIZATION
    std::unique_lock<std::mutex> entry_point_lock(global);
#endif
    size_t maxlevelcopy = maxlevel_;

#ifdef ENABLE_PARALLELIZATION
    if (element_max_level <= maxlevelcopy)
        entry_point_lock.unlock();
#endif
    idType currObj = entrypoint_node_;

    // create the new element's metadata
    char tmpData[this->element_graph_data_size_] = {0};
    auto cur_meta =
        new (tmpData) element_graph_data(element_max_level, level_data_size_, this->allocator);

    if (cur_c % this->blockSize == 0) {
        this->vector_blocks.emplace_back(this->blockSize, element_data_size_, this->allocator);
        this->meta_blocks.emplace_back(this->blockSize, element_graph_data_size_, this->allocator);
    }

    // Insert the new element to the data block
    this->vector_blocks.back().addElement(vector_data);
    this->meta_blocks.back().addElement(cur_meta);

    // this condition only means that we are not inserting the first element.
    if (entrypoint_node_ != HNSW_INVALID_ID) {
        DistType cur_dist = std::numeric_limits<DistType>::max();
        if (element_max_level < maxlevelcopy) {
            cur_dist = this->dist_func(vector_data, getDataByInternalId(currObj), this->dim);
            for (size_t level = maxlevelcopy; level > element_max_level; level--) {
                // this is done for the levels which are above the max level
                // to which we are going to insert the new element. We do
                // a greedy search in the graph starting from the entry point
                // at each level, and move on with the closest element we can find.
                // When there is no improvement to do, we take a step down.
                greedySearchLevel<false>(vector_data, level, currObj, cur_dist);
            }
        }

        auto max_common_level = std::min(element_max_level, maxlevelcopy);
        if (this->num_marked_deleted) {
            if (element_max_level >= maxlevelcopy) {
                // `cur_dist` is not initialized yet.
                cur_dist = this->dist_func(vector_data, getDataByInternalId(currObj), this->dim);
            }
            for (size_t level = max_common_level; (int)level >= 0; level--) {

                candidatesMaxHeap<DistType> top_candidates =
                    searchLayer<true>(currObj, vector_data, level, ef_construction_);
                if (top_candidates.empty()) {
                    // This means that we haven't found any non-marked-deleted candidate in the
                    // layer.

                    // Get currObj and cur_dist ready for the next iteration.
                    greedySearchLevel<false>(vector_data, level, currObj, cur_dist);
                } else {
                    currObj = mutuallyConnectNewElement(cur_c, top_candidates, level);
                }
            }
        } else {
            for (size_t level = max_common_level; (int)level >= 0; level--) {

                candidatesMaxHeap<DistType> top_candidates =
                    searchLayer<false>(currObj, vector_data, level, ef_construction_);
                currObj = mutuallyConnectNewElement(cur_c, top_candidates, level);
            }
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
    return true;
}

template <typename DataType, typename DistType>
idType HNSWIndex<DataType, DistType>::searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                                          VecSimQueryResult_Code *rc) const {
    *rc = VecSim_QueryResult_OK;

    if (cur_element_count == 0) {
        return entrypoint_node_;
    }
    idType currObj = entrypoint_node_;
    DistType cur_dist =
        this->dist_func(query_data, getDataByInternalId(entrypoint_node_), this->dim);
    for (size_t level = maxlevel_; level > 0 && currObj != HNSW_INVALID_ID; level--) {
        greedySearchLevel<true>(query_data, level, currObj, cur_dist, timeoutCtx, rc);
    }
    return currObj;
}

template <typename DataType, typename DistType>
template <bool has_marked_deleted>
candidatesLabelsMaxHeap<DistType> *
HNSWIndex<DataType, DistType>::searchBottomLayer_WithTimeout(idType ep_id, const void *data_point,
                                                             size_t ef, size_t k, void *timeoutCtx,
                                                             VecSimQueryResult_Code *rc) const {

    auto visited_nodes = getVisitedList();

    tag_t visited_tag = visited_nodes->getFreshTag();

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

    visited_nodes->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();
        // Pre-fetch the neighbours list of the top candidate (the one that is going
        // to be processed in the next iteration) into memory cache, to improve performance.
        __builtin_prefetch(getMetaDataByInternalId(curr_el_pair.second));

        if ((-curr_el_pair.first) > lowerBound && top_candidates->size() >= ef) {
            break;
        }
        if (VECSIM_TIMEOUT(timeoutCtx)) {
            *rc = VecSim_QueryResult_TimedOut;
            return top_candidates;
        }
        candidate_set.pop();

        processCandidate<has_marked_deleted>(curr_el_pair.second, data_point, 0, ef, visited_nodes,
                                             visited_tag, *top_candidates, candidate_set,
                                             lowerBound);
    }
#ifdef ENABLE_PARALLELIZATION
    visited_nodes_handler_pool->returnVisitedNodesHandlerToPool(visited_nodes);
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

    DataType normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine.
    if (this->metric == VecSimMetric_Cosine) {
        memcpy(normalized_blob, query_data, this->dim * sizeof(DataType));
        normalizeVector(normalized_blob, this->dim);
        query_data = normalized_blob;
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
    idType ep_id, const void *data_point, double epsilon, DistType radius, void *timeoutCtx,
    VecSimQueryResult_Code *rc) const {

    *rc = VecSim_QueryResult_OK;
    auto res_container = getNewResultsContainer(10); // arbitrary initial cap.

    auto visited_nodes = getVisitedList();

    tag_t visited_tag = visited_nodes->getFreshTag();
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
    visited_nodes->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        pair<DistType, idType> curr_el_pair = candidate_set.top();
        // If the best candidate is outside the dynamic range in more than epsilon (relatively) - we
        // finish the search.

        // Pre-fetch the neighbours list of the top candidate (the one that is going
        // to be processed in the next iteration) into memory cache, to improve performance.
        __builtin_prefetch(getMetaDataByInternalId(curr_el_pair.second));

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
            curr_el_pair.second, data_point, 0, epsilon, visited_nodes, visited_tag, res_container,
            candidate_set, dynamic_range_search_boundaries, radius);
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

    DataType normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        memcpy(normalized_blob, query_data, this->dim * sizeof(DataType));
        normalizeVector(normalized_blob, this->dim);
        query_data = normalized_blob;
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
VecSimInfoIterator *HNSWIndex<DataType, DistType>::infoIterator() const {
    VecSimIndexInfo info = this->info();
    // For readability. Update this number when needed.
    size_t numberOfInfoFields = 12;
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields);

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
