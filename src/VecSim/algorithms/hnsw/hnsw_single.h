#pragma once

#include "hnsw.h"

template <typename DataType, typename DistType>
class HNSWIndex_Single : public HNSWIndex<DataType, DistType> {
private:
    vecsim_stl::unordered_map<labelType, idType> label_lookup_;

#ifdef BUILD_TESTS
    friend class HNSWIndex_SingleSerializer;
    // Allow the following test to access the index size private member.
    friend class HNSWTest_preferAdHocOptimization_Test;
    friend class HNSWTest_test_dynamic_hnsw_info_iterator_Test;
    friend class AllocatorTest_testIncomingEdgesSet_Test;
    friend class AllocatorTest_test_hnsw_reclaim_memory_Test;
    friend class HNSWTest_testSizeEstimation_Test;
#endif

    HNSWIndex_Single() = delete;                  // default constructor is blocked.
    HNSWIndex_Single(const HNSWIndex_Single &) = delete; // default (shallow) copy constructor is blocked.
    void setExternalLabel(idType internal_id, labelType label);
    labelType *getExternalLabelPtr(idType internal_id) const;
    size_t getRandomLevel(double reverse_size);
    vecsim_stl::vector<idType> *getIncomingEdgesPtr(idType internal_id, size_t level) const;
    void setIncomingEdgesPtr(idType internal_id, size_t level, void *set_ptr);
    inline linklistsizeint *get_linklist0(idType internal_id) const;
    inline linklistsizeint *get_linklist(idType internal_id, size_t level) const;
    void setListCount(linklistsizeint *ptr, unsigned short int size);
    void removeExtraLinks(linklistsizeint *node_ll, candidatesMaxHeap<DistType> candidates,
                          size_t Mcurmax, idType *node_neighbors,
                          const vecsim_stl::vector<bool> &bitmap, idType *removed_links,
                          size_t *removed_links_num);
    inline DistType processCandidate(idType curNodeId, const void *data_point, size_t layer,
                                     size_t ef, tag_t visited_tag,
                                     candidatesMaxHeap<DistType> &top_candidates,
                                     candidatesMaxHeap<DistType> &candidates_set,
                                     DistType lowerBound) const;
    inline void processCandidate_RangeSearch(idType curNodeId, const void *data_point, size_t layer,
                                             double epsilon, tag_t visited_tag,
                                             VecSimQueryResult **top_candidates,
                                             candidatesMaxHeap<DistType> &candidate_set,
                                             DistType lowerBound, DistType radius) const;
    candidatesMaxHeap<DistType> searchLayer(idType ep_id, const void *data_point, size_t layer,
                                            size_t ef) const;
    candidatesLabelsMaxHeap<DistType>
    searchBottomLayer_WithTimeout(idType ep_id, const void *data_point, size_t ef, size_t k,
                                  void *timeoutCtx, VecSimQueryResult_Code *rc) const;
    VecSimQueryResult *searchRangeBottomLayer_WithTimeout(idType ep_id, const void *data_point,
                                                          double epsilon, DistType radius,
                                                          void *timeoutCtx,
                                                          VecSimQueryResult_Code *rc) const;
    void getNeighborsByHeuristic2(candidatesMaxHeap<DistType> &top_candidates, size_t M);
    idType mutuallyConnectNewElement(idType cur_c, candidatesMaxHeap<DistType> &top_candidates,
                                     size_t level);
    void repairConnectionsForDeletion(idType element_internal_id, idType neighbour_id,
                                      idType *neighbours_list, idType *neighbour_neighbours_list,
                                      size_t level, vecsim_stl::vector<bool> &neighbours_bitmap);
    void replaceEntryPoint(idType element_internal_id);
    void SwapLastIdWithDeletedId(idType element_internal_id, idType last_element_internal_id);


    virtual inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;
    virtual inline void setVectorId(labelType label, idType id) override;

public:
    HNSWIndex_Single(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator,
              size_t random_seed = 100, size_t initial_pool_size = 1);
    virtual ~HNSWIndex_Single() {};

    VisitedNodesHandler *getVisitedList() const;
    virtual VecSimIndexInfo info() const override;
    virtual VecSimInfoIterator *infoIterator() const override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;
    char *getDataByInternalId(idType internal_id) const;
    inline linklistsizeint *get_linklist_at_level(idType internal_id, size_t level) const;
    unsigned short int getListCount(const linklistsizeint *ptr) const;
    void resizeIndex(size_t new_max_elements);
    int deleteVector(labelType label) override;
    int addVector(const void *vector_data, labelType label) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;
    idType searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                               VecSimQueryResult_Code *rc) const;
    VecSimQueryResult_List topKQuery(const void *query_data, size_t k,
                                     VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *query_data, DistType radius,
                                      VecSimQueryParams *queryParams) override;
};

/**
 * getters and setters of index data
 */

template <typename DataType, typename DistType>
size_t HNSWIndex_Single<DataType, DistType>::indexLabelCount() const {
    return label_lookup_.size();
}

template <typename DataType, typename DistType>
double HNSWIndex_Single<DataType, DistType>::getDistanceFrom(labelType label,
                                                      const void *vector_data) const {
    auto id = label_lookup_.find(label);
    if (id == label_lookup_.end()) {
        return INVALID_SCORE;
    }
    DistType d = this->dist_func(vector_data, getDataByInternalId(id->second), this->dim);
    return d;
}

/**
 * helper functions
 */

template <typename DataType, typename DistType>
void HNSWIndex_Single<DataType, DistType>::replaceIdOfLabel(labelType label, idType new_id, idType old_id) {
    label_lookup_[label] = new_id;
}

////////////////////////////////////////////

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
HNSWIndex_Single<DataType, DistType>::HNSWIndex_Single(const HNSWParams *params,
                                         std::shared_ptr<VecSimAllocator> allocator,
                                         size_t random_seed, size_t pool_initial_size)
    : HNSWIndex<DataType, DistType>(params, allocator, random_seed, pool_initial_size), label_lookup_(this->max_elements_, allocator) {}

/**
 * Index API functions
 */
template <typename DataType, typename DistType>
void HNSWIndex_Single<DataType, DistType>::resizeIndex(size_t new_max_elements) {
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

template <typename DataType, typename DistType>
int HNSWIndex_Single<DataType, DistType>::deleteVector(const labelType label) {
    // check that the label actually exists in the graph, and update the number of elements.
    if (label_lookup_.find(label) == label_lookup_.end()) {
        return false;
    }
    idType element_internal_id = label_lookup_[label];
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
        replaceEntryPoint(element_internal_id);
    }

    // Swap the last id with the deleted one, and invalidate the last id data.
    idType last_element_internal_id = --cur_element_count;
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
int HNSWIndex_Single<DataType, DistType>::addVector(const void *vector_data, const labelType label) {

    idType cur_c;

    DataType normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine.
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        memcpy(normalized_data, vector_data, this->dim * sizeof(DataType));
        float_vector_normalize(normalized_data, this->dim);
        vector_data = normalized_data;
    }

    {
#ifdef ENABLE_PARALLELIZATION
        std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);
#endif

        // Checking if an element with the given label already exists. if so, remove it.
        if (label_lookup_.find(label) != label_lookup_.end()) {
            deleteVector(label);
        }
        if (cur_element_count >= max_elements_) {
            size_t vectors_to_add = this->blockSize - max_elements_ % this->blockSize;
            resizeIndex(max_elements_ + vectors_to_add);
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
    memcpy(getExternalLabelPtr(cur_c), &label, sizeof(labelType));
    memcpy(getDataByInternalId(cur_c), vector_data, data_size_);

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
idType HNSWIndex_Single<DataType, DistType>::searchBottomLayerEP(const void *query_data, void *timeoutCtx,
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
candidatesLabelsMaxHeap<DistType>
HNSWIndex_Single<DataType, DistType>::searchBottomLayer_WithTimeout(idType ep_id, const void *data_point,
                                                             size_t ef, size_t k, void *timeoutCtx,
                                                             VecSimQueryResult_Code *rc) const {
    candidatesLabelsMaxHeap<DistType> results(this->allocator);

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
        std::pair<DistType, idType> curr_el_pair = candidate_set.top();
        if ((-curr_el_pair.first) > lowerBound) {
            break;
        }
        if (__builtin_expect(VecSimIndexAbstract<DistType>::timeoutCallback(timeoutCtx), 0)) {
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

template <typename DataType, typename DistType>
VecSimQueryResult_List HNSWIndex_Single<DataType, DistType>::topKQuery(const void *query_data, size_t k,
                                                                VecSimQueryParams *queryParams) {

    VecSimQueryResult_List rl = {0};
    this->last_mode = STANDARD_KNN;

    if (cur_element_count == 0) {
        rl.code = VecSim_QueryResult_OK;
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    void *timeoutCtx = nullptr;

    DataType normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine.
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        memcpy(normalized_data, query_data, this->dim * sizeof(DataType));
        float_vector_normalize(normalized_data, this->dim);
        query_data = normalized_data;
    }
    // Get original efRuntime and store it.
    size_t originalEF = ef_;

    if (queryParams) {
        timeoutCtx = queryParams->timeoutCtx;
        if (queryParams->hnswRuntimeParams.efRuntime != 0) {
            ef_ = queryParams->hnswRuntimeParams.efRuntime;
        }
    }

    idType bottom_layer_ep = searchBottomLayerEP(query_data, timeoutCtx, &rl.code);
    if (VecSim_OK != rl.code) {
        ef_ = originalEF;
        return rl;
    }

    candidatesLabelsMaxHeap<DistType> results = searchBottomLayer_WithTimeout(
        bottom_layer_ep, query_data, std::max(ef_, k), k, timeoutCtx, &rl.code);

    // Restore efRuntime.
    ef_ = originalEF;

    if (VecSim_OK != rl.code) {
        return rl;
    }

    rl.results = array_new_len<VecSimQueryResult>(results.size(), results.size());
    for (int i = (int)results.size() - 1; i >= 0; --i) {
        VecSimQueryResult_SetId(rl.results[i], results.top().second);
        VecSimQueryResult_SetScore(rl.results[i], results.top().first);
        results.pop();
    }
    return rl;
}

template <typename DataType, typename DistType>
VecSimQueryResult *HNSWIndex_Single<DataType, DistType>::searchRangeBottomLayer_WithTimeout(
    idType ep_id, const void *data_point, double epsilon, DistType radius, void *timeoutCtx,
    VecSimQueryResult_Code *rc) const {
    auto *results = array_new<VecSimQueryResult>(10); // arbitrary initial cap.

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
        auto new_result = VecSimQueryResult{};
        VecSimQueryResult_SetId(new_result, getExternalLabel(ep_id));
        VecSimQueryResult_SetScore(new_result, ep_dist);
        results = array_append(results, new_result);
        dynamic_range = radius; // to ensure that dyn_range >= radius.
    }

    DistType dynamic_range_search_boundaries = dynamic_range * (1.0 + epsilon);
    candidate_set.emplace(-ep_dist, ep_id);
    this->visited_nodes_handler->tagNode(ep_id, visited_tag);

    while (!candidate_set.empty()) {
        std::pair<DistType, idType> curr_el_pair = candidate_set.top();
        // If the best candidate is outside the dynamic range in more than epsilon (relatively) - we
        // finish the search.
        if ((-curr_el_pair.first) > dynamic_range_search_boundaries) {
            break;
        }
        if (__builtin_expect(VecSimIndexAbstract<DistType>::timeoutCallback(timeoutCtx), 0)) {
            *rc = VecSim_QueryResult_TimedOut;
            return results;
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
        processCandidate_RangeSearch(curr_el_pair.second, data_point, 0, epsilon, visited_tag,
                                     &results, candidate_set, dynamic_range_search_boundaries,
                                     radius);
    }
#ifdef ENABLE_PARALLELIZATION
    visited_nodes_handler_pool->returnVisitedNodesHandlerToPool(this->visited_nodes_handler);
#endif

    *rc = VecSim_QueryResult_OK;
    return results;
}

template <typename DataType, typename DistType>
VecSimQueryResult_List HNSWIndex_Single<DataType, DistType>::rangeQuery(const void *query_data,
                                                                 DistType radius,
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
        // TODO: need more generic when other types will be supported.
        memcpy(normalized_blob, query_data, this->dim * sizeof(DataType));
        float_vector_normalize(normalized_blob, this->dim);
        query_data = normalized_blob;
    }

    double originalEpsilon = epsilon_;
    if (queryParams) {
        timeoutCtx = queryParams->timeoutCtx;
        if (queryParams->hnswRuntimeParams.epsilon != 0.0) {
            epsilon_ = queryParams->hnswRuntimeParams.epsilon;
        }
    }

    idType bottom_layer_ep = searchBottomLayerEP(query_data, timeoutCtx, &rl.code);
    if (VecSim_OK != rl.code) {
        epsilon_ = originalEpsilon;
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    // search bottom layer
    rl.results = searchRangeBottomLayer_WithTimeout(bottom_layer_ep, query_data, this->epsilon_,
                                                    radius, timeoutCtx, &rl.code);

    // Restore the default epsilon.
    epsilon_ = originalEpsilon;
    return rl;
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
HNSWIndex_Single<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                VecSimQueryParams *queryParams) {
    // As this is the only supported type, we always allocate 4 bytes for every element in the
    // vector.
    assert(this->vecType == VecSimType_FLOAT32);
    auto *queryBlobCopy = this->allocator->allocate(sizeof(DataType) * this->dim);
    memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
    if (this->metric == VecSimMetric_Cosine) {
        float_vector_normalize((DataType *)queryBlobCopy, this->dim);
    }
    // Ownership of queryBlobCopy moves to HNSW_BatchIterator that will free it at the end.
    return HNSWFactory::newBatchIterator(queryBlobCopy, queryParams, this->allocator, this);
}

template <typename DataType, typename DistType>
VecSimIndexInfo HNSWIndex_Single<DataType, DistType>::info() const {

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
    info.hnswInfo.memory = this->allocator->getAllocationSize();
    info.hnswInfo.last_mode = this->last_mode;
    return info;
}

template <typename DataType, typename DistType>
VecSimInfoIterator *HNSWIndex_Single<DataType, DistType>::infoIterator() const {
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
                         .fieldValue = {FieldValue{.uintegerValue = info.bfInfo.isMulti}}});
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
bool HNSWIndex_Single<DataType, DistType>::preferAdHocSearch(size_t subsetSize, size_t k,
                                                      bool initial_check) {
    // This heuristic is based on sklearn decision tree classifier (with 20 leaves nodes) -
    // see scripts/HNSW_batches_clf.py
    size_t index_size = this->indexSize();
    if (subsetSize > index_size) {
        throw std::runtime_error("internal error: subset size cannot be larger than index size");
    }
    size_t d = this->dim;
    size_t M = this->getM();
    float r = (index_size == 0) ? 0.0f : (float)(subsetSize) / (float)index_size;
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
