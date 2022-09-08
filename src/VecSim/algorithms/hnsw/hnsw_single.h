#pragma once

#include "hnsw.h"

template <typename DataType, typename DistType>
class HNSWIndex_Single : public HNSWIndex<DataType, DistType> {
private:
    vecsim_stl::unordered_map<labelType, idType> label_lookup_;

#ifdef BUILD_TESTS
    friend class HNSWIndexSerializer;
    // Allow the following test to access the index size private member.
    friend class AllocatorTest_testIncomingEdgesSet_Test;
    friend class AllocatorTest_test_hnsw_reclaim_memory_Test;
    friend class HNSWTest_testSizeEstimation_Test;
#endif

    // VecSimQueryResult *searchRangeBottomLayer_WithTimeout(idType ep_id, const void *data_point,
    //                                                       double epsilon, DistType radius,
    //                                                       void *timeoutCtx,
    //                                                       VecSimQueryResult_Code *rc) const;

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;
    inline void setVectorId(labelType label, idType id) override { label_lookup_[label] = id; }
    inline void resizeLabelLookup(size_t new_max_elements) override;
    inline vecsim_stl::abstract_priority_queue<DistType, labelType> *
    getNewMaxPriorityQueue() const override {
        return new (this->allocator)
            vecsim_stl::max_priority_queue<DistType, labelType>(this->allocator);
    }

public:
    HNSWIndex_Single(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator,
                     size_t random_seed = 100, size_t initial_pool_size = 1)
        : HNSWIndex<DataType, DistType>(params, allocator, random_seed, initial_pool_size),
          label_lookup_(this->max_elements_, allocator) {}

    ~HNSWIndex_Single() {}

    inline size_t indexLabelCount() const override;

    int deleteVector(labelType label) override;
    int addVector(const void *vector_data, labelType label) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;
    // VecSimQueryResult_List rangeQuery(const void *query_data, DistType radius,
    //                                   VecSimQueryParams *queryParams) override;
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
    return this->dist_func(vector_data, this->getDataByInternalId(id->second), this->dim);
}

/**
 * helper functions
 */

template <typename DataType, typename DistType>
void HNSWIndex_Single<DataType, DistType>::replaceIdOfLabel(labelType label, idType new_id,
                                                            idType old_id) {
    label_lookup_[label] = new_id;
}

template <typename DataType, typename DistType>
void HNSWIndex_Single<DataType, DistType>::resizeLabelLookup(size_t new_max_elements) {
    label_lookup_.reserve(new_max_elements);
}

/**
 * Index API functions
 */

template <typename DataType, typename DistType>
int HNSWIndex_Single<DataType, DistType>::deleteVector(const labelType label) {
    // check that the label actually exists in the graph, and update the number of elements.
    if (label_lookup_.find(label) == label_lookup_.end()) {
        return false;
    }
    idType element_internal_id = label_lookup_[label];
    label_lookup_.erase(label);
    return this->removeVector(element_internal_id);
}

template <typename DataType, typename DistType>
int HNSWIndex_Single<DataType, DistType>::addVector(const void *vector_data,
                                                    const labelType label) {

    // Checking if an element with the given label already exists. if so, remove it.
    if (label_lookup_.find(label) != label_lookup_.end()) {
        deleteVector(label);
    }

    return this->appendVector(vector_data, label);
}

// template <typename DataType, typename DistType>
// VecSimQueryResult *HNSWIndex_Single<DataType, DistType>::searchRangeBottomLayer_WithTimeout(
//     idType ep_id, const void *data_point, double epsilon, DistType radius, void *timeoutCtx,
//     VecSimQueryResult_Code *rc) const {
//     auto *results = array_new<VecSimQueryResult>(10); // arbitrary initial cap.

// #ifdef ENABLE_PARALLELIZATION
//     this->visited_nodes_handler =
//         this->visited_nodes_handler_pool->getAvailableVisitedNodesHandler();
// #endif

//     tag_t visited_tag = this->visited_nodes_handler->getFreshTag();
//     candidatesMaxHeap<DistType> candidate_set(this->allocator);

//     // Set the initial effective-range to be at least the distance from the entry-point.
//     DistType ep_dist = this->dist_func(data_point, getDataByInternalId(ep_id), this->dim);
//     DistType dynamic_range = ep_dist;
//     if (ep_dist <= radius) {
//         // Entry-point is within the radius - add it to the results.
//         auto new_result = VecSimQueryResult{};
//         VecSimQueryResult_SetId(new_result, getExternalLabel(ep_id));
//         VecSimQueryResult_SetScore(new_result, ep_dist);
//         results = array_append(results, new_result);
//         dynamic_range = radius; // to ensure that dyn_range >= radius.
//     }

//     DistType dynamic_range_search_boundaries = dynamic_range * (1.0 + epsilon);
//     candidate_set.emplace(-ep_dist, ep_id);
//     this->visited_nodes_handler->tagNode(ep_id, visited_tag);

//     while (!candidate_set.empty()) {
//         std::pair<DistType, idType> curr_el_pair = candidate_set.top();
//         // If the best candidate is outside the dynamic range in more than epsilon (relatively) -
//         we
//         // finish the search.
//         if ((-curr_el_pair.first) > dynamic_range_search_boundaries) {
//             break;
//         }
//         if (__builtin_expect(VecSimIndexAbstract<DistType>::timeoutCallback(timeoutCtx), 0)) {
//             *rc = VecSim_QueryResult_TimedOut;
//             return results;
//         }
//         candidate_set.pop();

//         // Decrease the effective range, but keep dyn_range >= radius.
//         if (-curr_el_pair.first < dynamic_range && -curr_el_pair.first >= radius) {
//             dynamic_range = -curr_el_pair.first;
//             dynamic_range_search_boundaries = dynamic_range * (1.0 + epsilon);
//         }

//         // Go over the candidate neighbours, add them to the candidates list if they are within
//         the
//         // epsilon environment of the dynamic range, and add them to the results if they are in
//         the
//         // requested radius.
//         processCandidate_RangeSearch(curr_el_pair.second, data_point, 0, epsilon, visited_tag,
//                                      &results, candidate_set, dynamic_range_search_boundaries,
//                                      radius);
//     }
// #ifdef ENABLE_PARALLELIZATION
//     visited_nodes_handler_pool->returnVisitedNodesHandlerToPool(this->visited_nodes_handler);
// #endif

//     *rc = VecSim_QueryResult_OK;
//     return results;
// }

// template <typename DataType, typename DistType>
// VecSimQueryResult_List
// HNSWIndex_Single<DataType, DistType>::rangeQuery(const void *query_data, DistType radius,
//                                                  VecSimQueryParams *queryParams) {

//     VecSimQueryResult_List rl = {0};
//     this->last_mode = RANGE_QUERY;

//     if (cur_element_count == 0) {
//         rl.code = VecSim_QueryResult_OK;
//         rl.results = array_new<VecSimQueryResult>(0);
//         return rl;
//     }
//     void *timeoutCtx = nullptr;

//     DataType normalized_blob[this->dim]; // This will be use only if metric ==
//     VecSimMetric_Cosine if (this->metric == VecSimMetric_Cosine) {
//         // TODO: need more generic when other types will be supported.
//         memcpy(normalized_blob, query_data, this->dim * sizeof(DataType));
//         float_vector_normalize(normalized_blob, this->dim);
//         query_data = normalized_blob;
//     }

//     double originalEpsilon = epsilon_;
//     if (queryParams) {
//         timeoutCtx = queryParams->timeoutCtx;
//         if (queryParams->hnswRuntimeParams.epsilon != 0.0) {
//             epsilon_ = queryParams->hnswRuntimeParams.epsilon;
//         }
//     }

//     idType bottom_layer_ep = searchBottomLayerEP(query_data, timeoutCtx, &rl.code);
//     if (VecSim_OK != rl.code) {
//         epsilon_ = originalEpsilon;
//         rl.results = array_new<VecSimQueryResult>(0);
//         return rl;
//     }

//     // search bottom layer
//     rl.results = searchRangeBottomLayer_WithTimeout(bottom_layer_ep, query_data, this->epsilon_,
//                                                     radius, timeoutCtx, &rl.code);

//     // Restore the default epsilon.
//     epsilon_ = originalEpsilon;
//     return rl;
// }
