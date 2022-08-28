#pragma once

#include "VecSim/batch_iterator.h"
#include "hnsw_wrapper.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/vec_sim_common.h"    //labelType, idType
#include "VecSim/vec_sim_interface.h" // timeoutCallback
#include "VecSim/algorithms/hnsw/hnsw_wrapper.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/algorithms/hnsw/visited_nodes_handler.h"
#include <utility> //pair
#include <limits>

using hnswlib::linklistsizeint;
using spaces::dist_func_t;

template <typename DataType, typename DistType>
class HNSW_BatchIterator : public VecSimBatchIterator {
private:
    HNSWIndex<DataType, DistType> *index_wrapper;
    std::shared_ptr<hnswlib::HierarchicalNSW<DistType>> hnsw_index;
    dist_func_t<DistType> dist_func;
    size_t dim;
    hnswlib::VisitedNodesHandler *visited_list; // Pointer to the hnsw visitedList structure.
    unsigned short visited_tag;                 // Used to mark nodes that were scanned.
    idType entry_point;                         // Internal id of the node to begin the scan from.
    bool depleted;
    size_t orig_ef_runtime; // Original default parameter to reproduce.

    // Data structure that holds the search state between iterations.
    using candidatesMinHeap = vecsim_stl::min_priority_queue<std::pair<DistType, idType>>;
    using candidatesMaxHeap = vecsim_stl::max_priority_queue<std::pair<DistType, idType>>;
    DistType lower_bound;
    candidatesMinHeap top_candidates_extras;
    candidatesMinHeap candidates;

    candidatesMaxHeap scanGraph(candidatesMinHeap &candidates,
                                candidatesMinHeap &spare_top_candidates, DistType &lower_bound,
                                idType entry_point, VecSimQueryResult_Code *rc);
    VecSimQueryResult_List prepareResults(candidatesMaxHeap top_candidates, size_t n_res);
    inline void visitNode(idType node_id) {
        this->visited_list->tagNode(node_id, this->visited_tag);
    }
    inline bool hasVisitedNode(idType node_id) const {
        return this->visited_list->getNodeTag(node_id) == this->visited_tag;
    }

public:
    HNSW_BatchIterator(void *query_vector, HNSWIndex<DataType, DistType> *index,
                       VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator);

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~HNSW_BatchIterator() override;
};

/******************** Implementation **************/

template <typename DataType, typename DistType>
VecSimQueryResult_List
HNSW_BatchIterator<DataType, DistType>::prepareResults(candidatesMaxHeap top_candidates,
                                                       size_t n_res) {
    VecSimQueryResult_List rl = {0};
    // size_t initial_results_num = array_len(batch_results);
    // Put the "spare" results (if exist) in the results heap.
    while (top_candidates.size() > n_res) {
        this->top_candidates_extras.emplace(top_candidates.top().first,
                                            top_candidates.top().second); // (distance, id)
        top_candidates.pop();
    }
    rl.results = array_new_len<VecSimQueryResult>(top_candidates.size(), top_candidates.size());
    // Return results from the top candidates heap, put them in reverse order in the batch results
    // array.
    for (int i = (int)(top_candidates.size() - 1); i >= 0; i--) {
        labelType label = this->hnsw_index->getExternalLabel(top_candidates.top().second);
        VecSimQueryResult_SetId(rl.results[i], label);
        VecSimQueryResult_SetScore(rl.results[i], top_candidates.top().first);
        top_candidates.pop();
    }
    return rl;
}

template <typename DataType, typename DistType>
HNSW_BatchIterator<DataType, DistType>::candidatesMaxHeap
HNSW_BatchIterator<DataType, DistType>::scanGraph(candidatesMinHeap &candidates,
                                                  candidatesMinHeap &top_candidates_extras,
                                                  DistType &lower_bound, idType entry_point,
                                                  VecSimQueryResult_Code *rc) {

    candidatesMaxHeap top_candidates(this->allocator);
    if (entry_point == HNSW_INVALID_ID) {
        this->depleted = true;
        return top_candidates;
    }

    // In the first iteration, add the entry point to the empty candidates set.
    if (this->getResultsCount() == 0 && this->top_candidates_extras.empty() &&
        this->candidates.empty()) {
        DistType dist = dist_func(this->getQueryBlob(),
                                  this->hnsw_index->getDataByInternalId(entry_point), dim);
        lower_bound = dist;
        this->visitNode(entry_point);
        candidates.emplace(dist, entry_point);
    }
    // Checks that we didn't got timeout between iterations.
    if (__builtin_expect(VecSimIndex::timeoutCallback(this->getTimeoutCtx()), 0)) {
        *rc = VecSim_QueryResult_TimedOut;
        return top_candidates;
    }

    // Move extras from previous iteration to the top candidates.
    while (top_candidates.size() < this->hnsw_index->getEf() && !top_candidates_extras.empty()) {
        top_candidates.emplace(top_candidates_extras.top().first,
                               top_candidates_extras.top().second);
        top_candidates_extras.pop();
    }
    if (top_candidates.size() == this->hnsw_index->getEf()) {
        return top_candidates;
    }

    while (!candidates.empty()) {
        DistType curr_node_dist = candidates.top().first;
        idType curr_node_id = candidates.top().second;
        // If the closest element in the candidates set is further than the furthest element in the
        // top candidates set, and we have enough results, we finish the search.
        if (curr_node_dist > lower_bound && top_candidates.size() >= this->hnsw_index->getEf()) {
            break;
        }
        if (__builtin_expect(VecSimIndex::timeoutCallback(this->getTimeoutCtx()), 0)) {
            *rc = VecSim_QueryResult_TimedOut;
            return top_candidates;
        }
        if (top_candidates.size() < this->hnsw_index->getEf() || lower_bound > curr_node_dist) {
            top_candidates.emplace(curr_node_dist, curr_node_id);
            if (top_candidates.size() > this->hnsw_index->getEf()) {
                // If the top candidates queue is full, pass the "worst" results to the "extras",
                // for the next iterations.
                top_candidates_extras.emplace(top_candidates.top().first,
                                              top_candidates.top().second);
                top_candidates.pop();
            }
            if (!top_candidates.empty()) {
                lower_bound = top_candidates.top().first;
            }
        }

        // Take the current node out of the candidates queue and go over his neighbours.
        candidates.pop();
        linklistsizeint *cur_node_links_header =
            this->hnsw_index->get_linklist_at_level(curr_node_id, 0);
        unsigned short links_num = this->hnsw_index->getListCount(cur_node_links_header);
        auto *node_links = (linklistsizeint *)(cur_node_links_header + 1);

        __builtin_prefetch(visited_list->getElementsTags() + *node_links);
        __builtin_prefetch(hnsw_index->getDataByInternalId(*node_links));

        for (size_t j = 0; j < links_num; j++) {
            linklistsizeint candidate_id = *(node_links + j);

            __builtin_prefetch(visited_list->getElementsTags() + *(node_links + j + 1));
            __builtin_prefetch(hnsw_index->getDataByInternalId(*(node_links + j + 1)));

            if (this->hasVisitedNode(candidate_id)) {
                continue;
            }
            this->visitNode(candidate_id);
            char *candidate_data = this->hnsw_index->getDataByInternalId(candidate_id);
            DistType candidate_dist =
                dist_func(this->getQueryBlob(), (const void *)candidate_data, dim);
            candidates.emplace(candidate_dist, candidate_id);
            __builtin_prefetch(hnsw_index->get_linklist_at_level(candidates.top().second, 0));
        }
    }

    // If we found fewer results than wanted, mark the search as depleted.
    if (top_candidates.size() < this->hnsw_index->getEf()) {
        this->depleted = true;
    }
    return top_candidates;
}

template <typename DataType, typename DistType>
HNSW_BatchIterator<DataType, DistType>::HNSW_BatchIterator(
    void *query_vector, HNSWIndex<DataType, DistType> *index_wrapper,
    VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimBatchIterator(query_vector, queryParams ? queryParams->timeoutCtx : nullptr,
                          std::move(allocator)),
      index_wrapper(index_wrapper), depleted(false), top_candidates_extras(this->allocator),
      candidates(this->allocator) {

    this->hnsw_index = index_wrapper->getHNSWIndex();
    this->dist_func = index_wrapper->GetDistFunc();
    this->dim = index_wrapper->GetDim();
    this->entry_point = hnsw_index->getEntryPointId();
    // Use "fresh" tag to mark nodes that were visited along the search in some iteration.
    this->visited_list = hnsw_index->getVisitedList();
    this->visited_tag = this->visited_list->getFreshTag();
    this->orig_ef_runtime = this->hnsw_index->getEf();
    if (queryParams && queryParams->hnswRuntimeParams.efRuntime > 0) {
        this->hnsw_index->setEf(queryParams->hnswRuntimeParams.efRuntime);
    }
}

template <typename DataType, typename DistType>
VecSimQueryResult_List
HNSW_BatchIterator<DataType, DistType>::getNextResults(size_t n_res,
                                                       VecSimQueryResult_Order order) {

    VecSimQueryResult_List batch = {0};
    // If ef_runtime lower than the number of results to return, increase it. Therefore, we assume
    // that the number of results that return from the graph scan is at least n_res (if exist).
    size_t orig_ef = this->hnsw_index->getEf();
    if (orig_ef < n_res) {
        this->hnsw_index->setEf(n_res);
    }

    // In the first iteration, we search the graph from top bottom to find the initial entry point,
    // and then we scan the graph to get results (layer 0).
    if (this->getResultsCount() == 0) {
        idType bottom_layer_ep = this->hnsw_index->searchBottomLayerEP(
            this->getQueryBlob(), this->getTimeoutCtx(), &batch.code);
        if (VecSim_OK != batch.code) {
            return batch;
        }
        this->entry_point = bottom_layer_ep;
    }
    // We ask for at least n_res candidate from the scan. In fact, at most ef results will return,
    // and it could be that ef > n_res.
    auto top_candidates = this->scanGraph(this->candidates, this->top_candidates_extras,
                                          this->lower_bound, this->entry_point, &batch.code);
    if (VecSim_OK != batch.code) {
        return batch;
    }
    // Move the spare results to the "extras" queue if needed, and create the batch results array.
    batch = this->prepareResults(top_candidates, n_res);

    this->updateResultsCount(VecSimQueryResult_Len(batch));
    if (this->getResultsCount() == this->index_wrapper->indexLabelCount()) {
        this->depleted = true;
    }
    // By default, results are ordered by score.
    if (order == BY_ID) {
        sort_results_by_id(batch);
    }
    this->hnsw_index->setEf(orig_ef);
    return batch;
}

template <typename DataType, typename DistType>
bool HNSW_BatchIterator<DataType, DistType>::isDepleted() {
    return this->depleted && this->top_candidates_extras.empty();
}

template <typename DataType, typename DistType>
void HNSW_BatchIterator<DataType, DistType>::reset() {
    this->resetResultsCount();
    this->depleted = false;
    this->visited_tag = this->visited_list->getFreshTag();
    // Clear the queues.
    this->candidates = candidatesMinHeap(this->allocator);
    this->top_candidates_extras = candidatesMinHeap(this->allocator);
}

template <typename DataType, typename DistType>
HNSW_BatchIterator<DataType, DistType>::~HNSW_BatchIterator() {
    this->hnsw_index->setEf(this->orig_ef_runtime);
}
