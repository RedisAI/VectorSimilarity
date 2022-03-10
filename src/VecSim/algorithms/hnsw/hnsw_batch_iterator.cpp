#include "hnsw_batch_iterator.h"

#include <utility>
#include "VecSim/query_result_struct.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/algorithms/hnsw/visited_nodes_handler.h"
#include <limits>

inline void HNSW_BatchIterator::visitNode(idType node_id) {
    this->visited_list->tagNode(node_id, this->visited_tag);
}

inline bool HNSW_BatchIterator::hasVisitedNode(idType node_id) const {
    return this->visited_list->getNodeTag(node_id) == this->visited_tag;
}

VecSimQueryResult_List HNSW_BatchIterator::prepareResults(candidatesMaxHeap top_candidates,
                                                          size_t n_res) {
    // size_t initial_results_num = array_len(batch_results);
    // Put the "spare" results (if exist) in the results heap.
    while (top_candidates.size() > n_res) {
        this->top_candidates_extras.emplace(top_candidates.top().first,
                                            top_candidates.top().second); // (distance, id)
        top_candidates.pop();
    }
    auto *batch_results =
        array_new_len<VecSimQueryResult>(top_candidates.size(), top_candidates.size());
    // Return results from the top candidates heap, put them in reverse order in the batch results
    // array.
    for (int i = (int)(top_candidates.size() - 1); i >= 0; i--) {
        labelType label = this->hnsw_index->getExternalLabel(top_candidates.top().second);
        VecSimQueryResult_SetId(batch_results[i], label);
        VecSimQueryResult_SetScore(batch_results[i], top_candidates.top().first);
        top_candidates.pop();
    }
    return batch_results;
}

candidatesMaxHeap HNSW_BatchIterator::scanGraph(candidatesMinHeap &candidates,
                                                candidatesMinHeap &top_candidates_extras,
                                                float &lower_bound, idType entry_point) {

    candidatesMaxHeap top_candidates(this->allocator);
    if (entry_point == HNSW_INVALID_ID) {
        this->depleted = true;
        return top_candidates;
    }
    auto dist_func = this->space->get_dist_func();

    // In the first iteration, add the entry point to the empty candidates set.
    if (this->getResultsCount() == 0) {
        float dist =
            dist_func(this->getQueryBlob(), this->hnsw_index->getDataByInternalId(entry_point),
                      this->space->get_data_dim());
        lower_bound = dist;
        this->visitNode(entry_point);
        candidates.emplace(dist, entry_point);
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
        float curr_node_dist = candidates.top().first;
        idType curr_node_id = candidates.top().second;
        // If the closest element in the candidates set is further than the furthest element in the
        // top candidates set, and we have enough results, we finish the search.
        if (curr_node_dist > lower_bound && top_candidates.size() >= this->hnsw_index->getEf()) {
            break;
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
        unsigned int *cur_node_links_header =
            this->hnsw_index->get_linklist_at_level(curr_node_id, 0);
        unsigned short links_num = this->hnsw_index->getListCount(cur_node_links_header);
        auto *node_links = (unsigned int *)(cur_node_links_header + 1);
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_list->getElementsTags() + *node_links), _MM_HINT_T0);
        _mm_prefetch((char *)(visited_list->getElementsTags() + *node_links + 64), _MM_HINT_T0);
        _mm_prefetch(hnsw_index.getDataByInternalId(*node_links), _MM_HINT_T0);
        _mm_prefetch(hnsw_index.getDataByInternalId(*(node_links + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < links_num; j++) {
            unsigned int candidate_id = *(node_links + j);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_list->getElementsTags() + *(node_links + j + 1)),
                         _MM_HINT_T0);
            _mm_prefetch(hnsw_index.getDataByInternalId(*(node_links + j + 1)), _MM_HINT_T0);
#endif
            if (this->hasVisitedNode(candidate_id)) {
                continue;
            }
            this->visitNode(candidate_id);
            char *candidate_data = this->hnsw_index->getDataByInternalId(candidate_id);
            float candidate_dist = dist_func(this->getQueryBlob(), (const void *)candidate_data,
                                             this->space->get_data_dim());
            candidates.emplace(candidate_dist, candidate_id);
        }
    }

    // If we found fewer results than wanted, mark the search as depleted.
    if (top_candidates.size() < this->hnsw_index->getEf()) {
        this->depleted = true;
    }
    return top_candidates;
}

HNSW_BatchIterator::HNSW_BatchIterator(void *query_vector, HNSWIndex *index_wrapper,
                                       std::shared_ptr<VecSimAllocator> allocator)
    : VecSimBatchIterator(query_vector, std::move(allocator)), index_wrapper(index_wrapper),
      depleted(false), top_candidates_extras(this->allocator), candidates(this->allocator) {
    this->space = index_wrapper->getSpace();

    this->hnsw_index = index_wrapper->getHNSWIndex();
    this->entry_point = hnsw_index->getEntryPointId();
    // Use "fresh" tag to mark nodes that were visited along the search in some iteration.
    this->visited_list = hnsw_index->getVisitedList();
    this->visited_tag = this->visited_list->getFreshTag();
}

VecSimQueryResult_List HNSW_BatchIterator::getNextResults(size_t n_res,
                                                          VecSimQueryResult_Order order) {

    // If ef_runtime lower than the number of results to return, increase it. Therefore, we assume
    // that the number of results that return from the graph scan is at least n_res (if exist).
    size_t orig_ef = this->hnsw_index->getEf();
    if (orig_ef < n_res) {
        this->hnsw_index->setEf(n_res);
    }

    // In the first iteration, we search the graph from top bottom to find the initial entry point,
    // and then we scan the graph to get results (layer 0).
    if (this->getResultsCount() == 0) {
        idType bottom_layer_ep = this->hnsw_index->searchBottomLayerEP(this->getQueryBlob());
        this->entry_point = bottom_layer_ep;
    }
    // We ask for at least n_res candidate from the scan. In fact, at most ef results will return,
    // and it could be that ef > n_res.
    auto top_candidates = this->scanGraph(this->candidates, this->top_candidates_extras,
                                          this->lower_bound, this->entry_point);
    // Move the spare results to the "extras" queue if needed, and create the batch results array.
    auto batch_results = this->prepareResults(top_candidates, n_res);

    this->updateResultsCount(array_len(batch_results));
    if (this->getResultsCount() == this->index_wrapper->indexSize()) {
        this->depleted = true;
    }
    // By default, results are ordered by score.
    if (order == BY_ID) {
        sort_results_by_id(batch_results);
    }
    this->hnsw_index->setEf(orig_ef);
    return batch_results;
}

bool HNSW_BatchIterator::isDepleted() {
    return this->depleted && this->top_candidates_extras.empty();
}

void HNSW_BatchIterator::reset() {
    this->resetResultsCount();
    this->depleted = false;
    this->visited_tag = this->visited_list->getFreshTag();
    // Clear the queues.
    this->candidates = candidatesMinHeap(this->allocator);
    this->top_candidates_extras = candidatesMinHeap(this->allocator);
}
