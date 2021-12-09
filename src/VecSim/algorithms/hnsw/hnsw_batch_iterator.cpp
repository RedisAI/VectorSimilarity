#include "hnsw_batch_iterator.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/hnsw/visited_list_pool.h"

bool HNSW_BatchIterator::hasReturned(idType node_id) const {
    return this->visited_list[node_id] == this->c
}

CandidatesHeap HNSW_BatchIterator::scanGraph() {

    CandidatesHeap top_candidates(this->allocator);
    CandidatesHeap candidate_set(this->allocator);

    auto &hnsw_index = this->index->hnsw;
    auto space = this->index->space.get();

    hnswlib::vl_type *visited_array = this->visited_list->visitedElements;
    // Replace the visited tag to be the different one than the previous iteration tag.
    hnswlib::vl_type cur_visited_tag = this->visited_tag == this->minimal_tag ? this->minimal_tag+1 :
                                       this->minimal_tag;
    hnswlib::vl_type cur_returned_and_visited_tag = this->visited_and_returned_tag == this->minimal_tag+2 ?
            this->minimal_tag+2 : this->minimal_tag+3;
    auto dist_func = space->get_dist_func();

    float dist = dist_func(this->getQueryBlob(), hnsw_index.getDataByInternalId(this->entry_point),
                           space->get_data_dim());
    float lowerBound = dist;
    top_candidates.emplace(dist, this->entry_point);
    // the candidates distances are saved negatively, so we will have O(1) access to the closest candidate
    // from the max heap, which is the one with the largest (negative) value
    candidate_set.emplace(-dist, this->entry_point);

    visited_array[this->entry_point] = cur_visited_tag;

    while (!candidate_set.empty()) {
        pair<float, idType> curr_el_pair = candidate_set.top();
        // If the closest element in the candidates set is further than the furthest element in the top candidates
        // set, we finish the search.
        if ((-curr_el_pair.first) > lowerBound) {
            // If we found fewer results than wanted, allow re-visiting nodes from previous iterations.
            if (top_candidates.size() < hnsw_index.getEf()) {
                if (!this->allow_marked_candidates) {
                    this->allow_marked_candidates = true;
                } else {
                    // If this options was enabled already, there are no more reachable results
                    this->depleted = true;
                }
            }
            break;
        }
        candidate_set.pop();
        idType cur_node_id = curr_el_pair.second;

        unsigned int *cur_node_links_header = hnsw_index.get_linklist_at_level(cur_node_id, 0);
        unsigned short links_num = hnsw_index.getListCount(cur_node_links_header);
        auto *node_links = (unsigned int *)(cur_node_links_header + 1);
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(node_ll + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(visited_array + *(node_ll + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*node_links), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(node_links + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < links_num; j++) {
            uint candidate_id = *(node_links + j);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(node_links + j + 1)), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(node_links + j + 1)), _MM_HINT_T0);
#endif
            if (visited_array[candidate_id] == cur_visited_tag ||
            visited_array[candidate_id] == cur_returned_and_visited_tag) {
                continue;
            }
            if (!this->allow_marked_candidates && this->hasReturned(candidate_id)) {
                continue;
            }
            if (hasReturned(candidate_id)) {
                visited_array[candidate_id] = cur_returned_and_visited_tag;
            } else {
                visited_array[candidate_id] = cur_visited_tag;
            }
            char *candidate_data = hnsw_index.getDataByInternalId(candidate_id);

            float candidate_dist = dist_func(this->getQueryBlob(), (const void *)candidate_data,
                                             space->get_data_dim());
            if (top_candidates.size() < hnsw_index.getEf() || lowerBound > candidate_dist) {
                candidate_set.emplace(-candidate_dist, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
                if (!this->hasReturned(candidate_id)) {
                    top_candidates.emplace(candidate_dist, candidate_id);
                }

                if (top_candidates.size() > hnsw_index.getEf()) {
                    // set as entry point for next iterations the best node found but hasn't returned
                    this->entry_point = top_candidates.top().second;
                    top_candidates.pop();
                }

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

HNSW_BatchIterator::HNSW_BatchIterator(const void *query_vector, const HNSWIndex *hnsw_index,
                   std::shared_ptr<VecSimAllocator> allocator) : VecSimBatchIterator(query_vector, allocator),
                   // the search_id and the visited list is determined in the first iteration.
                   index(hnsw_index), allow_marked_candidates(false),
                   depleted(false) {
    entry_point = hnsw_index->getEntryPointId();

    // Save the current state of the visited list, and derive tags in which we are going to use
    // from the current tag. We will use these "fresh" tags to mark returned results and visited nodes.
    visited_list = this->index->hnsw.getVisitedList();
    // reset again to obtain 2 alternating tags for marking visited nodes in a specific iteration.
    minimal_tag = visited_tag = this->visited_list->curTag;
    this->visited_list->reset();
    // reset again twice to get another 2 alternating tags, to mark nodes that were returned and scanned
    // in a specific iteration.
    this->visited_list->reset();
    visited_and_returned_tag = this->visited_list->curTag;
    this->visited_list->reset();
}

VecSimQueryResult_List HNSW_BatchIterator::getNextResults(size_t n_res, VecSimQueryResult_Order order) {

    if (this->getResultsCount() == 0) {
        idType bottom_layer_ep = this->index->hnsw.searchBottomLayerEP(this->getQueryBlob());
        this->entry_point = bottom_layer_ep;
        auto initial_results = make_unique<CandidatesHeap>(this->scanGraph());
        results = *initial_results;
    }

    // First take results from the results set that we already found
    auto *batch_results = array_new<VecSimQueryResult>(n_res);
    for (int i = 0; i < min(results.size(), n_res); i++) {
        this->markReturned(results.top().second);
        batch_results = array_append(batch_results, VecSimQueryResult{});
        VecSimQueryResult_SetId(batch_results[i], results.top().second);
        VecSimQueryResult_SetScore(batch_results[i], results.top().first);
        results.pop();
    }
    if (array_len(batch_results) == n_res) {
        this->updateResultsCount(n_res);
        return batch_results;
    }

    // Otherwise, scan graph for more results.
    auto more_results = make_unique<CandidatesHeap>(this->scanGraph());
    results = *more_results;
}

bool HNSW_BatchIterator::isDepleted() {
    return this->depleted;
}