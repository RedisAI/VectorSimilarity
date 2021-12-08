#include "hnsw_batch_iterator.h"
#include "VecSim/query_result_struct.h"


CandidatesHeap HNSW_BatchIterator::scanGraph() {
    // We want to reset the visited list (i.e., set all nodes as unvisited), unless the following holds:
    // - we are running search in batches
    // - we are in the first phase, where we do not search again in nodes that were visited in previous iterations.
    VisitedList *vl = index.hnsw.visited_list_pool_->getFreeVisitedList(reset);
    vl_type *visited_array = vl->visitedElements;
    vl_type visited_array_tag = vl->curVisitedTag;

    if (b_iter) {
        if (b_iter->getSearchId() == INVALID_SEARCH_ID) {
            b_iter->setSearchId(visited_array_tag); // set the current tag as the search id for next iterations
        } else {
            visited_array_tag = b_iter->getSearchId(); // use the search id from previous iterations as the tag
        }
    }

    CandidatesQueue<dist_t> top_candidates(this->allocator);
    CandidatesQueue<dist_t> candidate_set(this->allocator);

    dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
    dist_t lowerBound = dist;
    top_candidates.emplace(dist, ep_id);
    // the candidates distances are saved negatively, so we will have O(1) access to the closest candidate
    // from the max heap, which is the one with the largest (negative) value
    candidate_set.emplace(-dist, ep_id);

    visited_array[ep_id] = visited_array_tag;

    while (!candidate_set.empty()) {
        pair<dist_t, tableint> curr_el_pair = candidate_set.top();
        // if the closest element in the candidates set is further than the furthest element in the top candidates
        // set, we finish the search.
        if ((-curr_el_pair.first) > lowerBound) {
            // If we found fewer results than wanted, allow
            if (b_iter && top_candidates.size() < ef) {
                b_iter->setAllowMarkedCandidates();
            }
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
    visited_list_pool_->returnVisitedListToPool(vl);
#endif

    return top_candidates;
}

HNSW_BatchIterator::HNSW_BatchIterator(const void *query_vector, const HNSWIndex *hnsw_index,
                   std::shared_ptr<VecSimAllocator> allocator) : VecSimBatchIterator(query_vector, allocator),
                   // the search_id is determined in the first iteration.
                   search_id(INVALID_SEARCH_ID), index(hnsw_index), allow_marked_candidates(false) {
    entry_point = hnsw_index->getEntryPointId();
}

VecSimQueryResult_List HNSW_BatchIterator::getNextResults(size_t n_res, VecSimQueryResult_Order order) {

    if (this->getResultsCount() == 0) {
        auto initial_results = this->index->hnsw.searchKnn(this->getQueryBlob(),
                                                           this->index->hnsw.getEf());
    }
    // First take results from the results set that we already found
    auto *batch_results = array_new<VecSimQueryResult>(n_res);
    for (int i = 0; i < min(results.size(), n_res); i++) {
        batch_results = array_append(batch_results, VecSimQueryResult{});
        VecSimQueryResult_SetId(batch_results[i], results.top().second);
        VecSimQueryResult_SetScore(batch_results[i], results.top().first);
        results.pop();
    }
    if (array_len(batch_results) == n_res) {
        this->updateResultsCount(n_res);
        return batch_results;
    }
}

