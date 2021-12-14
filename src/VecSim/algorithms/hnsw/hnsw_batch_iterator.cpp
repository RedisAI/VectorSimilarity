#include "hnsw_batch_iterator.h"

#include <utility>
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/hnsw/visited_nodes_handler.h"

inline bool HNSW_BatchIterator::hasReturned(idType node_id) const {
    return this->visited_list->getNodeTag(node_id) - this->tag_range_start % 2 == 1;
}

inline void HNSW_BatchIterator::markReturned (uint node_id) {
    this->visited_list->visitNode(node_id, cur_returned_visited_tag);
}

inline void HNSW_BatchIterator::unmarkReturned (uint node_id) {
    this->visited_list->visitNode(node_id, cur_visited_tag);
}

unique_ptr<CandidatesHeap> HNSW_BatchIterator::scanGraph() {

    CandidatesHeap top_candidates(this->allocator);
    CandidatesHeap candidate_set(this->allocator);

    auto &hnsw_index = this->index->hnsw;
    auto space = this->index->space.get();

    // Get fresh visited tag and returned_visited (different from the previous iteration).
    this->cur_visited_tag = this->visited_list->getFreshTag();
    this->cur_returned_visited_tag = this->visited_list->getFreshTag();

    auto dist_func = space->get_dist_func();
    float dist = dist_func(this->getQueryBlob(), hnsw_index.getDataByInternalId(this->entry_point),
                           space->get_data_dim());
    float lowerBound = dist;
    top_candidates.emplace(dist, this->entry_point);
    // the candidates distances are saved negatively, so we will have O(1) access to the closest candidate
    // from the max heap, which is the one with the largest (negative) value
    candidate_set.emplace(-dist, this->entry_point);

    this->visited_list->visitNode(this->entry_point,  cur_visited_tag);

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

        uint *cur_node_links_header = hnsw_index.get_linklist_at_level(cur_node_id, 0);
        ushort links_num = hnsw_index.getListCount(cur_node_links_header);
        auto *node_links = (uint *)(cur_node_links_header + 1);
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
            if (this->visited_list->getNodeTag(candidate_id) == this->cur_visited_tag ||
                    this->visited_list->getNodeTag(candidate_id) == this->cur_returned_visited_tag) {
                continue;
            }
            if (!this->allow_marked_candidates && this->hasReturned(candidate_id)) {
                continue;
            }
            if (hasReturned(candidate_id)) {
                this->visited_list->visitNode(candidate_id, cur_returned_visited_tag);
            } else {
                this->visited_list->visitNode(candidate_id, cur_visited_tag);
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
                    this->markReturned(candidate_id);
                }

                if (top_candidates.size() > hnsw_index.getEf()) {
                    // set as entry point for next iterations the best node found but hasn't returned
                    this->entry_point = top_candidates.top().second;
                    this->unmarkReturned(top_candidates.top().second);
                    top_candidates.pop();
                }

                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }
    this->iterations_counter++;
    return make_unique<CandidatesHeap>(top_candidates);
}

HNSW_BatchIterator::HNSW_BatchIterator(const void *query_vector, const HNSWIndex *hnsw_index,
                   std::shared_ptr<VecSimAllocator> allocator) :
                   VecSimBatchIterator(query_vector, std::move(allocator)),
                   // the search_id and the visited list is determined in the first iteration.
                   index(hnsw_index), allow_marked_candidates(false), iterations_counter(0),
                   depleted(false), results(nullptr) {

    this->entry_point = hnsw_index->getEntryPointId();
    // Save the current state of the visited list, and derive tags in which we are going to use
    // from the current tag. We will use these "fresh" tags to mark returned results and visited nodes.
    this->visited_list = this->index->hnsw.getVisitedList();
    this->tag_range_start = this->visited_list->getFreshTag();
    // Note: we assume that the number of iterations will be at most 500. We want to ensure that
    // tags will not reset during the iterations
    if (USHRT_MAX-this->tag_range_start < 1000) {
        this->visited_list->reset();
        this->tag_range_start = this->visited_list->getFreshTag();
    }
    this->cur_visited_tag = this->tag_range_start;
    // reset to get another tag to mark nodes that were returned and scanned
    this->cur_returned_visited_tag = this->visited_list->getFreshTag();
}

VecSimQueryResult_List HNSW_BatchIterator::getNextResults(size_t n_res, VecSimQueryResult_Order order) {

    // In the first iteration, we search the graph from top bottom to find the initial entry point,
    // and then we scan the graph to get results (layer 0).
    if (this->getResultsCount() == 0) {
        idType bottom_layer_ep = this->index->hnsw.searchBottomLayerEP(this->getQueryBlob());
        this->entry_point = bottom_layer_ep;
        results = this->scanGraph();
    }
    auto *batch_results = array_new<VecSimQueryResult>(n_res);
    while (array_len(batch_results) < n_res) {
        for (int i = 0; i < min(results->size(), n_res-array_len(batch_results)); i++) {
            batch_results = array_append(batch_results, VecSimQueryResult{});
            VecSimQueryResult_SetId(batch_results[i], results->top().second);
            VecSimQueryResult_SetScore(batch_results[i], results->top().first);
            results->pop();
        }
        if (array_len(batch_results) == n_res || this->depleted) {
            this->updateResultsCount(array_len(batch_results));
            return batch_results;
        }
        // Otherwise, scan graph for more results, and save them.
        results.reset(this->scanGraph().get());
    }
    return batch_results;
}

bool HNSW_BatchIterator::isDepleted() {
    return this->depleted;
}
