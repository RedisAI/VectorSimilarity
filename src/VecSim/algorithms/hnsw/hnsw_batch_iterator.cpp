#include "hnsw_batch_iterator.h"

#include <utility>
#include "VecSim/query_result_struct.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/algorithms/hnsw/visited_nodes_handler.h"

// Every tag which is greater than "tag_range_start" with an even difference,
// was meant to mark returned nodes in previous iterations.
inline bool HNSW_BatchIterator::hasReturned(idType node_id) const {
    return (this->visited_list->getNodeTag(node_id) > this->tag_range_start) &&
           ((this->visited_list->getNodeTag(node_id) - this->tag_range_start) % 2 == 0);
}

inline bool HNSW_BatchIterator::hasVisitedInCurIteration(idType node_id) const {
    return (this->visited_list->getNodeTag(node_id) == this->cur_visited_tag) ||
           (this->visited_list->getNodeTag(node_id) == this->cur_returned_visited_tag);
}

inline void HNSW_BatchIterator::visitNode(idType node_id) {
    if (hasReturned(node_id)) {
        this->visited_list->tagNode(node_id, cur_returned_visited_tag);
    } else {
        this->visited_list->tagNode(node_id, cur_visited_tag);
    }
}

inline void HNSW_BatchIterator::markReturned(idType node_id) {
    this->visited_list->tagNode(node_id, cur_returned_visited_tag);
}

inline void HNSW_BatchIterator::unmarkReturned(idType node_id) {
    this->visited_list->tagNode(node_id, cur_visited_tag);
}

vecsim_stl::max_priority_queue<pair<float, idType>> HNSW_BatchIterator::scanGraph() {

    vecsim_stl::max_priority_queue<pair<float, idType>> top_candidates(this->allocator);
    if (this->entry_point == -1) {
        this->depleted = true;
        return top_candidates;
    }
    bool ep_changed = false;
    vecsim_stl::max_priority_queue<pair<float, idType>> candidate_set(this->allocator);

    auto &hnsw_index = this->index->hnsw;
    auto space = this->index->space.get();

    // Get fresh visited tag and returned_visited (different from the previous iteration).
    this->cur_visited_tag = this->visited_list->getFreshTag();
    this->cur_returned_visited_tag = this->visited_list->getFreshTag();

    auto dist_func = space->get_dist_func();

    float dist = dist_func(this->getQueryBlob(), hnsw_index.getDataByInternalId(this->entry_point),
                           space->get_data_dim());
    float lowerBound = dist;
    this->visitNode(this->entry_point);
    if (!hasReturned(this->entry_point)) {
        top_candidates.emplace(dist, this->entry_point);
        this->markReturned(this->entry_point);
    }
    // The candidates distances are saved negatively, so we will have O(1) access to the closest
    // candidate from the max heap, which is the one with the largest (negative) value.
    candidate_set.emplace(-dist, this->entry_point);

    while (!candidate_set.empty()) {
        pair<float, idType> curr_el_pair = candidate_set.top();
        // If the closest element in the candidates set is further than the furthest element in the
        // top candidates set, we finish the search.
        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() >= hnsw_index.getEf()) {
            break;
        }
        candidate_set.pop();
        idType cur_node_id = curr_el_pair.second;

        uint *cur_node_links_header = hnsw_index.get_linklist_at_level(cur_node_id, 0);
        ushort links_num = hnsw_index.getListCount(cur_node_links_header);
        auto *node_links = (uint *)(cur_node_links_header + 1);
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_list->getElementsTags() + *node_links), _MM_HINT_T0);
        _mm_prefetch((char *)(visited_list->getElementsTags() + *node_links + 64), _MM_HINT_T0);
        _mm_prefetch(hnsw_index.getDataByInternalId(*node_links), _MM_HINT_T0);
        _mm_prefetch(hnsw_index.getDataByInternalId(*(node_links + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < links_num; j++) {
            uint candidate_id = *(node_links + j);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_list->getElementsTags() + *(node_links + j + 1)),
                         _MM_HINT_T0);
            _mm_prefetch(hnsw_index.getDataByInternalId(*(node_links + j + 1)), _MM_HINT_T0);
#endif
            if (this->hasVisitedInCurIteration(candidate_id)) {
                continue;
            }
            if (!this->allow_returned_candidates && this->hasReturned(candidate_id)) {
                continue;
            }
            this->visitNode(candidate_id);

            char *candidate_data = hnsw_index.getDataByInternalId(candidate_id);
            float candidate_dist = dist_func(this->getQueryBlob(), (const void *)candidate_data,
                                             space->get_data_dim());
            if (top_candidates.size() < hnsw_index.getEf() || lowerBound > candidate_dist) {
                candidate_set.emplace(-candidate_dist, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(hnsw_index.getDataByInternalId(candidate_set.top().second),
                             _MM_HINT_T0);
#endif
                if (!this->hasReturned(candidate_id)) {
                    top_candidates.emplace(candidate_dist, candidate_id);
                    this->markReturned(candidate_id);
                }

                if (top_candidates.size() > hnsw_index.getEf()) {
                    // Set as entry point for next iterations the best node found but hasn't
                    // returned.
                    this->entry_point = top_candidates.top().second;
                    ep_changed = true;
                    this->unmarkReturned(top_candidates.top().second);
                    top_candidates.pop();
                }

                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }

    // If we found fewer results than wanted, allow re-visiting nodes from previous iterations.
    if (top_candidates.size() < hnsw_index.getEf()) {
        if (!this->allow_returned_candidates) {
            this->allow_returned_candidates = true;
        } else {
            // If this options was enabled already, there are no more reachable results.
            this->depleted = true;
        }
    }
    // If the entry point hadn't changed, set it to be "worst" result that we return.
    if (!ep_changed && !top_candidates.empty()) {
        this->entry_point = top_candidates.top().second;
    }
    return top_candidates;
}

HNSW_BatchIterator::HNSW_BatchIterator(const void *query_vector, const HNSWIndex *hnsw_index,
                                       std::shared_ptr<VecSimAllocator> allocator,
                                       short max_iterations)
    : VecSimBatchIterator(query_vector, std::move(allocator)),
      // the search_id and the visited list is determined in the first iteration.
      index(hnsw_index), allow_returned_candidates(false), depleted(false), iteration_num(0),
      results(this->allocator) {

    this->entry_point = this->index->hnsw.getEntryPointId();
    // Save the current state of the visited list, and derive tags in which we are going to use
    // from the current tag. We will use these "fresh" tags to mark returned results and visited
    // nodes.
    this->visited_list = this->index->hnsw.getVisitedList();
    this->tag_range_start = this->visited_list->getFreshTag();
    // The number of iterations is bounded, as we want to ensure that tags will not reset during the
    // iterations.
    if (USHRT_MAX - this->tag_range_start < 2 * max_iterations) {
        this->visited_list->reset();
        this->tag_range_start = this->visited_list->getFreshTag();
    } else if (max_iterations <= 0) {
        throw std::runtime_error("Invalid argument given for max_iterations: should be a positive "
                                 "number lower than SHRT_MAX");
    }
    this->max_iterations = max_iterations;
}

VecSimQueryResult_List HNSW_BatchIterator::getNextResults(size_t n_res,
                                                          VecSimQueryResult_Order order) {

    auto *batch_results = array_new<VecSimQueryResult>(n_res);
    if (++iteration_num == this->max_iterations) {
        this->depleted = true;
    }
    // In the first iteration, we search the graph from top bottom to find the initial entry point,
    // and then we scan the graph to get results (layer 0).
    if (this->getResultsCount() == 0) {
        idType bottom_layer_ep = this->index->hnsw.searchBottomLayerEP(this->getQueryBlob());
        this->entry_point = bottom_layer_ep;
        auto top_candidates = this->scanGraph();
        // Get the results and insert them to a min heap.
        while (!top_candidates.empty()) {
            hnswlib::labeltype label =
                this->index->hnsw.getExternalLabel(top_candidates.top().second);
            this->results.emplace(top_candidates.top().first, label); // (distance, label)
            top_candidates.pop();
        }
    }

    while (array_len(batch_results) < n_res) {
        size_t iteration_res_num = array_len(batch_results);
        size_t num_results_to_add = min(this->results.size(), n_res - iteration_res_num);
        for (int i = 0; i < num_results_to_add; i++) {
            batch_results = array_append(batch_results, VecSimQueryResult{});
            VecSimQueryResult_SetId(batch_results[iteration_res_num], this->results.top().second);
            VecSimQueryResult_SetScore(batch_results[iteration_res_num++],
                                       this->results.top().first);
            this->results.pop();
        }
        if (iteration_res_num == n_res || this->depleted) {
            this->updateResultsCount(array_len(batch_results));
            if (this->getResultsCount() == this->index->indexSize()) {
                this->depleted = true;
            }
            // By default, results are ordered by score.
            if (order == BY_ID) {
                sort_results_by_id(batch_results);
            }
            return batch_results;
        }
        // Otherwise, scan graph for more results, and save them.
        auto top_candidates = this->scanGraph();
        // Get the results and insert them to a min heap.
        while (!top_candidates.empty()) {
            hnswlib::labeltype label =
                this->index->hnsw.getExternalLabel(top_candidates.top().second);
            this->results.emplace(top_candidates.top().first, label); // (distance, label)
            top_candidates.pop();
        }
    }
    return batch_results;
}

bool HNSW_BatchIterator::isDepleted() { return this->depleted && this->results.empty(); }

void HNSW_BatchIterator::reset() {
    this->resetResultsCount();
    this->iteration_num = 0;
    this->depleted = false;
    this->allow_returned_candidates = false;
    this->tag_range_start = this->visited_list->getFreshTag();
    if (USHRT_MAX - this->tag_range_start < 2 * max_iterations) {
        this->visited_list->reset();
        this->tag_range_start = this->visited_list->getFreshTag();
    }
    this->results = vecsim_stl::min_priority_queue<pair<float, labelType>>(
        this->allocator); // clear the results queue
}
