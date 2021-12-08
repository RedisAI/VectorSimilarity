#include "hnsw_batch_iterator.h"
#include "VecSim/query_result_struct.h"


HNSW_BatchIterator::HNSW_BatchIterator(const void *query_vector, const HNSWIndex *hnsw_index,
                   std::shared_ptr<VecSimAllocator> allocator) : VecSimBatchIterator(query_vector, allocator),
                   // the search_id is determined in the first iteration.
                   search_id(INVALID_SEARCH_ID), index(hnsw_index), allow_marked_candidates(false) {
    entry_point = hnsw_index->getEntryPointId();
}

VecSimQueryResult_List HNSW_BatchIterator::getNextResults(size_t n_res, VecSimQueryResult_Order order) {

    if (this->getResultsCount() == 0) {
        auto initial_results = this->index->hnsw.searchKnn(this->getQueryBlob(),
                                                           this->index->hnsw.getEf(), this);
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

