/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "hnsw_batch_iterator.h"

template <typename DataType, typename DistType>
class HNSWSingle_BatchIterator : public HNSW_BatchIterator<DataType, DistType> {
private:
    inline void fillFromExtras(candidatesLabelsMinMaxHeap<DistType> *top_candidates) override;
    inline VecSimQueryResult_List
    prepareResults(candidatesLabelsMinMaxHeap<DistType> *top_candidates, size_t n_res) override;

    inline void updateHeaps(candidatesLabelsMinMaxHeap<DistType> *top_candidates, DistType dist,
                            idType id) override;

public:
    HNSWSingle_BatchIterator(void *query_vector, const HNSWIndex<DataType, DistType> *index,
                             VecSimQueryParams *queryParams,
                             std::shared_ptr<VecSimAllocator> allocator)
        : HNSW_BatchIterator<DataType, DistType>(query_vector, index, queryParams, allocator) {}

    ~HNSWSingle_BatchIterator() override {}
};

/******************** Implementation **************/

template <typename DataType, typename DistType>
VecSimQueryResult_List HNSWSingle_BatchIterator<DataType, DistType>::prepareResults(
    candidatesLabelsMinMaxHeap<DistType> *top_candidates_, size_t n_res) {
    VecSimQueryResult_List rl = {0};

    auto top_candidates =
        static_cast<vecsim_stl::min_max_heap<std::pair<DistType, labelType>> *>(top_candidates_);
    size_t size = std::min(top_candidates->size(), n_res);
    rl.results = array_new_len<VecSimQueryResult>(size, size);
    // Return results from the top candidates heap, put them in reverse order in the batch results
    // array.
    for (size_t i = 0; i < size; i++) {
        auto top = top_candidates->pop_min();
        VecSimQueryResult_SetId(rl.results[i], top.second);
        VecSimQueryResult_SetScore(rl.results[i], top.first);
    }
    // Put the "spare" results (if exist) in the extra candidates heap.
    for (auto &spare : *top_candidates) {
        this->top_candidates_extras.emplace(spare); // (distance, label)
    }
    return rl;
}

template <typename DataType, typename DistType>
void HNSWSingle_BatchIterator<DataType, DistType>::fillFromExtras(
    candidatesLabelsMinMaxHeap<DistType> *top_candidates) {
    while (top_candidates->size() < this->ef && !this->top_candidates_extras.empty()) {
        top_candidates->insert(this->top_candidates_extras.top());
        this->top_candidates_extras.pop();
    }
}

template <typename DataType, typename DistType>
void HNSWSingle_BatchIterator<DataType, DistType>::updateHeaps(
    candidatesLabelsMinMaxHeap<DistType> *top_candidates, DistType dist, idType id) {
    if (top_candidates->size() < this->ef) {
        top_candidates->emplace(dist, this->index->getExternalLabel(id));
        this->lower_bound = top_candidates->peek_max().first;
    } else if (this->lower_bound > dist) {
        // If the top candidates queue is full, pass the "worst" results to the "extras",
        // for the next iterations.
        auto extra = top_candidates->exchange_max(dist, this->index->getExternalLabel(id));
        this->top_candidates_extras.emplace(extra);
        this->lower_bound = top_candidates->peek_max().first;
    }
}
