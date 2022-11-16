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
    inline void fillFromExtras(candidatesLabelsMaxHeap<DistType> *top_candidates) override;
    inline VecSimQueryResult_List prepareResults(candidatesLabelsMaxHeap<DistType> *top_candidates,
                                                 size_t n_res) override;

    inline void updateHeaps(candidatesLabelsMaxHeap<DistType> *top_candidates, DistType dist,
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
    candidatesLabelsMaxHeap<DistType> *top_candidates, size_t n_res) {
    VecSimQueryResult_List rl = {0};

    // Put the "spare" results (if exist) in the extra candidates heap.
    while (top_candidates->size() > n_res) {
        this->top_candidates_extras.emplace(top_candidates->top().first,
                                            top_candidates->top().second); // (distance, label)
        top_candidates->pop();
    }
    rl.results = array_new_len<VecSimQueryResult>(top_candidates->size(), top_candidates->size());
    // Return results from the top candidates heap, put them in reverse order in the batch results
    // array.
    for (int i = (int)(top_candidates->size() - 1); i >= 0; i--) {
        VecSimQueryResult_SetId(rl.results[i], top_candidates->top().second);
        VecSimQueryResult_SetScore(rl.results[i], top_candidates->top().first);
        top_candidates->pop();
    }
    return rl;
}

template <typename DataType, typename DistType>
void HNSWSingle_BatchIterator<DataType, DistType>::fillFromExtras(
    candidatesLabelsMaxHeap<DistType> *top_candidates) {
    while (top_candidates->size() < this->ef && !this->top_candidates_extras.empty()) {
        top_candidates->emplace(this->top_candidates_extras.top().first,
                                this->top_candidates_extras.top().second);
        this->top_candidates_extras.pop();
    }
}

template <typename DataType, typename DistType>
void HNSWSingle_BatchIterator<DataType, DistType>::updateHeaps(
    candidatesLabelsMaxHeap<DistType> *top_candidates, DistType dist, idType id) {
    if (top_candidates->size() < this->ef) {
        top_candidates->emplace(dist, this->index->getExternalLabel(id));
        this->lower_bound = top_candidates->top().first;
    } else if (this->lower_bound > dist) {
        top_candidates->emplace(dist, this->index->getExternalLabel(id));
        // If the top candidates queue is full, pass the "worst" results to the "extras",
        // for the next iterations.
        this->top_candidates_extras.emplace(top_candidates->top().first,
                                            top_candidates->top().second);
        top_candidates->pop();
        this->lower_bound = top_candidates->top().first;
    }
}
