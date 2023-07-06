/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "hnsw_batch_iterator.h"

template <typename DataType, typename DistType>
class HNSWMulti_BatchIterator : public HNSW_BatchIterator<DataType, DistType> {
private:
    vecsim_stl::unordered_set<labelType> returned;

    inline void fillFromExtras(candidatesLabelsMinMaxHeap<DistType> *top_candidates) override;
    inline VecSimQueryResult_List
    prepareResults(candidatesLabelsMinMaxHeap<DistType> *top_candidates, size_t n_res) override;
    inline void updateHeaps(candidatesLabelsMinMaxHeap<DistType> *top_candidates, DistType dist,
                            idType id) override;

public:
    HNSWMulti_BatchIterator(void *query_vector, const HNSWIndex<DataType, DistType> *index,
                            VecSimQueryParams *queryParams,
                            std::shared_ptr<VecSimAllocator> allocator)
        : HNSW_BatchIterator<DataType, DistType>(query_vector, index, queryParams, allocator),
          returned(this->index->indexSize(), this->allocator) {}

    ~HNSWMulti_BatchIterator() override {}

    void reset() override;
};

/******************** Implementation **************/

template <typename DataType, typename DistType>
VecSimQueryResult_List HNSWMulti_BatchIterator<DataType, DistType>::prepareResults(
    candidatesLabelsMinMaxHeap<DistType> *top_candidates, size_t n_res) {
    VecSimQueryResult_List rl = {0};

    // Put the "spare" results (if exist) in the extra candidates heap.
    while (top_candidates->size() > n_res) {
        this->top_candidates_extras.emplace(top_candidates->pop_max()); // (distance, label)
    }
    size_t size = top_candidates->size();
    rl.results = array_new_len<VecSimQueryResult>(size, size);
    // Return results from the top candidates heap, put them in reverse order in the batch results
    // array.
    for (size_t i = 0; i < size; i++) {
        auto top = top_candidates->pop_min();
        VecSimQueryResult_SetId(rl.results[i], top.second);
        VecSimQueryResult_SetScore(rl.results[i], top.first);
        this->returned.insert(top.second);
    }
    return rl;
}

template <typename DataType, typename DistType>
void HNSWMulti_BatchIterator<DataType, DistType>::fillFromExtras(
    candidatesLabelsMinMaxHeap<DistType> *top_candidates) {
    while (top_candidates->size() < this->ef && !this->top_candidates_extras.empty()) {
        auto top = this->top_candidates_extras.top();
        if (returned.find(top.second) == returned.end()) {
            top_candidates->insert(top);
        }
        this->top_candidates_extras.pop();
    }
}

template <typename DataType, typename DistType>
void HNSWMulti_BatchIterator<DataType, DistType>::updateHeaps(
    candidatesLabelsMinMaxHeap<DistType> *top_candidates, DistType dist, idType id) {

    if (this->lower_bound > dist || top_candidates->size() < this->ef) {
        labelType label = this->index->getExternalLabel(id);
        if (returned.find(label) == returned.end()) {
            top_candidates->emplace(dist, label);
            if (top_candidates->size() > this->ef) {
                // If the top candidates queue is full, pass the "worst" results to the "extras",
                // for the next iterations.
                this->top_candidates_extras.emplace(top_candidates->pop_max());
            }
            this->lower_bound = top_candidates->peek_max().first;
        }
    }
}

template <typename DataType, typename DistType>
void HNSWMulti_BatchIterator<DataType, DistType>::reset() {
    this->returned.clear();
    HNSW_BatchIterator<DataType, DistType>::reset();
}
