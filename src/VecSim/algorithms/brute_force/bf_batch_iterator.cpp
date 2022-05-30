#include <cassert>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <functional>
#include "bf_batch_iterator.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/query_result_struct.h"

unsigned char BF_BatchIterator::next_id = 0;

// heuristics: decide if using heap or select search, based on the ratio between the
// number of remaining results and the index size.
VecSimQueryResult_List BF_BatchIterator::searchByHeuristics(size_t n_res,
                                                            VecSimQueryResult_Order order) {
    if ((this->index->indexSize() - this->getResultsCount()) / 1000 > n_res) {
        // Heap based search always returns the results ordered by score
        return this->heapBasedSearch(n_res);
    }
    VecSimQueryResult_List rl = this->selectBasedSearch(n_res);
    if (order == BY_SCORE) {
        sort_results_by_score(rl);
    }
    return rl;
}

void BF_BatchIterator::swapScores(
    const vecsim_stl::unordered_map<size_t, size_t> &TopCandidatesIndices, size_t res_num) {
    // Create a set of the indices in the scores array for every results that we return.
    set<size_t> indices;
    for (auto pos : TopCandidatesIndices) {
        indices.insert(pos.second);
    }
    // Get the first valid position in the next iteration.
    size_t next_scores_valid_start_pos = this->scores_valid_start_pos + res_num;
    // Get the first index of a results in this iteration which is greater or equal to
    // next_scores_valid_start_pos.
    auto reuse_index_it = indices.lower_bound(next_scores_valid_start_pos);
    auto it = indices.begin();
    size_t ind = this->scores_valid_start_pos;
    // Swap elements which are in the first res_num positions in the scores array, and place them
    // in indices of results that we return now (reuse these indices).
    while (ind < next_scores_valid_start_pos) {
        // don't swap if there is a result in one of the heading indices which will be invalid from
        // next iteration.
        if (*it == ind) {
            it++;
        } else {
            this->scores[*reuse_index_it] = this->scores[ind];
            reuse_index_it++;
        }
        ind++;
    }
    this->scores_valid_start_pos = next_scores_valid_start_pos;
}

VecSimQueryResult_List BF_BatchIterator::heapBasedSearch(size_t n_res) {
    VecSimQueryResult_List rl = {0};
    float upperBound = std::numeric_limits<float>::lowest();
    vecsim_stl::max_priority_queue<pair<float, labelType>> TopCandidates(this->allocator);
    // map vector's label to its index in the scores vector.
    vecsim_stl::unordered_map<size_t, size_t> TopCandidatesIndices(n_res, this->allocator);
    for (size_t i = this->scores_valid_start_pos; i < this->scores.size(); i++) {
        if (TopCandidates.size() < n_res) {
            TopCandidates.emplace(this->scores[i].first, this->scores[i].second);
            TopCandidatesIndices[this->scores[i].second] = i;
            upperBound = TopCandidates.top().first;
        } else {
            if (this->scores[i].first >= upperBound) {
                continue;
            } else {
                // remove the furthest vector from the candidates and from the label->index mappings
                TopCandidatesIndices.erase(TopCandidates.top().second);
                TopCandidates.pop();
                TopCandidatesIndices[this->scores[i].second] = i;
                TopCandidates.emplace(this->scores[i].first, this->scores[i].second);
                upperBound = TopCandidates.top().first;
            }
        }
    }

    // Save the top results to return.
    rl.results = array_new_len<VecSimQueryResult>(TopCandidates.size(), TopCandidates.size());
    for (int i = (int)TopCandidates.size() - 1; i >= 0; --i) {
        VecSimQueryResult_SetId(rl.results[i], TopCandidates.top().second);
        VecSimQueryResult_SetScore(rl.results[i], TopCandidates.top().first);
        TopCandidates.pop();
    }
    swapScores(TopCandidatesIndices, array_len(rl.results));
    return rl;
}

VecSimQueryResult_List BF_BatchIterator::selectBasedSearch(size_t n_res) {
    VecSimQueryResult_List rl = {0};
    size_t remaining_vectors_count = this->scores.size() - this->scores_valid_start_pos;
    // Get an iterator to the effective first element in the scores array, which is the first
    // element that hasn't been returned in previous iterations.
    auto valid_begin_it = this->scores.begin() + (int)(this->scores_valid_start_pos);
    // We return up to n_res vectors, the remaining vectors size is an upper bound.
    if (n_res > remaining_vectors_count) {
        n_res = remaining_vectors_count;
    }
    auto n_th_element_pos = valid_begin_it + (int)n_res;
    // This will perform an in-place partition of the elements in the slice of the array that
    // contains valid results, based on the n-th element as the pivot - every element with a lower
    // will be placed before it, and all the rest will be placed after.
    std::nth_element(valid_begin_it, n_th_element_pos, this->scores.end());

    rl.results = array_new<VecSimQueryResult>(n_res);
    for (size_t i = this->scores_valid_start_pos; i < this->scores_valid_start_pos + n_res; i++) {
        rl.results = array_append(rl.results, VecSimQueryResult{});
        VecSimQueryResult_SetId(rl.results[array_len(rl.results) - 1], this->scores[i].second);
        VecSimQueryResult_SetScore(rl.results[array_len(rl.results) - 1], this->scores[i].first);
    }
    // Update the valid results start position after returning the results.
    this->scores_valid_start_pos += array_len(rl.results);
    return rl;
}

BF_BatchIterator::BF_BatchIterator(void *query_vector, const BruteForceIndex *bf_index,
                                   VecSimQueryParams *queryParams,
                                   std::shared_ptr<VecSimAllocator> allocator)
    : VecSimBatchIterator(query_vector, queryParams ? queryParams->timeoutCtx : nullptr, allocator),
      index(bf_index), scores_valid_start_pos(0) {
    BF_BatchIterator::next_id++;
}

VecSimQueryResult_List BF_BatchIterator::getNextResults(size_t n_res,
                                                        VecSimQueryResult_Order order) {
    assert((order == BY_ID || order == BY_SCORE) &&
           "Possible order values are only 'BY_ID' or 'BY_SCORE'");
    // Only in the first iteration we need to compute all the scores
    if (this->scores.empty()) {
        assert(getResultsCount() == 0);
        this->scores.reserve(this->index->indexSize());
        vecsim_stl::vector<VectorBlock *> blocks = this->index->getVectorBlocks();
        VecSimQueryResult_Code rc;
        for (auto &block : blocks) {
            // compute the scores for the vectors in every block and extend the scores array.
            auto block_scores = this->index->computeBlockScores(
                block, this->getQueryBlob(), this->getTimeoutCtx(), &rc);
            if (VecSim_OK != rc) {
                return {NULL, rc};
            }
            for (size_t i = 0; i < block_scores.size(); i++) {
                this->scores.emplace_back(block_scores[i], block->getMember(i)->label);
            }
        }
    }
    if (__builtin_expect(VecSimIndex::timeoutCallback(this->getTimeoutCtx()), 0)) {
        return {NULL, VecSim_QueryResult_TimedOut};
    }
    VecSimQueryResult_List rl = searchByHeuristics(n_res, order);

    this->updateResultsCount(array_len(rl.results));
    if (order == BY_ID) {
        sort_results_by_id(rl);
    }
    return rl;
}

bool BF_BatchIterator::isDepleted() {
    assert(this->getResultsCount() <= this->index->indexSize());
    bool depleted = this->getResultsCount() == this->index->indexSize();
    return depleted;
}

void BF_BatchIterator::reset() {
    this->scores.clear();
    this->resetResultsCount();
    this->scores_valid_start_pos = 0;
}
