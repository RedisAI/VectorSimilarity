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
VecSimQueryResult *BF_BatchIterator::searchByHeuristics(size_t n_res,
                                                        VecSimQueryResult_Order order) {
    if ((this->index->indexSize() - this->getResultsCount()) / 1000 > n_res) {
        // Heap based search always returns the results ordered by score
        return this->heapBasedSearch(n_res);
    }
    VecSimQueryResult *res = this->selectBasedSearch(n_res);
    if (order == BY_SCORE) {
        sort_results_by_score(res);
    }
    return res;
}

VecSimQueryResult *BF_BatchIterator::heapBasedSearch(size_t n_res) {
    float upperBound = std::numeric_limits<float>::lowest();
    CandidatesHeap TopCandidates(this->allocator);
    // map vector's label to its index in the scores vector.
    unordered_map<size_t, size_t> TopCandidatesIndices(n_res);
    for (size_t i = this->scores_valid_start_pos; i < this->scores.size(); i++) {
        if (TopCandidates.size() < n_res) {
            TopCandidates.emplace(this->scores[i].first, this->scores[i].second);
            TopCandidatesIndices[this->scores[i].second] = i;
            upperBound = TopCandidates.top().first;
        } else {
            if (this->scores[i].first >= upperBound) {
                continue;
            } else {
                TopCandidates.emplace(this->scores[i].first, this->scores[i].second);
                TopCandidatesIndices[this->scores[i].second] = i;
                // remove the furthest vector from the candidates and from the label->index mappings
                TopCandidatesIndices.erase(TopCandidates.top().second);
                TopCandidates.pop();
                upperBound = TopCandidates.top().first;
            }
        }
    }

    auto *results = array_new_len<VecSimQueryResult>(TopCandidates.size(), TopCandidates.size());
    for (int i = (int)TopCandidates.size() - 1; i >= 0; --i) {
        VecSimQueryResult_SetId(results[i], TopCandidates.top().second);
        VecSimQueryResult_SetScore(results[i], TopCandidates.top().first);
        // Move the first valid entry in the scores array to the current vector position,
        // and advance the scores array's head (for next iterations).
        this->scores[TopCandidatesIndices[TopCandidates.top().second]] =
            this->scores[this->scores_valid_start_pos++];
        TopCandidates.pop();
    }
    return results;
}

VecSimQueryResult *BF_BatchIterator::selectBasedSearch(size_t n_res) {
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

    auto *results = array_new<VecSimQueryResult>(n_res);
    for (size_t i = this->scores_valid_start_pos; i < this->scores_valid_start_pos + n_res; i++) {
        results = array_append(results, VecSimQueryResult{});
        VecSimQueryResult_SetId(results[array_len(results) - 1], this->scores[i].second);
        VecSimQueryResult_SetScore(results[array_len(results) - 1], this->scores[i].first);
    }
    // Update the valid results start position after returning the results.
    this->scores_valid_start_pos += array_len(results);
    return results;
}

BF_BatchIterator::BF_BatchIterator(const void *query_vector, const BruteForceIndex *bf_index,
                                   std::shared_ptr<VecSimAllocator> allocator)
    : VecSimBatchIterator(query_vector, allocator), id(BF_BatchIterator::next_id), index(bf_index),
      scores_valid_start_pos(0) {
    BF_BatchIterator::next_id++;
}

VecSimQueryResult_List BF_BatchIterator::getNextResults(size_t n_res,
                                                        VecSimQueryResult_Order order) {
    assert((order == BY_ID || order == BY_SCORE) &&
           "Possible order values are only 'BY_ID' or 'BY_SCORE'");
    // Only in the first iteration we need to compute all the scores
    if (getResultsCount() == 0) {
        assert(this->scores.empty());
        this->scores.reserve(this->index->indexSize());
        vecsim_stl::vector<VectorBlock *> blocks = this->index->getVectorBlocks();
        for (auto &block : blocks) {
            // compute the scores for the vectors in every block and extend the scores array.
            vecsim_stl::vector<std::pair<float, labelType>> block_scores =
                block->computeBlockScores(getIndex()->distFunc(), getQueryBlob());
            this->scores.insert(this->scores.end(), block_scores.begin(), block_scores.end());
        }
    }
    VecSimQueryResult *results = searchByHeuristics(n_res, order);

    this->updateResultsCount(array_len(results));
    if (order == BY_ID) {
        sort_results_by_id(results);
    }
    return results;
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
