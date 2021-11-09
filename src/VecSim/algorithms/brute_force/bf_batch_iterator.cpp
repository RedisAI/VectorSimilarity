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

// Compare function that handle NaN values (consider these as the highest, above inf)
bool cmp(const pair<float, labelType> &a, const pair<float, labelType> &b) {
    if (isnan(a.first)) {
        return false;
    }
    if (isnan(b.first)) {
        return true;
    }
    return a.first < b.first;
}

// heuristics: decide if using heap or select search, based on the ratio between the
// number of remaining results and the index size.
VecSimQueryResult *BF_BatchIterator::searchByHeuristics(size_t n_res, VecSimQueryResult_Order order) {
    if (getIndex()->indexSize() - getResultsCount() / 1000 > n_res) {
        return heapBasedSearch(n_res);
    }
    VecSimQueryResult *res = selectBasedSearch(n_res);
    if (order == BY_SCORE) {
        sort_results_by_score(res);
    }
    return res;
}

VecSimQueryResult *BF_BatchIterator::heapBasedSearch(size_t n_res) {
    float upperBound = std::numeric_limits<float>::lowest();
    CandidatesHeap TopCandidates;
    // map vector's label to its index in the scores vector.
    unordered_map<size_t, size_t> TopCandidatesIndices(n_res);
    for (int i = 0; i < scores.size(); i++) {
        if (isnan(scores[i].first)) {
            continue;
        }
        if (TopCandidates.size() < n_res) {
            TopCandidates.emplace(scores[i].first, scores[i].second);
            TopCandidatesIndices[scores[i].second] = i;
            upperBound = TopCandidates.top().first;
        } else {
            if (scores[i].first >= upperBound) {
                continue;
            } else {
                TopCandidates.emplace(scores[i].first, scores[i].second);
                TopCandidatesIndices[scores[i].second] = i;
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
        // Invalidate vector score, so we won't return it again in the next iterations.
        scores[TopCandidatesIndices[TopCandidates.top().second]].first = INVALID_SCORE;
        TopCandidates.pop();
    }
    return results;
}

VecSimQueryResult *BF_BatchIterator::selectBasedSearch(size_t n_res) {
    // We return up to n_res vectors, the remaining vectors size is an upper bound.
    if (n_res > scores.size()) {
        n_res = scores.size();
    }
    auto n_th_element_pos = scores.begin() + (int)n_res;
    // This will perform an in-place partition of the elements in the array, based on the
    // n-th element as the pivot - every element with a lower will be placed before it, and
    // all the rest will be placed after.
    std::nth_element(scores.begin(), n_th_element_pos, scores.end(), cmp);

    auto *results = array_new<VecSimQueryResult>(n_res);
    for (size_t i = 0; i < n_res; i++) {
        if (isnan(scores[i].first)) {
            continue;
        }
        results = array_append(results, VecSimQueryResult{});
        VecSimQueryResult_SetId(results[array_len(results) - 1], scores[i].second);
        VecSimQueryResult_SetScore(results[array_len(results) - 1], scores[i].first);
    }
    // Erase the returned results slice from the scores array
    scores.erase(scores.begin(), n_th_element_pos);
    return results;
}

BF_BatchIterator::BF_BatchIterator(const void *query_vector, const BruteForceIndex *bf_index)
    : VecSimBatchIterator(query_vector), id(BF_BatchIterator::next_id), index(bf_index) {
    BF_BatchIterator::next_id++;
}

VecSimQueryResult_List BF_BatchIterator::getNextResults(size_t n_res,
                                                        VecSimQueryResult_Order order) {
    assert((order == BY_ID || order == BY_SCORE) &&
           "Possible order values are only 'BY_ID' or 'BY_SCORE'");
    // Only in the first iteration we need to compute all the scores
    if (scores.empty() && getResultsCount() == 0) {
        scores.reserve(getIndex()->indexSize());
        vector<VectorBlock *> blocks = getIndex()->getVectorBlocks();
        for (auto& block : blocks) {
            // compute the scores for the vectors in every block and extend the scores array.
            std::vector<std::pair<float, labelType>> block_scores =
                block->computeBlockScores(getIndex()->distFunc(), getQueryBlob());
            scores.insert(scores.end(), block_scores.begin(), block_scores.end());
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
    assert(this->getResultsCount() <= index->indexSize());
    bool depleted = this->getResultsCount() == index->indexSize();
    return depleted;
}

void BF_BatchIterator::reset() {
    scores.clear();
    resetResultsCount();
}
