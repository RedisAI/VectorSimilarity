#include <cassert>
#include <algorithm>
#include <iostream>
#include "bf_batch_iterator.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/query_result_struct.h"

unsigned char BF_BatchIterator::next_id = 0;

VecSimQueryResult *BF_BatchIterator::heapBasedSearch(size_t n_res) {
    float upperBound = std::numeric_limits<float>::lowest();
    CandidatesHeap TopCandidates;
    vector<VectorBlock *> blocks = getIndex()->getVectorBlocks();
    for (size_t i = 0; i < blocks.size(); i++) {
        blocks[i]->heapBasedSearch(scores[i], upperBound, n_res, TopCandidates);
    }

    auto *results = array_new_len<VecSimQueryResult>(TopCandidates.size(), TopCandidates.size());
    for (int i = (int)TopCandidates.size() - 1; i >= 0; --i) {
        VecSimQueryResult_SetId(results[i], TopCandidates.top().second);
        VecSimQueryResult_SetScore(results[i], TopCandidates.top().first);
        // Erase vector score, so we won't return it again in the next iterations.
        pair<size_t, size_t> coordinates = labelToScoreCoordinates[TopCandidates.top().second];
        scores[coordinates.first].erase(scores[coordinates.first].begin() +
                                        (int)coordinates.second);
        TopCandidates.pop();
    }
    return results;
}

VecSimQueryResult *BF_BatchIterator::selectBasedSearch(size_t n_res) {
    // Flatten the score matrix, keep the labels to have stable sorting.
    vector<pair<float, labelType>> allScores;
    for (auto vectorBlockScores : scores) {
        allScores.insert(allScores.end(), vectorBlockScores.begin(), vectorBlockScores.end());
    }

    auto n_th_element_pos = allScores.begin() + (int)n_res;
    std::nth_element(allScores.begin(), n_th_element_pos, allScores.end());

    auto *results = array_new<VecSimQueryResult>(n_res);
    for (size_t i = 0; i < n_res; i++) {
        results = array_append(results, VecSimQueryResult{});
        VecSimQueryResult_SetId(results[array_len(results) - 1], allScores[i].second);
        VecSimQueryResult_SetScore(results[array_len(results) - 1], allScores[i].first);
        // Erase vector score, so we won't return it again in the next iterations.
        pair<size_t, size_t> coordinates = labelToScoreCoordinates[allScores[i].second];
        scores[coordinates.first].erase(scores[coordinates.first].begin() +
                                        (int)coordinates.second);
    }
    return results;
}

BF_BatchIterator::BF_BatchIterator(const void *query_vector, const BruteForceIndex *bf_index)
    : VecSimBatchIterator(query_vector), id(BF_BatchIterator::next_id) {
    index = bf_index;
    BF_BatchIterator::next_id++;
}

VecSimQueryResult_List BF_BatchIterator::getNextResults(size_t n_res,
                                                        VecSimQueryResult_Order order) {
    assert((order == BY_ID || order == BY_SCORE) &&
           "Possible order values are only 'BY_ID' or 'BY_SCORE'");
    // Only in the first iteration we need to compute all the scores
    if (scores.empty()) {
        vector<VectorBlock *> blocks = getIndex()->getVectorBlocks();
        for (size_t i = 0; i < blocks.size(); i++) {
            // compute the scores for the vectors in every block and create a score matrix.
            std::vector<std::pair<float, labelType>> block_scores =
                blocks[i]->ComputeScores(getIndex()->distFunc(), getQueryBlob());
            // Save the "coordinates" of every label in the scores matrix
            for (size_t j = 0; j < block_scores.size(); j++) {
                labelType label = blocks[i]->getMember(j)->label;
                labelToScoreCoordinates[label] = {i, j};
            }
            scores.push_back(block_scores);
        }
    }
    // heuristics: decide if using heap or select search, based on the ratio between the
    // number of remaining results and the index size.
    VecSimQueryResult *results;
    if ((getIndex()->indexSize() - getResultsCount()) / 1000 > n_res) {
        results = heapBasedSearch(n_res);
    } else {
        results = selectBasedSearch(n_res);
    }
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
