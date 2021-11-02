#include <cassert>
#include "bf_batch_iterator.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/query_result_struct.h"

unsigned char BF_BatchIterator::next_id = 0;

BF_BatchIterator::BF_BatchIterator(const void *query_vector, const BruteForceIndex *bf_index) :
    VecSimBatchIterator(query_vector), id(BF_BatchIterator::next_id) {
        index = bf_index;
        lower_bound = std::numeric_limits<float>::lowest();
        BF_BatchIterator::next_id++;
}

VecSimQueryResult_List BF_BatchIterator::getNextResults(size_t n_res, VecSimQueryResult_Order order) {
    assert((order == BY_ID || order == BY_SCORE) &&
           "Possible order values are only 'BY_ID' or 'BY_SCORE'");
    // Only in the first iteration we need to compute all the scores
    if (scores.empty()) {
        for (auto vectorBlock : getIndex()->getVectorBlocks()) {
            // compute the scores for the vectors in every block and append them to the scores vectors.
            std::vector<float> block_scores = vectorBlock->ComputeScores(getIndex()->distFunc(), getQueryBlob());
            scores.push_back(block_scores);
        }
    }
    // todo: decide if having heap or select search
    float upperBound = std::numeric_limits<float>::lowest();
    CandidatesHeap TopCandidates;
    vector<VectorBlock *> blocks = getIndex()->getVectorBlocks();
    for (size_t i = 0; i < blocks.size(); i++) {
        blocks[i]->heapBasedSearch(scores[i], lower_bound, upperBound, n_res, TopCandidates);
    }

    // update the lower bound for next iteration as the highest score in this iteration
    lower_bound = upperBound;
    this->updateResultsCount(TopCandidates.size());

    auto *results = array_new_len<VecSimQueryResult>(TopCandidates.size(), TopCandidates.size());
    for (int i = (int)TopCandidates.size() - 1; i >= 0; --i) {
        VecSimQueryResult_SetId(results[i], TopCandidates.top().second);
        VecSimQueryResult_SetScore(results[i], TopCandidates.top().first);
        TopCandidates.pop();
    }
    if (order == BY_ID) {
        sort_results_by_id(results);
    }
    return results;
}

bool BF_BatchIterator::isDepleted() {
    bool depleted = this->getResultsCount() == index->indexSize();
    return depleted;
}

void BF_BatchIterator::reset() {
    lower_bound = std::numeric_limits<float>::max();
    resetResultsCount();
}
