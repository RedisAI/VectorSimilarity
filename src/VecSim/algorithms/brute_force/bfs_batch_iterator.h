#pragma once
#include "bf_batch_iterator.h"

#include <vector>
#include <limits>

class BFS_BatchIterator : public BF_BatchIterator {
public:
    BFS_BatchIterator(void *query_vector, const BruteForceIndex<float, float> *index,
                      VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator)
        : BF_BatchIterator(query_vector, index, queryParams, allocator) {}

    ~BFS_BatchIterator() override = default;

private:
    inline VecSimQueryResult_Code calculateScores() override {

        this->scores.reserve(this->index->indexLabelCount());
        vecsim_stl::vector<VectorBlock *> blocks = this->index->getVectorBlocks();
        VecSimQueryResult_Code rc;

        idType curr_id = 0;
        for (auto &block : blocks) {
            // compute the scores for the vectors in every block and extend the scores array.
            auto block_scores = this->index->computeBlockScores(block, this->getQueryBlob(),
                                                                this->getTimeoutCtx(), &rc);
            if (VecSim_OK != rc) {
                return rc;
            }
            for (size_t i = 0; i < block_scores.size(); i++) {
                this->scores.emplace_back(block_scores[i], index->getVectorLabel(curr_id));
                ++curr_id;
            }
        }
        assert(curr_id == index->indexSize());
        return VecSim_QueryResult_OK;
    }
};
