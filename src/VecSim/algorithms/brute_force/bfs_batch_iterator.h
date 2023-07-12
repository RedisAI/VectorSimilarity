/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "bf_batch_iterator.h"

#include <limits>

template <typename DataType, typename DistType>
class BFS_BatchIterator : public BF_BatchIterator<DataType, DistType> {
public:
    BFS_BatchIterator(void *query_vector, const BruteForceIndex<DataType, DistType> *index,
                      VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator)
        : BF_BatchIterator<DataType, DistType>(query_vector, index, queryParams, allocator) {}

    ~BFS_BatchIterator() override = default;

private:
    inline VecSimQueryResult_Code calculateScores() override {
        this->index_label_count = this->index->indexLabelCount();
        this->scores.reserve(this->index_label_count);
        auto &blocks = this->index->getVectorBlocks();
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
                this->scores.emplace_back(block_scores[i], this->index->getVectorLabel(curr_id));
                ++curr_id;
            }
        }
        assert(curr_id == this->index->indexSize());
        return VecSim_QueryResult_OK;
    }
};
