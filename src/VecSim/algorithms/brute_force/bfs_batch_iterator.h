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
    inline VecSimQueryReply_Code calculateScores() override {
        this->index_label_count = this->index->indexLabelCount();
        this->scores.reserve(this->index_label_count);
        auto dim = this->index->getDim();
        auto distFunc = this->index->getDistFunc();

        DataType *cur_vec = this->index->getDataByInternalId(0);
        for (idType curr_id = 0; curr_id < this->index_label_count; curr_id++, cur_vec += dim) {
            if (VECSIM_TIMEOUT(this->getTimeoutCtx())) {
                return VecSim_QueryReply_TimedOut;
            }
            DistType cur_dist = distFunc(cur_vec, this->getQueryBlob(), dim);
            labelType curr_label = this->index->getVectorLabel(curr_id);
            this->scores.emplace_back(cur_dist, curr_label);
        }

        return VecSim_QueryReply_OK;
    }
};
