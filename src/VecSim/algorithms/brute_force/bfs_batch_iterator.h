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
        idType size = this->index->indexSize();
        this->scores.reserve(this->index_label_count);
        auto distFunc = this->index->getDistFunc();
        auto dim = this->index->getDim();

        DataType PORTABLE_ALIGN cur_vec[dim];
        for (idType curr_id = 0; curr_id < size; curr_id++) {
            this->index->getDataByInternalId(curr_id, cur_vec);
            DistType curr_dist = distFunc(this->getQueryBlob(), cur_vec, dim);
            this->scores.emplace_back(curr_dist, this->index->getVectorLabel(curr_id));

            if (VECSIM_TIMEOUT(this->getTimeoutCtx())) {
                return VecSim_QueryReply_TimedOut;
            }
        }
        return VecSim_QueryReply_OK;
    }
};
