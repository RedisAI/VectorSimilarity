/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "bf_batch_iterator.h"

#include <limits>

template <typename DataType, typename DistType>
class BFM_BatchIterator : public BF_BatchIterator<DataType, DistType> {
public:
    BFM_BatchIterator(void *query_vector, const BruteForceIndex<DataType, DistType> *index,
                      VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator)
        : BF_BatchIterator<DataType, DistType>(query_vector, index, queryParams, allocator) {}

    ~BFM_BatchIterator() override = default;

private:
    inline VecSimQueryReply_Code calculateScores() override {
        this->index_label_count = this->index->indexLabelCount();
        this->scores.reserve(this->index_label_count);
        vecsim_stl::unordered_map<labelType, DistType> tmp_scores(this->index_label_count,
                                                                  this->allocator);
        auto distFunc = this->index->getDistFunc();
        auto dim = this->index->getDim();
        DataType PORTABLE_ALIGN cur_vec[this->index->getDim()];
        for (idType curr_id = 0; curr_id < this->index->indexSize(); curr_id++) {
            // compute the scores for the vectors in every block and extend the scores array.
            this->index->getDataByInternalId(curr_id, cur_vec);
            DistType curr_dist = distFunc(this->getQueryBlob(), cur_vec, dim);
            labelType curr_label = this->index->getVectorLabel(curr_id);
            auto curr_pair = tmp_scores.find(curr_label);
            // For each score, emplace or update the score of the label.
            if (curr_pair == tmp_scores.end()) {
                tmp_scores.emplace(curr_label, curr_dist);
            } else if (curr_pair->second > curr_dist) {
                curr_pair->second = curr_dist;
            }

            if (VECSIM_TIMEOUT(this->getTimeoutCtx())) {
                return VecSim_QueryReply_TimedOut;
            }
        }

        for (auto p : tmp_scores) {
            this->scores.emplace_back(p.second, p.first);
        }
        return VecSim_QueryReply_OK;
    }
};
