#pragma once

#include "VecSim/batch_iterator.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/query_result_struct.h"

using std::pair;
template <typename DataType, typename DistType>
class BF_BatchIterator : public VecSimBatchIterator {
private:
    const BruteForceIndex<DataType, DistType> *index;
    vecsim_stl::vector<pair<DistType, labelType>> scores; // vector of scores for every label.
    size_t scores_valid_start_pos; // the first index in the scores vector that contains a vector
                                   // that hasn't been returned already.

    VecSimQueryResult_List searchByHeuristics(size_t n_res, VecSimQueryResult_Order order);
    VecSimQueryResult_List selectBasedSearch(size_t n_res);
    VecSimQueryResult_List heapBasedSearch(size_t n_res);
    void swapScores(const vecsim_stl::unordered_map<size_t, size_t> &TopCandidatesIndices,
                    size_t res_num);

    inline VecSimQueryResult_Code calculateScores();

public:
    BF_BatchIterator(void *query_vector, const BruteForceIndex<DataType, DistType> *bf_index,
                     VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator);

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~BF_BatchIterator() override = default;
};
