#pragma once

#include "VecSim/batch_iterator.h"
#include "VecSim/utils/vecsim_stl.h"
#include "vector_block.h"
#include <vector>
#include <limits>

using namespace std;

template <typename DataType, typename DistType>
class BruteForceIndex;
class BF_BatchIterator : public VecSimBatchIterator {
protected:
    const BruteForceIndex<float, float> *index;
    vector<pair<float, labelType>> scores; // vector of scores for every label.
    size_t scores_valid_start_pos; // the first index in the scores vector that contains a vector
                                   // that hasn't been returned already.

    VecSimQueryResult_List searchByHeuristics(size_t n_res, VecSimQueryResult_Order order);
    VecSimQueryResult_List selectBasedSearch(size_t n_res);
    VecSimQueryResult_List heapBasedSearch(size_t n_res);
    void swapScores(const vecsim_stl::unordered_map<size_t, size_t> &TopCandidatesIndices,
                    size_t res_num);

    virtual inline VecSimQueryResult_Code calculateScores() = 0;

public:
    BF_BatchIterator(void *query_vector, const BruteForceIndex<float, float> *index,
                     VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator);

    inline const BruteForceIndex<float, float> *getIndex() const { return index; };

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~BF_BatchIterator() override = default;
};
