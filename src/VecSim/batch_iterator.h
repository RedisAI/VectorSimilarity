#pragma once

#include "VecSim/vec_sim.h"

class VecSimBatchIterator {

    const VecSimIndex *index;
    const void *query_vector;
    size_t returned_results_count;

public:

    explicit VecSimBatchIterator(const void *query_vector, const VecSimIndex *index)
    : index(index), query_vector(query_vector), returned_results_count(0) {};

    virtual VecSimQueryResult_List getNextResults(size_t n_res) = 0;

    virtual bool isDepleted() = 0;

    virtual void reset() = 0;

    virtual ~VecSimBatchIterator() = default;

};