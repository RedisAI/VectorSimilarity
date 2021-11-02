#pragma once

#include "VecSim/vec_sim.h"

class VecSimBatchIterator {

    const void *query_vector;
    size_t returned_results_count;

public:

    explicit VecSimBatchIterator(const void *query_vector)
    : query_vector(query_vector), returned_results_count(0) {};

    inline const void *getQueryBlob() const {
        return query_vector;
    }

    inline size_t getResultsCount() const {
        return returned_results_count;
    }

    inline void updateResultsCount(size_t num) {
        returned_results_count += num;
    }

    inline void resetResultsCount() {
        returned_results_count = 0;
    }

    virtual VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) = 0;

    virtual bool isDepleted() = 0;

    virtual void reset() = 0;

    virtual ~VecSimBatchIterator() = default;

};