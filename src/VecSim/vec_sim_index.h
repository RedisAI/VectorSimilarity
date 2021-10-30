#pragma once
#include "vec_sim_common.h"
#include "query_results.h"
#include <stddef.h>

class VecSimIndex {
public:
    VecSimIndex(const VecSimParams *params)
        : dim(params->size), vecType(params->type), metric(params->metric) {}
    virtual int addVector(const void *blob, size_t id) = 0;
    virtual int deleteVector(size_t id) = 0;
    virtual size_t indexSize() = 0;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) = 0;
    virtual VecSimIndexInfo info() = 0;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob) = 0;

    virtual ~VecSimIndex() {}

    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;
};