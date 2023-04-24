/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/vec_sim_interface.h"
#include "VecSim/utils/vec_utils.h"
#include <cstdarg>
#include <iostream>

// Print log messages to stdout
void Vecsim_Log(void *ctx, const char *message) { std::cout << message << std::endl; }

timeoutCallbackFunction VecSimIndexInterface::timeoutCallback = [](void *ctx) { return 0; };
logCallbackFunction VecSimIndexInterface::logCallback = Vecsim_Log;

int VecSimIndexInterface::addVectorWrapper(const void *blob, labelType label, void *auxiliaryCtx) {
    const void *processed_blob = processBlob(blob);
    int ret = addVector(processed_blob, label, auxiliaryCtx);
    returnProcessedBlob(processed_blob);
    return ret;
}

VecSimQueryResult_List VecSimIndexInterface::topKQueryWrapper(const void *queryBlob, size_t k,
                                                              VecSimQueryParams *queryParams) {
    const void *processed_blob = processBlob(queryBlob);
    VecSimQueryResult_List ret = topKQuery(processed_blob, k, queryParams);
    returnProcessedBlob(processed_blob);
    return ret;
}

VecSimQueryResult_List VecSimIndexInterface::rangeQueryWrapper(const void *queryBlob, double radius,
                                                               VecSimQueryParams *queryParams) {
    const void *processed_blob = processBlob(queryBlob);
    VecSimQueryResult_List ret = rangeQuery(processed_blob, radius, queryParams);
    returnProcessedBlob(processed_blob);
    return ret;
}

VecSimBatchIterator *
VecSimIndexInterface::newBatchIteratorWrapper(const void *queryBlob,
                                              VecSimQueryParams *queryParams) const {
    const void *processed_query = processBlob(queryBlob);
    VecSimBatchIterator *ret = newBatchIterator(processed_query, queryParams);
    returnProcessedBlob(processed_query);
    return ret;
}
