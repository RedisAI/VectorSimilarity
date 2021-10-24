#pragma once

#include "VecSim/vecsim.h"

#ifdef __cplusplus
extern "C" {
#endif

VecSimIndex *BruteForce_New(const VecSimParams *params);

void BruteForce_Free(VecSimIndex *index);

int BruteForce_AddVector(VecSimIndex *index, const void *vector_data, size_t id);

int BruteForce_DeleteVector(VecSimIndex *index, size_t id);

size_t BruteForce_Size(VecSimIndex *index);

VecSimQueryResult_Collection *BruteForce_TopKQuery(VecSimIndex *index, const void *queryBlob, size_t k,
                                         VecSimQueryParams *queryParams);

// TODO

VecSimQueryResult_Collection *BruteForce_DistanceQuery(VecSimIndex *index, const void *queryBlob,
                                             float distance, VecSimQueryParams queryParams);

void BruteForce_ClearDeleted(VecSimIndex *index);

#ifdef __cplusplus
}
#endif
