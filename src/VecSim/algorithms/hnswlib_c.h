
#pragma once

#include <stdlib.h>
#include <stdbool.h>

#include "VecSim/vecsim.h"

#ifdef __cplusplus
extern "C" {
#endif

struct HNSWIndex;

VecSimIndex *HNSWLIB_New(const VecSimParams *params);

void HNSWLIB_Free(VecSimIndex *index);

int HNSWLIB_AddVector(VecSimIndex *index, const void *vector_data, size_t id);

int HNSWLIB_DeleteVector(VecSimIndex *index, size_t id);

size_t HNSWLIB_Size(VecSimIndex *index);

VecSimQueryResult *HNSWLIB_TopKQuery(VecSimIndex *index, const void *queryBlob, size_t k,
                                     VecSimQueryParams *queryParams);

// TODO

VecSimQueryResult *HNSWLIB_DistanceQuery(VecSimIndex *index, const void *queryBlob, float distance,
                                         VecSimQueryParams queryParams);

void HNSWLIB_ClearDeleted(VecSimIndex *index);

#ifdef __cplusplus
}
#endif
