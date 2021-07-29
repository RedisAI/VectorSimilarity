
#pragma once

#include <stdbool.h>
#include <stdlib.h>

#include "VecSim/vecsim.h"

#ifdef __cplusplus
extern "C" {
#endif

struct HNSWIndex;

VecSimIndex *HNSW_New(VecSimParams *params);

void HNSW_Free(VecSimIndex *index);

int HNSW_AddVector(VecSimIndex *index, const void *vector_data, size_t id);

int HNSW_DeleteVector(VecSimIndex *index, size_t id);

size_t HNSW_Size(VecSimIndex *index);

VecSimQueryResult *HNSW_TopKQuery(VecSimIndex *index, const void *queryBlob, size_t k,
                                  VecSimQueryParams *queryParams);

// TODO

VecSimQueryResult *HNSW_DistanceQuery(VecSimIndex *index, const void *queryBlob, float distance,
                                      VecSimQueryParams queryParams);

void HNSW_ClearDeleted(VecSimIndex *index);

#ifdef __cplusplus
}
#endif
