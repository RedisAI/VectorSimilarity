#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include "vecsim.h"

#ifdef __cplusplus
extern "C" {
#endif

VecSimIndex* HNSW_New(AlgorithmParams params, DISTANCE_METRIC distanceMetric, VECTOR_TYPE vectorType, size_t vectorLen);

int HNSW_AddVector(VecSimIndex *index, const void* vector_data, size_t id);

int HNSW_DeleteVector(VecSimIndex *index, size_t id);

size_t HNSW_Size(VecSimIndex *index);

QueryResult* HNSW_TopKQuery(VecSimIndex* index, const void* queryBlob, size_t k);

void HNSW_Free(VecSimIndex *index);

// TODO

QueryResult* HNSW_DistnaceQuery(VecSimIndex* index, const void* queryBlob, float distance);

void HNSW_ClearDeleted(VecSimIndex* index);

#ifdef __cplusplus
}
#endif