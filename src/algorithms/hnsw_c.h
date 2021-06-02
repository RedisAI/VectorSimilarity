
#pragma once

#include <stdlib.h>
#include <stdbool.h>

#include "VectorSimilarity/src/vecsim.h"

#ifdef __cplusplus
extern "C" {
#endif

struct HNSWIndex;

VecSimIndex* HNSW_New(VecSimAlgoParams *params, VecSimMetric metric, VecSimVecType vectype, size_t veclen);

void HNSW_Free(VecSimIndex *index);

int HNSW_AddVector(VecSimIndex *index, const void* vector_data, size_t id);

int HNSW_DeleteVector(VecSimIndex *index, size_t id);

size_t HNSW_Size(VecSimIndex *index);

VecSimQueryResult* HNSW_TopKQuery(VecSimIndex* index, const void* queryBlob, size_t k);

// TODO

VecSimQueryResult* HNSW_DistnaceQuery(VecSimIndex* index, const void* queryBlob, float distance);

void HNSW_ClearDeleted(VecSimIndex* index);

#ifdef __cplusplus
}
#endif
