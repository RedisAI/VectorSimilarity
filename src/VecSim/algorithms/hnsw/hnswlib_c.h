
#pragma once

#include "VecSim/vec_sim.h"
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

struct HNSWIndex;

VecSimIndex *HNSWLib_New(const VecSimParams *params);

void HNSWLib_Free(VecSimIndex *index);

int HNSWLib_AddVector(VecSimIndex *index, const void *vector_data, size_t id);

int HNSWLib_DeleteVector(VecSimIndex *index, size_t id);

size_t HNSWLib_Size(VecSimIndex *index);

void HNSWLib_SetQueryRuntimeEf(VecSimIndex *index, size_t ef);

VecSimQueryResult_List *HNSWLib_TopKQuery(VecSimIndex *index, const void *queryBlob, size_t k,
                                          VecSimQueryParams *queryParams);

// TODO?
VecSimQueryResult_List *HNSWLib_DistanceQuery(VecSimIndex *index, const void *queryBlob,
                                              float distance, VecSimQueryParams queryParams);

#ifdef __cplusplus
}
#endif
