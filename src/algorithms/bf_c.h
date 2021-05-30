#pragma once

#include <stdlib.h>
#include <stdbool.h>

typedef struct BFIndex BFIndex;

#ifdef __cplusplus
extern "C" {
#endif
#include "vecsim.h"

BFIndex *InitBFIndex(size_t max_elements, int d);

bool AddVectorToBFIndex(BFIndex *index, const void* vector_data, size_t id);

bool RemoveVectorFromBFIndex(BFIndex *index, size_t id);

size_t GetBFIndexSize(BFIndex *index);

QueryResult *BFSearch(BFIndex *index, const void* query_data, size_t k);

void RemoveBFIndex(BFIndex *index);

#ifdef __cplusplus
}
#endif
