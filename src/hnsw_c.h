#pragma once

#include <stdlib.h>
#include <stdbool.h>

typedef struct BFIndex BFIndex;
typedef struct HNSWIndex HNSWIndex;
typedef struct Vector {
    size_t id;
    float dist;
} Vector;

#ifdef __cplusplus
extern "C" {
#endif

BFIndex *InitBFIndex(void);

HNSWIndex *InitHNSWIndex(void);

bool AddVectorToBFIndex(BFIndex *index, const void* vector_data, size_t id);

bool AddVectorToHNSWIndex(HNSWIndex *index, const void* vector_data, size_t id);

bool RemoveVectorFromBFIndex(BFIndex *index, size_t id);

bool RemoveVectorFromHNSWIndex(HNSWIndex *index, size_t id);

size_t GetBFIndexSize(BFIndex *index);

size_t GetHNSWIndexSize(HNSWIndex *index);

Vector *BFSearch(BFIndex *index, const void* query_data, size_t k);

Vector *HNSWSearch(HNSWIndex *index, const void* query_data, size_t k);

void RemoveBFIndex(BFIndex *index);

void RemoveHNSWIndex(HNSWIndex *index);

#ifdef __cplusplus
}
#endif