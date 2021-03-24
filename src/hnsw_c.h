#pragma once

#include <stdlib.h>
#include <stdbool.h>

typedef struct BFIndex BFIndex;
typedef struct HNSWIndex HNSWIndex;

#ifdef __cplusplus
extern "C" {
#endif

BFIndex *InitBFIndex(void);

HNSWIndex *InitHNSWIndex(void);

bool AddVectorToBFIndex(BFIndex *index, const char* vector_data, size_t id);

bool AddVectorToHNSWIndex(HNSWIndex *index, const char* vector_data, size_t id);

bool RemoveVectorFromBFIndex(BFIndex *index, size_t id);

bool RemoveVectorFromHNSWIndex(HNSWIndex *index, size_t id);

size_t GetBFIndexSize(BFIndex *index);

size_t GetHNSWIndexSize(HNSWIndex *index);


#ifdef __cplusplus
}
#endif