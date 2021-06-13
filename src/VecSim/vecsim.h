
#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VecSimType_FLOAT32,
    VecSimType_FLOAT64,
    VecSimType_INT32,
    VecSimType_INT64
} VecSimType;

typedef enum {
    VecSimAlgo_BF,
    VecSimAlgo_HNSW
} VecSimAlgo;

typedef enum {
    VecSimMetric_L2,
    VecSimMetric_IP
} VecSimMetric;

typedef struct {
    union {
        struct {
            size_t initialCapacity;
            size_t M;
            size_t efConstruction;
        } hnswParams;
        struct {
            size_t initialCapacity;
        } bfParams;
    };
    VecSimType type;
    size_t size;
    VecSimMetric metric;
    VecSimAlgo algo;
} VecSimParams;


typedef struct {
    size_t id;
    float score;
} VecSimQueryResult;

typedef struct VecSimIndex VecSimIndex;

typedef VecSimIndex* (*Index_New)(VecSimParams *params);
typedef int (*Index_AddVector)(VecSimIndex* index, const void* blob, size_t id);
typedef int (*Index_DeleteVector) (VecSimIndex* index, size_t id);
typedef size_t (*Index_IndexSize)(VecSimIndex* index);
typedef VecSimQueryResult* (*Index_TopKQuery)(VecSimIndex* index, const void* queryBlob, size_t k);
typedef void (*Index_Free)(VecSimIndex *index);
typedef VecSimQueryResult* (*Index_DistanceQuery)(VecSimIndex* index, const void* queryBlob, float distance);
typedef void (*Index_ClearDeleted)(VecSimIndex* index);

typedef struct VecSimIndex {
    Index_AddVector AddFn;
    Index_DeleteVector DeleteFn;
    Index_IndexSize SizeFn;
    Index_TopKQuery TopKQueryFn;
    Index_DistanceQuery DistanceQueryFn;
    Index_ClearDeleted ClearDeletedFn;
    Index_Free FreeFn;
} VecSimIndex;

VecSimIndex* VecSimIndex_New(VecSimParams *params);

void VecSimIndex_Free(VecSimIndex *index);

int VecSimIndex_AddVector(VecSimIndex* index, const void* blob, size_t id);

int VecSimIndex_DeleteVector(VecSimIndex* index, size_t id);

size_t VecSimIndex_IndexSize(VecSimIndex* index);

VecSimQueryResult* VecSimIndex_TopKQuery(VecSimIndex* index, const void* queryBlob, size_t k);

// TODO

VecSimQueryResult* VecSimIndex_DistanceQuery(VecSimIndex* index, const void* queryBlob, float distance);

void VecSimIndex_ClearDeleted(VecSimIndex* index);

#ifdef __cplusplus
}
#endif
