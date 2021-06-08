
#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VecSimVecType_FLOAT32,
    VecSimVecType_FLOAT64,
    VecSimVecType_INT32,
    VecSimVecType_INT64
} VecSimVecType;

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
            size_t efConstuction;
        } hnswParams;
        struct {
            size_t initialCapacity;
        } bfParams;
    };
    VecSimAlgo algorithmType;
} VecSimAlgoParams;


typedef struct {
    size_t id;
    float score;
} VecSimQueryResult;

typedef struct VecSimIndex VecSimIndex;

typedef VecSimIndex* (*Index_New)(VecSimAlgoParams *params, VecSimMetric distanceMetric, VecSimVecType vectorType, size_t vectorLen);
typedef int (*Index_AddVector)(VecSimIndex* index, const void* blob, size_t id);
typedef int (*Index_DeleteVector) (VecSimIndex* index, size_t id);
typedef size_t (*Index_IndexSize)(VecSimIndex* index);
typedef VecSimQueryResult* (*Index_TopKQuery)(VecSimIndex* index, const void* queryBlob, size_t k);
typedef void (*Index_Free)(VecSimIndex *index);
typedef VecSimQueryResult* (*Index_DistnaceQuery)(VecSimIndex* index, const void* queryBlob, float distance);
typedef void (*Index_ClearDeleted)(VecSimIndex* index);

typedef struct VecSimIndex {
    Index_AddVector AddFn;
    Index_DeleteVector DeleteFn;
    Index_IndexSize SizeFn;
    Index_TopKQuery TopKQueryFn;
    Index_DistnaceQuery DistanceQueryFn;
    Index_ClearDeleted ClearDeletedFn;
    Index_Free FreeFn;
} VecSimIndex;

VecSimIndex* VecSimIndex_New(VecSimAlgoParams *params, VecSimMetric metric, VecSimVecType vectype, size_t veclen);

void VecSimIndex_Free(VecSimIndex *index);

int VecSimIndex_AddVector(VecSimIndex* index, const void* blob, size_t id);

int VecSimIndex_DeleteVector(VecSimIndex* index, size_t id);

size_t VecSimIndex_IndexSize(VecSimIndex* index);

VecSimQueryResult* VecSimIndex_TopKQuery(VecSimIndex* index, const void* queryBlob, size_t k);

// TODO

VecSimQueryResult* VecSimIndex_DistnaceQuery(VecSimIndex* index, const void* queryBlob, float distance);

void VecSimIndex_ClearDeleted(VecSimIndex* index);

#ifdef __cplusplus
}
#endif
