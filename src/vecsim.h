#pragma once
#include "stdlib.h"

typedef enum {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64
} VECTOR_TYPE;

typedef enum {
    BF,
    HNSW
} ALGORITHM;

typedef enum {
    L2,
    IP
} DISTANCE_METRIC;

typedef struct {
    size_t initialCapacity;
    size_t M;
    size_t efConstuction;
} HNSWParams;

typedef struct {
    size_t initialCapacity;
} BFParams;

typedef struct {
    union {
        HNSWParams hnswParams;
        BFParams bfParams;
    };
    ALGORITHM algorithmType;
} AlgorithmParams;


typedef struct {
    size_t id;
    float score;
} QueryResult;

typedef struct VecSimIndex VecSimIndex;

typedef VecSimIndex* (*Index_New)(AlgorithmParams params, DISTANCE_METRIC distanceMetric, VECTOR_TYPE vectorType, size_t vectorLen);
typedef int (*Index_AddVector)(VecSimIndex* index, const void* blob, size_t id);
typedef int (*Index_DeleteVector) (VecSimIndex* index, size_t id);
typedef size_t (*Index_IndexSize)(VecSimIndex* index);
typedef QueryResult* (*Index_TopKQuery)(VecSimIndex* index, const void* queryBlob, size_t k);
typedef void (*Index_Free)(VecSimIndex *index);
typedef QueryResult* (*Index_DistnaceKQuery)(VecSimIndex* index, const void* queryBlob, float distance);
typedef void (*Index_ClearDeleted)(VecSimIndex* index);


typedef struct VecSimIndex {
    Index_AddVector AddFn;
    Index_DeleteVector DeleteFn;
    Index_IndexSize SizeFn;
    Index_TopKQuery TopKQueryFn;
    Index_DistnaceKQuery DistanceQueryFn;
    Index_ClearDeleted ClearDeletedFn;
    Index_Free FreeFn;
} VecSimIndex;

VecSimIndex* VecSimIndex_New(AlgorithmParams params, DISTANCE_METRIC distanceMetric, VECTOR_TYPE vectorType, size_t vectorLen);

int VecSimIndex_AddVector(VecSimIndex* index, const void* blob, size_t id);

int VecSimIndex_DeleteVector(VecSimIndex* index, size_t id);

size_t VecSimIndex_IndexSize(VecSimIndex* index);

QueryResult* VecSimIndex_TopKQuery(VecSimIndex* index, const void* queryBlob, size_t k);

void VecSimIndex_Free(VecSimIndex *index);

// TODO

QueryResult* VecSimIndex_DistnaceKQuery(VecSimIndex* index, const void* queryBlob, float distance);

void VecSimIndex_ClearDeleted(VecSimIndex* index);
