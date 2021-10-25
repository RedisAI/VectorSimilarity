#pragma once

#include <stdlib.h>
#include "query_results.h"

// HNSW default parameters
#define HNSW_DEFAULT_M        16
#define HNSW_DEFAULT_EF_C     200
#define HNSW_DEFAULT_EF_RT    10
#define BF_DEFAULT_BLOCK_SIZE 1024 * 1024

// Datatypes for indexing.
typedef enum {
    VecSimType_FLOAT32,
    VecSimType_FLOAT64,
    VecSimType_INT32,
    VecSimType_INT64
} VecSimType;

// Algorithm type/library.
typedef enum { VecSimAlgo_BF, VecSimAlgo_HNSWLIB } VecSimAlgo;

// Distance metric
typedef enum { VecSimMetric_L2, VecSimMetric_IP, VecSimMetric_Cosine } VecSimMetric;

/**
 * @brief Index initialization parameters.
 *
 */
typedef struct {
    size_t initialCapacity;
    size_t M;
    size_t efConstruction;
    size_t efRuntime;
} HNSWParams;

typedef struct {
    size_t initialCapacity;
    size_t blockSize;
} BFParams;
typedef struct {
    union {
        HNSWParams hnswParams;
        BFParams bfParams;
    };
    VecSimType type;     // Datatype to index.
    size_t size;         // Vector size (dimension).
    VecSimMetric metric; // Distance metric to use in the index.
    VecSimAlgo algo;     // Algorithm to use.
} VecSimParams;

/**
 * @brief Query Runtime parameters.
 *
 */
typedef struct {
    union {
        struct {
            size_t efRuntime; // EF parameter for HNSW graph accuracy/latency for search.
        } hnswRuntimeParams;
    };
} VecSimQueryParams;

/**
 * @brief Index information. Mainly used for debug/testing.
 *
 */
typedef struct {
    union {
        struct {
            size_t indexSize;      // Current count of vectors.
            size_t M;              // Number of allowed edges per node in graph.
            size_t efConstruction; // EF parameter for HNSW graph accuracy/latency for indexing.
            size_t efRuntime;      // EF parameter for HNSW graph accuracy/latency for search.
            size_t levels;         // Number of graph levels.
        } hnswInfo;
        struct {
            size_t indexSize; // Current count of vectors.
            size_t blockSize; // Brute force algorithm vector block (mini matrix) size
        } bfInfo;
    };
    VecSimType type;     // Datatype the index holds.
    size_t d;            // Vector size (dimension).
    VecSimAlgo algo;     // Algorithm being used.
    VecSimMetric metric; // Index distance metric
    // TODO:
    // size_t memory;
} VecSimIndexInfo;

typedef struct VecSimIndex VecSimIndex;

// The methods of the abstract VecSimIndex object
typedef int (*Index_AddVector)(VecSimIndex *index, const void *blob, size_t id);
typedef int (*Index_DeleteVector)(VecSimIndex *index, size_t id);
typedef size_t (*Index_IndexSize)(VecSimIndex *index);
typedef void (*Index_Free)(VecSimIndex *index);
typedef VecSimQueryResult_List *(*Index_TopKQuery)(VecSimIndex *index, const void *queryBlob,
                                                         size_t k, VecSimQueryParams *queryParams);
typedef VecSimQueryResult_List *(*Index_TopKQueryByID)(VecSimIndex *index,
                                                             const void *queryBlob, size_t k,
                                                             VecSimQueryParams *queryParams);
typedef VecSimQueryResult_List *(*Index_DistanceQuery)(VecSimIndex *index,
                                                             const void *queryBlob, float distance,
                                                             VecSimQueryParams *queryParams);
typedef void (*Index_ClearDeleted)(VecSimIndex *index);
typedef VecSimIndexInfo (*Index_Info)(VecSimIndex *index);
typedef VecSimBatchIterator *(*Index_IteratorNew)(VecSimIndex *index, const void *queryBlob);

struct VecSimIndex {

    // Index-specific callbacks
    Index_AddVector AddFn;
    Index_DeleteVector DeleteFn;
    Index_IndexSize SizeFn;
    Index_TopKQuery TopKQueryFn;
    Index_DistanceQuery DistanceQueryFn;
    Index_ClearDeleted ClearDeletedFn;
    Index_Free FreeFn;
    Index_Info InfoFn;
    Index_IteratorNew IteratorNewFn;

    // Index meta-data
    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;
};

/**
 * @brief Create a new VecSim index based on the given params.
 * @params
 */
VecSimIndex *VecSimIndex_New(const VecSimParams *params);

void VecSimIndex_Free(VecSimIndex *index);

int VecSimIndex_AddVector(VecSimIndex *index, const void *blob, size_t id);

int VecSimIndex_DeleteVector(VecSimIndex *index, size_t id);

size_t VecSimIndex_IndexSize(VecSimIndex *index);

VecSimQueryResult_List *VecSimIndex_TopKQuery(VecSimIndex *index, const void *queryBlob,
                                                    size_t k, VecSimQueryParams *queryParams);

VecSimQueryResult_List *VecSimIndex_TopKQueryByID(VecSimIndex *index, const void *queryBlob,
                                                        size_t k, VecSimQueryParams *queryParams);

// TODO?
VecSimQueryResult_List *VecSimIndex_DistanceQuery(VecSimIndex *index, const void *queryBlob,
                                                        float distance,
                                                        VecSimQueryParams *queryParams);

VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index);
