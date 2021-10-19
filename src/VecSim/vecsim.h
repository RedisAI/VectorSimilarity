#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

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

// Users should access this struct directly, but with VecSimQueryResults_<X> API
typedef struct {
    size_t id;
    float score;
} VecSimQueryResults_Item;

// An opaque object from which results can be obtained via iterator
typedef struct VecSimQueryResults VecSimQueryResults;

typedef struct VecSimQueryResults_Iterator VecSimQueryResult_Iterator;

typedef struct VecSimBatchIterator VecSimBatchIterator;

typedef enum { BY_DISTANCE, BY_ID } VecSimQueryResults_Order;

typedef struct VecSimIndex VecSimIndex;

typedef int (*Index_AddVector)(VecSimIndex *index, const void *blob, size_t id);
typedef int (*Index_DeleteVector)(VecSimIndex *index, size_t id);
typedef size_t (*Index_IndexSize)(VecSimIndex *index);
typedef void (*Index_Free)(VecSimIndex *index);
typedef VecSimQueryResults *(*Index_TopKQuery)(VecSimIndex *index, const void *queryBlob, size_t k,
                                              VecSimQueryParams *queryParams);
typedef VecSimQueryResults *(*Index_TopKQueryByID)(VecSimIndex *index, const void *queryBlob,
                                                  size_t k, VecSimQueryParams *queryParams);
typedef VecSimQueryResults *(*Index_DistanceQuery)(VecSimIndex *index, const void *queryBlob,
                                                  float distance, VecSimQueryParams *queryParams);
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


typedef VecSimQueryResult_Iterator *(*BatchIterator_Next)(VecSimBatchIterator *iterator, size_t n_results);

typedef void (*BatchIterator_Free)(VecSimBatchIterator *iterator);

typedef VecSimQueryResult_Iterator *(*BatchIterator_Reset)(VecSimBatchIterator *iterator);

struct VecSimBatchIterator {
    VecSimIndex *index;
    unsigned char iterator_id;
    size_t returned_results_count;

    BatchIterator_Next IteratorNext;
    BatchIterator_Free IteratorFree;
    BatchIterator_Reset IteratorReset;
};

VecSimIndex *VecSimIndex_New(const VecSimParams *params);

void VecSimIndex_Free(VecSimIndex *index);

int VecSimIndex_AddVector(VecSimIndex *index, const void *blob, size_t id);

int VecSimIndex_DeleteVector(VecSimIndex *index, size_t id);

size_t VecSimIndex_IndexSize(VecSimIndex *index);

VecSimQueryResults *VecSimIndex_TopKQuery(VecSimIndex *index, const void *queryBlob, size_t k,
                                         VecSimQueryParams *queryParams);

VecSimQueryResults *VecSimIndex_TopKQueryByID(VecSimIndex *index, const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams);

// TODO?
VecSimQueryResults *VecSimIndex_DistanceQuery(VecSimIndex *index, const void *queryBlob,
                                             float distance, VecSimQueryParams *queryParams);

VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index);

// Query results iterator API
size_t VecSimQueryResults_Len(VecSimQueryResults *results_iterator);

VecSimQueryResults_Iterator *VecSimQueryResults_GetIterator(VecSimQueryResults *results);

// Advance the iterator, so it will point to the next item. If this is the last item,
// the returned iterator will point to NULL.
VecSimQueryResults_Iterator *VecSimQueryResults_IteratorNext(VecSimQueryResults_Iterator *iterator);

int VecSimQueryResults_GetId(VecSimQueryResults_Iterator *iterator);

float VecSimQueryResults_GetScore(VecSimQueryResults_Iterator *iterator);

void VecSimQueryResults_IteratorFree(VecSimQueryResults_Iterator *iterator);

void VecSimQueryResults_Free(VecSimQueryResults *results);

// Batch iterator API
VecSimBatchIterator *VecSimIterator_New(VecSimIndex *index, const void *queryBlob);

VecSimQueryResults *VecSimIterator_Next(VecSimBatchIterator *iterator, size_t n_results, VecSimQueryResults_Order order);

void VecSimIterator_Free(VecSimBatchIterator *iterator);

void VecSimIterator_Reset(VecSimBatchIterator *iterator);

#ifdef __cplusplus
}
#endif
