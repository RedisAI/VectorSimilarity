#pragma once

#ifdef __cplusplus
extern "C" {
#endif

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
typedef VecSimQueryResult_List *(*Index_TopKQueryByID)(VecSimIndex *index, const void *queryBlob,
                                                       size_t k, VecSimQueryParams *queryParams);
typedef VecSimQueryResult_List *(*Index_DistanceQuery)(VecSimIndex *index, const void *queryBlob,
                                                       float distance,
                                                       VecSimQueryParams *queryParams);
typedef void (*Index_ClearDeleted)(VecSimIndex *index);
typedef VecSimIndexInfo (*Index_Info)(VecSimIndex *index);
typedef VecSimBatchIterator *(*Index_BatchIteratorNew)(VecSimIndex *index, const void *queryBlob);

/**
 * @brief abstract struct that represents an index. The supported index types use
 * this structure as the base, and add their specific functionality on top of it.
 */
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
    Index_BatchIteratorNew IteratorNewFn;

    // Index meta-data
    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;
};

/**
 * @brief Create a new VecSim index based on the given params.
 * @param params index configurations (initial size, data type, dimension, metric, algorithm and the
 * algorithm-related params).
 * @return A pointer to the created index.
 */
// todo: why aren't we using CreateFn for this, as we do in the rest of the api functions?
VecSimIndex *VecSimIndex_New(const VecSimParams *params);

/**
 * @brief Release an index and its internal data using the FreeFn method.
 * @param index the index to release.
 */
void VecSimIndex_Free(VecSimIndex *index);

/**
 * @brief Add a vector to an index using its the AddFn.
 * @param index the index to which the vector is added.
 * @param blob binary representation of the vector. Blob size should match the index data type and
 * dimension (todo: validate it).
 * @param id the id of the added vector
 * @return ?
 */
int VecSimIndex_AddVector(VecSimIndex *index, const void *blob, size_t id);

/**
 * @brief Remove a vector from an index using its DeleteFn.
 * @param index the index from which the vector is removed.
 * @param id the id of the removed vector
 * @return ?
 */
int VecSimIndex_DeleteVector(VecSimIndex *index, size_t id);

/**
 * @brief Return the number of vectors in the index using irs SizeFn.
 * @param index the index whose size is requested.
 * @return index size.
 */
size_t VecSimIndex_IndexSize(VecSimIndex *index);

/**
 * @brief Search for the k (approximate) closest vectors to a given vector in the index using the
 * index TopKQuery callback. The results are ordered by their score.
 * @param index the index to query in.
 * @param queryBlob binary representation of the query vector. Blob size should match the index data
 * type and dimension. (todo: validate it)
 * @param k the number of "nearest neighbours" to return (upper bound).
 * @param queryParams run time params for the search, which are algorithm-specific.
 * @return An opaque object the represents a list of results. User can access the id and score
 * (which is the distance according to the index metric) of every result through
 * VecSimQueryResult_Iterator.
 */
VecSimQueryResult_List *VecSimIndex_TopKQuery(VecSimIndex *index, const void *queryBlob, size_t k,
                                              VecSimQueryParams *queryParams);

/**
 * @brief Search for the k (approximate) closest vectors to a given vector in the index using the
 * index TopKQuery callback. The results are ordered by their id.
 * @param index the index to query in.
 * @param queryBlob binary representation of the query vector. Blob size should match the index data
 * type and dimension. (todo: validate it)
 * @param k the number of "nearest neighbours" to return (upper bound).
 * @param queryParams run time params for the search, which are algorithm-specific.
 * @return An opaque object the represents a list of results. User can access the id and score
 * (which is the distance according to the index metric) of every result through
 * VecSimQueryResult_Iterator.
 */
// Todo: consider unify this with TopKQuery and add VecSimQueryResult_Order as input.
VecSimQueryResult_List *VecSimIndex_TopKQueryByID(VecSimIndex *index, const void *queryBlob,
                                                  size_t k, VecSimQueryParams *queryParams);

// TODO?
VecSimQueryResult_List *VecSimIndex_DistanceQuery(VecSimIndex *index, const void *queryBlob,
                                                  float distance, VecSimQueryParams *queryParams);

/**
 * @brief Return index information.
 * @param index the index to return its info.
 * @return Index general and specific meta-data.
 */
VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index);

#ifdef __cplusplus
}
#endif