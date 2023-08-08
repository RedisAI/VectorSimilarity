/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <stdbool.h>

// Common definitions
#define INVALID_ID    UINT_MAX
#define INVALID_LABEL SIZE_MAX

// HNSW default parameters
#define HNSW_DEFAULT_M       16
#define HNSW_DEFAULT_EF_C    200
#define HNSW_DEFAULT_EF_RT   10
#define HNSW_DEFAULT_EPSILON 0.01
#define DEFAULT_BLOCK_SIZE   1024

#define HNSW_INVALID_LEVEL SIZE_MAX
#define INVALID_JOB_ID     UINT_MAX
#define INVALID_INFO       UINT_MAX

// Datatypes for indexing.
typedef enum {
    VecSimType_FLOAT32,
    VecSimType_FLOAT64,
    VecSimType_INT32,
    VecSimType_INT64
} VecSimType;

// Algorithm type/library.
typedef enum { VecSimAlgo_BF, VecSimAlgo_HNSWLIB, VecSimAlgo_TIERED } VecSimAlgo;

// Distance metric
typedef enum { VecSimMetric_L2, VecSimMetric_IP, VecSimMetric_Cosine } VecSimMetric;

typedef size_t labelType;
typedef unsigned int idType;

/**
 * @brief Query Runtime raw parameters.
 * Use VecSimIndex_ResolveParams to generate VecSimQueryParams from array of VecSimRawParams.
 *
 */
typedef struct {
    const char *name;
    size_t nameLen;
    const char *value;
    size_t valLen;
} VecSimRawParam;

#define VecSim_OK 0

typedef enum {
    VecSimParamResolver_OK = VecSim_OK, // for returning VecSim_OK as an enum value
    VecSimParamResolverErr_NullParam,
    VecSimParamResolverErr_AlreadySet,
    VecSimParamResolverErr_UnknownParam,
    VecSimParamResolverErr_BadValue,
    VecSimParamResolverErr_InvalidPolicy_NExits,
    VecSimParamResolverErr_InvalidPolicy_NHybrid,
    VecSimParamResolverErr_InvalidPolicy_NRange,
    VecSimParamResolverErr_InvalidPolicy_AdHoc_With_BatchSize,
    VecSimParamResolverErr_InvalidPolicy_AdHoc_With_EfRuntime
} VecSimResolveCode;

typedef struct AsyncJob AsyncJob; // forward declaration.

// Write async/sync mode
typedef enum { VecSim_WriteAsync, VecSim_WriteInPlace } VecSimWriteMode;

/**
 * Callback signatures for asynchronous tiered index.
 */
typedef void (*JobCallback)(AsyncJob *);
typedef int (*SubmitCB)(void *job_queue, void *index_ctx, AsyncJob **jobs, JobCallback *CBs,
                        size_t jobs_len);

/**
 * @brief Index initialization parameters.
 *
 */
typedef struct VecSimParams VecSimParams;
typedef struct {
    VecSimType type;     // Datatype to index.
    size_t dim;          // Vector's dimension.
    VecSimMetric metric; // Distance metric to use in the index.
    bool multi;          // Determines if the index should multi-index or not.
    size_t initialCapacity;
    size_t blockSize;
    size_t M;
    size_t efConstruction;
    size_t efRuntime;
    double epsilon;
} HNSWParams;

typedef struct {
    VecSimType type;     // Datatype to index.
    size_t dim;          // Vector's dimension.
    VecSimMetric metric; // Distance metric to use in the index.
    bool multi;          // Determines if the index should multi-index or not.
    size_t initialCapacity;
    size_t blockSize;
} BFParams;

// A struct that contains HNSW tiered index specific params.
typedef struct {
    size_t swapJobThreshold; // The minimum number of swap jobs to accumulate before applying
                             // all the ready swap jobs in a batch.
} TieredHNSWParams;

// A struct that contains the common tiered index params.
typedef struct {
    void *jobQueue;         // External queue that holds the jobs.
    void *jobQueueCtx;      // External context to be sent to the submit callback.
    SubmitCB submitCb;      // A callback that submits an array of jobs into a given jobQueue.
    size_t flatBufferLimit; // Maximum size allowed for the flat buffer. If flat buffer is full, use
                            // in-place insertion.
    VecSimParams *primaryIndexParams; // Parameters to initialize the index.
    union {
        TieredHNSWParams tieredHnswParams;
    } specificParams;
} TieredIndexParams;

typedef union {
    HNSWParams hnswParams;
    BFParams bfParams;
    TieredIndexParams tieredParams;
} AlgoParams;

struct VecSimParams {
    VecSimAlgo algo; // Algorithm to use.
    AlgoParams algoParams;
    void *logCtx; // External context that stores the index log.
};

/**
 * The specific job types in use (to be extended in the future by demand)
 */
typedef enum {
    HNSW_INSERT_VECTOR_JOB,
    HNSW_REPAIR_NODE_CONNECTIONS_JOB,
    HNSW_SEARCH_JOB,
    HNSW_SWAP_JOB,
    INVALID_JOB // to indicate that finding a JobType >= INVALID_JOB is an error
} JobType;

typedef struct {
    size_t efRuntime; // EF parameter for HNSW graph accuracy/latency for search.
    double epsilon;   // Epsilon parameter for HNSW graph accuracy/latency for range search.
} HNSWRuntimeParams;

/**
 * @brief Query runtime information - the search mode in RediSearch (used for debug/testing).
 *
 */
typedef enum {
    EMPTY_MODE,      // Default value to initialize the "lastMode" field with.
    STANDARD_KNN,    // Run k-nn query over the entire vector index.
    HYBRID_ADHOC_BF, // Measure ad-hoc the distance for every result that passes the filters,
                     // and take the top k results.
    HYBRID_BATCHES,  // Get the top vector results in batches upon demand, and keep the results that
                     // passes the filters until we reach k results.
    HYBRID_BATCHES_TO_ADHOC_BF, // Start with batches and dynamically switched to ad-hoc BF.
    RANGE_QUERY, // Run range query, to return all vectors that are within a given range from the
                 // query vector.
} VecSearchMode;

typedef enum {
    QUERY_TYPE_NONE, // Use when no params are given.
    QUERY_TYPE_KNN,
    QUERY_TYPE_HYBRID,
    QUERY_TYPE_RANGE,
} VecsimQueryType;

/**
 * @brief Query Runtime parameters.
 *
 */
typedef struct {
    union {
        HNSWRuntimeParams hnswRuntimeParams;
    };
    size_t batchSize;
    VecSearchMode searchMode;
    void *timeoutCtx; // This parameter is not exposed directly to the user, and we shouldn't expect
                      // to get it from the parameters resolve function.
} VecSimQueryParams;

/**
 * Index info that is static and immutable (cannot be changed over time)
 */
typedef struct {
    VecSimAlgo algo;     // Algorithm being used.
    size_t blockSize;    // Brute force algorithm vector block (mini matrix) size
    VecSimMetric metric; // Index distance metric
    VecSimType type;     // Datatype the index holds.
    bool isMulti;        // Determines if the index should multi-index or not.
    size_t dim;          // Vector size (dimension).

    bool isTiered; // The algorithm for the tiered index (if algo is tiered).
} VecSimIndexBasicInfo;

typedef struct {
    VecSimIndexBasicInfo basicInfo; // Index immutable meta-data.
    size_t indexSize;               // Current count of vectors.
    size_t indexLabelCount;         // Current unique count of labels.
    uint64_t memory;                // Index memory consumption.
    VecSearchMode lastMode;         // The mode in which the last query ran.
} CommonInfo;

typedef struct {
    size_t M;              // Number of allowed edges per node in graph.
    size_t efConstruction; // EF parameter for HNSW graph accuracy/latency for indexing.
    size_t efRuntime;      // EF parameter for HNSW graph accuracy/latency for search.
    double epsilon;        // Epsilon parameter for HNSW graph accuracy/latency for range search.
    size_t max_level;      // Number of graph levels.
    size_t entrypoint;     // Entrypoint vector label.
    size_t visitedNodesPoolSize;       // The max number of parallel graph scans so far.
    size_t numberOfMarkedDeletedNodes; // The number of nodes that are marked as deleted.
} hnswInfoStruct;

typedef struct {
    char dummy; // For not having this as an empty struct, can be removed after we extend this.
} bfInfoStruct;

typedef struct HnswTieredInfo {
    size_t pendingSwapJobsThreshold;
} HnswTieredInfo;

typedef struct {

    // Since we cannot recursively have a struct that contains itself, we need this workaround.
    union {
        hnswInfoStruct hnswInfo;
    } backendInfo; // The backend index info.
    union {
        HnswTieredInfo hnswTieredInfo;
    } specificTieredBackendInfo;   // Info relevant for tiered index with a specific backend.
    CommonInfo backendCommonInfo;  // Common index info.
    CommonInfo frontendCommonInfo; // Common index info.
    bfInfoStruct bfInfo;           // The brute force index info.

    uint64_t management_layer_memory; // Memory consumption of the management layer.
    bool backgroundIndexing;          // Determines if the index is currently being indexed in the
                                      // background.
    size_t bufferLimit;               // Maximum number of vectors allowed in the flat buffer.
} tieredInfoStruct;

/**
 * @brief Index information. Mainly used for debug/testing.
 *
 */
typedef struct {
    CommonInfo commonInfo;
    union {
        bfInfoStruct bfInfo;
        hnswInfoStruct hnswInfo;
        tieredInfoStruct tieredInfo;
    };
} VecSimIndexInfo;

// Memory function declarations.
typedef void *(*allocFn)(size_t n);
typedef void *(*callocFn)(size_t nelem, size_t elemsz);
typedef void *(*reallocFn)(void *p, size_t n);
typedef void (*freeFn)(void *p);
typedef char *(*strdupFn)(const char *s);

/**
 * @brief A struct to pass 3rd party memory functions to Vecsimlib.
 *
 */
typedef struct {
    allocFn allocFunction;     // Malloc like function.
    callocFn callocFunction;   // Calloc like function.
    reallocFn reallocFunction; // Realloc like function.
    freeFn freeFunction;       // Free function.
} VecSimMemoryFunctions;

/**
 * @brief A struct to pass 3rd party timeout function to Vecsimlib.
 * @param ctx some generic context to pass to the function
 * @return the function should return a non-zero value on timeout
 */
typedef int (*timeoutCallbackFunction)(void *ctx);

/**
 * @brief A struct to pass 3rd party logging function to Vecsimlib.
 * @param ctx some generic context to pass to the function
 * @param level loglevel (in redis we should choose from: "warning", "notice", "verbose", "debug")
 * @param message the message to log
 */
typedef void (*logCallbackFunction)(void *ctx, const char *level, const char *message);

// Round up to the nearest multiplication of blockSize.
static inline size_t RoundUpInitialCapacity(size_t initialCapacity, size_t blockSize) {
    return initialCapacity % blockSize ? initialCapacity + blockSize - initialCapacity % blockSize
                                       : initialCapacity;
}

#define VECSIM_TIMEOUT(ctx) (__builtin_expect(VecSimIndexInterface::timeoutCallback(ctx), false))

#ifdef __cplusplus
}
#endif
