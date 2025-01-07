/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
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
#define UNUSED(x)     (void)(x)

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
    VecSimType_BFLOAT16,
    VecSimType_FLOAT16,
    VecSimType_INT8,
    VecSimType_UINT8,
    VecSimType_INT32,
    VecSimType_INT64
} VecSimType;

// Algorithm type/library.
typedef enum { VecSimAlgo_BF, VecSimAlgo_HNSWLIB, VecSimAlgo_TIERED, VecSimAlgo_SVS } VecSimAlgo;

typedef enum {
    VecSimOption_AUTO = 0,
    VecSimOption_ENABLE = 1,
    VecSimOption_DISABLE = 2,
} VecSimOptionMode;

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

typedef enum {
    VecSimDebugCommandCode_OK = VecSim_OK, // for returning VecSim_OK as an enum value
    VecSimDebugCommandCode_BadIndex,
    VecSimDebugCommandCode_LabelNotExists,
    VecSimDebugCommandCode_MultiNotSupported
} VecSimDebugCommandCode;

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
    VecSimType type;        // Datatype to index.
    size_t dim;             // Vector's dimension.
    VecSimMetric metric;    // Distance metric to use in the index.
    bool multi;             // Determines if the index should multi-index or not.
    size_t initialCapacity; // Deprecated
    size_t blockSize;
    size_t M;
    size_t efConstruction;
    size_t efRuntime;
    double epsilon;
} HNSWParams;

typedef struct {
    VecSimType type;        // Datatype to index.
    size_t dim;             // Vector's dimension.
    VecSimMetric metric;    // Distance metric to use in the index.
    bool multi;             // Determines if the index should multi-index or not.
    size_t initialCapacity; // Deprecated.
    size_t blockSize;
} BFParams;

typedef enum {
    VecSimSvsQuant_NONE = 0,            // No quantization.
    VecSimSvsQuant_8 = 8,               // 8-bit quantization
    VecSimSvsQuant_4 = 4,               // 4-bit quantization
    VecSimSvsQuant_4x4 = 4 | (4 << 10), // 4-bit quantization with 4-bit residuals
    VecSimSvsQuant_4x8 = 4 | (8 << 10)  // 4-bit quantization with 8-bit residuals
} VecSimSvsQuantBits;

typedef struct {
    VecSimType type;     // Datatype to index.
    size_t dim;          // Vector's dimension.
    VecSimMetric metric; // Distance metric to use in the index.
    size_t blockSize;

    /* SVS-Vamana specifics. See Intel ScalableVectorSearch documentation */
    VecSimSvsQuantBits quantBits;    // Quantization level.
    float alpha;                     // The pruning parameter.
    size_t graph_max_degree;         // Maximum degree in the graph.
    size_t construction_window_size; // Search window size to use during graph construction.
    size_t max_candidate_pool_size;  // Limit on the number of neighbors considered during pruning.
    size_t prune_to;                 // Amount that candidates will be pruned.
    VecSimOptionMode use_search_history; // Either the contents of the search buffer can be used or
                                         // the entire search history.
    size_t search_window_size;           // Search window size to use during search.
    double epsilon; // Epsilon parameter for SVS graph accuracy/latency for range search.
} SVSParams;

// A struct that contains HNSW tiered index specific params.
typedef struct {
    size_t swapJobThreshold; // The minimum number of swap jobs to accumulate before applying
                             // all the ready swap jobs in a batch.
} TieredHNSWParams;

// A struct that contains SVS tiered index specific params.
typedef struct {
    size_t updateJobThreshold; // The flat index size threshold to trigger the update job.
} TieredSVSParams;

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
        TieredSVSParams tieredSVSParams;
    } specificParams;
} TieredIndexParams;

typedef union {
    HNSWParams hnswParams;
    BFParams bfParams;
    TieredIndexParams tieredParams;
    SVSParams svsParams;
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

typedef struct {
    size_t windowSize;              // Search window size for Vamana graph accuracy/latency tune.
    VecSimOptionMode searchHistory; // Enabling of the visited set for search.
    double epsilon; // Epsilon parameter for SVS graph accuracy/latency for range search.
} SVSRuntimeParams;

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
        SVSRuntimeParams svsRuntimeParams;
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
    VecSimAlgo algo;     // Algorithm being used (if index is tiered, this is the backend index).
    VecSimMetric metric; // Index distance metric
    VecSimType type;     // Datatype the index holds.
    bool isMulti;        // Determines if the index should multi-index or not.
    bool isTiered;       // Is the index is tiered or not.
    size_t blockSize;    // Brute force algorithm vector block (mini matrix) size
    size_t dim;          // Vector size (dimension).
} VecSimIndexBasicInfo;

/**
 * Index info for statistics - a thin and efficient (no locks, no calculations) info. Can be used in
 * production without worrying about performance
 */
typedef struct {
    size_t memory;
    size_t numberOfMarkedDeleted; // The number of vectors that are marked as deleted (HNSW/tiered
                                  // only).
} VecSimIndexStatsInfo;

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
 * @brief Index information. Should only be used for debug/testing.
 *
 */
typedef struct {
    CommonInfo commonInfo;
    union {
        bfInfoStruct bfInfo;
        hnswInfoStruct hnswInfo;
        tieredInfoStruct tieredInfo;
    };
} VecSimIndexDebugInfo;

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
