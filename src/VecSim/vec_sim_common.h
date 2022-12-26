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

// HNSW default parameters
#define HNSW_DEFAULT_M       16
#define HNSW_DEFAULT_EF_C    200
#define HNSW_DEFAULT_EF_RT   10
#define HNSW_DEFAULT_EPSILON 0.01
#define DEFAULT_BLOCK_SIZE   1024

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

// Vectors flags (for marking a specific vector)
typedef enum { DELETE_MARK = 0x01 } Flags;

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

/**
 * Callback signatures for asynchronous tiered index.
 */
typedef int (*SubmitCB)(void *job_queue, void **jobs, size_t jobs_len);
typedef int (*UpdateMemoryCB)(void *memory_ctx, size_t memory);
typedef void (*JobCallback)(void *);

/**
 * @brief Index initialization parameters.
 *
 */
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

// A struct that contains the common tiered index params.
typedef struct {
    void *jobQueue;             // External queue that holds the jobs.
    SubmitCB submitCb;          // A callback that submits an array of jobs into a given jobQueue.
    void *memoryCtx;            // External context that stores the index memory consumption.
    UpdateMemoryCB UpdateMemCb; // A callback that updates the memoryCtx
                                // with a given memory (number).
} TieredIndexParams;

typedef struct {
    HNSWParams hnswParams;
    TieredIndexParams tieredParams;
} TieredHNSWParams;

typedef struct {
    VecSimAlgo algo; // Algorithm to use.
    union {
        HNSWParams hnswParams;
        BFParams bfParams;
        TieredHNSWParams tieredHNSWParams;
    };
} VecSimParams;

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

/**
 * Definition of generic job structure for asynchronous tiered index.
 */
typedef struct AsyncJob {
    JobType jobType;
    JobCallback Execute; // A callback that receives a job as its input and executes the job.
} AsyncJob;

/**
 * Definition of a job that inserts a new vector from flat into HNSW Index.
 */
typedef struct HNSWInsertJob {
    AsyncJob base;
    void *index;
    labelType label;
    idType id;
} HNSWInsertJob;

/**
 * Definition of a job that swaps last id with a deleted id in HNSW Index after delete operation.
 */
typedef struct HNSWSwapJob {
    AsyncJob base;
    void *index;
    idType deleted_id;
    long pending_repair_jobs_counter; // number of repair jobs left to complete before this job
                                      // is ready to be executed (atomic counter).
} HNSWSwapJob;

/**
 * Definition of a job that repairs a certain node's connection in HNSW Index after delete
 * operation.
 */
typedef struct HNSWRepairJob {
    AsyncJob base;
    void *index;
    idType node_id;
    unsigned short level;
    HNSWSwapJob *assosiated_swap_job;
} HNSWRepairJob;

typedef struct {
    size_t efRuntime; // EF parameter for HNSW graph accuracy/latency for search.
    double epsilon;   // Epsilon parameter for HNSW graph accuracy/latency for range search.
} HNSWRuntimeParams;

/**
 * @brief Query runtime information - the search mode in RediSearch (used for debug/testing).
 *
 */
typedef enum {
    EMPTY_MODE,      // Default value to initialize the "last_mode" field with.
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
 * @brief Index information. Mainly used for debug/testing.
 *
 */
typedef struct {
    union {
        struct {
            size_t indexSize;       // Current count of vectors.
            size_t indexLabelCount; // Current unique count of labels.
            size_t blockSize;       // Sets the amount to grow when resizing
            size_t M;               // Number of allowed edges per node in graph.
            size_t efConstruction;  // EF parameter for HNSW graph accuracy/latency for indexing.
            size_t efRuntime;       // EF parameter for HNSW graph accuracy/latency for search.
            double epsilon;   // Epsilon parameter for HNSW graph accuracy/latency for range search.
            size_t max_level; // Number of graph levels.
            size_t entrypoint;           // Entrypoint vector label.
            VecSimMetric metric;         // Index distance metric
            uint64_t memory;             // Index memory consumption.
            VecSimType type;             // Datatype the index holds.
            bool isMulti;                // Determines if the index should multi-index or not.
            size_t dim;                  // Vector size (dimension).
            VecSearchMode last_mode;     // The mode in which the last query ran.
            size_t visitedNodesPoolSize; // The max number of parallel graph scans so far.
        } hnswInfo;
        struct {
            size_t indexSize;        // Current count of vectors.
            size_t indexLabelCount;  // Current unique count of labels.
            size_t blockSize;        // Brute force algorithm vector block (mini matrix) size
            VecSimMetric metric;     // Index distance metric
            uint64_t memory;         // Index memory consumption.
            VecSimType type;         // Datatype the index holds.
            bool isMulti;            // Determines if the index should multi-index or not.
            size_t dim;              // Vector size (dimension).
            VecSearchMode last_mode; // The mode in which the last query ran.
        } bfInfo;
    };
    VecSimAlgo algo; // Algorithm being used.
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

typedef enum {
    VecSim_QueryResult_OK = VecSim_OK,
    VecSim_QueryResult_TimedOut,
} VecSimQueryResult_Code;

#define VECSIM_TIMEOUT(ctx) (__builtin_expect(VecSimIndexInterface::timeoutCallback(ctx), false))

#ifdef __cplusplus
}
#endif
