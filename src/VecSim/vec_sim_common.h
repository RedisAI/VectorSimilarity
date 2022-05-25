#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>
#include <stdint.h>

// HNSW default parameters
#define HNSW_DEFAULT_M     16
#define HNSW_DEFAULT_EF_C  200
#define HNSW_DEFAULT_EF_RT 10
#define DEFAULT_BLOCK_SIZE 1024

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
    VecSimParamResolverErr_InvalidPolicy_AdHoc_With_BatchSize,
    VecSimParamResolverErr_InvalidPolicy_AdHoc_With_EfRuntime
} VecSimResolveCode;

/**
 * @brief Index initialization parameters.
 *
 */
typedef struct {
    VecSimType type;     // Datatype to index.
    size_t dim;          // Vector's dimension.
    VecSimMetric metric; // Distance metric to use in the index.
    size_t initialCapacity;
    size_t blockSize;
    size_t M;
    size_t efConstruction;
    size_t efRuntime;
} HNSWParams;

typedef struct {
    VecSimType type;     // Datatype to index.
    size_t dim;          // Vector's dimension.
    VecSimMetric metric; // Distance metric to use in the index.
    size_t initialCapacity;
    size_t blockSize;
} BFParams;

typedef struct {
    VecSimAlgo algo; // Algorithm to use.
    union {
        HNSWParams hnswParams;
        BFParams bfParams;
    };
} VecSimParams;

typedef struct {
    size_t efRuntime; // EF parameter for HNSW graph accuracy/latency for search.
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
    HYBRID_BATCHES_TO_ADHOC_BF // Start with batches and dynamically switched to ad-hoc BF.
} VecSearchMode;

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
    void *timeoutCtx;
} VecSimQueryParams;

/**
 * @brief Index information. Mainly used for debug/testing.
 *
 */
typedef struct {
    union {
        struct {
            size_t indexSize;        // Current count of vectors.
            size_t blockSize;        // Sets the amount to grow when resizing
            size_t M;                // Number of allowed edges per node in graph.
            size_t efConstruction;   // EF parameter for HNSW graph accuracy/latency for indexing.
            size_t efRuntime;        // EF parameter for HNSW graph accuracy/latency for search.
            size_t max_level;        // Number of graph levels.
            size_t entrypoint;       // Entrypoint vector label.
            VecSimMetric metric;     // Index distance metric
            uint64_t memory;         // Index memory consumption.
            VecSimType type;         // Datatype the index holds.
            size_t dim;              // Vector size (dimension).
            VecSearchMode last_mode; // The mode in which the last query ran.
        } hnswInfo;
        struct {
            size_t indexSize;        // Current count of vectors.
            size_t blockSize;        // Brute force algorithm vector block (mini matrix) size
            VecSimMetric metric;     // Index distance metric
            uint64_t memory;         // Index memory consumption.
            VecSimType type;         // Datatype the index holds.
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

typedef int (*timeoutCallbackFunction)(void *ctx);

#ifdef __cplusplus
}
#endif
