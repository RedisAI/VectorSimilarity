#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>
#include <stdint.h>

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
    int64_t memory;      // Index memory consumption.
} VecSimIndexInfo;

// Memory function declerations.
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

#ifdef __cplusplus
}
#endif
