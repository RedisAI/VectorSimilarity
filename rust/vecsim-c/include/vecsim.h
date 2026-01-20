/**
 * @file vecsim.h
 * @brief C API for the VecSim vector similarity search library.
 *
 * This header provides a C-compatible interface to the Rust VecSim library,
 * enabling high-performance vector similarity search from C/C++ applications.
 *
 * @example
 * ```c
 * #include "vecsim.h"
 *
 * int main() {
 *     // Create a BruteForce index
 *     BFParams params = {0};
 *     params.base.algo = VecSimAlgo_BF;
 *     params.base.type_ = VecSimType_FLOAT32;
 *     params.base.metric = VecSimMetric_L2;
 *     params.base.dim = 4;
 *     params.base.multi = false;
 *     params.base.initialCapacity = 100;
 *
 *     VecSimIndex *index = VecSimIndex_NewBF(&params);
 *
 *     // Add vectors
 *     float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
 *     VecSimIndex_AddVector(index, v1, 1);
 *
 *     // Query
 *     float query[] = {1.0f, 0.1f, 0.0f, 0.0f};
 *     VecSimQueryReply *reply = VecSimIndex_TopKQuery(index, query, 10, NULL, BY_SCORE);
 *
 *     // Iterate results
 *     VecSimQueryReply_Iterator *iter = VecSimQueryReply_GetIterator(reply);
 *     while (VecSimQueryReply_IteratorHasNext(iter)) {
 *         VecSimQueryResult *result = VecSimQueryReply_IteratorNext(iter);
 *         printf("Label: %llu, Score: %f\n",
 *                VecSimQueryResult_GetId(result),
 *                VecSimQueryResult_GetScore(result));
 *     }
 *
 *     // Cleanup
 *     VecSimQueryReply_IteratorFree(iter);
 *     VecSimQueryReply_Free(reply);
 *     VecSimIndex_Free(index);
 *     return 0;
 * }
 * ```
 */

#ifndef VECSIM_H
#define VECSIM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants and Macros
 * ========================================================================== */

/**
 * @brief Default block size for vector storage.
 */
#define DEFAULT_BLOCK_SIZE 1024

/**
 * @brief Macro to suppress unused variable warnings.
 */
#define UNUSED(x) (void)(x)

/**
 * @brief String constant for ad-hoc brute force policy.
 */
#define VECSIM_POLICY_ADHOC_BF "adhoc_bf"

/**
 * @brief HNSW default parameters.
 */
#define HNSW_DEFAULT_M       16
#define HNSW_DEFAULT_EF_C    200
#define HNSW_DEFAULT_EF_RT   10
#define HNSW_DEFAULT_EPSILON 0.01

/**
 * @brief SVS Vamana default parameters.
 */
#define SVS_VAMANA_DEFAULT_ALPHA_L2                 1.2f
#define SVS_VAMANA_DEFAULT_ALPHA_IP                 0.95f
#define SVS_VAMANA_DEFAULT_GRAPH_MAX_DEGREE         32
#define SVS_VAMANA_DEFAULT_CONSTRUCTION_WINDOW_SIZE 200
#define SVS_VAMANA_DEFAULT_USE_SEARCH_HISTORY       true
#define SVS_VAMANA_DEFAULT_NUM_THREADS              1
#define SVS_VAMANA_DEFAULT_TRAINING_THRESHOLD       (10 * DEFAULT_BLOCK_SIZE)
#define SVS_VAMANA_DEFAULT_UPDATE_THRESHOLD         (1 * DEFAULT_BLOCK_SIZE)
#define SVS_VAMANA_DEFAULT_SEARCH_WINDOW_SIZE       10
#define SVS_VAMANA_DEFAULT_LEANVEC_DIM              0
#define SVS_VAMANA_DEFAULT_EPSILON                  0.01f

/**
 * @brief General success code.
 */
#define VecSim_OK 0

/* ============================================================================
 * Type Definitions
 * ========================================================================== */

/**
 * @brief Label type for vectors (64-bit unsigned integer).
 */
typedef uint64_t labelType;

/**
 * @brief Vector element data type.
 */
typedef enum VecSimType {
    VecSimType_FLOAT32 = 0,   /**< 32-bit floating point (float) */
    VecSimType_FLOAT64 = 1,   /**< 64-bit floating point (double) */
    VecSimType_BFLOAT16 = 2,  /**< Brain floating point (16-bit) */
    VecSimType_FLOAT16 = 3,   /**< IEEE 754 half-precision (16-bit) */
    VecSimType_INT8 = 4,      /**< 8-bit signed integer */
    VecSimType_UINT8 = 5,     /**< 8-bit unsigned integer */
    VecSimType_INT32 = 6,     /**< 32-bit signed integer */
    VecSimType_INT64 = 7      /**< 64-bit signed integer */
} VecSimType;

/**
 * @brief Index algorithm type.
 */
typedef enum VecSimAlgo {
    VecSimAlgo_BF = 0,      /**< Brute Force (exact, linear scan) */
    VecSimAlgo_HNSWLIB = 1, /**< HNSW (approximate, logarithmic) */
    VecSimAlgo_TIERED = 2,  /**< Tiered (BruteForce frontend + HNSW backend) */
    VecSimAlgo_SVS = 3      /**< SVS/Vamana (approximate, single-layer graph) */
} VecSimAlgo;

/**
 * @brief Distance metric type.
 */
typedef enum VecSimMetric {
    VecSimMetric_L2 = 0,     /**< L2 (Euclidean) squared distance */
    VecSimMetric_IP = 1,     /**< Inner Product (dot product) */
    VecSimMetric_Cosine = 2  /**< Cosine distance (1 - cosine similarity) */
} VecSimMetric;

/**
 * @brief Query result ordering.
 */
typedef enum VecSimQueryReply_Order {
    BY_SCORE = 0,  /**< Order by distance/score (ascending) */
    BY_ID = 1      /**< Order by label ID (ascending) */
} VecSimQueryReply_Order;

/**
 * @brief Query reply status code.
 */
typedef enum VecSimQueryReply_Code {
    VecSim_QueryReply_OK = 0,      /**< Query completed successfully */
    VecSim_QueryReply_TimedOut = 1 /**< Query was aborted due to timeout */
} VecSimQueryReply_Code;

/**
 * @brief Search mode for queries (internal Rust representation).
 * Note: RediSearch defines its own VecSimSearchMode with VECSIM_ prefix.
 * This enum is used internally by the Rust library.
 */
typedef enum VecSimSearchMode_Internal {
    VECSIM_SEARCH_STANDARD = 0,  /**< Standard search mode */
    VECSIM_SEARCH_HYBRID = 1,    /**< Hybrid search mode */
    VECSIM_SEARCH_RANGE = 2      /**< Range search mode */
} VecSimSearchMode_Internal;

/**
 * @brief Hybrid search policy.
 */
typedef enum VecSimHybridPolicy {
    BATCHES = 0,  /**< Batch-based hybrid search */
    ADHOC = 1     /**< Ad-hoc hybrid search */
} VecSimHybridPolicy;

/**
 * @brief Index resolution codes.
 */
typedef enum VecSimResolveCode {
    VecSim_Resolve_OK = 0,   /**< Operation successful */
    VecSim_Resolve_ERR = 1   /**< Operation failed */
} VecSimResolveCode;

/**
 * @brief Write mode for tiered index operations.
 *
 * Controls whether vector additions/deletions go through the async
 * buffering path or directly to the backend index.
 */
typedef enum VecSimWriteMode {
    VecSim_WriteAsync = 0,    /**< Async: vectors go to flat buffer, migrated via background jobs */
    VecSim_WriteInPlace = 1   /**< InPlace: vectors go directly to the backend index */
} VecSimWriteMode;

/**
 * @brief Parameter resolution error codes.
 *
 * Returned by VecSimIndex_ResolveParams to indicate the result of parsing
 * runtime query parameters.
 */
typedef enum VecSimParamResolveCode {
    VecSimParamResolver_OK = 0,                         /**< Resolution succeeded */
    VecSimParamResolverErr_NullParam = 1,               /**< Null parameter pointer */
    VecSimParamResolverErr_AlreadySet = 2,              /**< Parameter already set */
    VecSimParamResolverErr_UnknownParam = 3,            /**< Unknown parameter name */
    VecSimParamResolverErr_BadValue = 4,                /**< Invalid parameter value */
    VecSimParamResolverErr_InvalidPolicy_NExits = 5,    /**< Policy does not exist */
    VecSimParamResolverErr_InvalidPolicy_NHybrid = 6,   /**< Not a hybrid query */
    VecSimParamResolverErr_InvalidPolicy_NRange = 7,    /**< Not a range query */
    VecSimParamResolverErr_InvalidPolicy_AdHoc_With_BatchSize = 8, /**< AdHoc with batch size */
    VecSimParamResolverErr_InvalidPolicy_AdHoc_With_EfRuntime = 9  /**< AdHoc with ef_runtime */
} VecSimParamResolveCode;

/**
 * @brief Query type for parameter resolution.
 */
typedef enum VecsimQueryType {
    QUERY_TYPE_NONE = 0,   /**< No specific query type */
    QUERY_TYPE_KNN = 1,    /**< Standard KNN query */
    QUERY_TYPE_HYBRID = 2, /**< Hybrid query (vector + filters) */
    QUERY_TYPE_RANGE = 3   /**< Range query */
} VecsimQueryType;

/**
 * @brief Option mode for various settings.
 */
typedef enum VecSimOptionMode {
    VecSimOption_AUTO = 0,     /**< Automatic mode */
    VecSimOption_ENABLE = 1,   /**< Enable the option */
    VecSimOption_DISABLE = 2   /**< Disable the option */
} VecSimOptionMode;

/**
 * @brief Tri-state boolean for optional settings.
 */
typedef enum VecSimBool {
    VecSimBool_TRUE = 1,   /**< True */
    VecSimBool_FALSE = 0,  /**< False */
    VecSimBool_UNSET = -1  /**< Not set */
} VecSimBool;

/**
 * @brief Search mode for queries (used for debug/testing).
 */
typedef enum VecSearchMode {
    EMPTY_MODE = 0,               /**< Empty/unset mode */
    STANDARD_KNN = 1,             /**< Standard KNN search */
    HYBRID_ADHOC_BF = 2,          /**< Hybrid ad-hoc brute force */
    HYBRID_BATCHES = 3,           /**< Hybrid batches search */
    HYBRID_BATCHES_TO_ADHOC_BF = 4, /**< Hybrid batches to ad-hoc BF */
    RANGE_QUERY = 5               /**< Range query */
} VecSearchMode;

/**
 * @brief Debug command result codes.
 */
typedef enum VecSimDebugCommandCode {
    VecSimDebugCommandCode_OK = 0,              /**< Command succeeded */
    VecSimDebugCommandCode_BadIndex = 1,        /**< Invalid index */
    VecSimDebugCommandCode_LabelNotExists = 2,  /**< Label does not exist */
    VecSimDebugCommandCode_MultiNotSupported = 3 /**< Multi-value not supported */
} VecSimDebugCommandCode;

/**
 * @brief Raw parameter for runtime query configuration.
 *
 * Used to pass string-based parameters that are resolved into typed
 * VecSimQueryParams by VecSimIndex_ResolveParams.
 */
typedef struct VecSimRawParam {
    const char *name;   /**< Parameter name */
    size_t nameLen;     /**< Length of parameter name */
    const char *value;  /**< Parameter value as string */
    size_t valLen;      /**< Length of parameter value */
} VecSimRawParam;

/**
 * @brief Timeout callback function type.
 *
 * Returns non-zero on timeout.
 */
typedef int (*timeoutCallbackFunction)(void *ctx);

/**
 * @brief Log callback function type.
 */
typedef void (*logCallbackFunction)(void *ctx, const char *level, const char *message);

// ============================================================================
// C++-Compatible Structures (for drop-in API compatibility)
// ============================================================================

/**
 * @brief HNSW parameters (C++-compatible layout).
 */
typedef struct {
    VecSimType type;
    size_t dim;
    VecSimMetric metric;
    bool multi;
    size_t initialCapacity;
    size_t blockSize;
    size_t M;
    size_t efConstruction;
    size_t efRuntime;
    double epsilon;
} HNSWParams_C;

/**
 * @brief BruteForce parameters (C++-compatible layout).
 */
typedef struct {
    VecSimType type;
    size_t dim;
    VecSimMetric metric;
    bool multi;
    size_t initialCapacity;
    size_t blockSize;
} BFParams_C;

/**
 * @brief SVS quantization bits.
 */
typedef enum {
    VecSimSvsQuant_NONE = 0,
    VecSimSvsQuant_Scalar = 1,
    VecSimSvsQuant_4 = 4,
    VecSimSvsQuant_8 = 8,
    VecSimSvsQuant_4x4 = 4 | (4 << 8),
    VecSimSvsQuant_4x8 = 4 | (8 << 8),
    VecSimSvsQuant_4x8_LeanVec = 4 | (8 << 8) | (1 << 16),
    VecSimSvsQuant_8x8_LeanVec = 8 | (8 << 8) | (1 << 16)
} VecSimSvsQuantBits;

/**
 * @brief SVS parameters (C++-compatible layout).
 */
typedef struct {
    VecSimType type;
    size_t dim;
    VecSimMetric metric;
    bool multi;
    size_t blockSize;
    VecSimSvsQuantBits quantBits;
    float alpha;
    size_t graph_max_degree;
    size_t construction_window_size;
    size_t max_candidate_pool_size;
    size_t prune_to;
    VecSimOptionMode use_search_history;
    size_t num_threads;
    size_t search_window_size;
    size_t search_buffer_capacity;
    size_t leanvec_dim;
    double epsilon;
} SVSParams_C;

/**
 * @brief Tiered HNSW specific parameters.
 */
typedef struct {
    size_t swapJobThreshold;
} TieredHNSWParams_C;

/**
 * @brief Tiered SVS specific parameters.
 */
typedef struct {
    size_t trainingTriggerThreshold;
    size_t updateTriggerThreshold;
    size_t updateJobWaitTime;
} TieredSVSParams_C;

/**
 * @brief Tiered HNSW Disk specific parameters.
 */
typedef struct {
    char _placeholder;
} TieredHNSWDiskParams_C;

/* Forward declaration */
typedef struct VecSimParams_C VecSimParams_C;

/**
 * @brief Callback for submitting async jobs.
 */
typedef int (*SubmitCB)(void *job_queue, void *index_ctx, void **jobs, void **cbs, size_t jobs_len);

/**
 * @brief Tiered index parameters (C++-compatible layout).
 */
typedef struct {
    void *jobQueue;
    void *jobQueueCtx;
    SubmitCB submitCb;
    size_t flatBufferLimit;
    VecSimParams_C *primaryIndexParams;
    union {
        TieredHNSWParams_C tieredHnswParams;
        TieredSVSParams_C tieredSVSParams;
        TieredHNSWDiskParams_C tieredHnswDiskParams;
    } specificParams;
} TieredIndexParams_C;

/**
 * @brief Union of algorithm parameters (C++-compatible layout).
 */
typedef union {
    HNSWParams_C hnswParams;
    BFParams_C bfParams;
    TieredIndexParams_C tieredParams;
    SVSParams_C svsParams;
} AlgoParams_C;

/**
 * @brief VecSimParams (C++-compatible layout).
 *
 * This structure matches the C++ VecSim API exactly for drop-in compatibility.
 */
struct VecSimParams_C {
    VecSimAlgo algo;
    AlgoParams_C algoParams;
    void *logCtx;
};

/**
 * @brief Disk context (C++-compatible layout).
 */
typedef struct {
    void *storage;
    const char *indexName;
    size_t indexNameLen;
} VecSimDiskContext_C;

/**
 * @brief Disk parameters (C++-compatible layout).
 */
typedef struct {
    VecSimParams_C *indexParams;
    VecSimDiskContext_C *diskContext;
} VecSimParamsDisk_C;

/* ============================================================================
 * C++ API Compatibility Typedefs
 * ============================================================================
 * These typedefs provide drop-in compatibility with the C++ VecSim API.
 */
typedef VecSimDiskContext_C VecSimDiskContext;
typedef VecSimParamsDisk_C VecSimParamsDisk;
typedef TieredIndexParams_C TieredIndexParams;
typedef AlgoParams_C AlgoParams;
typedef HNSWParams_C HNSWParams;
typedef BFParams_C BFParams;
typedef SVSParams_C SVSParams;
typedef VecSimParams_C VecSimParams;

/* Forward declarations for Rust-native types (defined later in this header) */
struct TieredParams_Rust;
struct DiskParams_Rust;
typedef struct TieredParams_Rust TieredParams;
typedef struct DiskParams_Rust DiskParams;

/**
 * @brief HNSW runtime parameters (C++-compatible layout).
 */
typedef struct {
    size_t efRuntime;
    double epsilon;
} HNSWRuntimeParams_C;

typedef HNSWRuntimeParams_C HNSWRuntimeParams;

/**
 * @brief SVS runtime parameters (C++-compatible layout).
 */
typedef struct {
    size_t windowSize;
    size_t bufferCapacity;
    VecSimOptionMode searchHistory;
    double epsilon;
} SVSRuntimeParams_C;

typedef SVSRuntimeParams_C SVSRuntimeParams;

/**
 * @brief Query parameters (C++-compatible layout).
 */
typedef struct {
    union {
        HNSWRuntimeParams_C hnswRuntimeParams;
        SVSRuntimeParams_C svsRuntimeParams;
    };
    size_t batchSize;
    VecSearchMode searchMode;
    void *timeoutCtx;
} VecSimQueryParams_C;

/* ============================================================================
 * Memory Function Types
 * ========================================================================== */

/**
 * @brief Function pointer type for malloc-style allocation.
 */
typedef void *(*allocFn)(size_t n);

/**
 * @brief Function pointer type for calloc-style allocation.
 */
typedef void *(*callocFn)(size_t nelem, size_t elemsz);

/**
 * @brief Function pointer type for realloc-style reallocation.
 */
typedef void *(*reallocFn)(void *p, size_t n);

/**
 * @brief Function pointer type for free-style deallocation.
 */
typedef void (*freeFn)(void *p);

/**
 * @brief Memory functions struct for custom memory management.
 *
 * This allows integration with external memory management systems like Redis.
 * Pass this struct to VecSim_SetMemoryFunctions to use custom allocators.
 */
typedef struct VecSimMemoryFunctions {
    allocFn allocFunction;     /**< Malloc-like allocation function */
    callocFn callocFunction;   /**< Calloc-like allocation function */
    reallocFn reallocFunction; /**< Realloc-like reallocation function */
    freeFn freeFunction;       /**< Free function */
} VecSimMemoryFunctions;

/* ============================================================================
 * Opaque Handle Types
 * ========================================================================== */

/**
 * @brief Opaque handle to a vector similarity index.
 */
typedef struct VecSimIndex VecSimIndex;

/**
 * @brief Opaque handle to a query reply.
 */
typedef struct VecSimQueryReply VecSimQueryReply;

/**
 * @brief Opaque handle to a single query result.
 */
typedef struct VecSimQueryResult VecSimQueryResult;

/**
 * @brief Opaque handle to a query reply iterator.
 */
typedef struct VecSimQueryReply_Iterator VecSimQueryReply_Iterator;

/**
 * @brief Opaque handle to a batch iterator.
 */
typedef struct VecSimBatchIterator VecSimBatchIterator;

/* ============================================================================
 * Parameter Structures
 * ========================================================================== */

/**
 * @brief Common base parameters for all index types (Rust-native API).
 */
typedef struct VecSimBaseParams {
    VecSimAlgo algo;           /**< Algorithm type */
    VecSimType type_;          /**< Vector element data type */
    VecSimMetric metric;       /**< Distance metric */
    size_t dim;                /**< Vector dimension */
    bool multi;                /**< Whether multiple vectors per label are allowed */
    size_t initialCapacity;    /**< Initial capacity (number of vectors) */
    size_t blockSize;          /**< Block size for storage (0 for default) */
} VecSimBaseParams;

/**
 * @brief Parameters for BruteForce index creation (Rust-native API).
 */
typedef struct BFParams_Rust {
    VecSimBaseParams base;  /**< Common parameters */
} BFParams_Rust;

/**
 * @brief Parameters for HNSW index creation (Rust-native API).
 */
typedef struct HNSWParams_Rust {
    VecSimBaseParams base;       /**< Common parameters */
    size_t M;                /**< Max connections per element per layer (default: 16) */
    size_t efConstruction;   /**< Dynamic candidate list size during construction (default: 200) */
    size_t efRuntime;        /**< Dynamic candidate list size during search (default: 10) */
    double epsilon;          /**< Approximation factor (0 = exact) */
} HNSWParams_Rust;

/**
 * @brief Parameters for SVS (Vamana) index creation (Rust-native API).
 *
 * SVS (Search via Satellite) is a graph-based approximate nearest neighbor
 * index using the Vamana algorithm with robust pruning.
 */
typedef struct SVSParams_Rust {
    VecSimBaseParams base;           /**< Common parameters */
    size_t graphMaxDegree;       /**< Maximum neighbors per node (R, default: 32) */
    float alpha;                 /**< Pruning parameter for diversity (default: 1.2) */
    size_t constructionWindowSize; /**< Beam width during construction (L, default: 200) */
    size_t searchWindowSize;     /**< Default beam width during search (default: 100) */
    bool twoPassConstruction;    /**< Enable two-pass construction (default: true) */
} SVSParams_Rust;

/**
 * @brief Parameters for Tiered index creation (Rust-native API).
 *
 * The tiered index combines a BruteForce frontend (for fast writes) with
 * an HNSW backend (for efficient queries). Vectors are first added to the
 * flat buffer, then migrated to HNSW via VecSimTieredIndex_Flush() or
 * automatically when the buffer is full.
 */
typedef struct TieredParams_Rust {
    VecSimBaseParams base;           /**< Common parameters */
    size_t M;                    /**< HNSW M parameter (default: 16) */
    size_t efConstruction;       /**< HNSW ef_construction (default: 200) */
    size_t efRuntime;            /**< HNSW ef_runtime (default: 10) */
    size_t flatBufferLimit;      /**< Max flat buffer size before in-place writes (default: 10000) */
    uint32_t writeMode;          /**< 0 = Async (buffer first), 1 = InPlace (direct to HNSW) */
} TieredParams_Rust;

/**
 * @brief Backend type for disk-based indices.
 */
typedef enum DiskBackend {
    DiskBackend_BruteForce = 0,  /**< Linear scan (exact results) */
    DiskBackend_Vamana = 1       /**< Vamana graph (approximate, fast) */
} DiskBackend;

/**
 * @brief Parameters for disk-based index creation (Rust-native API).
 *
 * Disk indices store vectors in memory-mapped files for persistence.
 * They support two backends:
 * - BruteForce: Linear scan (exact results, O(n))
 * - Vamana: Graph-based approximate search (fast, O(log n))
 */
typedef struct DiskParams_Rust {
    VecSimBaseParams base;           /**< Common parameters */
    const char *dataPath;        /**< Path to the data file (null-terminated) */
    DiskBackend backend;         /**< Backend algorithm (default: BruteForce) */
    size_t graphMaxDegree;       /**< Graph max degree for Vamana (default: 32) */
    float alpha;                 /**< Alpha parameter for Vamana (default: 1.2) */
    size_t constructionL;        /**< Construction window size for Vamana (default: 200) */
    size_t searchL;              /**< Search window size for Vamana (default: 100) */
} DiskParams_Rust;

/**
 * @brief HNSW-specific runtime parameters (Rust-native layout).
 */
typedef struct HNSWRuntimeParams_Rust {
    size_t efRuntime;  /**< Dynamic candidate list size during search */
    double epsilon;    /**< Approximation factor */
} HNSWRuntimeParams_Rust;

/**
 * @brief SVS-specific runtime parameters (Rust-native layout).
 */
typedef struct SVSRuntimeParams_Rust {
    size_t windowSize;       /**< Search window size for graph search */
    size_t bufferCapacity;   /**< Search buffer capacity */
    int searchHistory;       /**< Whether to use search history (0/1) */
    double epsilon;          /**< Approximation factor for range search */
} SVSRuntimeParams_Rust;

/**
 * @brief Query parameters (Rust-native layout).
 *
 * Note: For C++ API compatibility, use VecSimQueryParams_C instead.
 */
typedef struct VecSimQueryParams_Rust {
    HNSWRuntimeParams hnswRuntimeParams;  /**< HNSW-specific parameters */
    SVSRuntimeParams svsRuntimeParams;    /**< SVS-specific parameters */
    VecSimSearchMode_Internal searchMode; /**< Search mode */
    VecSimHybridPolicy hybridPolicy;      /**< Hybrid policy */
    size_t batchSize;                     /**< Batch size for iteration */
    void *timeoutCtx;                     /**< Timeout context (opaque) */
} VecSimQueryParams_Rust;

/**
 * @brief Query parameters (C++ API compatible).
 *
 * This typedef provides compatibility with the C++ VecSim API.
 */
typedef VecSimQueryParams_C VecSimQueryParams;

/* ============================================================================
 * Index Info Structures
 * ========================================================================== */

/**
 * @brief HNSW-specific index information.
 */
typedef struct VecSimHnswInfo {
    size_t M;               /**< M parameter */
    size_t efConstruction;  /**< ef_construction parameter */
    size_t efRuntime;       /**< ef_runtime parameter */
    size_t maxLevel;        /**< Maximum level in the graph */
    int64_t entrypoint;     /**< Entry point ID (-1 if none) */
    double epsilon;         /**< Epsilon parameter */
} VecSimHnswInfo;

/**
 * @brief Comprehensive index information.
 */
typedef struct VecSimIndexInfo {
    size_t indexSize;           /**< Current number of vectors */
    size_t indexLabelCount;     /**< Current number of unique labels */
    size_t dim;                 /**< Vector dimension */
    VecSimType type_;           /**< Data type */
    VecSimAlgo algo;            /**< Algorithm type */
    VecSimMetric metric;        /**< Distance metric */
    bool isMulti;               /**< Whether multi-value index */
    size_t blockSize;           /**< Block size */
    size_t memory;              /**< Memory usage in bytes */
    VecSimHnswInfo hnswInfo;    /**< HNSW-specific info (if applicable) */
} VecSimIndexInfo;

/**
 * @brief Index info that is static and immutable.
 */
typedef struct VecSimIndexBasicInfo {
    VecSimAlgo algo;            /**< Algorithm type */
    VecSimMetric metric;        /**< Distance metric */
    VecSimType type;            /**< Data type */
    bool isMulti;               /**< Whether multi-value index */
    bool isTiered;              /**< Whether tiered index */
    bool isDisk;                /**< Whether disk-based index */
    size_t blockSize;           /**< Block size */
    size_t dim;                 /**< Vector dimension */
} VecSimIndexBasicInfo;

/**
 * @brief Index info for statistics - thin and efficient.
 */
typedef struct VecSimIndexStatsInfo {
    size_t memory;                  /**< Memory usage in bytes */
    size_t numberOfMarkedDeleted;   /**< Number of marked deleted entries */
} VecSimIndexStatsInfo;

/**
 * @brief Common index information.
 */
typedef struct CommonInfo {
    VecSimIndexBasicInfo basicInfo; /**< Basic index information */
    size_t indexSize;               /**< Current number of vectors */
    size_t indexLabelCount;         /**< Current number of unique labels */
    uint64_t memory;                /**< Memory usage in bytes */
    VecSearchMode lastMode;         /**< Last search mode used */
} CommonInfo;

/**
 * @brief HNSW-specific debug information.
 */
typedef struct hnswInfoStruct {
    size_t M;                           /**< M parameter */
    size_t efConstruction;              /**< ef_construction parameter */
    size_t efRuntime;                   /**< ef_runtime parameter */
    double epsilon;                     /**< Epsilon parameter */
    size_t max_level;                   /**< Maximum level in the graph */
    size_t entrypoint;                  /**< Entry point ID */
    size_t visitedNodesPoolSize;        /**< Visited nodes pool size */
    size_t numberOfMarkedDeletedNodes;  /**< Number of marked deleted nodes */
} hnswInfoStruct;

/**
 * @brief BruteForce-specific debug information.
 */
typedef struct bfInfoStruct {
    int8_t dummy;   /**< Placeholder field */
} bfInfoStruct;

/**
 * @brief Debug information for an index.
 */
typedef struct VecSimIndexDebugInfo {
    CommonInfo commonInfo;      /**< Common index information */
    hnswInfoStruct hnswInfo;    /**< HNSW-specific info */
    bfInfoStruct bfInfo;        /**< BruteForce-specific info */
} VecSimIndexDebugInfo;

/* ============================================================================
 * Debug Info Iterator Types
 * ========================================================================== */

/**
 * @brief Opaque handle to a debug info iterator.
 */
typedef struct VecSimDebugInfoIterator VecSimDebugInfoIterator;

/**
 * @brief Field type for debug info fields.
 */
typedef enum {
    INFOFIELD_STRING = 0,   /**< String value */
    INFOFIELD_INT64 = 1,    /**< Signed 64-bit integer */
    INFOFIELD_UINT64 = 2,   /**< Unsigned 64-bit integer */
    INFOFIELD_FLOAT64 = 3,  /**< 64-bit floating point */
    INFOFIELD_ITERATOR = 4  /**< Nested iterator */
} VecSim_InfoFieldType;

/**
 * @brief Union of field values.
 */
typedef union {
    double floatingPointValue;              /**< 64-bit float value */
    int64_t integerValue;                   /**< Signed 64-bit integer */
    uint64_t uintegerValue;                 /**< Unsigned 64-bit integer */
    const char *stringValue;                /**< String value */
    VecSimDebugInfoIterator *iteratorValue; /**< Nested iterator */
} FieldValue;

/**
 * @brief A field in the debug info iterator.
 */
typedef struct {
    const char *fieldName;          /**< Field name */
    VecSim_InfoFieldType fieldType; /**< Field type */
    FieldValue fieldValue;          /**< Field value */
} VecSim_InfoField;

/* ============================================================================
 * Index Lifecycle Functions
 * ========================================================================== */

/**
 * @brief Create a new vector similarity index (C++-compatible API).
 *
 * This function provides drop-in compatibility with the C++ VecSim API.
 * It reads the algo field to determine which type of index to create,
 * then accesses the appropriate union variant in algoParams.
 *
 * @param params Index parameters with algorithm-specific params in the union.
 * @return A new index handle, or NULL on failure.
 *
 * @note For type-safe index creation, use VecSimIndex_NewBF(), VecSimIndex_NewHNSW(),
 *       VecSimIndex_NewSVS(), VecSimIndex_NewTiered(), or VecSimIndex_NewDisk().
 */
VecSimIndex *VecSimIndex_New(const VecSimParams_C *params);

/**
 * @brief Create a new BruteForce index.
 *
 * @param params Pointer to BruteForce-specific parameters
 * @return Pointer to the created index, or NULL on failure
 */
VecSimIndex *VecSimIndex_NewBF(const BFParams *params);

/**
 * @brief Create a new HNSW index.
 *
 * @param params Pointer to HNSW-specific parameters
 * @return Pointer to the created index, or NULL on failure
 */
VecSimIndex *VecSimIndex_NewHNSW(const HNSWParams *params);

/**
 * @brief Create a new SVS (Vamana) index.
 *
 * SVS provides an alternative to HNSW with a single-layer graph structure.
 * It uses robust pruning to maintain graph quality and provides good
 * recall with efficient memory usage.
 *
 * @param params Pointer to SVS-specific parameters
 * @return Pointer to the created index, or NULL on failure
 */
VecSimIndex *VecSimIndex_NewSVS(const SVSParams *params);

/**
 * @brief Create a new Tiered index.
 *
 * The tiered index combines a BruteForce frontend (for fast writes) with
 * an HNSW backend (for efficient queries). Vectors are first added to the
 * flat buffer, then migrated to HNSW via VecSimTieredIndex_Flush() or
 * automatically when the buffer is full.
 *
 * Currently only supports f32 vectors.
 *
 * @param params Pointer to Tiered-specific parameters
 * @return Pointer to the created index, or NULL on failure
 */
VecSimIndex *VecSimIndex_NewTiered(const TieredParams *params);

/**
 * @brief Free a vector similarity index.
 *
 * @param index Pointer to the index to free (may be NULL)
 */
void VecSimIndex_Free(VecSimIndex *index);

/* ============================================================================
 * Tiered Index Operations
 * ========================================================================== */

/**
 * @brief Flush the flat buffer to the HNSW backend.
 *
 * This migrates all vectors from the flat buffer to the HNSW index.
 *
 * @param index Pointer to a tiered index
 * @return Number of vectors flushed, or 0 if the index is not tiered
 */
size_t VecSimTieredIndex_Flush(VecSimIndex *index);

/**
 * @brief Get the number of vectors in the flat buffer.
 *
 * @param index Pointer to a tiered index
 * @return Number of vectors in the flat buffer, or 0 if not tiered
 */
size_t VecSimTieredIndex_FlatSize(const VecSimIndex *index);

/**
 * @brief Get the number of vectors in the HNSW backend.
 *
 * @param index Pointer to a tiered index
 * @return Number of vectors in the HNSW backend, or 0 if not tiered
 */
size_t VecSimTieredIndex_BackendSize(const VecSimIndex *index);

/**
 * @brief Run garbage collection on a tiered index.
 *
 * This cleans up deleted vectors and optimizes the index structure.
 *
 * @param index The tiered index handle.
 */
void VecSimTieredIndex_GC(VecSimIndex *index);

/**
 * @brief Acquire shared locks on a tiered index.
 *
 * This prevents modifications to the index while the locks are held.
 * Must be paired with VecSimTieredIndex_ReleaseSharedLocks.
 *
 * @param index The tiered index handle.
 */
void VecSimTieredIndex_AcquireSharedLocks(VecSimIndex *index);

/**
 * @brief Release shared locks on a tiered index.
 *
 * Must be called after VecSimTieredIndex_AcquireSharedLocks.
 *
 * @param index The tiered index handle.
 */
void VecSimTieredIndex_ReleaseSharedLocks(VecSimIndex *index);

/**
 * @brief Check if the index is a tiered index.
 *
 * @param index Pointer to an index
 * @return true if the index is tiered, false otherwise
 */
bool VecSimIndex_IsTiered(const VecSimIndex *index);

/**
 * @brief Create a new disk-based index.
 *
 * Disk indices store vectors in memory-mapped files for persistence.
 * They support two backends:
 * - BruteForce: Linear scan (exact results, O(n))
 * - Vamana: Graph-based approximate search (fast, O(log n))
 *
 * Currently only supports f32 vectors.
 *
 * @param params Pointer to Disk-specific parameters
 * @return Pointer to the created index, or NULL on failure
 */
VecSimIndex *VecSimIndex_NewDisk(const DiskParams *params);

/**
 * @brief Check if the index is a disk-based index.
 *
 * @param index Pointer to an index
 * @return true if the index is disk-based, false otherwise
 */
bool VecSimIndex_IsDisk(const VecSimIndex *index);

/**
 * @brief Flush changes to disk for a disk-based index.
 *
 * This ensures all pending changes are written to the underlying file.
 *
 * @param index Pointer to a disk-based index
 * @return true if flush succeeded, false otherwise
 */
bool VecSimDiskIndex_Flush(const VecSimIndex *index);

/* ============================================================================
 * Vector Operations
 * ========================================================================== */

/**
 * @brief Add a vector to the index.
 *
 * @param index Pointer to the index
 * @param vector Pointer to the vector data (must match index dimension and type)
 * @param label Label to associate with the vector
 * @return Number of vectors added (1 on success), or -1 on failure
 *
 * @note For single-value indices, adding a vector with an existing label
 *       replaces the previous vector.
 */
int VecSimIndex_AddVector(VecSimIndex *index, const void *vector, labelType label);

/**
 * @brief Delete all vectors with the given label.
 *
 * @param index Pointer to the index
 * @param label Label of vectors to delete
 * @return Number of vectors deleted, or 0 if label not found
 */
int VecSimIndex_DeleteVector(VecSimIndex *index, labelType label);

/**
 * @brief Get the distance from a stored vector to a query vector.
 *
 * @param index Pointer to the index
 * @param label Label of the stored vector
 * @param vector Pointer to the query vector
 * @return Distance value, or INFINITY if label not found
 *
 * @warning This function accesses internal storage directly. Use with caution.
 */
double VecSimIndex_GetDistanceFrom_Unsafe(VecSimIndex *index, labelType label, const void *vector);

/* ============================================================================
 * Query Functions
 * ========================================================================== */

/**
 * @brief Perform a top-k nearest neighbor query.
 *
 * @param index Pointer to the index
 * @param query Pointer to the query vector
 * @param k Maximum number of results to return
 * @param params Query parameters (may be NULL for defaults)
 * @param order Result ordering
 * @return Pointer to query reply, or NULL on failure
 *
 * @note The caller is responsible for freeing the reply with VecSimQueryReply_Free().
 */
VecSimQueryReply *VecSimIndex_TopKQuery(
    VecSimIndex *index,
    const void *query,
    size_t k,
    const VecSimQueryParams *params,
    VecSimQueryReply_Order order
);

/**
 * @brief Perform a range query.
 *
 * @param index Pointer to the index
 * @param query Pointer to the query vector
 * @param radius Maximum distance from query (inclusive)
 * @param params Query parameters (may be NULL for defaults)
 * @param order Result ordering
 * @return Pointer to query reply, or NULL on failure
 *
 * @note The caller is responsible for freeing the reply with VecSimQueryReply_Free().
 */
VecSimQueryReply *VecSimIndex_RangeQuery(
    VecSimIndex *index,
    const void *query,
    double radius,
    const VecSimQueryParams *params,
    VecSimQueryReply_Order order
);

/* ============================================================================
 * Query Reply Functions
 * ========================================================================== */

/**
 * @brief Get the number of results in a query reply.
 *
 * @param reply Pointer to the query reply
 * @return Number of results
 */
size_t VecSimQueryReply_Len(const VecSimQueryReply *reply);

/**
 * @brief Get the status code of a query reply.
 *
 * This is used to detect if the query timed out.
 *
 * @param reply Pointer to the query reply
 * @return The status code (VecSim_QueryReply_OK or VecSim_QueryReply_TimedOut)
 */
VecSimQueryReply_Code VecSimQueryReply_GetCode(const VecSimQueryReply *reply);

/**
 * @brief Free a query reply.
 *
 * @param reply Pointer to the query reply (may be NULL)
 */
void VecSimQueryReply_Free(VecSimQueryReply *reply);

/**
 * @brief Get an iterator over query results.
 *
 * @param reply Pointer to the query reply
 * @return Pointer to iterator, or NULL on failure
 *
 * @note The iterator is only valid while the reply exists.
 *       Free with VecSimQueryReply_IteratorFree().
 */
VecSimQueryReply_Iterator *VecSimQueryReply_GetIterator(VecSimQueryReply *reply);

/**
 * @brief Check if the iterator has more results.
 *
 * @param iter Pointer to the iterator
 * @return true if more results available, false otherwise
 */
bool VecSimQueryReply_IteratorHasNext(const VecSimQueryReply_Iterator *iter);

/**
 * @brief Get the next result from the iterator.
 *
 * @param iter Pointer to the iterator
 * @return Pointer to the next result, or NULL if no more results
 *
 * @note The returned pointer is valid until the next call to IteratorNext
 *       or until the reply is freed.
 */
const VecSimQueryResult *VecSimQueryReply_IteratorNext(VecSimQueryReply_Iterator *iter);

/**
 * @brief Reset the iterator to the beginning.
 *
 * @param iter Pointer to the iterator
 */
void VecSimQueryReply_IteratorReset(VecSimQueryReply_Iterator *iter);

/**
 * @brief Free an iterator.
 *
 * @param iter Pointer to the iterator (may be NULL)
 */
void VecSimQueryReply_IteratorFree(VecSimQueryReply_Iterator *iter);

/* ============================================================================
 * Query Result Functions
 * ========================================================================== */

/**
 * @brief Get the label (ID) from a query result.
 *
 * @param result Pointer to the query result
 * @return Label of the result
 */
labelType VecSimQueryResult_GetId(const VecSimQueryResult *result);

/**
 * @brief Get the score (distance) from a query result.
 *
 * @param result Pointer to the query result
 * @return Distance/score of the result
 */
double VecSimQueryResult_GetScore(const VecSimQueryResult *result);

/* ============================================================================
 * Batch Iterator Functions
 * ========================================================================== */

/**
 * @brief Create a batch iterator for incremental query processing.
 *
 * @param index Pointer to the index
 * @param query Pointer to the query vector
 * @param params Query parameters (may be NULL for defaults)
 * @return Pointer to batch iterator, or NULL on failure
 *
 * @note Batch iterators are useful for processing large result sets
 *       incrementally without loading all results into memory.
 */
VecSimBatchIterator *VecSimBatchIterator_New(
    VecSimIndex *index,
    const void *query,
    const VecSimQueryParams *params
);

/**
 * @brief Get the next batch of results.
 *
 * @param iter Pointer to the batch iterator
 * @param n Maximum number of results to return in this batch
 * @param order Result ordering
 * @return Pointer to query reply containing the batch, or NULL on failure
 *
 * @note The caller is responsible for freeing the reply with VecSimQueryReply_Free().
 */
VecSimQueryReply *VecSimBatchIterator_Next(
    VecSimBatchIterator *iter,
    size_t n,
    VecSimQueryReply_Order order
);

/**
 * @brief Check if the batch iterator has more results.
 *
 * @param iter Pointer to the batch iterator
 * @return true if more results available, false otherwise
 */
bool VecSimBatchIterator_HasNext(const VecSimBatchIterator *iter);

/**
 * @brief Reset the batch iterator to the beginning.
 *
 * @param iter Pointer to the batch iterator
 */
void VecSimBatchIterator_Reset(VecSimBatchIterator *iter);

/**
 * @brief Free a batch iterator.
 *
 * @param iter Pointer to the batch iterator (may be NULL)
 */
void VecSimBatchIterator_Free(VecSimBatchIterator *iter);

/* ============================================================================
 * Index Property Functions
 * ========================================================================== */

/**
 * @brief Get the current number of vectors in the index.
 *
 * @param index Pointer to the index
 * @return Number of vectors
 */
size_t VecSimIndex_IndexSize(const VecSimIndex *index);

/**
 * @brief Get the data type of the index.
 *
 * @param index Pointer to the index
 * @return Data type enum value
 */
VecSimType VecSimIndex_GetType(const VecSimIndex *index);

/**
 * @brief Get the distance metric of the index.
 *
 * @param index Pointer to the index
 * @return Metric enum value
 */
VecSimMetric VecSimIndex_GetMetric(const VecSimIndex *index);

/**
 * @brief Get the vector dimension of the index.
 *
 * @param index Pointer to the index
 * @return Vector dimension
 */
size_t VecSimIndex_GetDim(const VecSimIndex *index);

/**
 * @brief Check if the index is a multi-value index.
 *
 * @param index Pointer to the index
 * @return true if multi-value, false if single-value
 */
bool VecSimIndex_IsMulti(const VecSimIndex *index);

/**
 * @brief Check if a label exists in the index.
 *
 * @param index Pointer to the index
 * @param label Label to check
 * @return true if label exists, false otherwise
 */
bool VecSimIndex_ContainsLabel(const VecSimIndex *index, labelType label);

/**
 * @brief Get the count of vectors with the given label.
 *
 * @param index Pointer to the index
 * @param label Label to count
 * @return Number of vectors with this label (0 if not found)
 */
size_t VecSimIndex_LabelCount(const VecSimIndex *index, labelType label);

/**
 * @brief Get detailed index information.
 *
 * @param index Pointer to the index
 * @return VecSimIndexInfo structure with index details
 */
VecSimIndexInfo VecSimIndex_Info(const VecSimIndex *index);

/**
 * @brief Get basic immutable index information.
 *
 * @param index The index handle.
 * @return Basic index information.
 */
VecSimIndexBasicInfo VecSimIndex_BasicInfo(const VecSimIndex *index);

/**
 * @brief Get index statistics information.
 *
 * This is a thin and efficient info call with no locks or calculations.
 *
 * @param index The index handle.
 * @return Statistics information.
 */
VecSimIndexStatsInfo VecSimIndex_StatsInfo(const VecSimIndex *index);

/**
 * @brief Get detailed debug information for an index.
 *
 * This should only be used for debug/testing purposes.
 *
 * @param index The index handle.
 * @return Debug information.
 */
VecSimIndexDebugInfo VecSimIndex_DebugInfo(const VecSimIndex *index);

/**
 * @brief Create a debug info iterator for an index.
 *
 * The iterator provides a way to traverse all debug information fields
 * for an index, including nested information for tiered indices.
 *
 * @param index The index handle.
 * @return A debug info iterator, or NULL on failure.
 *         Must be freed with VecSimDebugInfoIterator_Free().
 */
VecSimDebugInfoIterator *VecSimIndex_DebugInfoIterator(const VecSimIndex *index);

/**
 * @brief Returns the number of fields in the info iterator.
 *
 * @param infoIterator The info iterator.
 * @return Number of fields.
 */
size_t VecSimDebugInfoIterator_NumberOfFields(VecSimDebugInfoIterator *infoIterator);

/**
 * @brief Check if the iterator has more fields.
 *
 * @param infoIterator The info iterator.
 * @return true if more fields are available, false otherwise.
 */
bool VecSimDebugInfoIterator_HasNextField(VecSimDebugInfoIterator *infoIterator);

/**
 * @brief Get the next field from the iterator.
 *
 * The returned pointer is valid until the next call to this function
 * or until the iterator is freed.
 *
 * @param infoIterator The info iterator.
 * @return Pointer to the next info field, or NULL if no more fields.
 */
VecSim_InfoField *VecSimDebugInfoIterator_NextField(VecSimDebugInfoIterator *infoIterator);

/**
 * @brief Free a debug info iterator.
 *
 * This also frees all nested iterators.
 *
 * @param infoIterator The info iterator to free.
 */
void VecSimDebugInfoIterator_Free(VecSimDebugInfoIterator *infoIterator);

/**
 * @brief Dump the neighbors of an element in HNSW index.
 *
 * Returns an array with <topLevel+2> entries, where each entry is an array itself.
 * Every internal array in a position <l> where <0<=l<=topLevel> corresponds to the neighbors
 * of the element in the graph in level <l>. It contains <n_l+1> entries, where <n_l> is the
 * number of neighbors in level l. The last entry in the external array is NULL (indicates its length).
 * The first entry in each internal array contains the number <n_l>, while the next
 * <n_l> entries are the labels of the elements neighbors in this level.
 *
 * Note: currently only HNSW indexes of type single are supported (multi not yet) - tiered included.
 * For cleanup, VecSimDebug_ReleaseElementNeighborsInHNSWGraph needs to be called with the value
 * pointed by neighborsData as returned from this call.
 *
 * @param index The index in which the element resides.
 * @param label The label to dump its neighbors in every level in which it exists.
 * @param neighborsData A pointer to a 2-dim array of integer which is a placeholder for the
 *                      output of the neighbors' labels that will be allocated and stored.
 * @return VecSimDebugCommandCode indicating success or failure reason.
 */
VecSimDebugCommandCode VecSimDebug_GetElementNeighborsInHNSWGraph(VecSimIndex *index, size_t label,
                                                                   int ***neighborsData);

/**
 * @brief Release the neighbors data allocated by VecSimDebug_GetElementNeighborsInHNSWGraph.
 *
 * @param neighborsData The 2-dim array returned in the placeholder to be de-allocated.
 */
void VecSimDebug_ReleaseElementNeighborsInHNSWGraph(int **neighborsData);

/**
 * @brief Determine if ad-hoc brute-force search is preferred over batched search.
 *
 * This is a heuristic function that helps decide the optimal search strategy
 * for hybrid queries based on the index size and the number of results needed.
 *
 * @param index The index handle.
 * @param subsetSize The estimated size of the subset to search.
 * @param k The number of results requested.
 * @param initial Whether this is the initial decision (true) or a re-evaluation (false).
 * @return true if ad-hoc search is preferred, false if batched search is preferred.
 */
bool VecSimIndex_PreferAdHocSearch(const VecSimIndex *index, size_t subsetSize, size_t k, bool initial);

/* ============================================================================
 * Parameter Resolution
 * ========================================================================== */

/**
 * @brief Resolve runtime query parameters from raw string parameters.
 *
 * Parses an array of VecSimRawParam structures and populates a VecSimQueryParams
 * structure with the resolved typed values.
 *
 * @param index Pointer to the index (used to determine algorithm-specific parameters)
 * @param rparams Array of raw parameters to resolve
 * @param paramNum Number of parameters in the array
 * @param qparams Pointer to VecSimQueryParams structure to populate
 * @param query_type Type of query (KNN, HYBRID, or RANGE)
 * @return VecSimParamResolver_OK on success, error code on failure
 *
 * Supported parameters:
 * - EF_RUNTIME: HNSW ef_runtime (positive integer, not for range queries)
 * - EPSILON: Approximation factor (positive float, range queries only, HNSW/SVS)
 * - BATCH_SIZE: Batch size for hybrid queries (positive integer)
 * - HYBRID_POLICY: "batches" or "adhoc_bf" (hybrid queries only)
 * - SEARCH_WINDOW_SIZE: SVS search window size (positive integer)
 * - SEARCH_BUFFER_CAPACITY: SVS search buffer capacity (positive integer)
 * - USE_SEARCH_HISTORY: SVS search history flag ("true"/"false"/"1"/"0")
 */
VecSimParamResolveCode VecSimIndex_ResolveParams(VecSimIndex *index,
                                                  VecSimRawParam *rparams,
                                                  int paramNum,
                                                  VecSimQueryParams *qparams,
                                                  VecsimQueryType query_type);

/* ============================================================================
 * Serialization Functions
 * ========================================================================== */

/**
 * @brief Save an index to a file.
 *
 * @param index Pointer to the index
 * @param path File path to save to (null-terminated C string)
 * @return true on success, false on failure
 *
 * @note Serialization is supported for:
 *       - BruteForce (f32 only)
 *       - HNSW (all data types)
 *       - SVS Single (f32 only)
 */
bool VecSimIndex_SaveIndex(const VecSimIndex *index, const char *path);

/**
 * @brief Load an index from a file.
 *
 * Reads the file header to determine the index type and data type,
 * then loads the appropriate index. The caller is responsible for
 * freeing the returned index with VecSimIndex_Free.
 *
 * @param path File path to load from (null-terminated C string)
 * @param params Optional parameters to override (may be NULL, currently unused)
 * @return Pointer to loaded index, or NULL on failure
 *
 * @note Supported index types for loading:
 *       - BruteForceSingle/Multi (f32)
 *       - HnswSingle/Multi (f32)
 *       - SvsSingle (f32)
 */
VecSimIndex *VecSimIndex_LoadIndex(const char *path, const VecSimParams *params);

/* ============================================================================
 * Memory Estimation Functions
 * ========================================================================== */

/**
 * @brief Estimate initial memory size for a BruteForce index.
 *
 * @param dim Vector dimension
 * @param initial_capacity Initial capacity (number of vectors)
 * @return Estimated memory size in bytes
 */
size_t VecSimIndex_EstimateBruteForceInitialSize(size_t dim, size_t initial_capacity);

/**
 * @brief Estimate memory size per element for a BruteForce index.
 *
 * @param dim Vector dimension
 * @return Estimated memory per element in bytes
 */
size_t VecSimIndex_EstimateBruteForceElementSize(size_t dim);

/**
 * @brief Estimate initial memory size for an HNSW index.
 *
 * @param dim Vector dimension
 * @param initial_capacity Initial capacity (number of vectors)
 * @param m M parameter (max connections per layer)
 * @return Estimated memory size in bytes
 */
size_t VecSimIndex_EstimateHNSWInitialSize(size_t dim, size_t initial_capacity, size_t m);

/**
 * @brief Estimate memory size per element for an HNSW index.
 *
 * @param dim Vector dimension
 * @param m M parameter (max connections per layer)
 * @return Estimated memory per element in bytes
 */
size_t VecSimIndex_EstimateHNSWElementSize(size_t dim, size_t m);

/**
 * @brief Estimate initial memory size for an index based on parameters.
 *
 * @param params The index parameters.
 * @return Estimated initial memory size in bytes.
 */
size_t VecSimIndex_EstimateInitialSize(const VecSimParams *params);

/**
 * @brief Estimate memory size per element for an index based on parameters.
 *
 * @param params The index parameters.
 * @return Estimated memory size per element in bytes.
 */
size_t VecSimIndex_EstimateElementSize(const VecSimParams *params);

/* ============================================================================
 * Write Mode Control
 * ========================================================================== */

/**
 * @brief Set the global write mode for tiered index operations.
 *
 * This controls whether vector additions/deletions in tiered indices go through
 * the async buffering path (VecSim_WriteAsync) or directly to the backend index
 * (VecSim_WriteInPlace).
 *
 * @param mode The write mode to set.
 *
 * @note In a tiered index scenario, this should be called from the main thread only
 *       (that is, the thread that is calling add/delete vector functions).
 */
void VecSim_SetWriteMode(VecSimWriteMode mode);

/**
 * @brief Get the current global write mode.
 *
 * @return The currently active write mode for tiered index operations.
 */
VecSimWriteMode VecSim_GetWriteMode(void);

/* ============================================================================
 * Memory Management Functions
 * ========================================================================== */

/**
 * @brief Set custom memory functions for all future allocations.
 *
 * This allows integration with external memory management systems like Redis.
 * The functions will be used for all memory allocations in the library.
 *
 * @param functions The memory functions struct containing custom allocators.
 *
 * @note This should be called once at initialization, before creating any indices.
 * @note The provided function pointers must be valid and thread-safe.
 * @note The functions must follow standard malloc/calloc/realloc/free semantics.
 */
void VecSim_SetMemoryFunctions(VecSimMemoryFunctions functions);

/**
 * @brief Set the timeout callback function.
 *
 * The callback will be called periodically during long operations to check
 * if the operation should be aborted. Return non-zero to abort.
 *
 * @param callback The timeout callback function, or NULL to disable.
 */
void VecSim_SetTimeoutCallbackFunction(timeoutCallbackFunction callback);

/**
 * @brief Set the log callback function.
 *
 * The callback will be called for logging messages from the library.
 *
 * @param callback The log callback function, or NULL to disable.
 */
void VecSim_SetLogCallbackFunction(logCallbackFunction callback);

/**
 * @brief Set the test log context.
 *
 * This is used for testing to identify which test is running.
 *
 * @param test_name The name of the test.
 * @param test_type The type of the test.
 */
void VecSim_SetTestLogContext(const char *test_name, const char *test_type);

/* ============================================================================
 * Vector Utility Functions
 * ========================================================================== */

/**
 * @brief Normalize a vector in-place.
 *
 * This normalizes the vector to unit length (L2 norm = 1).
 * This is useful for cosine similarity where vectors should be normalized.
 *
 * @param blob Pointer to the vector data.
 * @param dim The dimension of the vector.
 * @param type The data type of the vector elements.
 */
void VecSim_Normalize(void *blob, size_t dim, VecSimType type);

#ifdef __cplusplus
}
#endif

#endif /* VECSIM_H */
