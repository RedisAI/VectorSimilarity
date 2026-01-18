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
    VecSimType_UINT8 = 5      /**< 8-bit unsigned integer */
} VecSimType;

/**
 * @brief Index algorithm type.
 */
typedef enum VecSimAlgo {
    VecSimAlgo_BF = 0,      /**< Brute Force (exact, linear scan) */
    VecSimAlgo_HNSWLIB = 1, /**< HNSW (approximate, logarithmic) */
    VecSimAlgo_SVS = 2      /**< SVS/Vamana (approximate, single-layer graph) */
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
 * @brief Search mode for queries.
 */
typedef enum VecSimSearchMode {
    STANDARD = 0,  /**< Standard search mode */
    HYBRID = 1,    /**< Hybrid search mode */
    RANGE = 2      /**< Range search mode */
} VecSimSearchMode;

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
 * @brief Common base parameters for all index types.
 */
typedef struct VecSimParams {
    VecSimAlgo algo;           /**< Algorithm type */
    VecSimType type_;          /**< Vector element data type */
    VecSimMetric metric;       /**< Distance metric */
    size_t dim;                /**< Vector dimension */
    bool multi;                /**< Whether multiple vectors per label are allowed */
    size_t initialCapacity;    /**< Initial capacity (number of vectors) */
    size_t blockSize;          /**< Block size for storage (0 for default) */
} VecSimParams;

/**
 * @brief Parameters for BruteForce index creation.
 */
typedef struct BFParams {
    VecSimParams base;  /**< Common parameters */
} BFParams;

/**
 * @brief Parameters for HNSW index creation.
 */
typedef struct HNSWParams {
    VecSimParams base;       /**< Common parameters */
    size_t M;                /**< Max connections per element per layer (default: 16) */
    size_t efConstruction;   /**< Dynamic candidate list size during construction (default: 200) */
    size_t efRuntime;        /**< Dynamic candidate list size during search (default: 10) */
    double epsilon;          /**< Approximation factor (0 = exact) */
} HNSWParams;

/**
 * @brief Parameters for SVS (Vamana) index creation.
 *
 * SVS (Search via Satellite) is a graph-based approximate nearest neighbor
 * index using the Vamana algorithm with robust pruning.
 */
typedef struct SVSParams {
    VecSimParams base;           /**< Common parameters */
    size_t graphMaxDegree;       /**< Maximum neighbors per node (R, default: 32) */
    float alpha;                 /**< Pruning parameter for diversity (default: 1.2) */
    size_t constructionWindowSize; /**< Beam width during construction (L, default: 200) */
    size_t searchWindowSize;     /**< Default beam width during search (default: 100) */
    bool twoPassConstruction;    /**< Enable two-pass construction (default: true) */
} SVSParams;

/**
 * @brief HNSW-specific runtime parameters.
 */
typedef struct HNSWRuntimeParams {
    size_t efRuntime;  /**< Dynamic candidate list size during search */
    double epsilon;    /**< Approximation factor */
} HNSWRuntimeParams;

/**
 * @brief Query parameters.
 */
typedef struct VecSimQueryParams {
    HNSWRuntimeParams hnswRuntimeParams;  /**< HNSW-specific parameters */
    VecSimSearchMode searchMode;          /**< Search mode */
    VecSimHybridPolicy hybridPolicy;      /**< Hybrid policy */
    size_t batchSize;                     /**< Batch size for iteration */
    void *timeoutCtx;                     /**< Timeout context (opaque) */
} VecSimQueryParams;

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

/* ============================================================================
 * Index Lifecycle Functions
 * ========================================================================== */

/**
 * @brief Create a new vector similarity index.
 *
 * @param params Pointer to index parameters (VecSimParams, BFParams, HNSWParams, or SVSParams)
 * @return Pointer to the created index, or NULL on failure
 *
 * @note The params pointer is interpreted based on the algo field.
 *       For full control, use VecSimIndex_NewBF(), VecSimIndex_NewHNSW(), or VecSimIndex_NewSVS().
 */
VecSimIndex *VecSimIndex_New(const VecSimParams *params);

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
 * @brief Free a vector similarity index.
 *
 * @param index Pointer to the index to free (may be NULL)
 */
void VecSimIndex_Free(VecSimIndex *index);

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

/* ============================================================================
 * Serialization Functions
 * ========================================================================== */

/**
 * @brief Save an index to a file.
 *
 * @param index Pointer to the index
 * @param path File path to save to
 *
 * @note Currently not implemented (stub).
 */
void VecSimIndex_SaveIndex(const VecSimIndex *index, const char *path);

/**
 * @brief Load an index from a file.
 *
 * @param path File path to load from
 * @param params Optional parameters to override (may be NULL)
 * @return Pointer to loaded index, or NULL on failure
 *
 * @note Currently not implemented (stub).
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

#ifdef __cplusplus
}
#endif

#endif /* VECSIM_H */
