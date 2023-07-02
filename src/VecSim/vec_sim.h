/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include "query_results.h"
#include "vec_sim_common.h"
#include "info_iterator.h"

typedef struct VecSimIndexInterface VecSimIndex;

/**
 * @brief Create a new VecSim index based on the given params.
 * @param params index configurations (initial size, data type, dimension, metric, algorithm and the
 * algorithm-related params).
 * @return A pointer to the created index.
 */
VecSimIndex *VecSimIndex_New(const VecSimParams *params);

/**
 * @brief Estimates the size of an empty index according to the parameters.
 * @param params index configurations (initial size, data type, dimension, metric, algorithm and the
 * algorithm-related params).
 * @return Estimated index size.
 */
size_t VecSimIndex_EstimateInitialSize(const VecSimParams *params);

/**
 * @brief Estimates the size of a single vector and its metadata according to the parameters, WHEN
 * THE INDEX IS RESIZING BY A BLOCK. That is, this function estimates the allocation size of a new
 * block upon resizing all the internal data structures, and returns the size of a single vector in
 * that block. This value can be used later to decide what is the best block size for the block
 * size, when the memory limit is known.
 * ("memory limit for a block" / "size of a single vector in a block" = "block size")
 * @param params index configurations (initial size, data type, dimension, metric, algorithm and the
 * algorithm-related params).
 * @return The estimated single vector memory consumption, considering the parameters.
 */
size_t VecSimIndex_EstimateElementSize(const VecSimParams *params);

/**
 * @brief Release an index and its internal data.
 * @param index the index to release.
 */
void VecSimIndex_Free(VecSimIndex *index);

/**
 * @brief Add a vector to an index.
 * @param index the index to which the vector is added.
 * @param blob binary representation of the vector. Blob size should match the index data type and
 * dimension.
 * @param label the label of the added vector
 * @return the number of new vectors inserted (1 for new insertion, 0 for override).
 */
int VecSimIndex_AddVector(VecSimIndex *index, const void *blob, size_t label);

/**
 * @brief Remove a vector from an index.
 * @param index the index from which the vector is removed.
 * @param label the label of the removed vector
 * @return the number of vectors removed (0 if the label was not found)
 */
int VecSimIndex_DeleteVector(VecSimIndex *index, size_t label);

/**
 * @brief Calculate the distance of a vector from an index to a vector. This function assumes that
 * the vector fits the index - its type and dimension are the same as the index's, and if the
 * index's distance metric is cosine, the vector is already normalized.
 * @param index the index from which the first vector is located, and that defines the distance
 * metric.
 * @param label the label of the vector in the index.
 * @param blob binary representation of the second vector. Blob size should match the index data
 * type and dimension, and pre-normalized if needed.
 * @return The distance (according to the index's distance metric) between `blob` and the vector
 * with label  label`.
 */
double VecSimIndex_GetDistanceFrom(VecSimIndex *index, size_t label, const void *blob);

/**
 * @brief normalize the vector blob in place.
 * @param blob binary representation of a vector. Blob size should match the specified type and
 * dimension.
 * @param dim vector dimension.
 * @param type vector type.
 */
void VecSim_Normalize(void *blob, size_t dim, VecSimType type);

/**
 * @brief Return the number of vectors in the index.
 * @param index the index whose size is requested.
 * @return index size.
 */
size_t VecSimIndex_IndexSize(VecSimIndex *index);

/**
 * @brief Resolves VecSimRawParam array and generate VecSimQueryParams struct.
 * @param index the index whose size is requested.
 * @param rparams array of raw params to resolve.
 * @param paramNum number of params in rparams (or number of parames in rparams to resolve).
 * @param qparams pointer to VecSimQueryParams struct to set.
 * @param query_type indicates if query is hybrid, range or "standard" VSS query.
 * @return VecSim_OK if the resolve was successful, VecSimResolveCode error code if not.
 */
VecSimResolveCode VecSimIndex_ResolveParams(VecSimIndex *index, VecSimRawParam *rparams,
                                            int paramNum, VecSimQueryParams *qparams,
                                            VecsimQueryType query_type);

/**
 * @brief Search for the k closest vectors to a given vector in the index. The results can be
 * ordered by their score or id.
 * @param index the index to query in.
 * @param queryBlob binary representation of the query vector. Blob size should match the index data
 * type and dimension.
 * @param k the number of "nearest neighbours" to return (upper bound).
 * @param queryParams run time params for the search, which are algorithm-specific.
 * @param order the criterion to sort the results list by it. Options are by score, or by id.
 * @return An opaque object the represents a list of results. User can access the id and score
 * (which is the distance according to the index metric) of every result through
 * VecSimQueryResult_Iterator.
 */
VecSimQueryResult_List VecSimIndex_TopKQuery(VecSimIndex *index, const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams,
                                             VecSimQueryResult_Order);

/**
 * @brief Search for the vectors that are in a given range in the index with respect to a given
 * vector. The results can be ordered by their score or id.
 * @param index the index to query in.
 * @param queryBlob binary representation of the query vector. Blob size should match the index data
 * type and dimension.
 * @param radius the radius around the query vector to search vectors within it.
 * @param queryParams run time params for the search, which are algorithm-specific.
 * @param order the criterion to sort the results list by it. Options are by score, or by id.
 * @return An opaque object the represents a list of results. User can access the id and score
 * (which is the distance according to the index metric) of every result through
 * VecSimQueryResult_Iterator.
 */
VecSimQueryResult_List VecSimIndex_RangeQuery(VecSimIndex *index, const void *queryBlob,
                                              double radius, VecSimQueryParams *queryParams,
                                              VecSimQueryResult_Order);
/**
 * @brief Return index information.
 * @param index the index to return its info.
 * @return Index general and specific meta-data.
 */
VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index);

/**
 * @brief Return basic immutable index information.
 * @param index the index to return its info.
 * @return Index basic meta-data.
 */
VecSimIndexBasicInfo VecSimIndex_BasicInfo(VecSimIndex *index);

/**
 * @brief Returns an info iterator for generic reply purposes.
 *
 * @param index this index to return its info.
 * @return VecSimInfoIterator* An iterable containing the index general and specific meta-data.
 */
VecSimInfoIterator *VecSimIndex_InfoIterator(VecSimIndex *index);

/**
 * @brief Create a new batch iterator for a specific index, for a specific query vector,
 * using the Index_BatchIteratorNew method of the index. Should be released with
 * VecSimBatchIterator_Free call.
 * @param index the index in which the search will be done (in batches)
 * @param queryBlob binary representation of the vector. Blob size should match the index data type
 * and dimension.
 * @param queryParams run time params for the search, which are algorithm-specific.
 * @return Fresh batch iterator
 */
VecSimBatchIterator *VecSimBatchIterator_New(VecSimIndex *index, const void *queryBlob,
                                             VecSimQueryParams *queryParams);

/**
 * @brief Run async garbage collection for tiered async index.
 */
void VecSimTieredIndex_GC(VecSimIndex *index);

/**
 * @brief Return True if heuristics says that it is better to use ad-hoc brute-force
 * search over the index instead of using batch iterator.
 *
 * @param subsetSize the estimated number of vectors in the index that pass the filter
 * (that is, query results can be only from a subset of vector of this size).
 *
 * @param k the number of required results to return from the query.
 *
 * @param initial_check flag to indicate if this check is performed for the first time (upon
 * creating the hybrid iterator), or after running batches.
 */
bool VecSimIndex_PreferAdHocSearch(VecSimIndex *index, size_t subsetSize, size_t k,
                                   bool initial_check);

/**
 * @brief Allow 3rd party memory functions to be used for memory management.
 *
 * @param memoryfunctions VecSimMemoryFunctions struct.
 */
void VecSim_SetMemoryFunctions(VecSimMemoryFunctions memoryfunctions);

/**
 * @brief Allow 3rd party timeout callback to be used for limiting runtime of a query.
 *
 * @param callback timeoutCallbackFunction function. should get void* and return int.
 */
void VecSim_SetTimeoutCallbackFunction(timeoutCallbackFunction callback);

/**
 * @brief Allow 3rd party log callback to be used for logging.
 *
 * @param callback logCallbackFunction function. should get void* and return void.
 */
void VecSim_SetLogCallbackFunction(logCallbackFunction callback);

/**
 * @brief Allow 3rd party to set the write mode for tiered index - async insert/delete using
 * background jobs, or insert/delete inplace.
 *
 * @param mode VecSimWriteMode the mode in which we add/remove vectors (async or in-place).
 */
void VecSim_SetWriteMode(VecSimWriteMode mode);

#ifdef __cplusplus
}
#endif
