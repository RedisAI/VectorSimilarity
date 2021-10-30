#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include "query_results.h"
#include "vec_sim_common.h"

typedef struct VecSimIndex VecSimIndex;

/**
 * @brief abstract struct that represents an index. The supported index types use
 * this structure as the base, and add their specific functionality on top of it.
 */

/**
 * @brief Create a new VecSim index based on the given params.
 * @param params index configurations (initial size, data type, dimension, metric, algorithm and the
 * algorithm-related params).
 * @return A pointer to the created index.
 */
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
 * dimension.
 * @param id the id of the added vector
 * @return always returns true
 */
int VecSimIndex_AddVector(VecSimIndex *index, const void *blob, size_t id);

/**
 * @brief Remove a vector from an index using its DeleteFn.
 * @param index the index from which the vector is removed.
 * @param id the id of the removed vector
 * @return always returns true
 */
int VecSimIndex_DeleteVector(VecSimIndex *index, size_t id);

/**
 * @brief Return the number of vectors in the index using irs SizeFn.
 * @param index the index whose size is requested.
 * @return index size.
 */
size_t VecSimIndex_IndexSize(VecSimIndex *index);

/**
 * @brief Search for the k closest vectors to a given vector in the index using the
 * index TopKQuery callback. The results are ordered by their score.
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

// TODO?
VecSimQueryResult_List VecSimIndex_DistanceQuery(VecSimIndex *index, const void *queryBlob,
                                                 float distance, VecSimQueryParams *queryParams);

/**
 * @brief Return index information.
 * @param index the index to return its info.
 * @return Index general and specific meta-data.
 */
VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index);

/**
 * @brief Create a new batch iterator for a specific index, for a specific query vector,
 * using the Index_BatchIteratorNew method of the index. Should be released with
 * VecSimBatchIterator_Free call.
 * @param index the index in which the search will be done (in batches)
 * @param queryBlob binary representation of the vector. Blob size should match the index data type
 * and dimension.
 * @return Fresh batch iterator
 */
VecSimBatchIterator *VecSimBatchIterator_New(VecSimIndex *index, const void *queryBlob);

#ifdef __cplusplus
}
#endif
