#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// The possible ordering for results that return from a query
typedef enum { BY_SCORE, BY_ID } VecSimQueryResult_Order;

/**
 * @brief A single query result. This is an opaque object from which a user can get the result
 * vector id and score (comparing to the query vector).
 */
typedef struct VecSimQueryResult VecSimQueryResult;

/**
 * @brief Get the id of the result vector. If item is nullptr, return INVALID_ID (defined as the
 * -1).
 */
long VecSimQueryResult_GetId(VecSimQueryResult *item);

/**
 * @brief Get the score of the result vector. If item is nullptr, return INVALID_SCORE (defined as
 * the minimal value of float).
 */
float VecSimQueryResult_GetScore(VecSimQueryResult *item);

/**
 * @brief An opaque object from which results can be obtained via iterator.
 */
typedef VecSimQueryResult *VecSimQueryResult_List;

/**
 * @brief Iterator for going over the list of results that had returned form a query
 */
typedef struct VecSimQueryResult_Iterator VecSimQueryResult_Iterator;

/**
 * @brief Get the length of the result list that returned from a query.
 */
size_t VecSimQueryResult_Len(VecSimQueryResult_List results);

/**
 * @brief Create an iterator for going over the list of results. The iterator needs to be free
 * with VecSimQueryResult_IteratorFree.
 */
VecSimQueryResult_Iterator *VecSimQueryResult_List_GetIterator(VecSimQueryResult_List results);

/**
 * @brief Advance the iterator, so it will point to the next item, and return the value.
 * The first call will return the first result. This will return NULL once the iterator is depleted.
 */
VecSimQueryResult *VecSimQueryResult_IteratorNext(VecSimQueryResult_Iterator *iterator);

/**
 * @brief Return true while the iterator points to some result, false if it is depleted.
 */
bool VecSimQueryResult_IteratorHasNext(VecSimQueryResult_Iterator *iterator);

/**
 * @brief Release the iterator
 */
void VecSimQueryResult_IteratorFree(VecSimQueryResult_Iterator *iterator);

/**
 * @brief Release the entire query results list.
 */
void VecSimQueryResult_Free(VecSimQueryResult_List results);

/**
 * @brief Iterator for running the same query over an index, getting the in each iteration
 * the best results that hasn't returned in the previous iterations.
 */
typedef struct VecSimBatchIterator VecSimBatchIterator;

// Abstract batch iterator methods
typedef VecSimQueryResult_List *(*BatchIterator_Next)(VecSimBatchIterator *iterator,
                                                      size_t n_results);
typedef bool (*BatchIterator_HasNext)(VecSimBatchIterator *iterator);
typedef void (*BatchIterator_Free)(VecSimBatchIterator *iterator);
typedef void *(*BatchIterator_Reset)(VecSimBatchIterator *iterator);

/**
 * @brief Run TopKQuery over the underling index of the given iterator using BatchIterator_Next
 * method, and return n_results new results.
 * @param iterator the iterator that olds the current state of this "batched search".
 * @param n_results number of new results to return.
 * @param order enum - determine the returned results order (by id or by score).
 * @return List of (at most) new n_results vectors which are the "nearest neighbours" to the
 * underline query vector in the iterator.
 */
VecSimQueryResult_List *VecSimBatchIterator_Next(VecSimBatchIterator *iterator, size_t n_results,
                                                 VecSimQueryResult_Order order);

/**
 * @brief Return true while the iterator has new results to return, false if it is depleted
 * (using BatchIterator_HasNext method) .
 */
bool VecSimBatchIterator_HasNext(VecSimBatchIterator *iterator);

/**
 * @brief Release the iterator using BatchIterator_Free method
 */
void VecSimBatchIterator_Free(VecSimBatchIterator *iterator);

/**
 * @brief Reset the iterator - back to the initial state using BatchIterator_Reset method.
 */
void VecSimBatchIterator_Reset(VecSimBatchIterator *iterator);

#ifdef __cplusplus
}
#endif
