#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Users should not access this struct directly, but with VecSimQueryResult_<X> API
typedef struct VecSimQueryResult VecSimQueryResult;

// An opaque object from which results can be obtained via iterator
typedef struct VecSimQueryResult_List VecSimQueryResult_List;

typedef struct VecSimQueryResult_Iterator VecSimQueryResult_Iterator;

typedef struct VecSimBatchIterator VecSimBatchIterator;

typedef enum { BY_SCORE, BY_ID } VecSimQueryResult_Order;

// Query results iterator API
size_t VecSimQueryResult_Len(VecSimQueryResult_List *results_iterator);

VecSimQueryResult_Iterator *VecSimQueryResult_GetIterator(VecSimQueryResult_List *results);

// Advance the iterator, so it will point to the next item, and return the value.
// If this is the last item, this will return NULL.
VecSimQueryResult *VecSimQueryResult_IteratorNext(VecSimQueryResult_Iterator *iterator);

bool VecSimQueryResult_IteratorHasNext(VecSimQueryResult_Iterator *iterator);

int VecSimQueryResult_GetId(VecSimQueryResult *item);

float VecSimQueryResult_GetScore(VecSimQueryResult *item);

void VecSimQueryResult_IteratorFree(VecSimQueryResult_Iterator *iterator);

void VecSimQueryResult_Free(VecSimQueryResult_List *results);

// Batch iterator API
VecSimBatchIterator *VecSimBatchIterator_New(VecSimIndex *index, const void *queryBlob);

VecSimQueryResult_List *VecSimBatchIterator_Next(VecSimBatchIterator *iterator,
                                                 size_t n_results,
                                                 VecSimQueryResult_Order order);

bool VecSimBatchIterator_HasNext(VecSimBatchIterator *iterator);

void VecSimBatchIterator_Free(VecSimBatchIterator *iterator);

void VecSimBatchIterator_Reset(VecSimBatchIterator *iterator);

#ifdef __cplusplus
}
#endif


VecSimQueryResult VecSimQueryResult_Create(size_t id, float score);

void VecSimQueryResult_SetId(VecSimQueryResult result, size_t id);

void VecSimQueryResult_SetScore(VecSimQueryResult result, float score);