
#include "VecSim/vecsim.h"
#include "VecSim/algorithms/brute_force.h"
#include "VecSim/algorithms/hnsw/hnswlib_c.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/vec_utils.h"
#include "memory.h"

struct VecSimQueryResult_Iterator {
    VecSimQueryResult *curr_result;
    size_t index;
    size_t results_len;
};

struct VecSimBatchIterator {
    VecSimIndex *index;
    unsigned char iterator_id;
    size_t returned_results_count;

    BatchIterator_Next IteratorNext;
    BatchIterator_Free IteratorFree;
    BatchIterator_Reset IteratorReset;
};

int cmpVecSimQueryResult(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    return res1->id > res2->id ? 1 : res1->id < res2->id ? -1 : 0;
}

extern "C" VecSimIndex *VecSimIndex_New(const VecSimParams *params) {
    if (params->algo == VecSimAlgo_HNSWLIB) {
        return HNSWLib_New(params);
    }
    return BruteForce_New(params);
}

extern "C" int VecSimIndex_AddVector(VecSimIndex *index, const void *blob, size_t id) {
    if (index->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        float normalized_blob[index->dim];
        memcpy(normalized_blob, blob, index->dim * sizeof(float));
        float_vector_normalize(normalized_blob, index->dim);
        return index->AddFn(index, normalized_blob, id);
    }
    return index->AddFn(index, blob, id);
}

extern "C" int VecSimIndex_DeleteVector(VecSimIndex *index, size_t id) {
    return index->DeleteFn(index, id);
}

extern "C" size_t VecSimIndex_IndexSize(VecSimIndex *index) { return index->SizeFn(index); }

extern "C" VecSimQueryResult_Collection *VecSimIndex_TopKQuery(VecSimIndex *index,
                                                               const void *queryBlob, size_t k,
                                                               VecSimQueryParams *queryParams) {
    if (index->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        float normalized_blob[index->dim];
        memcpy(normalized_blob, queryBlob, index->dim * sizeof(float));
        float_vector_normalize(normalized_blob, index->dim);
        return index->TopKQueryFn(index, normalized_blob, k, queryParams);
    }
    return index->TopKQueryFn(index, queryBlob, k, queryParams);
}

extern "C" VecSimQueryResult_Collection *VecSimIndex_TopKQueryByID(VecSimIndex *index,
                                                                   const void *queryBlob, size_t k,
                                                                   VecSimQueryParams *queryParams) {
    VecSimQueryResult_Collection *results = VecSimIndex_TopKQuery(index, queryBlob, k, queryParams);
    qsort(results, VecSimQueryResult_Len(results), sizeof(VecSimQueryResult),
          (__compar_fn_t)cmpVecSimQueryResult);
    return results;
}

extern "C" void VecSimIndex_Free(VecSimIndex *index) { return index->FreeFn(index); }

extern "C" VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index) { return index->InfoFn(index); }

// TODO?
extern "C" VecSimQueryResult_Collection *VecSimIndex_DistanceQuery(VecSimIndex *index,
                                                                   const void *queryBlob,
                                                                   float distance,
                                                                   VecSimQueryParams *queryParams) {
    return index->DistanceQueryFn(index, queryBlob, distance, queryParams);
}

extern "C" size_t VecSimQueryResult_Len(VecSimQueryResult_Collection *result) {
    return array_len((VecSimQueryResult *)result);
}

extern "C" void VecSimQueryResult_Free(VecSimQueryResult_Collection *result) {
    array_free((VecSimQueryResult *)result);
}

extern "C" VecSimQueryResult_Iterator *
VecSimQueryResult_GetIterator(VecSimQueryResult_Collection *results) {
    if (VecSimQueryResult_Len(results) == 0) {
        return nullptr;
    }
    return new VecSimQueryResult_Iterator{(VecSimQueryResult *)results, 0,
                                          VecSimQueryResult_Len(results)};
}

extern "C" bool VecSimQueryResult_IteratorHasNext(VecSimQueryResult_Iterator *iterator) {
    if (iterator->index == iterator->results_len) {
        return false;
    }
    return true;
}

extern "C" VecSimQueryResult *VecSimQueryResult_IteratorNext(VecSimQueryResult_Iterator *iterator) {
    if (iterator->index == iterator->results_len) {
        return nullptr;
    }
    VecSimQueryResult *item = iterator->curr_result++;
    iterator->index++;

    return item;
}

extern "C" int VecSimQueryResult_GetId(VecSimQueryResult *res) {
    if (res == nullptr) {
        return -1;
    }
    return (int)res->id;
}

extern "C" float VecSimQueryResult_GetScore(VecSimQueryResult *res) {
    if (res == nullptr) {
        return -1;
    }
    return (float)res->score;
}

extern "C" void VecSimQueryResult_IteratorFree(VecSimQueryResult_Iterator *iterator) {
    delete iterator;
}
