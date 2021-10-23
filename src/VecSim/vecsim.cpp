
#include "VecSim/vecsim.h"
#include "VecSim/algorithms/brute_force.h"
#include "VecSim/algorithms/hnsw/hnswlib_c.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/vec_utils.h"
#include "memory.h"

struct VecSimQueryResults_Iterator {
    VecSimQueryResults_Item *curr_result;
    size_t index;
    size_t results_len;
};

int cmpVecSimQueryResult(const VecSimQueryResults_Item *res1, const VecSimQueryResults_Item *res2) {
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

extern "C" VecSimQueryResults *VecSimIndex_TopKQuery(VecSimIndex *index, const void *queryBlob,
                                                    size_t k, VecSimQueryParams *queryParams) {
    if (index->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        float normalized_blob[index->dim];
        memcpy(normalized_blob, queryBlob, index->dim * sizeof(float));
        float_vector_normalize(normalized_blob, index->dim);
        return index->TopKQueryFn(index, normalized_blob, k, queryParams);
    }
    return index->TopKQueryFn(index, queryBlob, k, queryParams);
}

extern "C" VecSimQueryResults *VecSimIndex_TopKQueryByID(VecSimIndex *index, const void *queryBlob,
                                                        size_t k, VecSimQueryParams *queryParams) {
    VecSimQueryResults *results = VecSimIndex_TopKQuery(index, queryBlob, k, queryParams);
    qsort(results, VecSimQueryResults_Len(results), sizeof(VecSimQueryResults_Item),
          (__compar_fn_t)cmpVecSimQueryResult);
    return results;
}

extern "C" void VecSimIndex_Free(VecSimIndex *index) { return index->FreeFn(index); }

extern "C" VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index) { return index->InfoFn(index); }

// TODO?
extern "C" VecSimQueryResults *VecSimIndex_DistanceQuery(VecSimIndex *index, const void *queryBlob,
                                                        float distance,
                                                        VecSimQueryParams *queryParams) {
    return index->DistanceQueryFn(index, queryBlob, distance, queryParams);
}

extern "C" size_t VecSimQueryResults_Len(VecSimQueryResults *result) {
    return array_len((VecSimQueryResults_Item *)result);
}

extern "C" void VecSimQueryResults_Free(VecSimQueryResults *result) {
    array_free((VecSimQueryResults_Item *)result);
}

extern "C" VecSimQueryResults_Iterator *VecSimQueryResults_GetIterator(VecSimQueryResults *results) {
    if (VecSimQueryResults_Len(results) == 0) {
        return nullptr;
    }
    return new VecSimQueryResults_Iterator{(VecSimQueryResults_Item *)results, 0, VecSimQueryResults_Len(results)};
}

extern "C" VecSimQueryResults_Iterator *VecSimQueryResults_IteratorNext(VecSimQueryResults_Iterator *iterator) {
    if (iterator == nullptr) {
        return nullptr;
    }
    iterator->curr_result++;
    iterator->index++;
    if (iterator->index == iterator->results_len) {
        delete iterator;
        return nullptr;
    }
    return iterator;
}

extern "C" int VecSimQueryResults_GetId(VecSimQueryResults_Iterator *iterator) {
    if (iterator == nullptr) {
        return -1;
    }
    return (int)iterator->curr_result->id;
}

extern "C" float VecSimQueryResults_GetScore(VecSimQueryResults_Iterator *iterator) {
    if (iterator == nullptr) {
        return -1;
    }
    return (float)iterator->curr_result->score;
}

extern "C" void VecSimQueryResults_IteratorFree(VecSimQueryResults_Iterator *iterator) {
    delete iterator;
}
