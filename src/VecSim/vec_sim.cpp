#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/algorithms/brute_force.h"
#include "VecSim/algorithms/hnsw/hnswlib_c.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/vec_utils.h"
#include "memory.h"

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

extern "C" VecSimQueryResult_List *VecSimIndex_TopKQuery(VecSimIndex *index, const void *queryBlob,
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

extern "C" VecSimQueryResult_List *VecSimIndex_TopKQueryByID(VecSimIndex *index,
                                                             const void *queryBlob, size_t k,
                                                             VecSimQueryParams *queryParams) {
    VecSimQueryResult_List *results = VecSimIndex_TopKQuery(index, queryBlob, k, queryParams);
    qsort(results, VecSimQueryResult_Len(results), sizeof(VecSimQueryResult),
          (__compar_fn_t)cmpVecSimQueryResult);
    return results;
}

extern "C" void VecSimIndex_Free(VecSimIndex *index) { return index->FreeFn(index); }

extern "C" VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index) { return index->InfoFn(index); }

// TODO?
extern "C" VecSimQueryResult_List *VecSimIndex_DistanceQuery(VecSimIndex *index,
                                                             const void *queryBlob, float distance,
                                                             VecSimQueryParams *queryParams) {
    return index->DistanceQueryFn(index, queryBlob, distance, queryParams);
}
