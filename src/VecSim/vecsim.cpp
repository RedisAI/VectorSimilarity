
#include "VecSim/vecsim.h"
#include "VecSim/algorithms/hnsw_c.h"
#include "VecSim/utils/arr_cpp.h"

int cmpVecSimQueryResult(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    return res1->id > res2->id ?  1 :
           res1->id < res2->id ? -1 :
           0;
}

extern "C" VecSimIndex* VecSimIndex_New(VecSimParams *params) {
    return HNSW_New(params);
}

extern "C" int VecSimIndex_AddVector(VecSimIndex* index, const void* blob, size_t id) {
    return index->AddFn(index, blob, id);
}

extern "C" int VecSimIndex_DeleteVector(VecSimIndex* index, size_t id) {
    return index->DeleteFn(index, id);
}

extern "C" size_t VecSimIndex_IndexSize(VecSimIndex* index) {
    return index->SizeFn(index);
}

extern "C"  VecSimQueryResult* VecSimIndex_TopKQuery(VecSimIndex* index, const void* queryBlob, size_t k) {
    return index->TopKQueryFn(index, queryBlob, k);
}

extern "C" VecSimQueryResult* VecSimIndex_TopKQueryByID(VecSimIndex* index, const void* queryBlob, size_t k) {
    VecSimQueryResult* results =  index->TopKQueryFn(index, queryBlob, k);
    qsort(results, k, sizeof(*results), (__compar_fn_t)cmpVecSimQueryResult);
    return results;
}

extern "C" void VecSimIndex_Free(VecSimIndex *index) {
    return index->FreeFn(index);
}

// TODO

extern "C" VecSimQueryResult* VecSimIndex_DistanceQuery(VecSimIndex* index, const void* queryBlob, float distance) {
    return index->DistanceQueryFn(index, queryBlob, distance);
}

extern "C" void VecSimIndex_ClearDeleted(VecSimIndex* index) {
    index->ClearDeletedFn(index);
}

extern "C" size_t VecSimQueryResult_Len(VecSimQueryResult* result) {
    return array_len(result);
}

extern "C" void VecSimQueryResult_Free(VecSimQueryResult* result) {
    array_free(result);
}
