
#include "VecSim/vecsim.h"
#include "VecSim/algorithms/hnsw_c.h"

int cmpVecSimQueryResult(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    return res1->id > res2->id ?  1 :
           res1->id < res2->id ? -1 :
           0;
}

VecSimIndex* VecSimIndex_New(VecSimParams *params) {
    return HNSW_New(params);
}

inline int VecSimIndex_AddVector(VecSimIndex* index, const void* blob, size_t id) {
    return index->AddFn(index, blob, id);
}

inline int VecSimIndex_DeleteVector(VecSimIndex* index, size_t id) {
    return index->DeleteFn(index, id);
}

inline size_t VecSimIndex_IndexSize(VecSimIndex* index) {
    return index->SizeFn(index);
}

inline VecSimQueryResult* VecSimIndex_TopKQuery(VecSimIndex* index, const void* queryBlob, size_t k) {
    return index->TopKQueryFn(index, queryBlob, k);
}

VecSimQueryResult* VecSimIndex_TopKQueryByID(VecSimIndex* index, const void* queryBlob, size_t k) {
    VecSimQueryResult* results =  index->TopKQueryFn(index, queryBlob, k);
    qsort(results, k, sizeof(*results), (__compar_fn_t)cmpVecSimQueryResult);
    return results;
}

inline void VecSimIndex_Free(VecSimIndex *index) {
    return index->FreeFn(index);
}

// TODO

inline VecSimQueryResult* VecSimIndex_DistanceQuery(VecSimIndex* index, const void* queryBlob, float distance) {
    return index->DistanceQueryFn(index, queryBlob, distance);
}

inline void VecSimIndex_ClearDeleted(VecSimIndex* index) {
    index->ClearDeletedFn(index);
}
