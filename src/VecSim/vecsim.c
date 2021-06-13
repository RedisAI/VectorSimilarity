
#include "VecSim/vecsim.h"
#include "VecSim/algorithms/hnsw_c.h"

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
