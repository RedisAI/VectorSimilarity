#include "vecsim.h"
#include "algorithms/hnsw_c.h"

VecSimIndex* VecSimIndex_New(AlgorithmParams params, DISTANCE_METRIC distanceMetric, VECTOR_TYPE vectorType, size_t vectorLen) {
    return HNSW_New(params, distanceMetric, vectorLen, vectorLen);
}

inline int VecSimIndex_AddVector(VecSimIndex* index, const void* blob, size_t id) {
    index->AddFn(index, blob, id);
}

inline int VecSimIndex_DeleteVector(VecSimIndex* index, size_t id) {
    index->DeleteFn(index, id);
}

inline size_t VecSimIndex_IndexSize(VecSimIndex* index) {
    index->SizeFn(index);
}

inline QueryResult* VecSimIndex_TopKQuery(VecSimIndex* index, const void* queryBlob, size_t k) {
    index->TopKQueryFn(index, queryBlob, k);
}

inline void VecSimIndex_Free(VecSimIndex *index) {
    index->FreeFn(index);
}

// TODO

inline QueryResult* VecSimIndex_DistnaceKQuery(VecSimIndex* index, const void* queryBlob, float distance) {
    index ->DistanceQueryFn(index, queryBlob, distance);
}

inline void VecSimIndex_ClearDeleted(VecSimIndex* index) {
    index->ClearDeletedFn(index);
}
