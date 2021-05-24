#include "hnsw_c.h"
#include "hnswlib/hnswalg.h"

using namespace std;
using namespace hnswlib;

struct HNSWIndex {
    VecSimIndex base;
    void* hnsw;
    void* space;
};

#ifdef __cplusplus
extern "C" {
#endif

static SpaceInterface<float>* HNSW_CreateDistanceSpace(DISTANCE_METRIC distanceMetric, size_t dim){
    if(distanceMetric == L2) {
        return new L2Space(dim);
    }
    else {
        return new InnerProductSpace(dim);
    }
}

static void* HNSW_CreateIndex(VECTOR_TYPE vectorType, SpaceInterface<float> *space, size_t max_elements, size_t M = 16, size_t ef_construction = 200) {
    // TODO support more types.
    return new HierarchicalNSW<float>(space, max_elements, M, ef_construction);
}

int HNSWIndex_AddVector(VecSimIndex *index, const void* vector_data, size_t id) {
    try {
        HierarchicalNSW<float>* hnsw = (HierarchicalNSW<float>*)((HNSWIndex*)index)->hnsw;
        hnsw->addPoint(vector_data, id);
    } catch (exception &ex) {
        return false;
    }
    return true;
}

int HNSW_DeleteVector(VecSimIndex *index, size_t id) {
    try {
        HierarchicalNSW<float>* hnsw = (HierarchicalNSW<float>*)((HNSWIndex*)index)->hnsw;
        hnsw->markDelete(id);
    } catch (exception &ex) {
        return false;
    }
    return true;
}

size_t HNSW_Size(VecSimIndex *index) {
    HierarchicalNSW<float>* hnsw = (HierarchicalNSW<float>*)((HNSWIndex*)index)->hnsw;
    return hnsw->cur_element_count;
}

QueryResult *HNSW_TopKQuery(VecSimIndex *index, const void* query_data, size_t k) {
    HierarchicalNSW<float>* hnsw = (HierarchicalNSW<float>*)((HNSWIndex*)index)->hnsw;
    priority_queue<pair<float, size_t>> knn_res = hnsw->searchKnn(query_data, k);
    auto *results = (QueryResult *)calloc(k ,sizeof(QueryResult));
    for (int i = k-1; i >= 0; --i) {
        results[i] = QueryResult {knn_res.top().second, knn_res.top().first};
        knn_res.pop();
    }
    return results;
}

void HNSW_Free(VecSimIndex *index) {
    HNSWIndex *hnswIndex = (HNSWIndex*)index;
    delete (HierarchicalNSW<float>*)hnswIndex->hnsw;
    delete (SpaceInterface<float>*)hnswIndex->space;
    delete hnswIndex;
}

VecSimIndex *HNSW_New(AlgorithmParams params, DISTANCE_METRIC distanceMetric, VECTOR_TYPE vectorType, size_t vectorLen) {

    auto space = HNSW_CreateDistanceSpace(distanceMetric, vectorLen);

    auto hnsw = HNSW_CreateIndex(vectorType, space, params.hnswParams.initialSize, params.hnswParams.M, params.hnswParams.efConstuction);
    VecSimIndex base = {
        .AddFn = HNSWIndex_AddVector, 
        .DeleteFn = HNSW_DeleteVector,
        .SizeFn = HNSW_Size,
        .TopKQueryFn = HNSW_TopKQuery,
        .FreeFn = HNSW_Free,

    };
    return &(new HNSWIndex{base, hnsw, space})->base;
}

#ifdef __cplusplus
}
#endif
