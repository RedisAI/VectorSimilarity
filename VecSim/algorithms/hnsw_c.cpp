
#include "hnsw_c.h"
#include <deque>
#include "hnswlib/hnswlib/hnswalg.h"

using namespace std;
using namespace hnswlib;

#ifdef __cplusplus
extern "C" {
#endif

struct HNSWIndex {
    VecSimIndex base;
    void* hnsw;
    void* space;
};

static SpaceInterface<float>* HNSW_CreateDistanceSpace(VecSimMetric metric, size_t dim) {
    if(metric == VecSimMetric_L2) {
        return new L2Space(dim);
    }
    else {
        return new InnerProductSpace(dim);
    }
}

static void* HNSW_CreateIndex(VecSimVecType vectype, SpaceInterface<float> *space, size_t max_elements, size_t M = 16, size_t ef_construction = 200) {
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

VecSimQueryResult *HNSW_TopKQuery(VecSimIndex *index, const void* query_data, size_t k) {
    HierarchicalNSW<float>* hnsw = (HierarchicalNSW<float>*)((HNSWIndex*)index)->hnsw;
    priority_queue<pair<float, size_t>> knn_res = hnsw->searchKnn(query_data, k);
    auto *results = (VecSimQueryResult *)calloc(k ,sizeof(VecSimQueryResult));
    for (int i = k-1; i >= 0; --i) {
        results[i] = VecSimQueryResult {knn_res.top().second, knn_res.top().first};
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

VecSimIndex *HNSW_New(VecSimAlgoParams *params, VecSimMetric VecSimMetric, VecSimVecType vectype, size_t vectorLen) {

    auto space = HNSW_CreateDistanceSpace(VecSimMetric, vectorLen);

    auto hnsw = HNSW_CreateIndex(vectype, space, params->hnswParams.initialCapacity, params->hnswParams.M, params->hnswParams.efConstuction);
    VecSimIndex base = {
        AddFn: HNSWIndex_AddVector, 
        DeleteFn: HNSW_DeleteVector,
        SizeFn: HNSW_Size,
        TopKQueryFn: HNSW_TopKQuery,
        DistanceQueryFn: NULL,
        ClearDeletedFn: NULL,
        FreeFn: HNSW_Free
    };
	
	auto p = new HNSWIndex{base, hnsw, space};
    return &p->base;
}

#ifdef __cplusplus
}
#endif
