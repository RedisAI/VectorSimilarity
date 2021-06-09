
#include "hnsw_c.h"
#include <deque>
#include <memory>
#include "hnswlib/hnswlib/hnswalg.h"

using namespace std;
using namespace hnswlib;

struct HNSWIndex {
    HNSWIndex(VecSimVecType vectype, VecSimMetric metric, size_t dim, size_t max_elements, 
        size_t M = 16, size_t ef_construction = 200);
        
    VecSimIndex base;
    unique_ptr<SpaceInterface<float>> space;
    HierarchicalNSW<float> hnsw;
};

#ifdef __cplusplus
extern "C" {
#endif

int HNSWIndex_AddVector(VecSimIndex *index, const void* vector_data, size_t id) {
    try {
        auto idx = reinterpret_cast<HNSWIndex*>(index);
        auto &hnsw = idx->hnsw;
        hnsw.addPoint(vector_data, id);
        return true;
    } catch (...) {
        return false;
    }
}

int HNSW_DeleteVector(VecSimIndex *index, size_t id) {
    try {
        auto idx = reinterpret_cast<HNSWIndex*>(index);
        auto &hnsw = idx->hnsw;
        hnsw.markDelete(id);
        return true;
    } catch (...) {
        return false;
    }
}

size_t HNSW_Size(VecSimIndex *index) {
    auto idx = reinterpret_cast<HNSWIndex*>(index);
    auto &hnsw = idx->hnsw;
    return hnsw.cur_element_count;
}

VecSimQueryResult *HNSW_TopKQuery(VecSimIndex *index, const void* query_data, size_t k) {
    try {
        auto idx = reinterpret_cast<HNSWIndex*>(index);
        auto &hnsw = idx->hnsw;
        typedef priority_queue<pair<float, size_t>> knn_queue_t;
        auto knn_res = make_unique<knn_queue_t>(std::move(hnsw.searchKnn(query_data, k)));
        auto *results = (VecSimQueryResult *) calloc(k, sizeof(VecSimQueryResult));
        for (int i = k - 1; i >= 0; --i) {
            results[i] = VecSimQueryResult{knn_res->top().second, knn_res->top().first};
            knn_res->pop();
        }
        return results;
    } catch (...) {
        return NULL;
    }
}

void HNSW_Free(VecSimIndex *index) {
    try {
        auto idx = reinterpret_cast<HNSWIndex*>(index);
        delete idx;
    } catch (...) {
    }
}

VecSimIndex *HNSW_New(VecSimAlgoParams *params, VecSimMetric VecSimMetric, VecSimVecType vectype, size_t vectorLen) {
    try {
        auto p = new HNSWIndex(vectype, VecSimMetric, vectorLen, params->hnswParams.
            initialCapacity, params->hnswParams.M, params->hnswParams.efConstuction);
        return &p->base;
    } catch (...) {
        return NULL;
    }
}

#ifdef __cplusplus
}
#endif

HNSWIndex::HNSWIndex(VecSimVecType vectype, VecSimMetric metric, size_t dim, size_t max_elements, 
        size_t M, size_t ef_construction) :
            space(metric == VecSimMetric_L2 ? static_cast<SpaceInterface<float>*>(new L2Space(dim)) : 
                static_cast<SpaceInterface<float>*>(new InnerProductSpace(dim))),
            hnsw(space.get(), max_elements, M, ef_construction)
{
    base = VecSimIndex{
        AddFn: HNSWIndex_AddVector, 
        DeleteFn: HNSW_DeleteVector,
        SizeFn: HNSW_Size,
        TopKQueryFn: HNSW_TopKQuery,
        DistanceQueryFn: NULL,
        ClearDeletedFn: NULL,
        FreeFn: HNSW_Free
    };
}
