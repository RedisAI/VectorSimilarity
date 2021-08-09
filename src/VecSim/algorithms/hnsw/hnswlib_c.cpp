#include "VecSim/algorithms/hnsw/hnswlib_c.h"
#include <deque>
#include <memory>
#include <cassert>
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/hnsw/hnsw_impl.h"
#include "VecSim/spaces/L2.h"
#include "VecSim/spaces/internal_product.h"

using namespace std;
using namespace hnswlib;

struct HNSWIndex {
    HNSWIndex(VecSimType vectype, VecSimMetric metric, size_t dim, size_t max_elements,
              size_t M = HNSW_DEFAULT_M, size_t ef_construction = HNSW_DEFAULT_EF_C,
              size_t ef_runtime = HNSW_DEFAULT_EF_RT);

    VecSimIndex base;
    unique_ptr<SpaceInterface<float>> space;
    HierarchicalNSW<float> hnsw;
};

#ifdef __cplusplus
extern "C" {
#endif

int HNSWLib_AddVector(VecSimIndex *index, const void *vector_data, size_t id) {
    try {
        auto idx = reinterpret_cast<HNSWIndex *>(index);
        auto &hnsw = idx->hnsw;
        if (hnsw.getIndexSize() == hnsw.getIndexCapacity()) {
            hnsw.resizeIndex(hnsw.getIndexCapacity() * 2);
        }
        hnsw.addPoint(vector_data, id);
        return true;
    } catch (...) {
        return false;
    }
}

int HNSWLib_DeleteVector(VecSimIndex *index, size_t id) {
    auto idx = reinterpret_cast<HNSWIndex *>(index);
    auto &hnsw = idx->hnsw;
    return hnsw.removePoint(id);
}

size_t HNSWLib_Size(VecSimIndex *index) {
    auto idx = reinterpret_cast<HNSWIndex *>(index);
    auto &hnsw = idx->hnsw;
    return hnsw.getIndexSize();
}

VecSimQueryResult *HNSWLib_TopKQuery(VecSimIndex *index, const void *query_data, size_t k,
                                  VecSimQueryParams *queryParams) {
    try {
        auto idx = reinterpret_cast<HNSWIndex *>(index);
        auto &hnsw = idx->hnsw;
        // Get original efRuntime and store it.
        size_t originalEF = hnsw.getEf();

        if (queryParams) {
            if (queryParams->hnswRuntimeParams.efRuntime != 0) {
                hnsw.setEf(queryParams->hnswRuntimeParams.efRuntime);
            }
        }

        typedef priority_queue<pair<float, size_t>> knn_queue_t;
        auto knn_res = make_unique<knn_queue_t>(std::move(hnsw.searchKnn(query_data, k)));
        auto *results = array_new_len<VecSimQueryResult>(knn_res->size(), knn_res->size());
        for (int i = k - 1; i >= 0; --i) {
            results[i] = VecSimQueryResult{knn_res->top().second, knn_res->top().first};
            knn_res->pop();
        }
        // Restore efRuntime
        hnsw.setEf(originalEF);
        assert(hnsw.getEf() == originalEF);

        return results;
    } catch (...) {
        return NULL;
    }
}

void HNSWLib_Free(VecSimIndex *index) {
    try {
        auto idx = reinterpret_cast<HNSWIndex *>(index);
        delete idx;
    } catch (...) {
    }
}

VecSimIndex *HNSWLib_New(const VecSimParams *params) {
    try {
        auto p = new HNSWIndex(
            params->type, params->metric, params->size, params->hnswParams.initialCapacity,
            params->hnswParams.M ? params->hnswParams.M : HNSW_DEFAULT_M,
            params->hnswParams.efConstruction ? params->hnswParams.efConstruction
                                              : HNSW_DEFAULT_EF_C,
            params->hnswParams.efRuntime ? params->hnswParams.efRuntime : HNSW_DEFAULT_EF_RT);
        return &p->base;
    } catch (...) {
        return NULL;
    }
}

VecSimIndexInfo HNSWLib_Info(VecSimIndex *index) {
    auto idx = reinterpret_cast<HNSWIndex *>(index);
    auto &hnsw = idx->hnsw;

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_HNSWLIB;
    info.d = *((size_t *)idx->space->get_data_dim());
    info.type = VecSimType_FLOAT32;
    info.hnswInfo.M = hnsw.getM();
    info.hnswInfo.efConstruction = hnsw.getEfConstruction();
    info.hnswInfo.efRuntime = hnsw.getEf();
    info.hnswInfo.indexSize = hnsw.getIndexSize();
    info.hnswInfo.levels = hnsw.getMaxLevel();
    return info;
}

#ifdef __cplusplus
}
#endif

HNSWIndex::HNSWIndex(VecSimType vectype, VecSimMetric metric, size_t dim, size_t max_elements,
                     size_t M, size_t ef_construction, size_t ef_runtime)
    : space(metric == VecSimMetric_L2
                ? static_cast<SpaceInterface<float> *>(new L2Space(dim))
                : static_cast<SpaceInterface<float> *>(new InnerProductSpace(dim))),
      hnsw(space.get(), max_elements, M, ef_construction) {
    hnsw.setEf(ef_runtime);
    base = VecSimIndex{
        .AddFn =  HNSWLib_AddVector,
        .DeleteFn =  HNSWLib_DeleteVector,
        .SizeFn =  HNSWLib_Size,
        .TopKQueryFn =  HNSWLib_TopKQuery,
        .DistanceQueryFn =  NULL,
        .ClearDeletedFn =  NULL,
        .FreeFn =  HNSWLib_Free,
        .InfoFn =  HNSWLib_Info
    };
}
