
#include "hnsw_c.h"
#include <deque>
#include <memory>
#include "VecSim/utils/arr_cpp.h"
#include "hnswlib/hnswlib/hnswalg.h"

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

int HNSWIndex_AddVector(VecSimIndex *index, const void *vector_data, size_t id) {
    try {
        auto idx = reinterpret_cast<HNSWIndex *>(index);
        auto &hnsw = idx->hnsw;
        if (hnsw.cur_element_count == hnsw.max_elements_) {
            hnsw.resizeIndex(hnsw.max_elements_ * 2);
        }
        hnsw.addPoint(vector_data, id);
        return true;
    } catch (...) {
        return false;
    }
}

int HNSW_DeleteVector(VecSimIndex *index, size_t id) {
    try {
        auto idx = reinterpret_cast<HNSWIndex *>(index);
        auto &hnsw = idx->hnsw;
        hnsw.markDelete(id);
        return true;
    } catch (...) {
        return false;
    }
}

size_t HNSW_Size(VecSimIndex *index) {
    auto idx = reinterpret_cast<HNSWIndex *>(index);
    auto &hnsw = idx->hnsw;
    return hnsw.cur_element_count;
}

VecSimQueryResult *HNSW_TopKQuery(VecSimIndex *index, const void *query_data, size_t k,
                                  VecSimQueryParams *queryParams) {
    try {
        auto idx = reinterpret_cast<HNSWIndex *>(index);
        auto &hnsw = idx->hnsw;
        // Get original efRuntime and store it.
        size_t originalEF = hnsw.ef_;

        if (queryParams) {
            if (queryParams->hnswRuntimeParams.efRuntime != 0) {
                hnsw.setEf(queryParams->hnswRuntimeParams.efRuntime);
            }
        }

        typedef priority_queue<pair<float, size_t>> knn_queue_t;
        auto knn_res = make_unique<knn_queue_t>(std::move(hnsw.searchKnn(query_data, k)));
        VecSimQueryResult *results =
            array_new_len<VecSimQueryResult>(knn_res->size(), knn_res->size());
        for (int i = k - 1; i >= 0; --i) {
            results[i] = VecSimQueryResult{knn_res->top().second, knn_res->top().first};
            knn_res->pop();
        }
        // Restore efRuntime
        hnsw.setEf(originalEF);
        assert(hnsw.ef_ == originalEF);

        return results;
    } catch (...) {
        return NULL;
    }
}

void HNSW_Free(VecSimIndex *index) {
    try {
        auto idx = reinterpret_cast<HNSWIndex *>(index);
        delete idx;
    } catch (...) {
    }
}

VecSimIndex *HNSW_New(VecSimParams *params) {
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

VecSimIndexInfo HNSW_Info(VecSimIndex *index) {
    auto idx = reinterpret_cast<HNSWIndex *>(index);
    auto &hnsw = idx->hnsw;

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_HNSWLIB;
    info.d = *((size_t *)idx->space.get()->get_dist_func_param());
    info.type = VecSimType_FLOAT32;
    info.hnswInfo.M = hnsw.M_;
    info.hnswInfo.efConstruction = hnsw.ef_construction_;
    info.hnswInfo.efRuntime = hnsw.ef_;
    info.hnswInfo.indexSize = hnsw.cur_element_count;
    info.hnswInfo.levels = hnsw.maxlevel_;
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
        AddFn : HNSWIndex_AddVector,
        DeleteFn : HNSW_DeleteVector,
        SizeFn : HNSW_Size,
        TopKQueryFn : HNSW_TopKQuery,
        DistanceQueryFn : NULL,
        ClearDeletedFn : NULL,
        FreeFn : HNSW_Free,
        InfoFn : HNSW_Info
    };
}
