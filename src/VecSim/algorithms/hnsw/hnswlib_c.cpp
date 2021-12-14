#include "VecSim/algorithms/hnsw/hnswlib_c.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/hnsw/hnsw_batch_iterator.h"


#include <deque>
#include <memory>
#include <cassert>

using namespace std;
using namespace hnswlib;

/******************** Ctor / Dtor **************/

HNSWIndex::HNSWIndex(const VecSimParams *params, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndex(params, allocator),
      space(params->metric == VecSimMetric_L2
                ? static_cast<SpaceInterface<float> *>(new (allocator)
                                                           L2Space(params->size, allocator))
                : static_cast<SpaceInterface<float> *>(
                      new (allocator) InnerProductSpace(params->size, allocator))),
      hnsw(space.get(), params->hnswParams.initialCapacity, allocator,
           params->hnswParams.M ? params->hnswParams.M : HNSW_DEFAULT_M,
           params->hnswParams.efConstruction ? params->hnswParams.efConstruction
                                             : HNSW_DEFAULT_EF_C) {
    hnsw.setEf(params->hnswParams.efRuntime ? params->hnswParams.efRuntime : HNSW_DEFAULT_EF_RT);
}

/******************** Implementation **************/
int HNSWIndex::addVector(const void *vector_data, size_t id) {
    try {
        if (hnsw.getIndexSize() == this->hnsw.getIndexCapacity()) {
            this->hnsw.resizeIndex(std::max<size_t>(this->hnsw.getIndexCapacity() * 2, 2));
        }
        this->hnsw.addPoint(vector_data, id);
        return true;
    } catch (...) {
        return false;
    }
}

int HNSWIndex::deleteVector(size_t id) { return this->hnsw.removePoint(id); }

size_t HNSWIndex::indexSize() const { return this->hnsw.getIndexSize(); }

void HNSWIndex::setEf(size_t ef) { this->hnsw.setEf(ef); }

VecSimQueryResult_List HNSWIndex::topKQuery(const void *query_data, size_t k,
                                            VecSimQueryParams *queryParams) {
    try {
        // Get original efRuntime and store it.
        size_t originalEF = this->hnsw.getEf();

        if (queryParams) {
            if (queryParams->hnswRuntimeParams.efRuntime != 0) {
                hnsw.setEf(queryParams->hnswRuntimeParams.efRuntime);
            }
        }
        typedef vecsim_stl::priority_queue<pair<float, size_t>> knn_queue_t;
        auto knn_res = make_unique<knn_queue_t>(std::move(hnsw.searchKnn(query_data, k)));
        auto *results = array_new_len<VecSimQueryResult>(knn_res->size(), knn_res->size());
        for (int i = (int)knn_res->size() - 1; i >= 0; --i) {
            VecSimQueryResult_SetId(results[i], knn_res->top().second);
            VecSimQueryResult_SetScore(results[i], knn_res->top().first);
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

VecSimIndexInfo HNSWIndex::info() {

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_HNSWLIB;
    info.d = this->dim;
    info.type = this->vecType;
    info.metric = this->metric;
    info.hnswInfo.M = this->hnsw.getM();
    info.hnswInfo.efConstruction = this->hnsw.getEfConstruction();
    info.hnswInfo.efRuntime = this->hnsw.getEf();
    info.hnswInfo.indexSize = this->hnsw.getIndexSize();
    info.hnswInfo.levels = this->hnsw.getMaxLevel();
    info.hnswInfo.entrypoint = this->hnsw.getEntryPointLabel();
    info.memory = this->allocator->getAllocationSize();
    return info;
}

tableint HNSWIndex::getEntryPointId() const {
    return hnsw.getEntryPointId();
}

VecSimBatchIterator *HNSWIndex::newBatchIterator(const void *queryBlob) {
    return new (this->allocator)HNSW_BatchIterator(queryBlob, this, this->allocator);
}
