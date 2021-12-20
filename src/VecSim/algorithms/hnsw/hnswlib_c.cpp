#include "VecSim/algorithms/hnsw/hnswlib_c.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/query_result_struct.h"

#include <deque>
#include <memory>
#include <cassert>

using namespace std;
using namespace hnswlib;

/******************** Ctor / Dtor **************/

HNSWIndex::HNSWIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndex(allocator), dim(params->dim), vecType(params->type), metric(params->metric),
      space(params->metric == VecSimMetric_L2
                ? static_cast<SpaceInterface<float> *>(new (allocator)
                                                           L2Space(params->dim, allocator))
                : static_cast<SpaceInterface<float> *>(
                      new (allocator) InnerProductSpace(params->dim, allocator))),
      hnsw(space.get(), params->initialCapacity, allocator, params->M ? params->M : HNSW_DEFAULT_M,
           params->efConstruction ? params->efConstruction : HNSW_DEFAULT_EF_C) {
    hnsw.setEf(params->efRuntime ? params->efRuntime : HNSW_DEFAULT_EF_RT);
}

/******************** Implementation **************/
int HNSWIndex::addVector(const void *vector_data, size_t id) {
    try {
        float normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
        if (this->metric == VecSimMetric_Cosine) {
            // TODO: need more generic
            memcpy(normalized_data, vector_data, this->dim * sizeof(float));
            float_vector_normalize(normalized_data, this->dim);
            vector_data = normalized_data;
        }
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
        float normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
        if (this->metric == VecSimMetric_Cosine) {
            // TODO: need more generic
            memcpy(normalized_data, query_data, this->dim * sizeof(float));
            float_vector_normalize(normalized_data, this->dim);
            query_data = normalized_data;
        }
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
    info.hnswInfo.dim = this->dim;
    info.hnswInfo.type = this->vecType;
    info.hnswInfo.metric = this->metric;
    info.hnswInfo.M = this->hnsw.getM();
    info.hnswInfo.efConstruction = this->hnsw.getEfConstruction();
    info.hnswInfo.efRuntime = this->hnsw.getEf();
    info.hnswInfo.indexSize = this->hnsw.getIndexSize();
    info.hnswInfo.levels = this->hnsw.getMaxLevel();
    info.hnswInfo.entrypoint = this->hnsw.getEntryPointLabel();
    info.hnswInfo.memory = this->allocator->getAllocationSize();
    return info;
}

VecSimInfoIterator *HNSWIndex::infoIterator() {
    VecSimIndexInfo info = this->info();
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(1);

    infoIterator->addInfoField(
        {.fieldName = "ALGORITHM", .fieldType = INFOFIELD_STR, .stringValue = "HNSW"});
    infoIterator->addInfoField({.fieldName = "TYPE",
                                .fieldType = INFOFIELD_STR,
                                .stringValue = VecSimType_ToString(info.hnswInfo.type)});
    infoIterator->addInfoField({.fieldName = "DIMENSION",
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.hnswInfo.dim});
    infoIterator->addInfoField({.fieldName = "METRIC",
                                .fieldType = INFOFIELD_STR,
                                .stringValue = VecSimMetric_ToString(info.hnswInfo.metric)});
    infoIterator->addInfoField({.fieldName = "INDEX_SIZE",
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.hnswInfo.indexSize});
    infoIterator->addInfoField(
        {.fieldName = "M", .fieldType = INFOFIELD_UINT64, .uintegerValue = info.hnswInfo.M});
    infoIterator->addInfoField({.fieldName = "efConstruction",
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.hnswInfo.efConstruction});
    infoIterator->addInfoField({.fieldName = "efRuntime",
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.hnswInfo.efRuntime});
    infoIterator->addInfoField({.fieldName = "LEVELS",
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.hnswInfo.levels});
    infoIterator->addInfoField({.fieldName = "ENTRY_POINT",
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.hnswInfo.entrypoint});
    infoIterator->addInfoField({.fieldName = "MEMORY",
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.hnswInfo.memory});

    return infoIterator;
}

VecSimBatchIterator *HNSWIndex::newBatchIterator(const void *queryBlob) { return nullptr; }
