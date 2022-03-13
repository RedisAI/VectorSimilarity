#include "VecSim/algorithms/hnsw/hnsw_wrapper.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/vec_utils.h"
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

HNSWIndex::HNSWIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndex(allocator), dim(params->dim), vecType(params->type), metric(params->metric),
      space(params->metric == VecSimMetric_L2
                ? static_cast<SpaceInterface<float> *>(new (allocator)
                                                           L2Space(params->dim, allocator))
                : static_cast<SpaceInterface<float> *>(
                      new (allocator) InnerProductSpace(params->dim, allocator))),
      hnsw(new (allocator) hnswlib::HierarchicalNSW<float>(
          space.get(), params->initialCapacity, allocator, params->M ? params->M : HNSW_DEFAULT_M,
          params->efConstruction ? params->efConstruction : HNSW_DEFAULT_EF_C)),
      last_mode(EMPTY_MODE) {
    hnsw->setEf(params->efRuntime ? params->efRuntime : HNSW_DEFAULT_EF_RT);
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
        if (hnsw->getIndexSize() == this->hnsw->getIndexCapacity()) {
            this->hnsw->resizeIndex(std::max<size_t>(std::ceil(this->hnsw->getIndexCapacity() * 1.1), 2));
        }
        this->hnsw->addPoint(vector_data, id);
        return true;
    } catch (...) {
        return false;
    }
}

int HNSWIndex::deleteVector(size_t id) { return this->hnsw->removePoint(id); }

VecSimResolveCode HNSWIndex::resolveParams(VecSimRawParam *rparams, int paramNum,
                                           VecSimQueryParams *qparams) {
    if (!qparams || (!rparams && (paramNum != 0))) {
        return VecSimParamResolverErr_NullParam;
    }
    bzero(qparams, sizeof(VecSimQueryParams));
    for (int i = 0; i < paramNum; i++) {
        if ((rparams[i].nameLen == strlen(VecSimCommonStrings::HNSW_EF_RUNTIME_STRING)) &&
            (!strcasecmp(rparams[i].name, VecSimCommonStrings::HNSW_EF_RUNTIME_STRING))) {
            if (qparams->hnswRuntimeParams.efRuntime != 0) {
                return VecSimParamResolverErr_AlreadySet;
            } else {
                char *ep; // For checking that strtoll used all rparams[i].valLen chars.
                errno = 0;
                long long val = strtoll(rparams[i].value, &ep, 0);
                // Here we verify that val is positive and strtoll was successful.
                // The last test checks that the entire rparams[i].value was used.
                // We catch here inputs like "3.14", "123text" and so on.
                if (val <= 0 || val == LLONG_MAX || errno != 0 ||
                    (rparams[i].value + rparams[i].valLen) != ep) {
                    return VecSimParamResolverErr_BadValue;
                }
                qparams->hnswRuntimeParams.efRuntime = (size_t)val;
            }
        } else {
            return VecSimParamResolverErr_UnknownParam;
        }
    }
    return (VecSimResolveCode)VecSim_OK;
}

double HNSWIndex::getDistanceFrom(size_t label, const void *vector_data) {
    return this->hnsw->getDistanceByLabelFromPoint(label, vector_data);
}

size_t HNSWIndex::indexSize() const { return this->hnsw->getIndexSize(); }

void HNSWIndex::setEf(size_t ef) { this->hnsw->setEf(ef); }

VecSimQueryResult_List HNSWIndex::topKQuery(const void *query_data, size_t k,
                                            VecSimQueryParams *queryParams) {
    try {
        this->last_mode = STANDARD_KNN;
        float normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
        if (this->metric == VecSimMetric_Cosine) {
            // TODO: need more generic
            memcpy(normalized_data, query_data, this->dim * sizeof(float));
            float_vector_normalize(normalized_data, this->dim);
            query_data = normalized_data;
        }
        // Get original efRuntime and store it.
        size_t originalEF = this->hnsw->getEf();

        if (queryParams) {
            if (queryParams->hnswRuntimeParams.efRuntime != 0) {
                hnsw->setEf(queryParams->hnswRuntimeParams.efRuntime);
            }
        }
        auto knn_res = hnsw->searchKnn(query_data, k);
        auto *results = array_new_len<VecSimQueryResult>(knn_res.size(), knn_res.size());
        for (int i = (int)knn_res.size() - 1; i >= 0; --i) {
            VecSimQueryResult_SetId(results[i], knn_res.top().second);
            VecSimQueryResult_SetScore(results[i], knn_res.top().first);
            knn_res.pop();
        }
        // Restore efRuntime
        hnsw->setEf(originalEF);
        assert(hnsw->getEf() == originalEF);

        return results;
    } catch (...) {
        return NULL;
    }
}

VecSimIndexInfo HNSWIndex::info() const {

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_HNSWLIB;
    info.hnswInfo.dim = this->dim;
    info.hnswInfo.type = this->vecType;
    info.hnswInfo.metric = this->metric;
    info.hnswInfo.M = this->hnsw->getM();
    info.hnswInfo.efConstruction = this->hnsw->getEfConstruction();
    info.hnswInfo.efRuntime = this->hnsw->getEf();
    info.hnswInfo.indexSize = this->hnsw->getIndexSize();
    info.hnswInfo.max_level = this->hnsw->getMaxLevel();
    info.hnswInfo.entrypoint = this->hnsw->getEntryPointLabel();
    info.hnswInfo.memory = this->allocator->getAllocationSize();
    info.hnswInfo.last_mode = this->last_mode;
    return info;
}

VecSimBatchIterator *HNSWIndex::newBatchIterator(const void *queryBlob) {
    // As this is the only supported type, we always allocate 4 bytes for every element in the
    // vector.
    assert(this->vecType == VecSimType_FLOAT32);
    auto *queryBlobCopy = this->allocator->allocate(sizeof(float) * this->dim);
    memcpy(queryBlobCopy, queryBlob, dim * sizeof(float));
    if (metric == VecSimMetric_Cosine) {
        float_vector_normalize((float *)queryBlobCopy, dim);
    }
    // Ownership of queryBlobCopy moves to HNSW_BatchIterator that will free it at the end.
    return new (this->allocator) HNSW_BatchIterator(queryBlobCopy, this, this->allocator);
}

VecSimInfoIterator *HNSWIndex::infoIterator() {
    VecSimIndexInfo info = this->info();
    // For readability. Update this number when needed;
    size_t numberOfInfoFields = 12;
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields);

    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::ALGORITHM_STRING,
                                                .fieldType = INFOFIELD_STRING,
                                                .stringValue = VecSimAlgo_ToString(info.algo)});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::TYPE_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .stringValue = VecSimType_ToString(info.hnswInfo.type)});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::DIMENSION_STRING,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.hnswInfo.dim});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::METRIC_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .stringValue = VecSimMetric_ToString(info.hnswInfo.metric)});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.hnswInfo.indexSize});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_M_STRING,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.hnswInfo.M});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_EF_CONSTRUCTION_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .uintegerValue = info.hnswInfo.efConstruction});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_EF_RUNTIME_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .uintegerValue = info.hnswInfo.efRuntime});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_MAX_LEVEL,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.hnswInfo.max_level});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_ENTRYPOINT,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.hnswInfo.entrypoint});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::MEMORY_STRING,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.hnswInfo.memory});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::SEARCH_MODE_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .stringValue = VecSimSearchMode_ToString(info.hnswInfo.last_mode)});

    return infoIterator;
}

bool HNSWIndex::preferAdHocSearch(size_t subsetSize, size_t k) {
    // This heuristic is based on sklearn decision tree classifier (with 20 leaves nodes) -
    // see scripts/HNSW_batches_clf.py
    size_t index_size = this->indexSize();
    if (subsetSize > index_size) {
        throw std::runtime_error("internal error: subset size cannot be larger than index size");
    }
    size_t d = this->dim;
    size_t M = this->hnsw->getM();
    float r = (index_size == 0) ? 0.0f : (float)(subsetSize) / (float)index_size;
    bool res;

    // node 0
    if (index_size <= 30000) {
        // node 1
        if (index_size <= 5500) {
            // node 5
            res = true;
        } else {
            // node 6
            if (r <= 0.17) {
                // node 11
                res = true;
            } else {
                // node 12
                if (k <= 12) {
                    // node 13
                    if (d <= 55) {
                        // node 17
                        res = false;
                    } else {
                        // node 18
                        if (M <= 10) {
                            // node 19
                            res = false;
                        } else {
                            // node 20
                            res = true;
                        }
                    }
                } else {
                    // node 14
                    res = true;
                }
            }
        }
    } else {
        // node 2
        if (r < 0.07) {
            // node 3
            if (index_size <= 750000) {
                // node 15
                res = true;
            } else {
                // node 16
                if (k <= 7) {
                    // node 21
                    res = false;
                } else {
                    // node 22
                    if (r <= 0.03) {
                        // node 23
                        res = true;
                    } else {
                        // node 24
                        res = false;
                    }
                }
            }
        } else {
            // node 4
            if (d <= 75) {
                // node 7
                res = false;
            } else {
                // node 8
                if (k <= 12) {
                    // node 9
                    if (r <= 0.21) {
                        // node 27
                        if (M <= 57) {
                            // node 29
                            if (index_size <= 75000) {
                                // node 31
                                res = true;
                            } else {
                                // node 32
                                res = false;
                            }
                        } else {
                            // node 30
                            res = true;
                        }
                    } else {
                        // node 28
                        res = false;
                    }
                } else {
                    // node 10
                    if (M <= 10) {
                        // node 25
                        if (r <= 0.17) {
                            // node 33
                            res = true;
                        } else {
                            // node 34
                            res = false;
                        }
                    } else {
                        // node 26
                        if (index_size <= 300000) {
                            // node 35
                            res = true;
                        } else {
                            // node 36
                            if (r <= 0.17) {
                                // node 37
                                res = true;
                            } else {
                                // node 38
                                res = false;
                            }
                        }
                    }
                }
            }
        }
    }
    this->last_mode = res ? HYBRID_ADHOC_BF : HYBRID_BATCHES;
    return res;
}
