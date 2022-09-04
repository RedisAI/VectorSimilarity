#include "VecSim/algorithms/hnsw/hnsw_wrapper.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/hnsw/hnsw_batch_iterator.h"

#include <deque>
#include <memory>
#include <cassert>

using namespace std;
using namespace hnswlib;

/******************** Ctor / Dtor **************/

HNSWIndex::HNSWIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndexAbstract(allocator, params->dim, params->type, params->metric, params->blockSize,
                          params->multi),
      hnsw(new (allocator) hnswlib::HierarchicalNSW<float>(params, this->dist_func, allocator)) {}

/******************** Implementation **************/

int HNSWIndex::addVector(const void *vector_data, size_t id) {

    // If id already exists remove and re-add
    if (this->hnsw->isLabelExist(id)) {
        this->hnsw->removePoint(id);
    }

    try {
        float normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine.
        if (this->metric == VecSimMetric_Cosine) {
            // TODO: need more generic
            memcpy(normalized_data, vector_data, this->dim * sizeof(float));
            float_vector_normalize(normalized_data, this->dim);
            vector_data = normalized_data;
        }
        size_t index_capacity = this->hnsw->getIndexCapacity();
        if (hnsw->getIndexSize() == index_capacity) {
            size_t vectors_to_add = blockSize - index_capacity % blockSize;
            this->hnsw->resizeIndex(index_capacity + vectors_to_add);
        }
        this->hnsw->addPoint(vector_data, id);
        return true;
    } catch (...) {
        return false;
    }
}

int HNSWIndex::deleteVector(size_t id) {

    // If id doesn't exist.
    if (!this->hnsw->isLabelExist(id)) {
        return false;
    }

    // Else, *delete* it from the graph.
    this->hnsw->removePoint(id);

    size_t index_size = hnsw->getIndexSize();
    size_t curr_capacity = this->hnsw->getIndexCapacity();

    // If we need to free a complete block & there is a least one block between the
    // capacity and the size.
    if (index_size % blockSize == 0 && index_size + blockSize <= curr_capacity) {

        // Check if the capacity is aligned to block size.
        size_t extra_space_to_free = curr_capacity % blockSize;

        // Remove one block from the capacity.
        this->hnsw->resizeIndex(curr_capacity - blockSize - extra_space_to_free);
    }
    return true;
}

double HNSWIndex::getDistanceFrom(size_t label, const void *vector_data) const {
    return this->hnsw->getDistanceByLabelFromPoint(label, vector_data);
}

size_t HNSWIndex::indexSize() const { return this->hnsw->getIndexSize(); }

size_t HNSWIndex::indexLabelCount() const { return this->hnsw->getIndexLabelCount(); }

void HNSWIndex::setEf(size_t ef) { this->hnsw->setEf(ef); }

VecSimQueryResult_List HNSWIndex::topKQuery(const void *query_data, size_t k,
                                            VecSimQueryParams *queryParams) {
    VecSimQueryResult_List rl = {0};
    void *timeoutCtx = nullptr;
    try {
        this->last_mode = STANDARD_KNN;
        float normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine.
        if (this->metric == VecSimMetric_Cosine) {
            // TODO: need more generic
            memcpy(normalized_data, query_data, this->dim * sizeof(float));
            float_vector_normalize(normalized_data, this->dim);
            query_data = normalized_data;
        }
        // Get original efRuntime and store it.
        size_t originalEF = this->hnsw->getEf();

        if (queryParams) {
            timeoutCtx = queryParams->timeoutCtx;
            if (queryParams->hnswRuntimeParams.efRuntime != 0) {
                hnsw->setEf(queryParams->hnswRuntimeParams.efRuntime);
            }
        }
        auto knn_res = hnsw->searchKnn(query_data, k, timeoutCtx, &rl.code);
        if (VecSim_OK != rl.code) {
            return rl;
        }
        rl.results = array_new_len<VecSimQueryResult>(knn_res.size(), knn_res.size());
        for (int i = (int)knn_res.size() - 1; i >= 0; --i) {
            VecSimQueryResult_SetId(rl.results[i], knn_res.top().second);
            VecSimQueryResult_SetScore(rl.results[i], knn_res.top().first);
            knn_res.pop();
        }
        // Restore efRuntime.
        hnsw->setEf(originalEF);
        assert(hnsw->getEf() == originalEF);

    } catch (...) {
        rl.code = VecSim_QueryResult_Err;
    }
    return rl;
}

VecSimQueryResult_List HNSWIndex::rangeQuery(const void *queryBlob, double radius,
                                             VecSimQueryParams *queryParams) {
    auto rl = (VecSimQueryResult_List){0};
    void *timeoutCtx = nullptr;
    this->last_mode = RANGE_QUERY;

    float normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic when other types will be supported.
        memcpy(normalized_blob, queryBlob, this->dim * sizeof(float));
        float_vector_normalize(normalized_blob, this->dim);
        queryBlob = normalized_blob;
    }

    double originalEpsilon = this->hnsw->getEpsilon();
    if (queryParams) {
        timeoutCtx = queryParams->timeoutCtx;
        if (queryParams->hnswRuntimeParams.epsilon != 0.0) {
            hnsw->setEpsilon(queryParams->hnswRuntimeParams.epsilon);
        }
    }
    // Call searchRange internally in HNSWlib to obtains results. This will return and set the
    // rl.code to "TimedOut" if timeout occurs.
    rl.results = hnsw->searchRange(queryBlob, radius, timeoutCtx, &rl.code);
    if (VecSim_QueryResult_OK != rl.code) {
        return rl;
    }

    // Restore the default epsilon.
    hnsw->setEpsilon(originalEpsilon);
    assert(hnsw->getEpsilon() == originalEpsilon);

    rl.code = VecSim_QueryResult_OK;
    return rl;
}

VecSimIndexInfo HNSWIndex::info() const {

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_HNSWLIB;
    info.hnswInfo.dim = this->dim;
    info.hnswInfo.type = this->vecType;
    info.hnswInfo.isMulti = this->isMulti;
    info.hnswInfo.metric = this->metric;
    info.hnswInfo.blockSize = this->blockSize;
    info.hnswInfo.M = this->hnsw->getM();
    info.hnswInfo.efConstruction = this->hnsw->getEfConstruction();
    info.hnswInfo.efRuntime = this->hnsw->getEf();
    info.hnswInfo.epsilon = this->hnsw->getEpsilon();
    info.hnswInfo.indexSize = this->hnsw->getIndexSize();
    info.hnswInfo.indexLabelCount = this->indexLabelCount();
    info.hnswInfo.max_level = this->hnsw->getMaxLevel();
    info.hnswInfo.entrypoint = this->hnsw->getEntryPointLabel();
    info.hnswInfo.memory = this->allocator->getAllocationSize();
    info.hnswInfo.last_mode = this->last_mode;
    return info;
}

VecSimBatchIterator *HNSWIndex::newBatchIterator(const void *queryBlob,
                                                 VecSimQueryParams *queryParams) {
    // As this is the only supported type, we always allocate 4 bytes for every element in the
    // vector.
    assert(this->vecType == VecSimType_FLOAT32);
    auto *queryBlobCopy = this->allocator->allocate(sizeof(float) * this->dim);
    memcpy(queryBlobCopy, queryBlob, dim * sizeof(float));
    if (metric == VecSimMetric_Cosine) {
        float_vector_normalize((float *)queryBlobCopy, dim);
    }
    // Ownership of queryBlobCopy moves to HNSW_BatchIterator that will free it at the end.
    return new (this->allocator)
        HNSW_BatchIterator(queryBlobCopy, this, queryParams, this->allocator);
}

VecSimInfoIterator *HNSWIndex::infoIterator() const {
    VecSimIndexInfo info = this->info();
    // For readability. Update this number when needed.
    size_t numberOfInfoFields = 12;
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields);

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::ALGORITHM_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimAlgo_ToString(info.algo)}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::TYPE_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimType_ToString(info.hnswInfo.type)}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::DIMENSION_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.dim}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::METRIC_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimMetric_ToString(info.hnswInfo.metric)}}});

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::IS_MULTI_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.bfInfo.isMulti}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.indexSize}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::INDEX_LABEL_COUNT_STRING,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.indexLabelCount}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_M_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.M}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::HNSW_EF_CONSTRUCTION_STRING,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.efConstruction}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_EF_RUNTIME_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.efRuntime}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_MAX_LEVEL,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.max_level}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_ENTRYPOINT,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.entrypoint}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::MEMORY_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.hnswInfo.memory}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::SEARCH_MODE_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .fieldValue = {FieldValue{
                             .stringValue = VecSimSearchMode_ToString(info.hnswInfo.last_mode)}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::HNSW_EPSILON_STRING,
                         .fieldType = INFOFIELD_FLOAT64,
                         .fieldValue = {FieldValue{.floatingPointValue = info.hnswInfo.epsilon}}});

    return infoIterator;
}

bool HNSWIndex::preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) {
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
    // Set the mode - if this isn't the initial check, we switched mode form batches to ad-hoc.
    this->last_mode =
        res ? (initial_check ? HYBRID_ADHOC_BF : HYBRID_BATCHES_TO_ADHOC_BF) : HYBRID_BATCHES;
    return res;
}
