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
      blockSize(params->blockSize ? params->blockSize : DEFAULT_BLOCK_SIZE),
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
size_t HNSWIndex::estimateInitialSize(const HNSWParams *params) {
    size_t est = sizeof(VecSimAllocator) + sizeof(HNSWIndex) + sizeof(size_t);
    est += (params->metric == VecSimMetric_L2 ? sizeof(L2Space) : sizeof(InnerProductSpace)) +
           sizeof(size_t);
    est += sizeof(*hnsw) + sizeof(size_t);
    est += sizeof(VisitedNodesHandler) + sizeof(size_t);
    // Used for synchronization only when parallel indexing / searching is enabled.
#ifdef ENABLE_PARALLELIZATION
    est += sizeof(VisitedNodesHandlerPool);
#endif
    est += sizeof(tag_t) * params->initialCapacity + sizeof(size_t); // visited nodes

    est += sizeof(void *) * params->initialCapacity + sizeof(size_t); // link lists (for levels > 0)
    est += sizeof(size_t) * params->initialCapacity + sizeof(size_t); // element level
    est += sizeof(size_t) * params->initialCapacity +
           sizeof(size_t); // Labels lookup hash table buckets.

    size_t size_links_level0 =
        sizeof(linklistsizeint) + params->M * 2 * sizeof(tableint) + sizeof(void *);
    size_t size_total_data_per_element =
        size_links_level0 + params->dim * sizeof(float) + sizeof(labeltype);
    est += params->initialCapacity * size_total_data_per_element + sizeof(size_t);

    return est;
}

size_t HNSWIndex::estimateElementMemory(const HNSWParams *params) {
    size_t size_links_level0 = sizeof(linklistsizeint) + params->M * 2 * sizeof(tableint) +
                               sizeof(void *) + sizeof(vecsim_stl::vector<tableint>);
    size_t size_links_higher_level = sizeof(linklistsizeint) + params->M * sizeof(tableint) +
                                     sizeof(void *) + sizeof(vecsim_stl::vector<tableint>);
    // The Expectancy for the random variable which is the number of levels per element equals
    // 1/ln(M). Since the max_level is rounded to the "floor" integer, the actual average number
    // of levels is lower (intuitively, we "loose" a level every time the random generated number
    // should have been rounded up to the larger integer). So, we "fix" the expectancy and take
    // 1/2*ln(M) instead as an approximation.
    size_t expected_size_links_higher_levels =
        ceil((1 / (2 * log(params->M))) * (float)size_links_higher_level);

    size_t size_total_data_per_element = size_links_level0 + expected_size_links_higher_levels +
                                         params->dim * sizeof(float) + sizeof(labeltype);

    // For every new vector, a new node of size 24 is allocated in a bucket of the hash table.
    size_t size_label_lookup_node =
        24 + sizeof(size_t); // 24 + VecSimAllocator::allocation_header_size
    // 1 entry in visited nodes + 1 entry in element levels + (approximately) 1 bucket in labels
    // lookup hash map.
    size_t size_meta_data =
        sizeof(tag_t) + sizeof(size_t) + sizeof(size_t) + size_label_lookup_node;

    /* Disclaimer: we are neglecting two additional factors that consume memory:
     * 1. The overall bucket size in labels_lookup hash table is usually higher than the number of
     * requested buckets (which is the index capacity), and it is auto selected according to the
     * hashing policy and the max load factor.
     * 2. The incoming edges that aren't bidirectional are stored in a dynamic array
     * (vecsim_stl::vector) Those edges' memory *is omitted completely* from this estimation.
     */
    return size_meta_data + size_total_data_per_element;
}

int HNSWIndex::addVector(const void *vector_data, size_t id) {

    // If id already exists
    if (this->hnsw->isLabelExist(id)) {
        return false;
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

    // If id doesnt exist.
    if (!this->hnsw->isLabelExist(id)) {
        return false;
    }

    // Else delte it from the graph.
    this->hnsw->removePoint(id);

    size_t index_size = hnsw->getIndexSize();
    size_t curr_capacity = this->hnsw->getIndexCapacity();

    // If we need to free a complete block & there is a least one block between the
    // capacity and the size.
    if (index_size % blockSize == 0 && index_size + blockSize <= curr_capacity) {

        // Check if the capacity is aligned to block size.
        size_t extra_space_to_free = curr_capacity % blockSize;

        // Remove one block from the capacity. TODO check that we dont go below zero
        this->hnsw->resizeIndex(curr_capacity - blockSize - extra_space_to_free);
    }
    return true;
}

double HNSWIndex::getDistanceFrom(size_t label, const void *vector_data) {
    return this->hnsw->getDistanceByLabelFromPoint(label, vector_data);
}

size_t HNSWIndex::indexSize() const { return this->hnsw->getIndexSize(); }

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

VecSimIndexInfo HNSWIndex::info() const {

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_HNSWLIB;
    info.hnswInfo.dim = this->dim;
    info.hnswInfo.type = this->vecType;
    info.hnswInfo.metric = this->metric;
    info.hnswInfo.blockSize = this->blockSize;
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

VecSimInfoIterator *HNSWIndex::infoIterator() {
    VecSimIndexInfo info = this->info();
    // For readability. Update this number when needed.
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
