#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnsw_wrapper.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/utils/arr_cpp.h"
#include <cassert>
#include "memory.h"

extern "C" void VecSim_SetTimeoutCallbackFunction(timeoutCallbackFunction callback) {
    VecSimIndexAbstract::setTimeoutCallbackFunction(callback);
}

static VecSimResolveCode _ResolveParams_EFRuntime(VecSimAlgo index_type, VecSimRawParam rparam,
                                                  VecSimQueryParams *qparams, bool hybrid) {
    long long num_val;
    // EF_RUNTIME is a valid parameter only in HNSW algorithm.
    if (index_type != VecSimAlgo_HNSWLIB) {
        return VecSimParamResolverErr_UnknownParam;
    }
    if (qparams->hnswRuntimeParams.efRuntime != 0) {
        return VecSimParamResolverErr_AlreadySet;
    }
    if (validate_positive_integer_param(rparam, &num_val) != VecSimParamResolver_OK) {
        return VecSimParamResolverErr_BadValue;
    }

    qparams->hnswRuntimeParams.efRuntime = (size_t)num_val;
    return VecSimParamResolver_OK;
}

static VecSimResolveCode _ResolveParams_BatchSize(VecSimRawParam rparam, VecSimQueryParams *qparams,
                                                  bool hybrid) {
    long long num_val;
    if (!hybrid) {
        return VecSimParamResolverErr_InvalidPolicy_NHybrid;
    }
    if (qparams->batchSize != 0) {
        return VecSimParamResolverErr_AlreadySet;
    }
    if (validate_positive_integer_param(rparam, &num_val) != VecSimParamResolver_OK) {
        return VecSimParamResolverErr_BadValue;
    }
    qparams->batchSize = (size_t)num_val;
    return VecSimParamResolver_OK;
}

static VecSimResolveCode _ResolveParams_HybridPolicy(VecSimRawParam rparam,
                                                     VecSimQueryParams *qparams, bool hybrid) {
    if (!hybrid) {
        return VecSimParamResolverErr_InvalidPolicy_NHybrid;
    }
    if (qparams->searchMode != 0) {
        return VecSimParamResolverErr_AlreadySet;
    }
    if (!strcasecmp(rparam.value, "batches")) {
        qparams->searchMode = HYBRID_BATCHES;
    } else if (!strcasecmp(rparam.value, "adhoc_bf")) {
        qparams->searchMode = HYBRID_ADHOC_BF;
    } else {
        return VecSimParamResolverErr_InvalidPolicy_NExits;
    }
    return VecSimParamResolver_OK;
}

extern "C" VecSimIndex *VecSimIndex_New(const VecSimParams *params) {
    VecSimIndex *index = NULL;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    try {
        switch (params->algo) {
        case VecSimAlgo_HNSWLIB:
            index = HNSWIndex::HNSWIndex_New(&params->hnswParams, allocator);
            break;
        case VecSimAlgo_BF:
            index = BruteForceIndex::BruteForceIndex_New(&params->bfParams, allocator);
            break;
        default:
            break;
        }
    } catch (...) {
        // Index will delete itself. For now, do nothing.
    }
    return index;
}

extern "C" size_t VecSimIndex_EstimateInitialSize(const VecSimParams *params) {
    switch (params->algo) {
    case VecSimAlgo_HNSWLIB:
        return HNSWIndex::estimateInitialSize(&params->hnswParams);
    case VecSimAlgo_BF:
        return BruteForceIndex::estimateInitialSize(&params->bfParams);
    }
    return -1;
}

extern "C" int VecSimIndex_AddVector(VecSimIndex *index, const void *blob, size_t id) {
    int64_t before = index->getAllocator()->getAllocationSize();
    index->addVector(blob, id);
    int64_t after = index->getAllocator()->getAllocationSize();
    return after - before;
}

extern "C" int VecSimIndex_DeleteVector(VecSimIndex *index, size_t id) {
    int64_t before = index->getAllocator()->getAllocationSize();
    index->deleteVector(id);
    int64_t after = index->getAllocator()->getAllocationSize();
    return after - before;
}

extern "C" double VecSimIndex_GetDistanceFrom(VecSimIndex *index, size_t id, const void *blob) {
    return index->getDistanceFrom(id, blob);
}

extern "C" size_t VecSimIndex_EstimateElementSize(const VecSimParams *params) {
    switch (params->algo) {
    case VecSimAlgo_HNSWLIB:
        return HNSWIndex::estimateElementMemory(&params->hnswParams);
    case VecSimAlgo_BF:
        return BruteForceIndex::estimateElementMemory(&params->bfParams);
    }
    return -1;
}

extern "C" void VecSim_Normalize(void *blob, size_t dim, VecSimType type) {
    // TODO: need more generic
    assert(type == VecSimType_FLOAT32);
    float_vector_normalize((float *)blob, dim);
}

extern "C" size_t VecSimIndex_IndexSize(VecSimIndex *index) { return index->indexSize(); }

extern "C" VecSimResolveCode VecSimIndex_ResolveParams(VecSimIndex *index, VecSimRawParam *rparams,
                                                       int paramNum, VecSimQueryParams *qparams,
                                                       bool hybrid) {

    if (!qparams || (!rparams && (paramNum != 0))) {
        return VecSimParamResolverErr_NullParam;
    }
    VecSimAlgo index_type = index->info().algo;
    bzero(qparams, sizeof(VecSimQueryParams));
    auto res = VecSimParamResolver_OK;
    for (int i = 0; i < paramNum; i++) {
        if (!strcasecmp(rparams[i].name, VecSimCommonStrings::HNSW_EF_RUNTIME_STRING)) {
            if ((res = _ResolveParams_EFRuntime(index_type, rparams[i], qparams, hybrid)) !=
                VecSimParamResolver_OK) {
                return res;
            }
        } else if (!strcasecmp(rparams[i].name, VecSimCommonStrings::BATCH_SIZE_STRING)) {
            if ((res = _ResolveParams_BatchSize(rparams[i], qparams, hybrid)) !=
                VecSimParamResolver_OK) {
                return res;
            }
        } else if (!strcasecmp(rparams[i].name, VecSimCommonStrings::HYBRID_POLICY_STRING)) {
            if ((res = _ResolveParams_HybridPolicy(rparams[i], qparams, hybrid)) !=
                VecSimParamResolver_OK) {
                return res;
            }
        } else {
            return VecSimParamResolverErr_UnknownParam;
        }
    }
    // The combination of AD-HOC with batch_size is invalid, as there are no batches in this policy.
    if (qparams->searchMode == HYBRID_ADHOC_BF && qparams->batchSize > 0) {
        return VecSimParamResolverErr_InvalidPolicy_AdHoc_With_BatchSize;
    }
    // Also, 'ef_runtime' is meaning less in AD-HOC policy, since it doesn't involve search in HNSW
    // graph.
    if (qparams->searchMode == HYBRID_ADHOC_BF && index_type == VecSimAlgo_HNSWLIB &&
        qparams->hnswRuntimeParams.efRuntime > 0) {
        return VecSimParamResolverErr_InvalidPolicy_AdHoc_With_EfRuntime;
    }
    if (qparams->searchMode != 0) {
        index->setLastSearchMode(qparams->searchMode);
    }
    return res;
}

extern "C" VecSimQueryResult_List VecSimIndex_TopKQuery(VecSimIndex *index, const void *queryBlob,
                                                        size_t k, VecSimQueryParams *queryParams,
                                                        VecSimQueryResult_Order order) {
    assert((order == BY_ID || order == BY_SCORE) &&
           "Possible order values are only 'BY_ID' or 'BY_SCORE'");
    VecSimQueryResult_List results;
    results = index->topKQuery(queryBlob, k, queryParams);

    if (order == BY_ID) {
        sort_results_by_id(results);
    }
    return results;
}

extern "C" VecSimQueryResult_List VecSimIndex_RangeQuery(VecSimIndex *index, const void *queryBlob,
                                                         float radius,
                                                         VecSimQueryParams *queryParams,
                                                         VecSimQueryResult_Order order) {
    if (order != BY_ID && order != BY_SCORE) {
        throw std::runtime_error("Possible order values are only 'BY_ID' or 'BY_SCORE'");
    }
    if (radius < 0) {
        throw std::runtime_error("radius must be non-negative");
    }
    VecSimQueryResult_List results = index->rangeQuery(queryBlob, radius, queryParams);

    if (order == BY_SCORE) {
        sort_results_by_score(results);
    } else {
        sort_results_by_id(results);
    }
    return results;
}

extern "C" void VecSimIndex_Free(VecSimIndex *index) {
    std::shared_ptr<VecSimAllocator> allocator =
        index->getAllocator(); // Save allocator so it will not deallocate itself
    delete index;
}

extern "C" VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index) { return index->info(); }

extern "C" VecSimInfoIterator *VecSimIndex_InfoIterator(VecSimIndex *index) {
    return index->infoIterator();
}

extern "C" VecSimBatchIterator *VecSimBatchIterator_New(VecSimIndex *index, const void *queryBlob,
                                                        VecSimQueryParams *queryParams) {
    return index->newBatchIterator(queryBlob, queryParams);
}

extern "C" void VecSim_SetMemoryFunctions(VecSimMemoryFunctions memoryfunctions) {
    VecSimAllocator::setMemoryFunctions(memoryfunctions);
}

extern "C" bool VecSimIndex_PreferAdHocSearch(VecSimIndex *index, size_t subsetSize, size_t k,
                                              bool initial_check) {
    return index->preferAdHocSearch(subsetSize, k, initial_check);
}
