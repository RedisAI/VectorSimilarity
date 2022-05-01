#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnsw_wrapper.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/utils/arr_cpp.h"
#include <cassert>
#include "memory.h"

extern "C" VecSimIndex *VecSimIndex_New(const VecSimParams *params) {
    VecSimIndex *index = NULL;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    try {
        switch (params->algo) {
        case VecSimAlgo_HNSWLIB:
            index = new (allocator) HNSWIndex(&params->hnswParams, allocator);
            break;
        case VecSimAlgo_BF:
            index = new (allocator) BruteForceIndex(&params->bfParams, allocator);
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
                                                       int paramNum, VecSimQueryParams *qparams) {

    if (!qparams || (!rparams && (paramNum != 0))) {
        return VecSimParamResolverErr_NullParam;
    }
    bzero(qparams, sizeof(VecSimQueryParams));
    long long num_val;
    for (int i = 0; i < paramNum; i++) {
        if (!strcasecmp(rparams[i].name, VecSimCommonStrings::HNSW_EF_RUNTIME_STRING)) {
            // EF_RUNTIME is a valid parameter only in HNSW algorithm.
            if (!dynamic_cast<HNSWIndex *>(index)) {
                return VecSimParamResolverErr_UnknownParam;
            }
            if (qparams->hnswRuntimeParams.efRuntime != 0) {
                return VecSimParamResolverErr_AlreadySet;
            }
            if (validate_positive_integer_param(rparams[i], &num_val) != VecSimParamResolver_OK) {
                return VecSimParamResolverErr_BadValue;
            }
            qparams->hnswRuntimeParams.efRuntime = (size_t)num_val;
        } else if (!strcasecmp(rparams[i].name, VecSimCommonStrings::BATCH_SIZE_STRING)) {
            if (qparams->batchSize != 0) {
                return VecSimParamResolverErr_AlreadySet;
            }
            if (validate_positive_integer_param(rparams[i], &num_val) != VecSimParamResolver_OK) {
                return VecSimParamResolverErr_BadValue;
            }
            qparams->batchSize = (size_t)num_val;
        } else if (!strcasecmp(rparams[i].name, VecSimCommonStrings::HYBRID_POLICY_STRING)) {
            if (qparams->searchMode != 0) {
                return VecSimParamResolverErr_AlreadySet;
            }
            if (!strcasecmp(rparams[i].value, "batches")) {
                qparams->searchMode = HYBRID_BATCHES;
            } else if (!strcasecmp(rparams[i].value, "adhoc_bf")) {
                qparams->searchMode = HYBRID_ADHOC_BF;
            } else {
                return VecSimParamResolverErr_InvalidPolicy;
            }
        } else {
            return VecSimParamResolverErr_UnknownParam;
        }
    }
    // The combination of AD-HOC with batch_size is invalid, as there are no batches in this policy.
    if (qparams->searchMode == HYBRID_ADHOC_BF && qparams->batchSize > 0) {
        return VecSimParamResolverErr_InvalidPolicy;
    }
    // Also, 'ef_runtime' is meaning less in AD-HOC policy, since it doesn't involve search in HNSW
    // graph.
    if (qparams->searchMode == HYBRID_ADHOC_BF && dynamic_cast<HNSWIndex *>(index) &&
        qparams->hnswRuntimeParams.efRuntime > 0) {
        return VecSimParamResolverErr_InvalidPolicy;
    }
    return (VecSimResolveCode)VecSim_OK;
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

extern "C" void VecSimIndex_Free(VecSimIndex *index) {
    std::shared_ptr<VecSimAllocator> allocator =
        index->getAllocator(); // Save allocator so it will not deallocate itself
    delete index;
}

extern "C" VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index) { return index->info(); }

extern "C" VecSimInfoIterator *VecSimIndex_InfoIterator(VecSimIndex *index) {
    return index->infoIterator();
}

extern "C" VecSimBatchIterator *VecSimBatchIterator_New(VecSimIndex *index, const void *queryBlob) {
    return index->newBatchIterator(queryBlob);
}

extern "C" void VecSim_SetMemoryFunctions(VecSimMemoryFunctions memoryfunctions) {
    VecSimAllocator::setMemoryFunctions(memoryfunctions);
}

extern "C" bool VecSimIndex_PreferAdHocSearch(VecSimIndex *index, size_t subsetSize, size_t k,
                                              bool initial_check) {
    return index->preferAdHocSearch(subsetSize, k, initial_check);
}
