/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/index_factories/index_factory.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/types/bfloat16.h"
#include <cassert>
#include "memory.h"

extern "C" void VecSim_SetTimeoutCallbackFunction(timeoutCallbackFunction callback) {
    VecSimIndex::setTimeoutCallbackFunction(callback);
}

extern "C" void VecSim_SetLogCallbackFunction(logCallbackFunction callback) {
    VecSimIndex::setLogCallbackFunction(callback);
}

extern "C" void VecSim_SetWriteMode(VecSimWriteMode mode) { VecSimIndex::setWriteMode(mode); }

static VecSimResolveCode _ResolveParams_EFRuntime(VecSimAlgo index_type, VecSimRawParam rparam,
                                                  VecSimQueryParams *qparams,
                                                  VecsimQueryType query_type) {
    long long num_val;
    // EF_RUNTIME is a valid parameter only in HNSW algorithm.
    if (index_type != VecSimAlgo_HNSWLIB) {
        return VecSimParamResolverErr_UnknownParam;
    }
    // EF_RUNTIME is invalid for range query
    if (query_type == QUERY_TYPE_RANGE) {
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
                                                  VecsimQueryType query_type) {
    long long num_val;
    if (query_type != QUERY_TYPE_HYBRID) {
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

static VecSimResolveCode _ResolveParams_Epsilon(VecSimAlgo index_type, VecSimRawParam rparam,
                                                VecSimQueryParams *qparams,
                                                VecsimQueryType query_type) {
    double num_val;
    // EPSILON is a valid parameter only in HNSW algorithm.
    if (index_type != VecSimAlgo_HNSWLIB) {
        return VecSimParamResolverErr_UnknownParam;
    }
    if (query_type != QUERY_TYPE_RANGE) {
        return VecSimParamResolverErr_InvalidPolicy_NRange;
    }
    if (qparams->hnswRuntimeParams.epsilon != 0) {
        return VecSimParamResolverErr_AlreadySet;
    }
    if (validate_positive_double_param(rparam, &num_val) != VecSimParamResolver_OK) {
        return VecSimParamResolverErr_BadValue;
    }
    qparams->hnswRuntimeParams.epsilon = num_val;
    return VecSimParamResolver_OK;
}

static VecSimResolveCode _ResolveParams_HybridPolicy(VecSimRawParam rparam,
                                                     VecSimQueryParams *qparams,
                                                     VecsimQueryType query_type) {
    if (query_type != QUERY_TYPE_HYBRID) {
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
    return VecSimFactory::NewIndex(params);
}

extern "C" size_t VecSimIndex_EstimateInitialSize(const VecSimParams *params) {
    return VecSimFactory::EstimateInitialSize(params);
}

extern "C" int VecSimIndex_AddVector(VecSimIndex *index, const void *blob, size_t label) {
    return index->addVectorWrapper(blob, label);
}

extern "C" int VecSimIndex_DeleteVector(VecSimIndex *index, size_t label) {
    return index->deleteVector(label);
}

extern "C" double VecSimIndex_GetDistanceFrom_Unsafe(VecSimIndex *index, size_t label,
                                                     const void *blob) {
    return index->getDistanceFrom_Unsafe(label, blob);
}

extern "C" size_t VecSimIndex_EstimateElementSize(const VecSimParams *params) {
    return VecSimFactory::EstimateElementSize(params);
}

extern "C" void VecSim_Normalize(void *blob, size_t dim, VecSimType type) {
    if (type == VecSimType_FLOAT32) {
        spaces::GetNormalizeFunc<float>()(blob, dim);
    } else if (type == VecSimType_FLOAT64) {
        spaces::GetNormalizeFunc<double>()(blob, dim);
    } else if (type == VecSimType_BFLOAT16) {
        spaces::GetNormalizeFunc<vecsim_types::bfloat16>()(blob, dim);
    } else if (type == VecSimType_FLOAT16) {
        spaces::GetNormalizeFunc<vecsim_types::float16>()(blob, dim);
    }
}

extern "C" size_t VecSimIndex_IndexSize(VecSimIndex *index) { return index->indexSize(); }

extern "C" VecSimResolveCode VecSimIndex_ResolveParams(VecSimIndex *index, VecSimRawParam *rparams,
                                                       int paramNum, VecSimQueryParams *qparams,
                                                       VecsimQueryType query_type) {

    if (!qparams || (!rparams && (paramNum != 0))) {
        return VecSimParamResolverErr_NullParam;
    }
    VecSimAlgo index_type = index->basicInfo().algo;

    bzero(qparams, sizeof(VecSimQueryParams));
    auto res = VecSimParamResolver_OK;
    for (int i = 0; i < paramNum; i++) {
        if (!strcasecmp(rparams[i].name, VecSimCommonStrings::HNSW_EF_RUNTIME_STRING)) {
            if ((res = _ResolveParams_EFRuntime(index_type, rparams[i], qparams, query_type)) !=
                VecSimParamResolver_OK) {
                return res;
            }
        } else if (!strcasecmp(rparams[i].name, VecSimCommonStrings::HNSW_EPSILON_STRING)) {
            if ((res = _ResolveParams_Epsilon(index_type, rparams[i], qparams, query_type)) !=
                VecSimParamResolver_OK) {
                return res;
            }
        } else if (!strcasecmp(rparams[i].name, VecSimCommonStrings::BATCH_SIZE_STRING)) {
            if ((res = _ResolveParams_BatchSize(rparams[i], qparams, query_type)) !=
                VecSimParamResolver_OK) {
                return res;
            }
        } else if (!strcasecmp(rparams[i].name, VecSimCommonStrings::HYBRID_POLICY_STRING)) {
            if ((res = _ResolveParams_HybridPolicy(rparams[i], qparams, query_type)) !=
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

extern "C" VecSimQueryReply *VecSimIndex_TopKQuery(VecSimIndex *index, const void *queryBlob,
                                                   size_t k, VecSimQueryParams *queryParams,
                                                   VecSimQueryReply_Order order) {
    assert((order == BY_ID || order == BY_SCORE) &&
           "Possible order values are only 'BY_ID' or 'BY_SCORE'");
    VecSimQueryReply *results;
    results = index->topKQueryWrapper(queryBlob, k, queryParams);

    if (order == BY_ID) {
        sort_results_by_id(results);
    }
    return results;
}

extern "C" VecSimQueryReply *VecSimIndex_RangeQuery(VecSimIndex *index, const void *queryBlob,
                                                    double radius, VecSimQueryParams *queryParams,
                                                    VecSimQueryReply_Order order) {
    if (order != BY_ID && order != BY_SCORE) {
        throw std::runtime_error("Possible order values are only 'BY_ID' or 'BY_SCORE'");
    }
    if (radius < 0) {
        throw std::runtime_error("radius must be non-negative");
    }
    return index->rangeQueryWrapper(queryBlob, radius, queryParams, order);
}

extern "C" void VecSimIndex_Free(VecSimIndex *index) {
    std::shared_ptr<VecSimAllocator> allocator =
        index->getAllocator(); // Save allocator so it will not deallocate itself
    delete index;
}

extern "C" VecSimIndexDebugInfo VecSimIndex_DebugInfo(VecSimIndex *index) {
    return index->debugInfo();
}

extern "C" VecSimDebugInfoIterator *VecSimIndex_DebugInfoIterator(VecSimIndex *index) {
    return index->debugInfoIterator();
}

extern "C" VecSimIndexBasicInfo VecSimIndex_BasicInfo(VecSimIndex *index) {
    return index->basicInfo();
}

extern "C" VecSimIndexStatsInfo VecSimIndex_StatsInfo(VecSimIndex *index) {
    return index->statisticInfo();
}

extern "C" VecSimBatchIterator *VecSimBatchIterator_New(VecSimIndex *index, const void *queryBlob,
                                                        VecSimQueryParams *queryParams) {
    return index->newBatchIteratorWrapper(queryBlob, queryParams);
}

extern "C" void VecSimTieredIndex_GC(VecSimIndex *index) {
    if (index->basicInfo().isTiered) {
        index->runGC();
    }
}

extern "C" void VecSimTieredIndex_AcquireSharedLocks(VecSimIndex *index) {
    index->acquireSharedLocks();
}

extern "C" void VecSimTieredIndex_ReleaseSharedLocks(VecSimIndex *index) {
    index->releaseSharedLocks();
}

extern "C" void VecSim_SetMemoryFunctions(VecSimMemoryFunctions memoryfunctions) {
    VecSimAllocator::setMemoryFunctions(memoryfunctions);
}

extern "C" bool VecSimIndex_PreferAdHocSearch(VecSimIndex *index, size_t subsetSize, size_t k,
                                              bool initial_check) {
    return index->preferAdHocSearch(subsetSize, k, initial_check);
}
