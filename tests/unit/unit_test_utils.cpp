/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "unit_test_utils.h"
#include "gtest/gtest.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

VecsimQueryType test_utils::query_types[4] = {QUERY_TYPE_NONE, QUERY_TYPE_KNN, QUERY_TYPE_HYBRID,
                                              QUERY_TYPE_RANGE};

static bool allUniqueResults(VecSimQueryReply *res) {
    size_t len = VecSimQueryReply_Len(res);
    auto it1 = VecSimQueryReply_GetIterator(res);
    for (size_t i = 0; i < len; i++) {
        auto ei = VecSimQueryReply_IteratorNext(it1);
        auto it2 = VecSimQueryReply_GetIterator(res);
        for (size_t j = 0; j < i; j++) {
            auto ej = VecSimQueryReply_IteratorNext(it2);
            if (VecSimQueryResult_GetId(ei) == VecSimQueryResult_GetId(ej)) {
                VecSimQueryReply_IteratorFree(it2);
                VecSimQueryReply_IteratorFree(it1);
                return false;
            }
        }
        VecSimQueryReply_IteratorFree(it2);
    }
    VecSimQueryReply_IteratorFree(it1);
    return true;
}

/*
 * helper function to run Top K search and iterate over the results. ResCB is a callback that takes
 * the id, score and index of a result, and performs test-specific logic for each.
 */

VecSimQueryParams CreateQueryParams(const HNSWRuntimeParams &RuntimeParams) {
    VecSimQueryParams QueryParams = {.hnswRuntimeParams = RuntimeParams};
    return QueryParams;
}

VecSimQueryParams CreateQueryParams(const SVSRuntimeParams &RuntimeParams) {
    VecSimQueryParams QueryParams = {.svsRuntimeParams = RuntimeParams};
    return QueryParams;
}

static bool is_async_index(VecSimIndex *index) {
    return dynamic_cast<VecSimTieredIndex<float, float> *>(index) != nullptr ||
           dynamic_cast<VecSimTieredIndex<bfloat16, float> *>(index) != nullptr ||
           dynamic_cast<VecSimTieredIndex<float16, float> *>(index) != nullptr ||
           dynamic_cast<VecSimTieredIndex<uint8_t, float> *>(index) != nullptr ||
           dynamic_cast<VecSimTieredIndex<int8_t, float> *>(index) != nullptr ||
           dynamic_cast<VecSimTieredIndex<double, double> *>(index) != nullptr;
}

void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k, size_t expected_num_res,
                       std::function<void(size_t, double, size_t)> ResCB, VecSimQueryParams *params,
                       VecSimQueryReply_Order order) {
    VecSimQueryReply *res = VecSimIndex_TopKQuery(index, query, k, params, order);
    if (is_async_index(index)) {
        // Async index may return more or less than the expected number of results,
        // depending on the number of results that were available at the time of the query.
        // We can estimate the number of results that should be returned to be roughly
        // `expected_num_res` +- number of threads in the pool of the job queue.

        // for now, lets only check that the number of results is not greater than k.
        ASSERT_LE(VecSimQueryReply_Len(res), k);
    } else {
        ASSERT_EQ(VecSimQueryReply_Len(res), expected_num_res);
    }
    ASSERT_TRUE(allUniqueResults(res));
    VecSimQueryReply_Iterator *iterator = VecSimQueryReply_GetIterator(res);
    int res_ind = 0;
    while (VecSimQueryReply_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryReply_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        double score = VecSimQueryResult_GetScore(item);
        ResCB(id, score, res_ind++);
    }
    VecSimQueryReply_IteratorFree(iterator);
    VecSimQueryReply_Free(res);
}

void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k,
                       std::function<void(size_t, double, size_t)> ResCB, VecSimQueryParams *params,
                       VecSimQueryReply_Order order) {
    size_t expected_num_res = std::min(VecSimIndex_IndexSize(index), k);
    runTopKSearchTest(index, query, k, expected_num_res, ResCB, params, order);
}

/*
 * helper function to run batch search iteration, and iterate over the results. ResCB is a callback
 * that takes the id, score and index of a result, and performs test-specific logic for each.
 */
void runBatchIteratorSearchTest(VecSimBatchIterator *batch_iterator, size_t n_res,
                                std::function<void(size_t, double, size_t)> ResCB,
                                VecSimQueryReply_Order order, size_t expected_n_res) {
    if (expected_n_res == SIZE_MAX)
        expected_n_res = n_res;
    VecSimQueryReply *res = VecSimBatchIterator_Next(batch_iterator, n_res, order);
    ASSERT_EQ(VecSimQueryReply_Len(res), expected_n_res);
    ASSERT_TRUE(allUniqueResults(res));
    VecSimQueryReply_Iterator *iterator = VecSimQueryReply_GetIterator(res);
    int res_ind = 0;
    while (VecSimQueryReply_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryReply_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        double score = VecSimQueryResult_GetScore(item);
        ResCB(id, score, res_ind++);
    }
    ASSERT_EQ(res_ind, expected_n_res);
    VecSimQueryReply_IteratorFree(iterator);
    VecSimQueryReply_Free(res);
}

void compareCommonInfo(CommonInfo info1, CommonInfo info2) {
    ASSERT_EQ(info1.basicInfo.dim, info2.basicInfo.dim);
    ASSERT_EQ(info1.basicInfo.metric, info2.basicInfo.metric);
    ASSERT_EQ(info1.indexSize, info2.indexSize);
    ASSERT_EQ(info1.basicInfo.type, info2.basicInfo.type);
    ASSERT_EQ(info1.memory, info2.memory);
    ASSERT_EQ(info1.basicInfo.blockSize, info2.basicInfo.blockSize);
    ASSERT_EQ(info1.basicInfo.isMulti, info2.basicInfo.isMulti);
    ASSERT_EQ(info1.lastMode, info2.lastMode);
    ASSERT_EQ(info1.indexLabelCount, info2.indexLabelCount);
}
void compareFlatInfo(bfInfoStruct info1, bfInfoStruct info2) {}

void compareHNSWInfo(hnswInfoStruct info1, hnswInfoStruct info2) {
    ASSERT_EQ(info1.efConstruction, info2.efConstruction);
    ASSERT_EQ(info1.efRuntime, info2.efRuntime);
    ASSERT_EQ(info1.entrypoint, info2.entrypoint);
    ASSERT_EQ(info1.epsilon, info2.epsilon);
    ASSERT_EQ(info1.M, info2.M);
    ASSERT_EQ(info1.max_level, info2.max_level);
    ASSERT_EQ(info1.visitedNodesPoolSize, info2.visitedNodesPoolSize);
}

/*
 * helper function to run range query and iterate over the results. ResCB is a callback that takes
 * the id, score and index of a result, and performs test-specific logic for each.
 */
void runRangeQueryTest(VecSimIndex *index, const void *query, double radius,
                       const std::function<void(size_t, double, size_t)> &ResCB,
                       size_t expected_res_num, VecSimQueryReply_Order order,
                       VecSimQueryParams *params) {
    VecSimQueryReply *res =
        VecSimIndex_RangeQuery(index, (const void *)query, radius, params, order);
    EXPECT_EQ(VecSimQueryReply_Len(res), expected_res_num);
    EXPECT_TRUE(allUniqueResults(res));
    VecSimQueryReply_Iterator *iterator = VecSimQueryReply_GetIterator(res);
    int res_ind = 0;
    while (VecSimQueryReply_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryReply_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        double score = VecSimQueryResult_GetScore(item);
        ResCB(id, score, res_ind++);
    }
    EXPECT_EQ(res_ind, expected_res_num);
    VecSimQueryReply_IteratorFree(iterator);
    VecSimQueryReply_Free(res);
}

void compareFlatIndexInfoToIterator(VecSimIndexDebugInfo info, VecSimDebugInfoIterator *infoIter) {
    ASSERT_EQ(10, VecSimDebugInfoIterator_NumberOfFields(infoIter));
    while (VecSimDebugInfoIterator_HasNextField(infoIter)) {
        VecSim_InfoField *infoField = VecSimDebugInfoIterator_NextField(infoIter);
        if (!strcmp(infoField->fieldName, VecSimCommonStrings::ALGORITHM_STRING)) {
            // Algorithm type.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimAlgo_ToString(info.commonInfo.basicInfo.algo));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::TYPE_STRING)) {
            // Vector type.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimType_ToString(info.commonInfo.basicInfo.type));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::DIMENSION_STRING)) {
            // Vector dimension.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.dim);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::METRIC_STRING)) {
            // Metric.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimMetric_ToString(info.commonInfo.basicInfo.metric));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::SEARCH_MODE_STRING)) {
            // Search mode.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimSearchMode_ToString(info.commonInfo.lastMode));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::INDEX_SIZE_STRING)) {
            // Index size.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.indexSize);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::INDEX_LABEL_COUNT_STRING)) {
            // Index label count.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.indexLabelCount);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::IS_MULTI_STRING)) {
            // Is the index multi value.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.isMulti);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::BLOCK_SIZE_STRING)) {
            // Block size.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.blockSize);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::MEMORY_STRING)) {
            // Memory.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.memory);
        } else {
            FAIL();
        }
    }
}

void compareHNSWIndexInfoToIterator(VecSimIndexDebugInfo info, VecSimDebugInfoIterator *infoIter) {
    ASSERT_EQ(17, VecSimDebugInfoIterator_NumberOfFields(infoIter));
    while (VecSimDebugInfoIterator_HasNextField(infoIter)) {
        VecSim_InfoField *infoField = VecSimDebugInfoIterator_NextField(infoIter);
        if (!strcmp(infoField->fieldName, VecSimCommonStrings::ALGORITHM_STRING)) {
            // Algorithm type.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimAlgo_ToString(info.commonInfo.basicInfo.algo));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::TYPE_STRING)) {
            // Vector type.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimType_ToString(info.commonInfo.basicInfo.type));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::DIMENSION_STRING)) {
            // Vector dimension.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.dim);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::METRIC_STRING)) {
            // Metric.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimMetric_ToString(info.commonInfo.basicInfo.metric));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::SEARCH_MODE_STRING)) {
            // Search mode.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimSearchMode_ToString(info.commonInfo.lastMode));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::INDEX_SIZE_STRING)) {
            // Index size.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.indexSize);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::INDEX_LABEL_COUNT_STRING)) {
            // Index label count.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.indexLabelCount);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::IS_MULTI_STRING)) {
            // Is the index multi value.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.isMulti);
        } else if (!strcmp(infoField->fieldName,
                           VecSimCommonStrings::HNSW_EF_CONSTRUCTION_STRING)) {
            // EF construction.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.hnswInfo.efConstruction);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::HNSW_EF_RUNTIME_STRING)) {
            // EF runtime.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.hnswInfo.efRuntime);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::HNSW_EPSILON_STRING)) {
            // Epsilon.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_FLOAT64);
            ASSERT_EQ(infoField->fieldValue.floatingPointValue, info.hnswInfo.epsilon);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::HNSW_M_STRING)) {
            // M.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.hnswInfo.M);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::HNSW_MAX_LEVEL)) {
            // Levels.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.hnswInfo.max_level);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::HNSW_ENTRYPOINT)) {
            // Entrypoint.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.hnswInfo.entrypoint);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::MEMORY_STRING)) {
            // Memory.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.memory);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::BLOCK_SIZE_STRING)) {
            // Block size.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.blockSize);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::HNSW_NUM_MARKED_DELETED)) {
            // Number of marked deleted.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue,
                      info.hnswInfo.numberOfMarkedDeletedNodes);
        } else {
            FAIL();
        }
    }
}

void compareTieredHNSWIndexInfoToIterator(VecSimIndexDebugInfo info,
                                          VecSimIndexDebugInfo frontendIndexInfo,
                                          VecSimIndexDebugInfo backendIndexInfo,
                                          VecSimDebugInfoIterator *infoIterator) {
    while (infoIterator->hasNext()) {
        VecSim_InfoField *infoField = VecSimDebugInfoIterator_NextField(infoIterator);

        if (!strcmp(infoField->fieldName, VecSimCommonStrings::ALGORITHM_STRING)) {
            // Algorithm type.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue, VecSimCommonStrings::TIERED_STRING);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::TYPE_STRING)) {
            // Vector type.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimType_ToString(info.commonInfo.basicInfo.type));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::DIMENSION_STRING)) {
            // Vector dimension.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.dim);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::METRIC_STRING)) {
            // Metric.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimMetric_ToString(info.commonInfo.basicInfo.metric));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::SEARCH_MODE_STRING)) {
            // Search mode.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimSearchMode_ToString(info.commonInfo.lastMode));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::INDEX_SIZE_STRING)) {
            // Index size.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.indexSize);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::INDEX_LABEL_COUNT_STRING)) {
            // Index label count.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.indexLabelCount);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::IS_MULTI_STRING)) {
            // Is the index multi value.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.isMulti);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::MEMORY_STRING)) {
            // Memory.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.memory);
        } else if (!strcmp(infoField->fieldName,
                           VecSimCommonStrings::TIERED_MANAGEMENT_MEMORY_STRING)) {
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.tieredInfo.management_layer_memory);
        } else if (!strcmp(infoField->fieldName,
                           VecSimCommonStrings::TIERED_BACKGROUND_INDEXING_STRING)) {
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.tieredInfo.backgroundIndexing);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::FRONTEND_INDEX_STRING)) {
            ASSERT_EQ(infoField->fieldType, INFOFIELD_ITERATOR);
            compareFlatIndexInfoToIterator(frontendIndexInfo, infoField->fieldValue.iteratorValue);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::BACKEND_INDEX_STRING)) {
            ASSERT_EQ(infoField->fieldType, INFOFIELD_ITERATOR);
            compareHNSWIndexInfoToIterator(backendIndexInfo, infoField->fieldValue.iteratorValue);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::TIERED_BUFFER_LIMIT_STRING)) {
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.tieredInfo.bufferLimit);
        } else if (!strcmp(infoField->fieldName,
                           VecSimCommonStrings::TIERED_HNSW_SWAP_JOBS_THRESHOLD_STRING)) {
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(
                infoField->fieldValue.uintegerValue,
                info.tieredInfo.specificTieredBackendInfo.hnswTieredInfo.pendingSwapJobsThreshold);
        } else {
            FAIL();
        }
    }
}

void compareSVSIndexInfoToIterator(VecSimIndexDebugInfo info, VecSimDebugInfoIterator *infoIter) {
    ASSERT_EQ(10, VecSimDebugInfoIterator_NumberOfFields(infoIter));
    while (VecSimDebugInfoIterator_HasNextField(infoIter)) {
        VecSim_InfoField *infoField = VecSimDebugInfoIterator_NextField(infoIter);
        if (!strcmp(infoField->fieldName, VecSimCommonStrings::ALGORITHM_STRING)) {
            // Algorithm type.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimAlgo_ToString(info.commonInfo.basicInfo.algo));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::TYPE_STRING)) {
            // Vector type.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimType_ToString(info.commonInfo.basicInfo.type));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::DIMENSION_STRING)) {
            // Vector dimension.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.dim);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::METRIC_STRING)) {
            // Metric.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimMetric_ToString(info.commonInfo.basicInfo.metric));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::SEARCH_MODE_STRING)) {
            // Search mode.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoField->fieldValue.stringValue,
                         VecSimSearchMode_ToString(info.commonInfo.lastMode));
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::INDEX_SIZE_STRING)) {
            // Index size.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.indexSize);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::INDEX_LABEL_COUNT_STRING)) {
            // Index label count.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.indexLabelCount);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::IS_MULTI_STRING)) {
            // Is the index multi value.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.isMulti);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::MEMORY_STRING)) {
            // Memory.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.memory);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::BLOCK_SIZE_STRING)) {
            // Block size.
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.commonInfo.basicInfo.blockSize);
        } else {
            FAIL();
        }
    }
}

size_t getLabelsLookupNodeSize() {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    auto dummy_lookup = vecsim_stl::unordered_map<size_t, unsigned int>(1, allocator);
    size_t memory_before = allocator->getAllocationSize();
    dummy_lookup.insert({1, 1}); // Insert a dummy {key, value} element pair.
    size_t memory_after = allocator->getAllocationSize();
    return memory_after - memory_before;
}
namespace test_utils {
size_t CalcVectorDataSize(VecSimIndex *index, VecSimType data_type) {
    switch (data_type) {
    case VecSimType_FLOAT32: {
        VecSimIndexAbstract<float, float> *abs_index =
            dynamic_cast<VecSimIndexAbstract<float, float> *>(index);
        assert(abs_index &&
               "dynamic_cast failed: can't convert index to VecSimIndexAbstract<float, float>");
        return abs_index->getDataSize();
    }
    case VecSimType_FLOAT64: {
        VecSimIndexAbstract<double, double> *abs_index =
            dynamic_cast<VecSimIndexAbstract<double, double> *>(index);
        assert(abs_index &&
               "dynamic_cast failed: can't convert index to VecSimIndexAbstract<double, double>");
        return abs_index->getDataSize();
    }
    case VecSimType_BFLOAT16: {
        VecSimIndexAbstract<vecsim_types::bfloat16, float> *abs_index =
            dynamic_cast<VecSimIndexAbstract<vecsim_types::bfloat16, float> *>(index);
        assert(abs_index && "dynamic_cast failed: can't convert index to "
                            "VecSimIndexAbstract<vecsim_types::bfloat16, float>");
        return abs_index->getDataSize();
    }
    case VecSimType_FLOAT16: {
        VecSimIndexAbstract<vecsim_types::float16, float> *abs_index =
            dynamic_cast<VecSimIndexAbstract<vecsim_types::float16, float> *>(index);
        assert(abs_index && "dynamic_cast failed: can't convert index to "
                            "VecSimIndexAbstract<vecsim_types::float16, float>");
        return abs_index->getDataSize();
    }
    case VecSimType_INT8: {
        VecSimIndexAbstract<int8_t, float> *abs_index =
            dynamic_cast<VecSimIndexAbstract<int8_t, float> *>(index);
        assert(abs_index &&
               "dynamic_cast failed: can't convert index to VecSimIndexAbstract<int8_t, float>");
        return abs_index->getDataSize();
    }
    case VecSimType_UINT8: {
        VecSimIndexAbstract<uint8_t, float> *abs_index =
            dynamic_cast<VecSimIndexAbstract<uint8_t, float> *>(index);
        assert(abs_index &&
               "dynamic_cast failed: can't convert index to VecSimIndexAbstract<uint8_t, float>");
        return abs_index->getDataSize();
    }
    default:
        return 0;
    }
}

TieredIndexParams CreateTieredParams(VecSimParams &primary_params,
                                     tieredIndexMock &mock_thread_pool) {
    TieredIndexParams tiered_params = {.jobQueue = &mock_thread_pool.jobQ,
                                       .jobQueueCtx = mock_thread_pool.ctx,
                                       .submitCb = tieredIndexMock::submit_callback,
                                       .flatBufferLimit = SIZE_MAX,
                                       .primaryIndexParams = &primary_params,
                                       .specificParams = {TieredHNSWParams{.swapJobThreshold = 0}}};

    return tiered_params;
}

VecSimIndex *CreateNewTieredHNSWIndex(const HNSWParams &hnsw_params,
                                      tieredIndexMock &mock_thread_pool) {
    VecSimParams primary_params = CreateParams(hnsw_params);
    auto tiered_params = CreateTieredParams(primary_params, mock_thread_pool);
    VecSimParams params = CreateParams(tiered_params);
    VecSimIndex *index = VecSimIndex_New(&params);
    mock_thread_pool.ctx->index_strong_ref.reset(index);

    return index;
}
} // namespace test_utils
