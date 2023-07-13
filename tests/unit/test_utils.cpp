/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "test_utils.h"
#include "gtest/gtest.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"

VecsimQueryType test_utils::query_types[4] = {QUERY_TYPE_NONE, QUERY_TYPE_KNN, QUERY_TYPE_HYBRID,
                                              QUERY_TYPE_RANGE};

static bool allUniqueResults(VecSimQueryResult_List res) {
    size_t len = VecSimQueryResult_Len(res);
    auto it1 = VecSimQueryResult_List_GetIterator(res);
    for (size_t i = 0; i < len; i++) {
        auto ei = VecSimQueryResult_IteratorNext(it1);
        auto it2 = VecSimQueryResult_List_GetIterator(res);
        for (size_t j = 0; j < i; j++) {
            auto ej = VecSimQueryResult_IteratorNext(it2);
            if (VecSimQueryResult_GetId(ei) == VecSimQueryResult_GetId(ej)) {
                VecSimQueryResult_IteratorFree(it2);
                VecSimQueryResult_IteratorFree(it1);
                return false;
            }
        }
        VecSimQueryResult_IteratorFree(it2);
    }
    VecSimQueryResult_IteratorFree(it1);
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

static bool is_async_index(VecSimIndex *index) {
    return dynamic_cast<VecSimTieredIndex<float, float> *>(index) != nullptr ||
           dynamic_cast<VecSimTieredIndex<double, double> *>(index) != nullptr;
}

void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k, size_t expected_num_res,
                       std::function<void(size_t, double, size_t)> ResCB, VecSimQueryParams *params,
                       VecSimQueryResult_Order order) {
    VecSimQueryResult_List res = VecSimIndex_TopKQuery(index, query, k, params, order);
    if (is_async_index(index)) {
        // Async index may return more or less than the expected number of results,
        // depending on the number of results that were available at the time of the query.
        // We can estimate the number of results that should be returned to be roughly
        // `expected_num_res` +- number of threads in the pool of the job queue.

        // for now, lets only check that the number of results is not greater than k.
        ASSERT_LE(VecSimQueryResult_Len(res), k);
    } else {
        ASSERT_EQ(VecSimQueryResult_Len(res), expected_num_res);
    }
    ASSERT_TRUE(allUniqueResults(res));
    VecSimQueryResult_Iterator *iterator = VecSimQueryResult_List_GetIterator(res);
    int res_ind = 0;
    while (VecSimQueryResult_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryResult_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        double score = VecSimQueryResult_GetScore(item);
        ResCB(id, score, res_ind++);
    }
    VecSimQueryResult_IteratorFree(iterator);
    VecSimQueryResult_Free(res);
}

void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k,
                       std::function<void(size_t, double, size_t)> ResCB, VecSimQueryParams *params,
                       VecSimQueryResult_Order order) {
    size_t expected_num_res = std::min(VecSimIndex_IndexSize(index), k);
    runTopKSearchTest(index, query, k, expected_num_res, ResCB, params, order);
}

/*
 * helper function to run batch search iteration, and iterate over the results. ResCB is a callback
 * that takes the id, score and index of a result, and performs test-specific logic for each.
 */
void runBatchIteratorSearchTest(VecSimBatchIterator *batch_iterator, size_t n_res,
                                std::function<void(size_t, double, size_t)> ResCB,
                                VecSimQueryResult_Order order, size_t expected_n_res) {
    if (expected_n_res == SIZE_MAX)
        expected_n_res = n_res;
    VecSimQueryResult_List res = VecSimBatchIterator_Next(batch_iterator, n_res, order);
    ASSERT_EQ(VecSimQueryResult_Len(res), expected_n_res);
    ASSERT_TRUE(allUniqueResults(res));
    VecSimQueryResult_Iterator *iterator = VecSimQueryResult_List_GetIterator(res);
    int res_ind = 0;
    while (VecSimQueryResult_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryResult_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        double score = VecSimQueryResult_GetScore(item);
        ResCB(id, score, res_ind++);
    }
    ASSERT_EQ(res_ind, expected_n_res);
    VecSimQueryResult_IteratorFree(iterator);
    VecSimQueryResult_Free(res);
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
                       size_t expected_res_num, VecSimQueryResult_Order order,
                       VecSimQueryParams *params) {
    VecSimQueryResult_List res =
        VecSimIndex_RangeQuery(index, (const void *)query, radius, params, order);
    EXPECT_EQ(VecSimQueryResult_Len(res), expected_res_num);
    EXPECT_TRUE(allUniqueResults(res));
    VecSimQueryResult_Iterator *iterator = VecSimQueryResult_List_GetIterator(res);
    int res_ind = 0;
    while (VecSimQueryResult_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryResult_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        double score = VecSimQueryResult_GetScore(item);
        ResCB(id, score, res_ind++);
    }
    EXPECT_EQ(res_ind, expected_res_num);
    VecSimQueryResult_IteratorFree(iterator);
    VecSimQueryResult_Free(res);
}

void compareFlatIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter) {
    ASSERT_EQ(10, VecSimInfoIterator_NumberOfFields(infoIter));
    while (VecSimInfoIterator_HasNextField(infoIter)) {
        VecSim_InfoField *infoField = VecSimInfoIterator_NextField(infoIter);
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

void compareHNSWIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter) {
    ASSERT_EQ(17, VecSimInfoIterator_NumberOfFields(infoIter));
    while (VecSimInfoIterator_HasNextField(infoIter)) {
        VecSim_InfoField *infoField = VecSimInfoIterator_NextField(infoIter);
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

size_t getLabelsLookupNodeSize() {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    auto dummy_lookup = vecsim_stl::unordered_map<size_t, unsigned int>(1, allocator);
    size_t memory_before = allocator->getAllocationSize();
    dummy_lookup.insert({1, 1}); // Insert a dummy {key, value} element pair.
    size_t memory_after = allocator->getAllocationSize();
    return memory_after - memory_before;
}
