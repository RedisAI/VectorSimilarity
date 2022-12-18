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

void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k,
                       std::function<void(size_t, double, size_t)> ResCB, VecSimQueryParams *params,
                       VecSimQueryResult_Order order) {
    VecSimQueryResult_List res = VecSimIndex_TopKQuery(index, query, k, params, order);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    ASSERT_TRUE(allUniqueResults(res));
    VecSimQueryResult_Iterator *iterator = VecSimQueryResult_List_GetIterator(res);
    int res_ind = 0;
    while (VecSimQueryResult_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryResult_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        double score = VecSimQueryResult_GetScore(item);
        ResCB(id, score, res_ind++);
    }
    ASSERT_EQ(res_ind, k);
    VecSimQueryResult_IteratorFree(iterator);
    VecSimQueryResult_Free(res);
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

void compareFlatIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter) {
    ASSERT_EQ(10, VecSimInfoIterator_NumberOfFields(infoIter));
    while (VecSimInfoIterator_HasNextField(infoIter)) {
        VecSim_InfoField *infoFiled = VecSimInfoIterator_NextField(infoIter);
        if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::ALGORITHM_STRING)) {
            // Algorithm type.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoFiled->fieldValue.stringValue, VecSimAlgo_ToString(info.algo));
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::TYPE_STRING)) {
            // Vector type.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoFiled->fieldValue.stringValue, VecSimType_ToString(info.bfInfo.type));
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::DIMENSION_STRING)) {
            // Vector dimension.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.bfInfo.dim);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::METRIC_STRING)) {
            // Metric.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoFiled->fieldValue.stringValue,
                         VecSimMetric_ToString(info.bfInfo.metric));
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::SEARCH_MODE_STRING)) {
            // Search mode.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoFiled->fieldValue.stringValue,
                         VecSimSearchMode_ToString(info.bfInfo.last_mode));
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::INDEX_SIZE_STRING)) {
            // Index size.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.bfInfo.indexSize);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::INDEX_LABEL_COUNT_STRING)) {
            // Index label count.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.bfInfo.indexLabelCount);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::IS_MULTI_STRING)) {
            // Is the index multi value.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.bfInfo.isMulti);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::BLOCK_SIZE_STRING)) {
            // Block size.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.bfInfo.blockSize);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::MEMORY_STRING)) {
            // Memory.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.bfInfo.memory);
        } else {
            ASSERT_TRUE(false);
        }
    }
}

void compareHNSWIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter) {
    ASSERT_EQ(15, VecSimInfoIterator_NumberOfFields(infoIter));
    while (VecSimInfoIterator_HasNextField(infoIter)) {
        VecSim_InfoField *infoFiled = VecSimInfoIterator_NextField(infoIter);
        if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::ALGORITHM_STRING)) {
            // Algorithm type.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoFiled->fieldValue.stringValue, VecSimAlgo_ToString(info.algo));
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::TYPE_STRING)) {
            // Vector type.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoFiled->fieldValue.stringValue,
                         VecSimType_ToString(info.hnswInfo.type));
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::DIMENSION_STRING)) {
            // Vector dimension.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.hnswInfo.dim);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::METRIC_STRING)) {
            // Metric.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoFiled->fieldValue.stringValue,
                         VecSimMetric_ToString(info.hnswInfo.metric));
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::SEARCH_MODE_STRING)) {
            // Search mode.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_STRING);
            ASSERT_STREQ(infoFiled->fieldValue.stringValue,
                         VecSimSearchMode_ToString(info.hnswInfo.last_mode));
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::INDEX_SIZE_STRING)) {
            // Index size.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.hnswInfo.indexSize);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::INDEX_LABEL_COUNT_STRING)) {
            // Index label count.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.hnswInfo.indexLabelCount);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::IS_MULTI_STRING)) {
            // Is the index multi value.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.hnswInfo.isMulti);
        } else if (!strcmp(infoFiled->fieldName,
                           VecSimCommonStrings::HNSW_EF_CONSTRUCTION_STRING)) {
            // EF construction.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.hnswInfo.efConstruction);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::HNSW_EF_RUNTIME_STRING)) {
            // EF runtime.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.hnswInfo.efRuntime);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::HNSW_EPSILON_STRING)) {
            // Epsilon.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_FLOAT64);
            ASSERT_EQ(infoFiled->fieldValue.floatingPointValue, info.hnswInfo.epsilon);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::HNSW_M_STRING)) {
            // M.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.hnswInfo.M);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::HNSW_MAX_LEVEL)) {
            // Levels.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.hnswInfo.max_level);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::HNSW_ENTRYPOINT)) {
            // Entrypoint.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.hnswInfo.entrypoint);
        } else if (!strcmp(infoFiled->fieldName, VecSimCommonStrings::MEMORY_STRING)) {
            // Memory.
            ASSERT_EQ(infoFiled->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoFiled->fieldValue.uintegerValue, info.hnswInfo.memory);
        } else {
            ASSERT_TRUE(false);
        }
    }
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
    ASSERT_EQ(VecSimQueryResult_Len(res), expected_res_num);
    ASSERT_TRUE(allUniqueResults(res));
    VecSimQueryResult_Iterator *iterator = VecSimQueryResult_List_GetIterator(res);
    int res_ind = 0;
    while (VecSimQueryResult_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryResult_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        double score = VecSimQueryResult_GetScore(item);
        ResCB(id, score, res_ind++);
    }
    ASSERT_EQ(res_ind, expected_res_num);
    VecSimQueryResult_IteratorFree(iterator);
    VecSimQueryResult_Free(res);
}

size_t getLabelsLookupNodeSize() {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    auto dummy_lookup = vecsim_stl::unordered_map<size_t, unsigned int>(1, allocator);
    size_t memory_before = allocator->getAllocationSize();
    dummy_lookup.insert({1, 1}); // Insert a dummy {key, value} element pair.
    size_t memory_after = allocator->getAllocationSize();
    return memory_after - memory_before;
}

/*
 * Mock callbacks for testing async tiered index. We use a simple std::queue to simulate the job
 * queue.
 */
int tiered_index_mock::submit_callback(void *job_queue, void **jobs, size_t len) {
    for (size_t i = 0; i < len; i++) {
        static_cast<JobQueue *>(job_queue)->push(jobs[i]);
    }
    return VecSim_OK;
}

int tiered_index_mock::update_mem_callback(void *mem_ctx, size_t mem) {
    *(size_t *)mem_ctx = mem;
    return VecSim_OK;
}
