#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/raft_ivf/ivf_factory.h"
#include "test_utils.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/query_result_struct.h"
#include <climits>
#include <unistd.h>
#include <random>
#include <thread>
#include <atomic>

using namespace tiered_index_mock;

template <typename index_type_t>
class RaftIvfTieredTest : public ::testing::Test {};

TieredRaftIVFPQParams createDefaultPQTieredParams(size_t dim) {
    RaftIVFPQParams params = {.dim = dim,
                              .metric = VecSimMetric_L2,
                              .nLists = 3,
                              .pqBits = 8,
                              .pqDim = 0,
                              .codebookKind = RaftIVFPQ_PerSubspace,
                              .kmeans_nIters = 20,
                              .kmeans_trainsetFraction = 0.5,
                              .nProbes = 3,
                              .lutType = CUDAType_R_32F,
                              .internalDistanceType = CUDAType_R_32F,
                              .preferredShmemCarveout = 1.0};
    TieredRaftIVFPQParams params_tiered = {.PQParams = params};
    return params_tiered;
}

TieredRaftIVFFlatParams createDefaultFlatTieredParams(size_t dim) {
    RaftIVFFlatParams params = {.dim = dim,
                                .metric = VecSimMetric_L2,
                                .nLists = 3,
                                .kmeans_nIters = 20,
                                .kmeans_trainsetFraction = 0.5,
                                .nProbes = 3};
    TieredRaftIVFFlatParams params_tiered = {.flatParams = params};
    return params_tiered;
}

using DataTypeSetFloat = ::testing::Types<IndexType<VecSimType_FLOAT32, float>>;

TYPED_TEST_SUITE(RaftIvfTieredTest, DataTypeSetFloat);

TYPED_TEST(RaftIvfTieredTest, RaftIVFTiered_PQ_add_sanity_test) {
    size_t dim = 4;

    auto *jobQ = new JobQueue();
    size_t memory_ctx = 0;
    TieredIndexParams tiered_params = {.jobQueue = jobQ,
                                       .submitCb = submit_callback,
                                       .memoryCtx = &memory_ctx,
                                       .UpdateMemCb = update_mem_callback};

    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    TieredRaftIVFPQParams params_tiered = createDefaultPQTieredParams(dim);
    params_tiered.tieredParams = tiered_params;

    VecSimIndex *index = RaftIVFPQFactory::NewTieredIndex(&params_tiered, allocator);
    VecSimQueryParams queryParams = {.batchSize = 1};

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T a[dim], b[dim], c[dim], d[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (TEST_DATA_T)0;
        b[i] = (TEST_DATA_T)1;
        c[i] = (TEST_DATA_T)4;
        d[i] = (TEST_DATA_T)5;
    }

    VecSimQueryResult_List resultQuery;
    VecSimQueryResult_Iterator *it = nullptr;
    // Add first vector.
    VecSimIndex_AddVector(index, a, 0);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    VecSimIndex_AddVector(index, b, 1);
    VecSimIndex_AddVector(index, c, 2);
    VecSimIndex_AddVector(index, d, 3);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 4);

    resultQuery = index->topKQuery(a, 1, &queryParams);
    it = VecSimQueryResult_List_GetIterator(resultQuery);
    VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 0);
    ASSERT_EQ(currentResult->score, 0);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_Free(resultQuery);
    VecSimQueryResult_IteratorFree(it);

    resultQuery = index->topKQuery(c, 3, &queryParams);
    sort_results_by_id(resultQuery);
    ASSERT_EQ(resultQuery.results[0].id, 1);
    ASSERT_EQ(resultQuery.results[0].score, 36);
    ASSERT_EQ(resultQuery.results[1].id, 2);
    ASSERT_EQ(resultQuery.results[1].score, 0);
    ASSERT_EQ(resultQuery.results[2].id, 3);
    ASSERT_EQ(resultQuery.results[2].score, 4);
    VecSimQueryResult_Free(resultQuery);
    VecSimIndex_Free(index);
}

