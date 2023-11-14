#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/algorithms/raft_ivf/ivf_tiered.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "test_utils.h"
#include <climits>
#include <unistd.h>
#include <random>

#include "mock_thread_pool.h"


template <typename index_type_t>
class RaftIvfTieredTest : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    TieredRaftIvfIndex<data_t, dist_t>* createTieredIndex(VecSimParams *params,
                                     tieredIndexMock &mock_thread_pool,
                                     size_t flat_buffer_limit = 0) {
        TieredIndexParams params_tiered = {
            .jobQueue = &mock_thread_pool.jobQ,
            .jobQueueCtx = mock_thread_pool.ctx,
            .submitCb = tieredIndexMock::submit_callback,
            .flatBufferLimit = flat_buffer_limit,
            .primaryIndexParams = params,
        };
        auto *tiered_index = TieredFactory::NewIndex(&params_tiered);
        // Set the created tiered index in the index external context (it will take ownership over
        // the index, and we'll need to release the ctx at the end of the test.
        mock_thread_pool.ctx->index_strong_ref.reset(tiered_index);

        return reinterpret_cast<TieredRaftIvfIndex<data_t, dist_t> *>(tiered_index);
    }
};

VecSimParams createDefaultPQParams(size_t dim, uint32_t nLists = 3, uint32_t nProbes = 3) {
    RaftIvfParams ivfparams = {.dim = dim,
                              .metric = VecSimMetric_L2,
                              .nLists = nLists,
                              .kmeans_nIters = 20,
                              .kmeans_trainsetFraction = 0.5,
                              .nProbes = nProbes,
                              .usePQ = true,
                              .pqBits = 8,
                              .pqDim = 0,
                              .codebookKind = RaftIVFPQCodebookKind_PerSubspace,
                              .lutType = CUDAType_R_32F,
                              .internalDistanceType = CUDAType_R_32F,
                              .preferredShmemCarveout = 1.0};
    VecSimParams params{.algo = VecSimAlgo_RAFT_IVFPQ, .algoParams = {.raftIvfParams = ivfparams}};
    return params;
}

VecSimParams createDefaultFlatParams(size_t dim, uint32_t nLists = 3, uint32_t nProbes = 3) {
    RaftIvfParams ivfparams = {.dim = dim,
                                .metric = VecSimMetric_L2,
                                .nLists = nLists,
                                .kmeans_nIters = 20,
                                .kmeans_trainsetFraction = 0.5,
                                .nProbes = nProbes,
                                .usePQ = false};
    VecSimParams params{.algo = VecSimAlgo_RAFT_IVFFLAT, .algoParams = {.raftIvfParams = ivfparams}};
    return params;
}

using DataTypeSetFloat = ::testing::Types<IndexType<VecSimType_FLOAT32, float>>;

TYPED_TEST_SUITE(RaftIvfTieredTest, DataTypeSetFloat);

TYPED_TEST(RaftIvfTieredTest, end_to_end) {
    size_t dim = 4;
    size_t flat_buffer_limit = 3;
    size_t nLists = 2;

    VecSimParams params = createDefaultFlatParams(dim, nLists, nLists);
    auto mock_thread_pool = tieredIndexMock();
    auto *index = this->createTieredIndex(&params, mock_thread_pool, flat_buffer_limit);
    mock_thread_pool.init_threads();
    
    VecSimQueryParams queryParams = {.batchSize = 1};

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T a[dim], b[dim], c[dim], d[dim], e[dim], zero[dim];
    std::vector<TEST_DATA_T> a_vec(dim, (TEST_DATA_T)1);
    std::vector<TEST_DATA_T> b_vec(dim, (TEST_DATA_T)2);
    std::vector<TEST_DATA_T> c_vec(dim, (TEST_DATA_T)4);
    std::vector<TEST_DATA_T> d_vec(dim, (TEST_DATA_T)5);
    std::vector<TEST_DATA_T> zero_vec(dim, (TEST_DATA_T)0);

    auto inserted_vectors = std::vector<std::vector<TEST_DATA_T>>{a_vec, b_vec, c_vec, d_vec};

    // Search for vectors when the index is empty.
    runTopKSearchTest(index, a_vec.data(), 1, nullptr);

    // Add vectors.
    VecSimIndex_AddVector(index, a_vec.data(), 0);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    VecSimIndex_AddVector(index, b_vec.data(), 1);
    VecSimIndex_AddVector(index, c_vec.data(), 2);
    VecSimIndex_AddVector(index, d_vec.data(), 3);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 4);


    mock_thread_pool.thread_pool_join();
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
    // Callbacks for verifying results.
    auto ver_res_0 = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, index);
        ASSERT_DOUBLE_EQ(score, dim * inserted_vectors[id][0] * inserted_vectors[id][0]);
    };
    size_t result_c[] = {2, 3, 1, 0}; // Order of results for query on c.
    auto ver_res_c = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, result_c[index]);
        double dist = inserted_vectors[id][0] - c_vec[0];
        ASSERT_DOUBLE_EQ(score, dim * dist * dist);
    };

    auto k = 4;
    runTopKSearchTest(index, zero_vec.data(), k, ver_res_0);
    runTopKSearchTest(index, c_vec.data(), k, ver_res_c);
}

