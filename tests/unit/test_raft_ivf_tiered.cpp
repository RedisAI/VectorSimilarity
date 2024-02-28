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

    TieredRaftIvfIndex<data_t, dist_t> *createTieredIndex(VecSimParams *params,
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
    VecSimParams params{.algo = VecSimAlgo_RAFT_IVFFLAT,
                        .algoParams = {.raftIvfParams = ivfparams}};
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

TYPED_TEST(RaftIvfTieredTest, transferJob) {
    // Create RAFT Tiered index instance with a mock queue.

    size_t dim = 4;
    size_t flat_buffer_limit = 3;
    size_t nLists = 1;

    VecSimParams params = createDefaultFlatParams(dim, nLists, nLists);
    auto mock_thread_pool = tieredIndexMock();
    auto *tiered_index = this->createTieredIndex(&params, mock_thread_pool, flat_buffer_limit);
    auto allocator = tiered_index->getAllocator();

    VecSimQueryParams queryParams = {.batchSize = 1};

    // Create a vector and add it to the tiered index.
    labelType vec_label = 1;
    TEST_DATA_T vector[dim];
    GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
    VecSimIndex_AddVector(tiered_index, vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->getDistanceFrom_Unsafe(vec_label, vector), 0);

    // Execute the insert job manually (in a synchronous manner).
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 1);
    auto *insertion_job = reinterpret_cast<RAFTTransferJob *>(mock_thread_pool.jobQ.front().job);
    ASSERT_EQ(insertion_job->jobType, RAFT_TRANSFER_JOB);

    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);
    // RAFT IVF index should have allocated a single block, while flat index should remove the
    // block.
    ASSERT_EQ(tiered_index->frontendIndex->indexCapacity(), 0);
    // After the execution, the job should be removed from the job queue.
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(RaftIvfTieredTest, transferJobAsync) {
    size_t dim = 32;
    size_t n = 500;
    size_t nLists = 120;
    size_t flat_buffer_limit = 160;

    size_t k = 1;

    // Create RaftIvfTiered index instance with a mock queue.
    VecSimParams params = createDefaultFlatParams(dim, nLists, 20);
    auto mock_thread_pool = tieredIndexMock();
    auto *tiered_index = this->createTieredIndex(&params, mock_thread_pool, flat_buffer_limit);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    // Insert vectors
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i, i);
    }

    mock_thread_pool.thread_pool_join();
    // Verify that the vectors were inserted to RaftIvf as expected, that the jobqueue is empty,
    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), n);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);
    // Verify that the vectors were inserted to RaftIvf as expected
    for (size_t i = 0; i < size_t{n / 10}; i++) {
        TEST_DATA_T expected_vector[dim];
        GenerateVector<TEST_DATA_T>(expected_vector, dim, i);
        VecSimQueryReply *res = VecSimIndex_TopKQuery(tiered_index->backendIndex, expected_vector,
                                                      k, nullptr, BY_SCORE);
        ASSERT_EQ(VecSimQueryReply_GetCode(res), VecSim_QueryReply_OK);
        ASSERT_EQ(VecSimQueryReply_Len(res), k);
        ASSERT_EQ(res->results[0].id, i);
        ASSERT_EQ(res->results[0].score, 0);
        VecSimQueryReply_Free(res);
    }
}

TYPED_TEST(RaftIvfTieredTest, transferJob_inplace) {
    size_t dim = 32;
    size_t n = 200;
    size_t nLists = 120;
    size_t flat_buffer_limit = 160;

    size_t k = 1;

    // Create RaftIvfTiered index instance with a mock queue.
    VecSimParams params = createDefaultFlatParams(dim, nLists, 20);
    auto mock_thread_pool = tieredIndexMock();
    auto *tiered_index = this->createTieredIndex(&params, mock_thread_pool, flat_buffer_limit);

    // In the absence of BG threads to takes jobs from the queue, the tiered index should
    // transfer in place when flat_buffer is over the limit.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i, i);
    }

    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), flat_buffer_limit);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), n - flat_buffer_limit);

    // Run another batch of insertion. The tiered index should transfer inplace again.
    for (size_t i = n; i < n * 2; i++) {
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i, i);
    }
    ASSERT_EQ(tiered_index->indexSize(), 2 * n);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), flat_buffer_limit * 2);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 2 * (n - flat_buffer_limit));
}

TYPED_TEST(RaftIvfTieredTest, deleteVector_backend) {
    size_t dim = 32;
    size_t n = 500;
    size_t nLists = 120;
    size_t nDelete = 10;
    size_t flat_buffer_limit = 1000;

    size_t k = 1;

    // Create RaftIvfTiered index instance with a mock queue.
    VecSimParams params = createDefaultFlatParams(dim, nLists, nLists);
    auto mock_thread_pool = tieredIndexMock();
    auto *tiered_index = this->createTieredIndex(&params, mock_thread_pool, flat_buffer_limit);

    labelType vec_label = 0;
    // Delete from an empty index.
    ASSERT_EQ(VecSimIndex_DeleteVector(tiered_index, vec_label), 0);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    // Insert vectors
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i, i);
    }
    // Use just one thread to transfer all the vectors
    mock_thread_pool.thread_pool_wait(100);

    // Check that the backend index has the first 12 vectors
    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), n);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    for (size_t i = 0; i < nDelete + 2; i++) {
        TEST_DATA_T expected_vector[dim];
        GenerateVector<TEST_DATA_T>(expected_vector, dim, i);
        VecSimQueryReply *res = VecSimIndex_TopKQuery(tiered_index->backendIndex, expected_vector,
                                                      k, nullptr, BY_SCORE);
        ASSERT_EQ(VecSimQueryReply_GetCode(res), VecSim_QueryReply_OK);
        ASSERT_EQ(VecSimQueryReply_Len(res), k);
        ASSERT_EQ(res->results[0].id, i);
        ASSERT_EQ(res->results[0].score, 0);
        VecSimQueryReply_Free(res);
    }

    // Delete 10 first vectors
    for (size_t i = 0; i < nDelete; i++) {
        VecSimIndex_DeleteVector(tiered_index, i);
    }

    ASSERT_EQ(tiered_index->indexSize(), n - nDelete);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), n - nDelete);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
}

TYPED_TEST(RaftIvfTieredTest, searchMetricCosine) {
    size_t dim = 32;
    size_t n = 25;
    size_t nLists = 5;
    size_t flat_buffer_limit = 100;

    size_t k = 10;

    // Create RaftIvfTiered index instance with a mock queue.
    VecSimParams params = createDefaultFlatParams(dim, nLists, nLists);

    // Set the metric to cosine.
    params.algoParams.raftIvfParams.metric = VecSimMetric_Cosine;

    auto mock_thread_pool = tieredIndexMock();
    auto *tiered_index = this->createTieredIndex(&params, mock_thread_pool, flat_buffer_limit);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    std::vector<std::vector<TEST_DATA_T>> inserted_vectors;

    for (size_t i = 0; i < n; i++) {
        inserted_vectors.push_back(std::vector<TEST_DATA_T>(dim));
        // Generate vectors
        for (size_t j = 0; j < dim; j++) {
            inserted_vectors.back()[j] = (TEST_DATA_T)i + j;
        }
        // Insert vectors
        VecSimIndex_AddVector(tiered_index, inserted_vectors.back().data(), i);
    }
    mock_thread_pool.thread_pool_wait(100);

    // The query is a vector with half of the values equal to 8.1 and the other half equal to 1.1.
    TEST_DATA_T query[dim];
    TEST_DATA_T query_norm[dim];
    GenerateVector<TEST_DATA_T>(query, dim / 2, 8.1f);
    GenerateVector<TEST_DATA_T>(query + dim / 2, dim / 2, 1.1f);
    memcpy(query_norm, query, dim * sizeof(TEST_DATA_T));
    VecSim_Normalize(query_norm, dim, VecSimType_FLOAT32);

    auto verify_cb = [&](size_t id, double score, size_t index) {
        TEST_DATA_T neighbor_norm[dim];
        memcpy(neighbor_norm, inserted_vectors[id].data(), dim * sizeof(TEST_DATA_T));
        VecSim_Normalize(neighbor_norm, dim, VecSimType_FLOAT32);

        // Use distance function of the bruteforce index to verify the score.
        double dist = tiered_index->frontendIndex->getDistFunc()(
            query_norm,
            neighbor_norm,
            dim);
        ASSERT_NEAR(score, dist, 1e-5);
    };

    runTopKSearchTest(tiered_index, query, k, verify_cb);
}

TYPED_TEST(RaftIvfTieredTest, searchMetricIP) {
    size_t dim = 4;
    size_t n = 25;
    size_t nLists = 5;
    size_t flat_buffer_limit = 100;

    size_t k = 10;

    // Create RaftIvfTiered index instance with a mock queue.
    VecSimParams params = createDefaultFlatParams(dim, nLists, nLists);

    // Set the metric to Inner Product.
    params.algoParams.raftIvfParams.metric = VecSimMetric_IP;

    auto mock_thread_pool = tieredIndexMock();
    auto *tiered_index = this->createTieredIndex(&params, mock_thread_pool, flat_buffer_limit);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    std::vector<std::vector<TEST_DATA_T>> inserted_vectors;

    for (size_t i = 0; i < n; i++) {
        inserted_vectors.push_back(std::vector<TEST_DATA_T>(dim));
        // Generate vectors
        for (size_t j = 0; j < dim; j++) {
            inserted_vectors.back()[j] = (TEST_DATA_T)i + j;
        }
        // Insert vectors
        VecSimIndex_AddVector(tiered_index, inserted_vectors.back().data(), i);
    }
    mock_thread_pool.thread_pool_wait(100);

    // The query is a vector with half of the values equal to 1.1 and the other half equal to 0.1.
    TEST_DATA_T query[dim] = {1.1f, 1.1f, 0.1f, 0.1f};

    auto verify_cb = [&](size_t id, double score, size_t index) {
        // Use distance function of the bruteforce index to verify the score.
        double dist = tiered_index->frontendIndex->getDistFunc()(
            query,
            inserted_vectors[id].data(),
            dim);
        ASSERT_NEAR(score, dist, 1e-5);
    };

    runTopKSearchTest(tiered_index, query, k, verify_cb);
}
