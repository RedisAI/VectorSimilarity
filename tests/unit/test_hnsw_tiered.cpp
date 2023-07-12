#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"
#include <string>
#include <array>

#include "test_utils.h"
#include "mock_thread_pool.h"

#include <thread>

// Runs the test for all combination of data type(float/double) - label type (single/multi)

template <typename index_type_t>
class HNSWTieredIndexTest : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        auto tiered_index = reinterpret_cast<TieredHNSWIndex<data_t, dist_t> *>(index);
        return tiered_index->getHNSWIndex();
    }
    TieredHNSWIndex<data_t, dist_t> *CreateTieredHNSWIndex(VecSimParams &hnsw_params,
                                                           tieredIndexMock &mock_thread_pool,
                                                           size_t swap_job_threshold = 0,
                                                           size_t flat_buffer_limit = SIZE_MAX) {
        TieredIndexParams tiered_params = {
            .jobQueue = &mock_thread_pool.jobQ,
            .jobQueueCtx = mock_thread_pool.ctx,
            .submitCb = tieredIndexMock::submit_callback,
            .flatBufferLimit = flat_buffer_limit,
            .primaryIndexParams = &hnsw_params,
            .specificParams = {TieredHNSWParams{.swapJobThreshold = swap_job_threshold}}};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<data_t, dist_t> *>(
            TieredFactory::NewIndex(&tiered_params));

        // Set the created tiered index in the index external context (it will take ownership over
        // the index, and we'll need to release the ctx at the end of the test.
        mock_thread_pool.ctx->index_strong_ref.reset(tiered_index);
        return tiered_index;
    }
};

TYPED_TEST_SUITE(HNSWTieredIndexTest, DataTypeSetExtended);

// Runs the test for each data type(float/double). The label type should be explicitly
// set in the test.

template <typename index_type_t>
class HNSWTieredIndexTestBasic : public HNSWTieredIndexTest<index_type_t> {};
TYPED_TEST_SUITE(HNSWTieredIndexTestBasic, DataTypeSet);

TYPED_TEST(HNSWTieredIndexTest, CreateIndexInstance) {
    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = 4,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti()};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();
    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);

    // Get the allocator from the tiered index.
    auto allocator = tiered_index->getAllocator();

    // Add a vector to the flat index.
    TEST_DATA_T vector[tiered_index->backendIndex->getDim()];
    GenerateVector<TEST_DATA_T>(vector, tiered_index->backendIndex->getDim());
    labelType vector_label = 1;
    VecSimIndex_AddVector(tiered_index->frontendIndex, vector, vector_label);

    // Create a mock job that inserts some vector into the HNSW index.
    auto insert_to_index = [](AsyncJob *job) {
        auto *my_insert_job = reinterpret_cast<HNSWInsertJob *>(job);
        auto my_index =
            reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(my_insert_job->index);

        // Move the vector from the temp flat index into the HNSW index.
        // Note that we access the vector via its internal id since in index of type MULTI,
        // this is the only way to do so (knowing the label is not enough...)
        VecSimIndex_AddVector(my_index->backendIndex,
                              my_index->frontendIndex->getDataByInternalId(my_insert_job->id),
                              my_insert_job->label);
        // TODO: enable deleting vectors by internal id for the case of moving a single vector
        //  from the flat buffer in MULTI.
        VecSimIndex_DeleteVector(my_index->frontendIndex, my_insert_job->label);
        auto it = my_index->labelToInsertJobs.at(my_insert_job->label).begin();
        ASSERT_EQ(job, *it); // Assert pointers equation
        // Here we update labelToInsertJobs mapping, as we except that for every insert job
        // there will be a corresponding item in the map.
        my_index->labelToInsertJobs.at(my_insert_job->label).erase(it);
        delete job;
    };

    auto job = new (allocator)
        HNSWInsertJob(tiered_index->allocator, vector_label, 0, insert_to_index, tiered_index);
    auto jobs_vec = vecsim_stl::vector<HNSWInsertJob *>(1, job, allocator);
    tiered_index->labelToInsertJobs.insert({vector_label, jobs_vec});

    // Wrap this job with an array and submit the jobs to the queue.
    // TODO: in the future this should be part of the tiered index "add_vector" flow, and
    //  we can replace this to avoid the breaking of the abstraction.
    tiered_index->submitSingleJob((AsyncJob *)job);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 1);

    // Execute the job from the queue and validate that the index was updated properly.
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->getDistanceFrom(1, vector), 0);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vector_label).size(), 0);
}

TYPED_TEST(HNSWTieredIndexTest, testSizeEstimation) {
    size_t dim = 128;
    size_t n = DEFAULT_BLOCK_SIZE;
    size_t M = 32;
    size_t bs = DEFAULT_BLOCK_SIZE;
    bool isMulti = TypeParam::isMulti();

    HNSWParams hnsw_params = {.type = TypeParam::get_index_type(),
                              .dim = dim,
                              .metric = VecSimMetric_L2,
                              .multi = isMulti,
                              .initialCapacity = n,
                              .M = M};
    VecSimParams vecsim_hnsw_params = CreateParams(hnsw_params);

    auto mock_thread_pool = tieredIndexMock();
    TieredIndexParams tiered_params = {.jobQueue = &mock_thread_pool.jobQ,
                                       .jobQueueCtx = mock_thread_pool.ctx,
                                       .submitCb = tieredIndexMock::submit_callback,
                                       .flatBufferLimit = SIZE_MAX,
                                       .primaryIndexParams = &vecsim_hnsw_params};
    VecSimParams params = CreateParams(tiered_params);
    auto *index = VecSimIndex_New(&params);
    mock_thread_pool.ctx->index_strong_ref.reset(index);
    auto allocator = index->getAllocator();

    size_t initial_size_estimation = VecSimIndex_EstimateInitialSize(&params);

    // labels_lookup hash table has additional memory, since STL implementation chooses "an
    // appropriate prime number" higher than n as the number of allocated buckets (for n=1000, 1031
    // buckets are created)
    auto hnsw_index = this->CastToHNSW(index);
    if (isMulti == false) {
        auto hnsw = reinterpret_cast<HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T> *>(hnsw_index);
        initial_size_estimation += (hnsw->labelLookup.bucket_count() - n) * sizeof(size_t);
    } else { // if its a multi value index cast to HNSW_Multi
        auto hnsw = reinterpret_cast<HNSWIndex_Multi<TEST_DATA_T, TEST_DIST_T> *>(hnsw_index);
        initial_size_estimation += (hnsw->labelLookup.bucket_count() - n) * sizeof(size_t);
    }

    ASSERT_EQ(initial_size_estimation, index->getAllocationSize());

    // Add vectors up to initial capacity (initial capacity == block size).
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
        mock_thread_pool.thread_iteration();
    }

    // Estimate memory delta for filling up the first block and adding another block.
    size_t estimation = VecSimIndex_EstimateElementSize(&params) * bs;

    size_t before = index->getAllocationSize();
    GenerateAndAddVector<TEST_DATA_T>(index, dim, bs + n, bs + n);
    mock_thread_pool.thread_iteration();
    size_t actual = index->getAllocationSize() - before;

    // Flat index should be empty, hence the index size includes only hnsw size.
    ASSERT_EQ(index->indexSize(), hnsw_index->indexSize());
    ASSERT_EQ(index->indexCapacity(), hnsw_index->indexCapacity());
    // We added n + 1 vectors
    ASSERT_EQ(index->indexSize(), n + 1);
    // We should have 2 blocks now
    ASSERT_EQ(index->indexCapacity(), 2 * bs);

    // We check that the actual size is within 1% of the estimation.
    ASSERT_GE(estimation, actual * 0.99);
    ASSERT_LE(estimation, actual * 1.01);
}

TYPED_TEST(HNSWTieredIndexTest, addVector) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    bool isMulti = TypeParam::isMulti();
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = isMulti};
    VecSimParams hnsw_params = CreateParams(params);

    auto mock_thread_pool = tieredIndexMock();
    TieredIndexParams tiered_params = {.jobQueue = &mock_thread_pool.jobQ,
                                       .jobQueueCtx = mock_thread_pool.ctx,
                                       .submitCb = tieredIndexMock::submit_callback,
                                       .flatBufferLimit = SIZE_MAX,
                                       .primaryIndexParams = &hnsw_params};
    auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
        TieredFactory::NewIndex(&tiered_params));
    // Get the allocator from the tiered index.
    auto allocator = tiered_index->getAllocator();
    // Set the created tiered index in the index external context.
    mock_thread_pool.ctx->index_strong_ref.reset(tiered_index);

    BFParams bf_params = {.type = TypeParam::get_index_type(),
                          .dim = dim,
                          .metric = VecSimMetric_L2,
                          .multi = isMulti};

    // Validate that memory upon creating the tiered index is as expected (no more than 2%
    // above te expected, since in different platforms there are some minor additional
    // allocations).
    size_t expected_mem = TieredFactory::EstimateInitialSize(&tiered_params);
    ASSERT_LE(expected_mem, tiered_index->getAllocationSize());
    ASSERT_GE(expected_mem * 1.02, tiered_index->getAllocationSize());

    // Create a vector and add it to the tiered index.
    labelType vec_label = 1;
    TEST_DATA_T vector[dim];
    GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
    VecSimIndex_AddVector(tiered_index, vector, vec_label);
    // Validate that the vector was inserted to the flat buffer properly.
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexCapacity(), DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(tiered_index->indexCapacity(), DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(tiered_index->frontendIndex->getDistanceFrom(vec_label, vector), 0);
    // Validate that the job was created properly
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label).size(), 1);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label)[0]->label, vec_label);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label)[0]->id, 0);

    // Account for the allocation of a new block due to the vector insertion.
    expected_mem += (BruteForceFactory::EstimateElementSize(&bf_params)) * DEFAULT_BLOCK_SIZE;
    // Account for the memory that was allocated in the labelToId map (approx.)
    expected_mem += sizeof(vecsim_stl::unordered_map<labelType, idType>::value_type) +
                    sizeof(void *) + sizeof(size_t);
    // Account for the memory that was allocated in the labelToInsertJobs map (approx.)
    expected_mem +=
        sizeof(
            vecsim_stl::unordered_map<labelType, vecsim_stl::vector<HNSWInsertJob *>>::value_type) +
        sizeof(void *) + sizeof(size_t);
    // Account for the inner buffer of the std::vector<HNSWInsertJob *> in the map.
    expected_mem += sizeof(void *) + sizeof(size_t);
    // Account for the insert job that was created.
    expected_mem += sizeof(HNSWInsertJob) + sizeof(size_t);
    ASSERT_GE(expected_mem * 1.02, tiered_index->getAllocationSize());
    ASSERT_LE(expected_mem, tiered_index->getAllocationSize());

    if (isMulti) {
        // Add another vector under the same label (create another insert job)
        VecSimIndex_AddVector(tiered_index, vector, vec_label);
        ASSERT_EQ(tiered_index->indexSize(), 2);
        ASSERT_EQ(tiered_index->indexLabelCount(), 1);
        ASSERT_EQ(tiered_index->backendIndex->indexSize(), 0);
        ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 2);
        // Validate that the second job was created properly
        ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label).size(), 2);
        ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label)[1]->label, vec_label);
        ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label)[1]->id, 1);
    }
}

TYPED_TEST(HNSWTieredIndexTest, manageIndexOwnership) {

    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti()};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);

    // Get the allocator from the tiered index.
    auto allocator = tiered_index->getAllocator();

    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);
    size_t initial_mem = allocator->getAllocationSize();

    // Create a dummy job callback that insert one vector to the underline HNSW index.
    auto dummy_job = [](AsyncJob *job) {
        auto *my_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(job->index);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        size_t dim = 4;
        TEST_DATA_T vector[dim];
        GenerateVector<TEST_DATA_T>(vector, dim);
        my_index->backendIndex->addVector(vector, my_index->backendIndex->indexSize());
    };

    std::atomic_int successful_executions(0);
    auto job1 =
        new (allocator) AsyncJob(allocator, HNSW_INSERT_VECTOR_JOB, dummy_job, tiered_index);
    auto job2 =
        new (allocator) AsyncJob(allocator, HNSW_INSERT_VECTOR_JOB, dummy_job, tiered_index);

    // Wrap this job with an array and submit the jobs to the queue.
    tiered_index->submitSingleJob(job1);
    tiered_index->submitSingleJob(job2);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 2);

    // Execute the job from the queue asynchronously, delete the index in the meantime.
    auto run_fn = [&successful_executions, &mock_thread_pool]() {
        // Create a temporary strong reference of the index from the weak reference that the
        // job holds, to ensure that the index is not deleted while the job is running.
        if (auto temp_ref = mock_thread_pool.jobQ.front().index_weak_ref.lock()) {
            // At this point we wish to validate that we have both the index strong ref (stored
            // in index_ctx) and the weak ref owned by the job (that we currently promoted).
            EXPECT_EQ(mock_thread_pool.jobQ.front().index_weak_ref.use_count(), 2);

            mock_thread_pool.jobQ.front().job->Execute(mock_thread_pool.jobQ.front().job);
            successful_executions++;
        }
        mock_thread_pool.jobQ.kick();
    };
    std::thread t1(run_fn);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // Delete the index while the job is still running, to ensure that the weak ref protects
    // the index.
    mock_thread_pool.reset_ctx();
    EXPECT_EQ(mock_thread_pool.jobQ.front().index_weak_ref.use_count(), 1);
    t1.join();
    // Expect that the first job will succeed.
    ASSERT_EQ(successful_executions, 1);

    // The second job should not run, since the weak reference is not supposed to become a
    // strong references now.
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 1);
    ASSERT_EQ(mock_thread_pool.jobQ.front().index_weak_ref.use_count(), 0);
    std::thread t2(run_fn);
    t2.join();
    // Expect that the second job is ot successful.
    ASSERT_EQ(successful_executions, 1);
}

TYPED_TEST(HNSWTieredIndexTest, insertJob) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti()};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Create a vector and add it to the tiered index.
    labelType vec_label = 1;
    TEST_DATA_T vector[dim];
    GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
    VecSimIndex_AddVector(tiered_index, vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);

    // Execute the insert job manually (in a synchronous manner).
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 1);
    auto *insertion_job = reinterpret_cast<HNSWInsertJob *>(mock_thread_pool.jobQ.front().job);
    ASSERT_EQ(insertion_job->label, vec_label);
    ASSERT_EQ(insertion_job->id, 0);
    ASSERT_EQ(insertion_job->jobType, HNSW_INSERT_VECTOR_JOB);

    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);
    // HNSW index should have allocated a single block, while flat index should remove the
    // block.
    ASSERT_EQ(tiered_index->backendIndex->indexCapacity(), DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(tiered_index->indexCapacity(), DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(tiered_index->frontendIndex->indexCapacity(), 0);
    ASSERT_EQ(tiered_index->backendIndex->getDistanceFrom(vec_label, vector), 0);
    // After the execution, the job should be removed from the labelToInsertJobs mapping.
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);
}

TYPED_TEST(HNSWTieredIndexTestBasic, insertJobAsync) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 5000;
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    // Insert vectors
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i, i);
    }

    mock_thread_pool.thread_pool_join();
    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), n);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);
    // Verify that the vectors were inserted to HNSW as expected
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T expected_vector[dim];
        GenerateVector<TEST_DATA_T>(expected_vector, dim, i);
        ASSERT_EQ(tiered_index->backendIndex->getDistanceFrom(i, expected_vector), 0);
    }
}

TYPED_TEST(HNSWTieredIndexTestBasic, insertJobAsyncMulti) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 5000;
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = true};
    VecSimParams hnsw_params = CreateParams(params);
    size_t per_label = 5;
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();

    // Create and insert vectors, store them in this continuous array.
    TEST_DATA_T vectors[n * dim];
    for (size_t i = 0; i < n / per_label; i++) {
        for (size_t j = 0; j < per_label; j++) {
            GenerateVector<TEST_DATA_T>(vectors + i * dim * per_label + j * dim, dim,
                                        i * per_label + j);
            tiered_index->addVector(vectors + i * dim * per_label + j * dim, i);
        }
    }

    mock_thread_pool.thread_pool_join();
    EXPECT_EQ(tiered_index->backendIndex->indexSize(), n);
    EXPECT_EQ(tiered_index->indexLabelCount(), n / per_label);
    EXPECT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
    // Verify that the vectors were inserted to HNSW as expected
    for (size_t i = 0; i < n / per_label; i++) {
        for (size_t j = 0; j < per_label; j++) {
            // The distance from every vector that is stored under the label i should be zero
            EXPECT_EQ(tiered_index->backendIndex->getDistanceFrom(i, vectors + i * per_label * dim +
                                                                         j * dim),
                      0);
        }
    }
}

TYPED_TEST(HNSWTieredIndexTestBasic, KNNSearch) {
    size_t dim = 4;
    size_t k = 10;

    size_t n = k * 3;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    size_t cur_memory_usage;

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();
    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    auto hnsw_index = tiered_index->backendIndex;
    auto flat_index = tiered_index->frontendIndex;

    TEST_DATA_T query_0[dim];
    GenerateVector<TEST_DATA_T>(query_0, dim, 0);
    TEST_DATA_T query_1mid[dim];
    GenerateVector<TEST_DATA_T>(query_1mid, dim, n / 3);
    TEST_DATA_T query_2mid[dim];
    GenerateVector<TEST_DATA_T>(query_2mid, dim, n * 2 / 3);
    TEST_DATA_T query_n[dim];
    GenerateVector<TEST_DATA_T>(query_n, dim, n - 1);

    // Search for vectors when the index is empty.
    runTopKSearchTest(tiered_index, query_0, k, nullptr);

    // Define the verification functions.
    auto ver_res_0 = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, index);
        ASSERT_DOUBLE_EQ(score, dim * id * id);
    };

    auto ver_res_1mid = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(std::abs(int(id - query_1mid[0])), (index + 1) / 2);
        ASSERT_DOUBLE_EQ(score, dim * pow((index + 1) / 2, 2));
    };

    auto ver_res_2mid = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(std::abs(int(id - query_2mid[0])), (index + 1) / 2);
        ASSERT_DOUBLE_EQ(score, dim * pow((index + 1) / 2, 2));
    };

    auto ver_res_n = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, n - 1 - index);
        ASSERT_DOUBLE_EQ(score, dim * index * index);
    };

    // Insert n/2 vectors to the main index.
    for (size_t i = 0; i < n / 2; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, i, i);
    }
    ASSERT_EQ(tiered_index->indexSize(), n / 2);
    ASSERT_EQ(tiered_index->indexSize(), hnsw_index->indexSize());

    // Search for k vectors with the flat index empty.
    cur_memory_usage = allocator->getAllocationSize();
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // Insert n/2 vectors to the flat index.
    for (size_t i = n / 2; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, i, i);
    }
    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->indexSize(), hnsw_index->indexSize() + flat_index->indexSize());

    cur_memory_usage = allocator->getAllocationSize();
    // Search for k vectors so all the vectors will be from the flat index.
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    // Search for k vectors so all the vectors will be from the main index.
    runTopKSearchTest(tiered_index, query_n, k, ver_res_n);
    // Search for k so some of the results will be from the main and some from the flat index.
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    runTopKSearchTest(tiered_index, query_2mid, k, ver_res_2mid);
    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // Add some overlapping vectors to the main and flat index.
    // adding directly to the underlying indexes to avoid jobs logic.
    // The main index will have vectors 0 - 2n/3 and the flat index will have vectors n/3 - n
    for (size_t i = n / 3; i < n / 2; i++) {
        GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, i, i);
    }
    for (size_t i = n / 2; i < n * 2 / 3; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, i, i);
    }

    cur_memory_usage = allocator->getAllocationSize();
    // Search for k vectors so all the vectors will be from the main index.
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    // Search for k vectors so all the vectors will be from the flat index.
    runTopKSearchTest(tiered_index, query_n, k, ver_res_n);
    // Search for k so some of the results will be from the main and some from the flat index.
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    runTopKSearchTest(tiered_index, query_2mid, k, ver_res_2mid);
    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // More edge cases:

    // Search for more vectors than the index size.
    k = n + 1;
    runTopKSearchTest(tiered_index, query_0, k, n, ver_res_0);
    runTopKSearchTest(tiered_index, query_n, k, n, ver_res_n);

    // Search for less vectors than the index size, but more than the flat and main index sizes.
    k = n * 5 / 6;
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_n, k, ver_res_n);

    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // Search for more vectors than the main index size, but less than the flat index size.
    for (size_t i = n / 2; i < n * 2 / 3; i++) {
        VecSimIndex_DeleteVector(hnsw_index, i);
    }
    ASSERT_EQ(flat_index->indexSize(), n * 2 / 3);
    ASSERT_EQ(hnsw_index->indexSize(), n / 2);
    k = n * 2 / 3;
    cur_memory_usage = allocator->getAllocationSize();
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_n, k, ver_res_n);
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    runTopKSearchTest(tiered_index, query_2mid, k, ver_res_2mid);
    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // Search for more vectors than the flat index size, but less than the main index size.
    for (size_t i = n / 2; i < n; i++) {
        VecSimIndex_DeleteVector(flat_index, i);
    }
    ASSERT_EQ(flat_index->indexSize(), n / 6);
    ASSERT_EQ(hnsw_index->indexSize(), n / 2);
    k = n / 4;
    cur_memory_usage = allocator->getAllocationSize();
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // Search for vectors when the flat index is not empty but the main index is empty.
    for (size_t i = 0; i < n * 2 / 3; i++) {
        VecSimIndex_DeleteVector(hnsw_index, i);
        GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, i, i);
    }
    ASSERT_EQ(flat_index->indexSize(), n * 2 / 3);
    ASSERT_EQ(hnsw_index->indexSize(), 0);
    k = n / 3;
    cur_memory_usage = allocator->getAllocationSize();
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // // // // // // // // // // // //
    // Check behavior upon timeout.  //
    // // // // // // // // // // // //

    VecSimQueryResult_List res;
    // Add a vector to the HNSW index so there will be a reason to query it.
    GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, n, n);

    // Set timeout callback to always return 1 (will fail while querying the flat buffer).
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out

    res = VecSimIndex_TopKQuery(tiered_index, query_0, k, nullptr, BY_SCORE);
    ASSERT_EQ(res.results, nullptr);
    ASSERT_EQ(res.code, VecSim_QueryResult_TimedOut);

    // Set timeout callback to return 1 after n checks (will fail while querying the HNSW index).
    // Brute-force index checks for timeout after each vector.
    size_t checks_in_flat = flat_index->indexSize();
    VecSimQueryParams qparams = {.timeoutCtx = &checks_in_flat};
    VecSim_SetTimeoutCallbackFunction([](void *ctx) {
        auto count = static_cast<size_t *>(ctx);
        if (*count == 0) {
            return 1;
        }
        (*count)--;
        return 0;
    });
    res = VecSimIndex_TopKQuery(tiered_index, query_0, k, &qparams, BY_SCORE);
    ASSERT_EQ(res.results, nullptr);
    ASSERT_EQ(res.code, VecSim_QueryResult_TimedOut);
    // Make sure we didn't get the timeout in the flat index.
    checks_in_flat = flat_index->indexSize(); // Reset the counter.
    res = VecSimIndex_TopKQuery(flat_index, query_0, k, &qparams, BY_SCORE);
    ASSERT_EQ(res.code, VecSim_QueryResult_OK);
    VecSimQueryResult_Free(res);

    // Clean up.
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; });
}

TYPED_TEST(HNSWTieredIndexTest, parallelSearch) {
    size_t dim = 4;
    size_t k = 10;
    size_t n = 2000;
    bool isMulti = TypeParam::isMulti();

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = isMulti,
        .efRuntime = n,
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();
    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    std::atomic_int successful_searches(0);
    auto parallel_knn_search = [](AsyncJob *job) {
        auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
        size_t k = search_job->k;
        size_t dim = search_job->dim;
        auto query = search_job->query;

        auto verify_res = [&](size_t id, double score, size_t res_index) {
            TEST_DATA_T element = *(TEST_DATA_T *)query;
            ASSERT_EQ(std::abs(id - element), (res_index + 1) / 2);
            ASSERT_EQ(score, dim * (id - element) * (id - element));
        };
        runTopKSearchTest(job->index, query, k, verify_res);
        (*search_job->successful_searches)++;
        delete job;
    };

    size_t per_label = isMulti ? 10 : 1;
    size_t n_labels = n / per_label;

    // Fill the job queue with insert and search jobs, while filling the flat index, before
    // initializing the thread pool.
    for (size_t i = 0; i < n; i++) {
        // Insert a vector to the flat index and add a job to insert it to the main index.
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i % n_labels, i);

        // Add a search job. Make sure the query element is between k and n - k.
        auto query = (TEST_DATA_T *)allocator->allocate(dim * sizeof(TEST_DATA_T));
        GenerateVector<TEST_DATA_T>(query, dim, (i % (n_labels - (2 * k))) + k);
        auto search_job = new (allocator) tieredIndexMock::SearchJobMock(
            allocator, parallel_knn_search, tiered_index, k, query, n, dim, &successful_searches);
        tiered_index->submitSingleJob(search_job);
    }

    EXPECT_EQ(tiered_index->indexSize(), n);
    EXPECT_EQ(tiered_index->indexLabelCount(), n_labels);
    EXPECT_EQ(tiered_index->labelToInsertJobs.size(), n_labels);
    for (auto &it : tiered_index->labelToInsertJobs) {
        EXPECT_EQ(it.second.size(), per_label);
    }
    EXPECT_EQ(tiered_index->frontendIndex->indexSize(), n);
    EXPECT_EQ(tiered_index->backendIndex->indexSize(), 0);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    // All the vectors are already in the tiered index, so we expect to find the expected
    // results from the get-go.
    mock_thread_pool.init_threads();
    mock_thread_pool.thread_pool_join();

    EXPECT_EQ(tiered_index->backendIndex->indexSize(), n);
    EXPECT_EQ(tiered_index->backendIndex->indexLabelCount(), n_labels);
    EXPECT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(HNSWTieredIndexTest, parallelInsertSearch) {
    size_t dim = 4;
    size_t k = 10;
    size_t n = 3000;

    size_t block_size = n / 100;

    bool isMulti = TypeParam::isMulti();

    // Create TieredHNSW index instance with a mock queue.
    size_t n_labels = isMulti ? n / 25 : n;
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = isMulti,
        .blockSize = block_size,
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();
    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(mock_thread_pool.thread_pool_size, 0);
    mock_thread_pool.init_threads();
    std::atomic_int successful_searches(0);

    auto parallel_knn_search = [](AsyncJob *job) {
        auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
        size_t k = search_job->k;
        auto query = search_job->query;
        // In this test we don't care about the results, just that the search doesn't crash
        // and returns the correct number of valid results.
        auto verify_res = [&](size_t id, double score, size_t res_index) {};
        runTopKSearchTest(job->index, query, k, verify_res);
        (*search_job->successful_searches)++;
        delete job;
    };

    // Insert vectors in parallel to search.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i % n_labels, i);
        auto query = (TEST_DATA_T *)allocator->allocate(dim * sizeof(TEST_DATA_T));
        GenerateVector<TEST_DATA_T>(query, dim, (TEST_DATA_T)n / 4 + (i % 1000) * M_PI);
        auto search_job = new (allocator) tieredIndexMock::SearchJobMock(
            allocator, parallel_knn_search, tiered_index, k, query, n, dim, &successful_searches);
        tiered_index->submitSingleJob(search_job);
    }

    mock_thread_pool.thread_pool_join();

    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(tiered_index->backendIndex->indexSize(), n);
    EXPECT_EQ(tiered_index->backendIndex->indexLabelCount(), n_labels);
    EXPECT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(HNSWTieredIndexTestBasic, MergeMulti) {
    size_t dim = 4;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = true,
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    auto hnsw_index = tiered_index->backendIndex;
    auto flat_index = tiered_index->frontendIndex;

    // Insert vectors with label 0 to HNSW only.
    GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, 0, 1);
    GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, 0, 2);
    // Insert vectors with label 1 to flat buffer only.
    GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, 1, 0);
    GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, 1, 1);
    GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, 1, 2);
    // Insert DIFFERENT vectors with label 2 to both HNSW and flat buffer.
    GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, 2, 0);
    GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, 2, 1);

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, 0);

    // Search in the tiered index for more vectors than it has. Merging the results from the two
    // indexes should result in a list of unique vectors, even if the scores of the duplicates are
    // different.
    runTopKSearchTest(tiered_index, query, 5, 3, [](size_t _, double __, size_t ___) {});
}

TYPED_TEST(HNSWTieredIndexTest, deleteFromHNSWBasic) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    bool isMulti = TypeParam::isMulti();

    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = isMulti};
    VecSimParams hnsw_params = CreateParams(params);

    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Delete a non existing label.
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(0), 0);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Insert one vector to HNSW and then delete it (it should have no neighbors to repair).
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 0);
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(0), 1);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Add another vector and remove it. Since the other vector in the index has marked deleted,
    // this vector should have no neighbors, and again, no neighbors to repair.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 1, 1);
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(1), 1);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Add two vectors and delete one, expect that at backendIndex one repair job will be created.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 2, 2);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 3, 3);
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(3), 1);

    // The first job should be a repair job of the first inserted non-deleted node id (2)
    // in level 0.
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 1);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_REPAIR_NODE_CONNECTIONS_JOB);
    ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->node_id, 2);
    ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->level, 0);
    ASSERT_EQ(tiered_index->idToRepairJobs.size(), 1);
    ASSERT_GE(tiered_index->idToRepairJobs.at(2).size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(2)[0]->associatedSwapJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(2)[0]->associatedSwapJobs[0]->deleted_id, 3);

    ASSERT_EQ(tiered_index->indexSize(), 4);
    ASSERT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 3);
    ASSERT_EQ(tiered_index->idToSwapJob.size(), 3);
}

TYPED_TEST(HNSWTieredIndexTestBasic, deleteFromHNSWMulti) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;

    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = true};
    VecSimParams hnsw_params = CreateParams(params);

    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Add two vectors and delete one, expect that at least one repair job will be created.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 1, 1);
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(0), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(1).size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(1)[0]->associatedSwapJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(1)[0]->associatedSwapJobs[0]->deleted_id, 0);
    ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->node_id, 1);
    ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->level, 0);
    mock_thread_pool.jobQ.pop();

    // Insert another vector under the label (1) that has not been deleted.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 1, 2);

    // Expect to see both ids stored under this label being deleted (1 and 2), and have both
    // ids need repair (as the connection between the two vectors is mutual). However, 1 has
    // also an outgoing edge to his other (deleted) neighbor (0), so there will be no new
    // repair job created for 1, since the previous repair job is expected to have both 0 and 2 in
    // its associated swap jobs. Also, there is an edge 0->1 whose going to be repaired as well.
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(1), 2);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 2);
    ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->node_id, 0);
    ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->level, 0);
    mock_thread_pool.jobQ.pop();
    ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->node_id, 2);
    ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->level, 0);
    mock_thread_pool.jobQ.pop();
    // No new job for deleting 1->2 edge, just another associated swap job for the existing repair
    // job of 1 (in addition to 0, we have 2).
    ASSERT_EQ(tiered_index->idToRepairJobs.size(), 3);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(1).size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(1)[0]->associatedSwapJobs.size(), 2);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(1)[0]->associatedSwapJobs[1]->deleted_id, 2);

    ASSERT_EQ(tiered_index->idToRepairJobs.at(0).size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(0)[0]->associatedSwapJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(0)[0]->associatedSwapJobs[0]->deleted_id, 1);

    ASSERT_EQ(tiered_index->idToRepairJobs.at(2).size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(2)[0]->associatedSwapJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(2)[0]->associatedSwapJobs[0]->deleted_id, 1);

    ASSERT_EQ(tiered_index->idToSwapJob.size(), 3);
}

TYPED_TEST(HNSWTieredIndexTestBasic, deleteFromHNSWMultiLevels) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;

    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Test that repair jobs are created for multiple levels.
    size_t num_elements_with_multiple_levels = 0;
    int vec_id = -1;
    do {
        vec_id++;
        GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, vec_id, vec_id);
        if (tiered_index->getHNSWIndex()->getGraphDataByInternalId(vec_id)->toplevel > 0) {
            num_elements_with_multiple_levels++;
        }
    } while (num_elements_with_multiple_levels < 2);

    // Delete the last inserted vector, which is in level 1.
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(vec_id), 1);
    ASSERT_EQ(tiered_index->getHNSWIndex()->getGraphDataByInternalId(vec_id)->toplevel, 1);
    // This should be an array of length 1.
    auto &level_one = tiered_index->getHNSWIndex()->getLevelData(vec_id, 1);
    ASSERT_EQ(level_one.numLinks, 1);

    size_t num_repair_jobs = mock_thread_pool.jobQ.size();
    // There should be at least two nodes to repair, the neighbors of next_id in levels 0 and 1
    ASSERT_GE(num_repair_jobs, 2);
    while (mock_thread_pool.jobQ.size() > 1) {
        // First we should have jobs for repairing nodes in level 0.
        ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->level, 0);
        mock_thread_pool.jobQ.pop();
    }

    // The last job should be repairing the single neighbor in level 1.
    ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->level, 1);
    ASSERT_EQ(((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->node_id, level_one.links[0]);
}

TYPED_TEST(HNSWTieredIndexTest, deleteFromHNSWWithRepairJobExec) {
    // Create TieredHNSW index instance with a mock queue.
    size_t n = 1000;
    size_t dim = 4;
    bool isMulti = TypeParam::isMulti();

    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = isMulti,
                         .M = 4};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector(tiered_index->getHNSWIndex(), dim, i, i);
    }

    // Delete vectors one by one and run the resulted repair jobs.
    while (tiered_index->getHNSWIndex()->getNumMarkedDeleted() < n) {
        // Choose the current entry point each time (it should be modified after the deletion).
        idType ep = tiered_index->getHNSWIndex()->safeGetEntryPointState().first;
        auto incoming_neighbors =
            tiered_index->getHNSWIndex()->safeCollectAllNodeIncomingNeighbors(ep);
        ASSERT_EQ(tiered_index->deleteLabelFromHNSW(ep), 1);
        ASSERT_EQ(mock_thread_pool.jobQ.size(), incoming_neighbors.size());
        ASSERT_EQ(tiered_index->getHNSWIndex()->checkIntegrity().connections_to_repair,
                  mock_thread_pool.jobQ.size());
        ASSERT_NE(tiered_index->getHNSWIndex()->safeGetEntryPointState().first, ep);

        // Execute synchronously all the repair jobs for the current deletion.
        while (!mock_thread_pool.jobQ.empty()) {
            idType repair_node_id = ((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->node_id;
            auto repair_node_level = ((HNSWRepairJob *)(mock_thread_pool.jobQ.front().job))->level;

            tiered_index->getHNSWIndex()->repairNodeConnections(repair_node_id, repair_node_level);
            LevelData &node_level =
                tiered_index->getHNSWIndex()->getLevelData(repair_node_id, repair_node_level);
            // This makes sure that the deleted node is no longer in the neighbors set of the
            // repaired node.
            ASSERT_TRUE(std::find(node_level.links, node_level.links + node_level.numLinks, ep) ==
                        node_level.links + node_level.numLinks);
            // Remove the job from the id -> repair_jobs lookup, so we won't think that it is
            // still pending and avoid creating new jobs for nodes that already been repaired
            // as they were pointing to deleted elements.
            tiered_index->idToRepairJobs.erase(repair_node_id);
            mock_thread_pool.jobQ.kick();
        }
        ASSERT_EQ(tiered_index->getHNSWIndex()->checkIntegrity().connections_to_repair, 0);
    }
}

TYPED_TEST(HNSWTieredIndexTest, manageIndexOwnershipWithPendingJobs) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    bool isMulti = TypeParam::isMulti();

    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = isMulti};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();
    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    // Add a vector and create a pending insert job.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);

    // Delete the index before the job was executed (this would delete the pending job as well).
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 1);
    EXPECT_EQ(mock_thread_pool.jobQ.front().index_weak_ref.use_count(), 1);
    mock_thread_pool.reset_ctx();
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 1);
    EXPECT_EQ(mock_thread_pool.jobQ.front().index_weak_ref.use_count(), 0);
    mock_thread_pool.jobQ.pop();

    // Recreate the index with a new ctx.
    auto *ctx = new tieredIndexMock::IndexExtCtx(&mock_thread_pool);
    mock_thread_pool.reset_ctx(ctx);
    tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    allocator = tiered_index->getAllocator();
    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    // Add two vectors directly to HNSW, and remove one vector to create a repair job.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 1, 1);
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(0), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.size(), 1);

    // Delete the index before the job was executed (this would delete the pending job as well).
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 1);
    EXPECT_EQ(mock_thread_pool.jobQ.front().index_weak_ref.use_count(), 1);
    mock_thread_pool.reset_ctx();
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 1);
    EXPECT_EQ(mock_thread_pool.jobQ.front().index_weak_ref.use_count(), 0);
}

TYPED_TEST(HNSWTieredIndexTestBasic, AdHocSingle) {
    size_t dim = 4;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    auto hnsw_index = tiered_index->backendIndex;
    auto flat_index = tiered_index->frontendIndex;

    TEST_DATA_T vec1[dim];
    GenerateVector<TEST_DATA_T>(vec1, dim, 1);
    TEST_DATA_T vec2[dim];
    GenerateVector<TEST_DATA_T>(vec2, dim, 2);
    TEST_DATA_T vec3[dim];
    GenerateVector<TEST_DATA_T>(vec3, dim, 3);
    TEST_DATA_T vec4[dim];
    GenerateVector<TEST_DATA_T>(vec4, dim, 4);

    // Insert vectors to the tiered index.
    VecSimIndex_AddVector(hnsw_index, vec1, 1); // vec1 is inserted to HNSW only.
    VecSimIndex_AddVector(flat_index, vec2, 2); // vec2 is inserted to flat only.

    // vec3 is inserted to both HNSW and flat, simulating a vector that was inserted
    // to HNSW and not yet removed from flat.
    VecSimIndex_AddVector(hnsw_index, vec3, 3);
    VecSimIndex_AddVector(flat_index, vec3, 3);

    // vec4 is not inserted to any index, simulating a non-existing vector.

    // copy memory context before querying the index.
    size_t cur_memory_usage = allocator->getAllocationSize();

    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 1, vec1), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 2, vec2), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 3, vec3), 0);
    ASSERT_TRUE(std::isnan(VecSimIndex_GetDistanceFrom(tiered_index, 4, vec4)));

    ASSERT_EQ(cur_memory_usage, allocator->getAllocationSize());
}

TYPED_TEST(HNSWTieredIndexTestBasic, AdHocMulti) {
    size_t dim = 4;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = true,
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);

    auto hnsw_index = tiered_index->backendIndex;
    auto flat_index = tiered_index->frontendIndex;
    auto allocator = tiered_index->getAllocator();

    TEST_DATA_T cur_element = 1;

    // vec1_* are inserted to HNSW only.
    TEST_DATA_T vec1_1[dim];
    GenerateVector<TEST_DATA_T>(vec1_1, dim, cur_element++);
    TEST_DATA_T vec1_2[dim];
    GenerateVector<TEST_DATA_T>(vec1_2, dim, cur_element++);
    TEST_DATA_T vec1_3[dim];
    GenerateVector<TEST_DATA_T>(vec1_3, dim, cur_element++);

    // vec2_* are inserted to flat only.
    TEST_DATA_T vec2_1[dim];
    GenerateVector<TEST_DATA_T>(vec2_1, dim, cur_element++);
    TEST_DATA_T vec2_2[dim];
    GenerateVector<TEST_DATA_T>(vec2_2, dim, cur_element++);
    TEST_DATA_T vec2_3[dim];
    GenerateVector<TEST_DATA_T>(vec2_3, dim, cur_element++);

    // vec3_* are inserted to both HNSW and flat (some to HNSW only, some to flat only)
    TEST_DATA_T vec3_1[dim];
    GenerateVector<TEST_DATA_T>(vec3_1, dim, cur_element++);
    TEST_DATA_T vec3_2[dim];
    GenerateVector<TEST_DATA_T>(vec3_2, dim, cur_element++);
    TEST_DATA_T vec3_3[dim];
    GenerateVector<TEST_DATA_T>(vec3_3, dim, cur_element++);

    // vec4_* are inserted to both HNSW and flat with some overlap.
    TEST_DATA_T vec4_1[dim];
    GenerateVector<TEST_DATA_T>(vec4_1, dim, cur_element++);
    TEST_DATA_T vec4_2[dim];
    GenerateVector<TEST_DATA_T>(vec4_2, dim, cur_element++);
    TEST_DATA_T vec4_3[dim];
    GenerateVector<TEST_DATA_T>(vec4_3, dim, cur_element++);

    // vec5 is not inserted to any index, simulating a non-existing vector.
    TEST_DATA_T vec5[dim];
    GenerateVector<TEST_DATA_T>(vec5, dim, cur_element++);

    // Insert vectors to the tiered index.
    VecSimIndex_AddVector(hnsw_index, vec1_1, 1);
    VecSimIndex_AddVector(hnsw_index, vec1_2, 1);
    VecSimIndex_AddVector(hnsw_index, vec1_3, 1);

    VecSimIndex_AddVector(flat_index, vec2_1, 2);
    VecSimIndex_AddVector(flat_index, vec2_2, 2);
    VecSimIndex_AddVector(flat_index, vec2_3, 2);

    VecSimIndex_AddVector(hnsw_index, vec3_1, 3);
    VecSimIndex_AddVector(flat_index, vec3_2, 3);
    VecSimIndex_AddVector(hnsw_index, vec3_3, 3);

    VecSimIndex_AddVector(hnsw_index, vec4_1, 4);
    VecSimIndex_AddVector(hnsw_index, vec4_2, 4);
    VecSimIndex_AddVector(flat_index, vec4_2, 4);
    VecSimIndex_AddVector(flat_index, vec4_3, 4);

    // vec5 is not inserted to any index, simulating a non-existing vector.

    // copy memory context before querying the index.
    size_t cur_memory_usage = allocator->getAllocationSize();

    // Distance from any vector to its label should be 0.
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 1, vec1_1), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 1, vec1_2), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 1, vec1_3), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 2, vec2_1), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 2, vec2_2), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 2, vec2_3), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 3, vec3_1), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 3, vec3_2), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 3, vec3_3), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 4, vec4_1), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 4, vec4_2), 0);
    ASSERT_EQ(VecSimIndex_GetDistanceFrom(tiered_index, 4, vec4_3), 0);
    // Distance from a non-existing label should be NaN.
    ASSERT_TRUE(std::isnan(VecSimIndex_GetDistanceFrom(tiered_index, 5, vec5)));

    ASSERT_EQ(cur_memory_usage, allocator->getAllocationSize());
}

TYPED_TEST(HNSWTieredIndexTest, parallelInsertAdHoc) {
    size_t dim = 4;
    size_t n = 1000;

    size_t block_size = n / 100;
    bool isMulti = TypeParam::isMulti();

    // Create TieredHNSW index instance with a mock queue.
    size_t n_labels = isMulti ? n / 50 : n;
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = isMulti,
        .blockSize = block_size,
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();
    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    std::atomic_int successful_searches(0);

    auto parallel_adhoc_search = [](AsyncJob *job) {
        auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
        auto query = search_job->query;
        size_t element = *(TEST_DATA_T *)query;
        size_t label = element % search_job->n;
        bool isMulti =
            reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(search_job->index)
                ->backendIndex->isMultiValue();

        ASSERT_EQ(0, VecSimIndex_GetDistanceFrom(search_job->index, label, query));

        (*search_job->successful_searches)++;
        delete job;
    };

    // Insert vectors in parallel to search.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i % n_labels, i);
        auto query = (TEST_DATA_T *)allocator->allocate(dim * sizeof(TEST_DATA_T));
        GenerateVector<TEST_DATA_T>(query, dim, i);
        auto search_job = new (allocator)
            tieredIndexMock::SearchJobMock(allocator, parallel_adhoc_search, tiered_index, 1, query,
                                           n_labels, dim, &successful_searches);
        tiered_index->submitSingleJob(search_job);
    }

    mock_thread_pool.thread_pool_join();

    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(tiered_index->backendIndex->indexSize(), n);
    EXPECT_EQ(tiered_index->backendIndex->indexLabelCount(), n_labels);
    EXPECT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(HNSWTieredIndexTest, deleteVector) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti()};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    labelType vec_label = 0;
    // Delete from an empty index.
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 0);

    // Create a vector and add it to the tiered index (expect it to go into the flat buffer).
    TEST_DATA_T vector[dim];
    GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
    VecSimIndex_AddVector(tiered_index, vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);

    // Expect to have one pending insert job.
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);
    auto *job = tiered_index->labelToInsertJobs.at(vec_label).back();

    // Remove vector from flat buffer.
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 1);
    ASSERT_EQ(tiered_index->indexSize(), 0);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    // The insert job should become invalid, and executing it should do nothing.
    ASSERT_EQ(job->isValid, false);
    ASSERT_EQ(reinterpret_cast<HNSWInsertJob *>(job)->id, 0);
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 0);

    // Create a vector and add it to HNSW in the tiered index.
    VecSimIndex_AddVector(tiered_index->backendIndex, vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);

    // Remove from main index.
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 1);
    ASSERT_EQ(tiered_index->indexLabelCount(), 0);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 1);

    // Re-insert a deleted label with a different vector.
    TEST_DATA_T new_vec_val = 2.0;
    GenerateVector<TEST_DATA_T>(vector, dim, new_vec_val);
    VecSimIndex_AddVector(tiered_index, vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 2);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);

    // Move the vector to HNSW by executing the insert job.
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 2);
    // Check that the distance from the deleted vector (of zeros) to the label is the distance
    // to the new vector (L2 distance).
    TEST_DATA_T deleted_vector[dim];
    GenerateVector<TEST_DATA_T>(deleted_vector, dim, 0);
    ASSERT_EQ(tiered_index->backendIndex->getDistanceFrom(vec_label, deleted_vector),
              dim * pow(new_vec_val, 2));
}

TYPED_TEST(HNSWTieredIndexTestBasic, deleteVectorMulti) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = true};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Test some more scenarios that are relevant only for multi value index.
    labelType vec_label = 0;
    labelType other_vec_val = 2.0;
    idType invalidJobsCounter = 0;
    // Create a vector and add it to HNSW in the tiered index.
    TEST_DATA_T vector[dim];
    GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
    VecSimIndex_AddVector(tiered_index->backendIndex, vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);

    // Test deleting a label for which one of its vector's is in the flat index while the
    // second one is in HNSW.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, other_vec_val);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->indexSize(), 2);
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 2);
    ASSERT_EQ(tiered_index->indexLabelCount(), 0);
    ASSERT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 1);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->isValid, false);
    ASSERT_EQ(reinterpret_cast<HNSWInsertJob *>(mock_thread_pool.jobQ.front().job)->id,
              invalidJobsCounter++);
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Test deleting a label for which both of its vector's is in the flat index.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, vec_label);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, other_vec_val);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 2);
    ASSERT_EQ(tiered_index->indexSize(), 3);
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 2);
    ASSERT_EQ(tiered_index->indexLabelCount(), 0);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->isValid, false);
    ASSERT_EQ(reinterpret_cast<HNSWInsertJob *>(mock_thread_pool.jobQ.front().job)->id,
              invalidJobsCounter++);
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->isValid, false);
    ASSERT_EQ(reinterpret_cast<HNSWInsertJob *>(mock_thread_pool.jobQ.front().job)->id,
              invalidJobsCounter++);
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Test deleting a label for which both of its vector's is in HNSW index.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, vec_label);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, other_vec_val);
    mock_thread_pool.thread_iteration();
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 3);
    ASSERT_EQ(tiered_index->backendIndex->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 2);
    ASSERT_EQ(tiered_index->backendIndex->indexLabelCount(), 0);
    ASSERT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 3);

    // Expect to see two repair jobs - one for each deleted vector internal id.
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 2);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_REPAIR_NODE_CONNECTIONS_JOB);
    ASSERT_EQ(reinterpret_cast<HNSWRepairJob *>(mock_thread_pool.jobQ.front().job)->node_id, 2);
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_REPAIR_NODE_CONNECTIONS_JOB);
    ASSERT_EQ(reinterpret_cast<HNSWRepairJob *>(mock_thread_pool.jobQ.front().job)->node_id, 1);
    mock_thread_pool.thread_iteration();
}

TYPED_TEST(HNSWTieredIndexTestBasic, deleteVectorMultiFromFlatAdvanced) {

    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = true};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Insert vectors to flat buffer under two distinct labels, so that ids 0, 2 will be associated
    // with the first label, and ids 1, 3, 4 will be associated with the second label.
    labelType vec_label_first = 0;
    labelType vec_label_second = 1;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label_first);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label_second);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label_first);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label_second);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label_second);

    // Remove the second label, expect to see that id 1 will hold id 2 eventually.
    ASSERT_EQ(tiered_index->labelToInsertJobs.erase(vec_label_second), 1);
    auto updated_ids = tiered_index->frontendIndex->deleteVectorAndGetUpdatedIds(vec_label_second);
    ASSERT_EQ(updated_ids.size(), 1);
    ASSERT_EQ(updated_ids.at(1).first, 2);
    for (auto &it : updated_ids) {
        tiered_index->updateInsertJobInternalId(it.second.first, it.first, it.second.second);
    }
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label_first).size(), 2);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label_first)[0]->label, vec_label_first);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label_first)[0]->id, 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label_first)[1]->label, vec_label_first);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label_first)[1]->id, 1);

    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->indexSize(), 2);

    // Remove the first label, expect an empty set.
    updated_ids = tiered_index->frontendIndex->deleteVectorAndGetUpdatedIds(vec_label_first);
    ASSERT_EQ(updated_ids.size(), 0);
    ASSERT_EQ(tiered_index->indexSize(), 0);
    tiered_index->labelToInsertJobs.clear();

    // Insert vectors to flat buffer under two distinct labels, so that ids 0, 3 will be associated
    // with the first label, and ids 1, 2, 4 will be associated with the second label. This should
    // test the case of multiple moves once we delete the second label:
    // {1->4} => {1->4, 2->3} => {1->3}
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label_first);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label_second);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label_second);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label_first);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label_second);
    ASSERT_EQ(tiered_index->labelToInsertJobs.erase(vec_label_second), 1);
    updated_ids = tiered_index->frontendIndex->deleteVectorAndGetUpdatedIds(vec_label_second);
    ASSERT_EQ(updated_ids.size(), 1);
    ASSERT_EQ(updated_ids.at(1).first, 3);
    for (auto &it : updated_ids) {
        tiered_index->updateInsertJobInternalId(it.second.first, it.first, it.second.second);
    }
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label_first).size(), 2);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label_first)[0]->label, vec_label_first);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label_first)[0]->id, 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label_first)[1]->label, vec_label_first);
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label_first)[1]->id, 1);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->indexSize(), 2);
    tiered_index->labelToInsertJobs.clear();

    // Clean jobs from queue
    while (!mock_thread_pool.jobQ.empty()) {
        mock_thread_pool.jobQ.kick();
    }
}

TYPED_TEST(HNSWTieredIndexTest, deleteVectorAndRepairAsync) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    for (size_t maxSwapJobs : {(int)n + 1, 10, 1}) {
        HNSWParams params = {.type = TypeParam::get_index_type(),
                             .dim = dim,
                             .metric = VecSimMetric_L2,
                             .multi = TypeParam::isMulti(),
                             .blockSize = 100};
        VecSimParams hnsw_params = CreateParams(params);
        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index =
            this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool, maxSwapJobs);
        auto allocator = tiered_index->getAllocator();

        size_t per_label = TypeParam::isMulti() ? 50 : 1;
        size_t n_labels = n / per_label;

        // Launch the BG threads loop that takes jobs from the queue and executes them.

        for (size_t i = 0; i < mock_thread_pool.thread_pool_size; i++) {
            mock_thread_pool.thread_pool.emplace_back(tieredIndexMock::thread_main_loop, i,
                                                      std::ref(mock_thread_pool));
        }

        // Create and insert vectors one by one, then delete them one by one.
        std::srand(10); // create pseudo random generator with any arbitrary seed.
        for (size_t i = 0; i < n; i++) {
            TEST_DATA_T vector[dim];
            for (size_t j = 0; j < dim; j++) {
                vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
            }
            VecSimIndex_AddVector(tiered_index, vector, i % n_labels);
        }
        // While a thread is ingesting a vector into HNSW, a vector may appear in both indexes
        // (hence it will be counted twice in the index size calculation).
        size_t index_size = tiered_index->indexSize();
        EXPECT_GE(index_size, n);
        EXPECT_LE(index_size, n + mock_thread_pool.thread_pool_size);
        EXPECT_EQ(tiered_index->indexLabelCount(), n_labels);
        for (size_t i = 0; i < n_labels; i++) {
            // Every vector associated with the label may appear in flat/HNSW index or in both if
            // its just being ingested.
            int num_deleted = tiered_index->deleteVector(i);
            EXPECT_GE(num_deleted, per_label);
            EXPECT_LE(num_deleted,
                      MIN(2 * per_label, per_label + mock_thread_pool.thread_pool_size));
            EXPECT_EQ(tiered_index->deleteVector(i), 0); // delete already deleted label
        }
        EXPECT_EQ(tiered_index->indexLabelCount(), 0);

        mock_thread_pool.thread_pool_join();

        EXPECT_EQ(tiered_index->getHNSWIndex()->checkIntegrity().connections_to_repair, 0);
        EXPECT_EQ(tiered_index->getHNSWIndex()->safeGetEntryPointState().first, INVALID_ID);
        // Verify that we have no pending jobs.
        EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0);
        EXPECT_EQ(tiered_index->idToRepairJobs.size(), 0);
        for (auto &it : tiered_index->idToSwapJob) {
            EXPECT_EQ(it.second->pending_repair_jobs_counter.load(), 0);
        }
        // Trigger swapping of vectors that hadn't been swapped yet.
        tiered_index->executeReadySwapJobs();
        ASSERT_EQ(tiered_index->idToSwapJob.size(), 0);
        EXPECT_EQ(tiered_index->indexSize(), 0);
    }
}

TYPED_TEST(HNSWTieredIndexTest, alternateInsertDeleteAsync) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 16;
    size_t n = 1000;
    for (size_t maxSwapJobs : {(int)n + 1, 10, 1}) {
        for (size_t M : {2, 16}) {
            HNSWParams params = {.type = TypeParam::get_index_type(),
                                 .dim = dim,
                                 .metric = VecSimMetric_L2,
                                 .multi = TypeParam::isMulti(),
                                 .blockSize = 100,
                                 .M = M};
            VecSimParams hnsw_params = CreateParams(params);
            auto mock_thread_pool = tieredIndexMock();

            auto *tiered_index =
                this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool, maxSwapJobs);
            auto allocator = tiered_index->getAllocator();

            size_t per_label = TypeParam::isMulti() ? 5 : 1;
            size_t n_labels = n / per_label;

            // Launch the BG threads loop that takes jobs from the queue and executes them.

            for (size_t i = 0; i < mock_thread_pool.thread_pool_size; i++) {
                mock_thread_pool.thread_pool.emplace_back(tieredIndexMock::thread_main_loop, i,
                                                          std::ref(mock_thread_pool));
            }

            // Create and insert 10 vectors, then delete them right after.
            size_t batch_size = 5;
            std::srand(10); // create pseudo random generator with any arbitrary seed.
            for (size_t i = 0; i < n / batch_size; i++) {
                for (size_t l = 0; l < batch_size; l++) {
                    TEST_DATA_T vector[dim];
                    for (size_t j = 0; j < dim; j++) {
                        vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
                    }
                    tiered_index->addVector(vector, (i * batch_size + l) / per_label);
                }
                for (size_t l = 0; l < batch_size / per_label; l++) {
                    // Every vector associated with the label may appear in flat/HNSW index or in
                    // both if its just being ingested.
                    int num_deleted = tiered_index->deleteVector(i * batch_size / per_label + l);
                    EXPECT_GE(num_deleted, per_label);
                    EXPECT_LE(num_deleted, 2 * per_label);
                }
            }
            // Vectors are deleted from flat buffer in place (in HNSW they are only marked as
            // deleted).
            EXPECT_GE(tiered_index->frontendIndex->indexSize(), 0);
            EXPECT_EQ(tiered_index->indexLabelCount(), 0);

            mock_thread_pool.thread_pool_join();

            EXPECT_EQ(tiered_index->getHNSWIndex()->checkIntegrity().connections_to_repair, 0);
            EXPECT_EQ(tiered_index->getHNSWIndex()->safeGetEntryPointState().first, INVALID_ID);
            // Verify that we have no pending jobs.
            EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0);
            EXPECT_EQ(tiered_index->idToRepairJobs.size(), 0);
            for (auto &it : tiered_index->idToSwapJob) {
                EXPECT_EQ(it.second->pending_repair_jobs_counter.load(), 0);
            }
            // Trigger swapping of vectors that hadn't been swapped yet.
            tiered_index->executeReadySwapJobs();
            ASSERT_LE(tiered_index->idToSwapJob.size(), 0);
            ASSERT_EQ(tiered_index->indexSize(), 0);
        }
    }
}

TYPED_TEST(HNSWTieredIndexTest, swapJobBasic) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti()};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Test initialization of the pendingSwapJobsThreshold value.
    ASSERT_EQ(tiered_index->pendingSwapJobsThreshold, DEFAULT_PENDING_SWAP_JOBS_THRESHOLD);
    mock_thread_pool.ctx->index_strong_ref.reset();

    tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool,
                                               MAX_PENDING_SWAP_JOBS_THRESHOLD + 1);
    allocator = tiered_index->getAllocator();
    ASSERT_EQ(tiered_index->pendingSwapJobsThreshold, MAX_PENDING_SWAP_JOBS_THRESHOLD);
    mock_thread_pool.ctx->index_strong_ref.reset();

    tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool, 1);
    ASSERT_EQ(tiered_index->pendingSwapJobsThreshold, 1);

    allocator = tiered_index->getAllocator();

    // Call reserve for the unordered maps that are going to be used, since upon initialization it
    // consumed 0 memory, but after insertion and deletion they will consume a minimal amount of
    // memory (that is equivalent to the memory consumption upon reserving 0 buckets).
    tiered_index->idToRepairJobs.reserve(0);
    tiered_index->idToSwapJob.reserve(0);
    TypeParam::isMulti() ? reinterpret_cast<HNSWIndex_Multi<TEST_DATA_T, TEST_DIST_T> *>(
                               tiered_index->getHNSWIndex())
                               ->labelLookup.reserve(0)
                         : reinterpret_cast<HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T> *>(
                               tiered_index->getHNSWIndex())
                               ->labelLookup.reserve(0);

    size_t initial_mem = tiered_index->getAllocationSize();
    size_t initial_mem_backend = tiered_index->backendIndex->getAllocationSize();
    size_t initial_mem_frontend = tiered_index->frontendIndex->getAllocationSize();

    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 1, 1);
    EXPECT_EQ(tiered_index->indexLabelCount(), 2);
    EXPECT_EQ(tiered_index->indexSize(), 2);
    // Delete both vectors.
    EXPECT_EQ(tiered_index->deleteVector(0), 1);
    EXPECT_EQ(tiered_index->deleteVector(1), 1);
    EXPECT_EQ(tiered_index->idToSwapJob.size(), 2);
    EXPECT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 2);
    // Expect to have pending repair jobs, so that swap job cannot be executed yet - for each
    // deleted vector there should be a single repair job.
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 2);
    EXPECT_EQ(tiered_index->idToSwapJob.at(0)->pending_repair_jobs_counter.load(), 1);
    EXPECT_EQ(tiered_index->idToSwapJob.at(1)->pending_repair_jobs_counter.load(), 1);
    mock_thread_pool.thread_iteration();
    mock_thread_pool.thread_iteration();
    EXPECT_EQ(tiered_index->idToSwapJob.size(), 2);
    // Insert another vector and remove it. expect it to have no neighbors.
    // Threshold for is set to be 1, so now we expect that a single deleted vector (which has no
    // pending repair jobs) will be swapped - and 2 will remain.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 2, 2);
    EXPECT_EQ(tiered_index->deleteVector(2), 1);
    EXPECT_EQ(tiered_index->idToSwapJob.size(), 2);
    EXPECT_EQ(tiered_index->indexSize(), 2);
    EXPECT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 2);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Now swap the rest of the jobs.
    tiered_index->executeReadySwapJobs();
    EXPECT_EQ(tiered_index->idToSwapJob.size(), 0);
    EXPECT_EQ(tiered_index->indexSize(), 0);
    EXPECT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 0);

    // Reserve manually 0 buckets in the hash tables so that memory would be as it was before we
    // started inserting vectors.
    tiered_index->idToRepairJobs.reserve(0);
    tiered_index->idToSwapJob.reserve(0);
    // Manually shrink the vectors so that memory would be as it was before we started inserting
    tiered_index->getHNSWIndex()->vectorBlocks.shrink_to_fit();
    tiered_index->getHNSWIndex()->graphDataBlocks.shrink_to_fit();

    EXPECT_EQ(tiered_index->backendIndex->getAllocationSize(), initial_mem_backend);
    EXPECT_EQ(tiered_index->frontendIndex->getAllocationSize(), initial_mem_frontend);
    EXPECT_EQ(tiered_index->getAllocationSize(), initial_mem);
    mock_thread_pool.reset_ctx();

    // VecSimAllocator::allocation_header_size = size_t, this should be the only memory that we
    // account for at this point.
    EXPECT_EQ(allocator->getAllocationSize(), sizeof(size_t));
}

TYPED_TEST(HNSWTieredIndexTest, swapJobBasic2) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti()};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool, 1);
    ASSERT_EQ(tiered_index->pendingSwapJobsThreshold, 1);
    auto allocator = tiered_index->getAllocator();

    // Insert 3 vectors, expect to have a fully connected graph.
    idType invalid_jobs_counter = 0;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 1, 1);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->backendIndex, dim, 2, 2);
    // Delete 0, expect to have two repair jobs pending for 1 and 2 and execute it.
    EXPECT_EQ(tiered_index->deleteVector(0), 1);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 2);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_REPAIR_NODE_CONNECTIONS_JOB);
    mock_thread_pool.thread_iteration();
    EXPECT_EQ(tiered_index->idToSwapJob.at(0)->pending_repair_jobs_counter.load(), 1);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_REPAIR_NODE_CONNECTIONS_JOB);
    mock_thread_pool.thread_iteration();
    EXPECT_EQ(tiered_index->idToSwapJob.at(0)->pending_repair_jobs_counter.load(), 0);
    // Delete 2, expect to create two repair job pending from 0 and 1. Also, expect that swap
    // job for 0 will be executed, so that 2 and 0 are swapped. Then, we should have only 1
    // pending repair job for the "new" 0 - for deleting the old 1->2, while the second job for
    // deleting the old 0->2 is invalid and reduced from the pending repair jobs counter.
    EXPECT_EQ(tiered_index->deleteVector(2), 1);
    EXPECT_EQ(tiered_index->indexSize(), 2);
    EXPECT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 1);

    EXPECT_EQ(tiered_index->idToSwapJob.at(0)->pending_repair_jobs_counter.load(), 1);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 2);
    // The first repair job should remove 1->0 (originally was 1->2).
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_REPAIR_NODE_CONNECTIONS_JOB);
    ASSERT_EQ(reinterpret_cast<HNSWRepairJob *>(mock_thread_pool.jobQ.front().job)->node_id, 1);
    ASSERT_EQ(reinterpret_cast<HNSWRepairJob *>(mock_thread_pool.jobQ.front().job)
                  ->associatedSwapJobs[0]
                  ->deleted_id,
              0);
    mock_thread_pool.thread_iteration();
    EXPECT_EQ(tiered_index->idToSwapJob.at(0)->pending_repair_jobs_counter.load(), 0);
    // The second repair job is invalid due to the removal of (the original) 0.
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_REPAIR_NODE_CONNECTIONS_JOB);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->isValid, false);
    ASSERT_EQ(reinterpret_cast<HNSWRepairJob *>(mock_thread_pool.jobQ.front().job)->node_id,
              invalid_jobs_counter++);
    ASSERT_EQ(reinterpret_cast<HNSWRepairJob *>(mock_thread_pool.jobQ.front().job)
                  ->associatedSwapJobs[0]
                  ->deleted_id,
              0);
    mock_thread_pool.thread_iteration();
    // Delete 1, that should still have 0->1 edge that should be repaired. This should cause
    // the swap and removal of 0 (that has no more pending jobs at that point) - so that 1 would
    // get id 0, and then the new 0 should have no pending repair jobs.
    EXPECT_EQ(tiered_index->deleteVector(1), 1);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 1);
    EXPECT_EQ(tiered_index->idToSwapJob.size(), 1);
    EXPECT_EQ(tiered_index->idToSwapJob.at(0)->deleted_id, 0);
    EXPECT_EQ(tiered_index->idToSwapJob.at(0)->pending_repair_jobs_counter.load(), 0);
    // The repair job is invalid due to the removal of (the previous) 0.
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_REPAIR_NODE_CONNECTIONS_JOB);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->isValid, false);
    ASSERT_EQ(reinterpret_cast<HNSWRepairJob *>(mock_thread_pool.jobQ.front().job)->node_id,
              invalid_jobs_counter);
    ASSERT_EQ(reinterpret_cast<HNSWRepairJob *>(mock_thread_pool.jobQ.front().job)
                  ->associatedSwapJobs[0]
                  ->deleted_id,
              0);
    mock_thread_pool.thread_iteration();
    EXPECT_EQ(tiered_index->indexSize(), 1);
    EXPECT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 1);
    EXPECT_EQ(tiered_index->getHNSWIndex()->safeGetEntryPointState().first, INVALID_ID);

    // Call delete again, this should only trigger the swap and removal of 1
    // (which has already deleted)
    EXPECT_EQ(tiered_index->deleteVector(1), 0);
    EXPECT_EQ(tiered_index->indexSize(), 0);
    EXPECT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 0);
}

// A set of lambdas that determine whether a vector should be inserted to the
// HNSW index (returns true) or to the flat index (returns false).
inline constexpr std::array<std::pair<std::string_view, bool (*)(size_t, size_t)>, 11> lambdas = {{
    {"100% HNSW,   0% FLAT ", [](size_t idx, size_t n) -> bool { return 1; }},
    {" 50% HNSW,  50% FLAT ", [](size_t idx, size_t n) -> bool { return idx % 2; }},
    {"  0% HNSW, 100% FLAT ", [](size_t idx, size_t n) -> bool { return 0; }},
    {" 90% HNSW,  10% FLAT ", [](size_t idx, size_t n) -> bool { return idx % 10; }},
    {" 10% HNSW,  90% FLAT ", [](size_t idx, size_t n) -> bool { return !(idx % 10); }},
    {" 99% HNSW,   1% FLAT ", [](size_t idx, size_t n) -> bool { return idx % 100; }},
    {"  1% HNSW,  99% FLAT ", [](size_t idx, size_t n) -> bool { return !(idx % 100); }},
    {"first 10% are in HNSW", [](size_t idx, size_t n) -> bool { return idx < (n / 10); }},
    {"first 10% are in FLAT", [](size_t idx, size_t n) -> bool { return idx >= (n / 10); }},
    {" last 10% are in FLAT", [](size_t idx, size_t n) -> bool { return idx < (9 * n / 10); }},
    {" last 10% are in HNSW", [](size_t idx, size_t n) -> bool { return idx >= (9 * n / 10); }},
}};

TYPED_TEST(HNSWTieredIndexTest, BatchIterator) {
    size_t d = 4;
    size_t M = 8;
    size_t ef = 20;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams hnsw_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .initialCapacity = n,
        .efConstruction = ef,
        .efRuntime = ef,
    };
    VecSimParams params = CreateParams(hnsw_params);

    // for (auto &[decider_name, decider] : lambdas) { // TODO: not supported by clang < 16
    for (auto &lambda : lambdas) {
        // manually deconstruct the pair to avoid the clang error
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredHNSWIndex(params, mock_thread_pool);
        auto allocator = tiered_index->getAllocator();

        auto *hnsw = tiered_index->backendIndex;
        auto *flat = tiered_index->frontendIndex;

        // For every i, add the vector (i,i,i,i) under the label i.
        for (size_t i = 0; i < n; i++) {
            auto cur = decider(i, n) ? hnsw : flat;
            GenerateAndAddVector<TEST_DATA_T>(cur, d, i % n_labels, i);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(tiered_index), n) << decider_name;

        // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
        TEST_DATA_T query[d];
        GenerateVector<TEST_DATA_T>(query, d, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(tiered_index, query, nullptr);
        size_t iteration_num = 0;

        // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet
        // in every iteration. The results order should be sorted by their score (distance from
        // the query vector), which means sorted from the largest id to the lowest.
        size_t n_res = 5;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            std::vector<size_t> expected_ids(n_res);
            for (size_t i = 0; i < n_res; i++) {
                expected_ids[i] = (n - iteration_num * n_res - i - 1) % n_labels;
            }
            auto verify_res = [&](size_t id, double score, size_t index) {
                ASSERT_EQ(expected_ids[index], id) << decider_name;
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num, n_labels / n_res) << decider_name;
        VecSimBatchIterator_Free(batchIterator);
    }
}

TYPED_TEST(HNSWTieredIndexTest, BatchIteratorReset) {
    size_t d = 4;
    size_t M = 8;
    size_t ef = 20;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams hnsw_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .initialCapacity = n,
        .efConstruction = ef,
        .efRuntime = ef,
    };
    VecSimParams params = CreateParams(hnsw_params);

    // for (auto &[decider_name, decider] : lambdas) { // TODO: not supported by clang < 16
    for (auto &lambda : lambdas) {
        // manually deconstruct the pair to avoid the clang error
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredHNSWIndex(params, mock_thread_pool);
        auto allocator = tiered_index->getAllocator();

        auto *hnsw = tiered_index->backendIndex;
        auto *flat = tiered_index->frontendIndex;

        // For every i, add the vector (i,i,i,i) under the label i.
        for (size_t i = 0; i < n; i++) {
            auto cur = decider(i, n) ? hnsw : flat;
            GenerateAndAddVector<TEST_DATA_T>(cur, d, i % n_labels, i);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(tiered_index), n) << decider_name;

        // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
        TEST_DATA_T query[d];
        GenerateVector<TEST_DATA_T>(query, d, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(tiered_index, query, nullptr);
        ASSERT_NO_FATAL_FAILURE(VecSimBatchIterator_Reset(batchIterator));

        // Get the 100 vectors whose ids are the maximal among those that hasn't been returned yet,
        // in every iteration. Run this flow for 3 times, and reset the iterator.
        size_t n_res = 100;
        size_t re_runs = 3;

        for (size_t take = 0; take < re_runs; take++) {
            size_t iteration_num = 0;
            while (VecSimBatchIterator_HasNext(batchIterator)) {
                std::vector<size_t> expected_ids(n_res);
                for (size_t i = 0; i < n_res; i++) {
                    expected_ids[i] = (n - iteration_num * n_res - i - 1) % n_labels;
                }
                auto verify_res = [&](size_t id, double score, size_t index) {
                    ASSERT_EQ(expected_ids[index], id) << decider_name;
                };
                runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_SCORE);
                iteration_num++;
            }
            ASSERT_EQ(iteration_num, n_labels / n_res) << decider_name;
            VecSimBatchIterator_Reset(batchIterator);
        }

        // Try resetting the iterator before it is depleted.
        n_res = 10;
        for (size_t take = 0; take < re_runs; take++) {
            size_t iteration_num = 0;
            do {
                ASSERT_TRUE(VecSimBatchIterator_HasNext(batchIterator)) << decider_name;
                std::vector<size_t> expected_ids(n_res);
                for (size_t i = 0; i < n_res; i++) {
                    expected_ids[i] = (n - iteration_num * n_res - i - 1) % n_labels;
                }
                auto verify_res = [&](size_t id, double score, size_t index) {
                    ASSERT_EQ(expected_ids[index], id) << decider_name;
                };
                runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_SCORE);
            } while (5 > iteration_num++);
            VecSimBatchIterator_Reset(batchIterator);
        }
        VecSimBatchIterator_Free(batchIterator);
    }
}

TYPED_TEST(HNSWTieredIndexTest, BatchIteratorSize1) {
    size_t d = 4;
    size_t M = 8;
    size_t ef = 20;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams hnsw_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .initialCapacity = n,
        .efConstruction = ef,
        .efRuntime = ef,
    };
    VecSimParams params = CreateParams(hnsw_params);

    // for (auto &[decider_name, decider] : lambdas) { // TODO: not supported by clang < 16
    for (auto &lambda : lambdas) {
        // manually deconstruct the pair to avoid the clang error
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredHNSWIndex(params, mock_thread_pool);
        auto allocator = tiered_index->getAllocator();

        auto *hnsw = tiered_index->backendIndex;
        auto *flat = tiered_index->frontendIndex;

        // For every i, add the vector (i,i,i,i) under the label `n_labels - (i % n_labels)`.
        for (size_t i = 0; i < n; i++) {
            auto cur = decider(i, n) ? hnsw : flat;
            GenerateAndAddVector<TEST_DATA_T>(cur, d, n_labels - (i % n_labels), i);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(tiered_index), n) << decider_name;

        // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
        TEST_DATA_T query[d];
        GenerateVector<TEST_DATA_T>(query, d, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(tiered_index, query, nullptr);

        size_t iteration_num = 0;
        size_t n_res = 1, expected_n_res = 1;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            iteration_num++;
            // Expect to get results in the reverse order of labels - which is the order of the
            // distance from the query vector. Get one result in every iteration.
            auto verify_res = [&](size_t id, double score, size_t index) {
                ASSERT_EQ(id, iteration_num) << decider_name;
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_SCORE, expected_n_res);
        }

        ASSERT_EQ(iteration_num, n_labels) << decider_name;
        VecSimBatchIterator_Free(batchIterator);
    }
}

TYPED_TEST(HNSWTieredIndexTest, BatchIteratorAdvanced) {
    size_t d = 4;
    size_t M = 8;
    size_t ef = 1000;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams hnsw_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .initialCapacity = n,
        .efConstruction = ef,
    };
    VecSimParams params = CreateParams(hnsw_params);
    HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
    VecSimQueryParams query_params = CreateQueryParams(hnswRuntimeParams);

    // for (auto &[decider_name, decider] : lambdas) { // TODO: not supported by clang < 16
    for (auto &lambda : lambdas) {
        // manually deconstruct the pair to avoid the clang error
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredHNSWIndex(params, mock_thread_pool);
        auto allocator = tiered_index->getAllocator();

        auto *hnsw = tiered_index->backendIndex;
        auto *flat = tiered_index->frontendIndex;

        TEST_DATA_T query[d];
        GenerateVector<TEST_DATA_T>(query, d, n);

        VecSimBatchIterator *batchIterator =
            VecSimBatchIterator_New(tiered_index, query, &query_params);

        // Try to get results even though there are no vectors in the index.
        VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, 10, BY_SCORE);
        ASSERT_EQ(VecSimQueryResult_Len(res), 0) << decider_name;
        VecSimQueryResult_Free(res);
        ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator)) << decider_name;

        // Insert one label and query again. The internal id will be 0.
        for (size_t j = 0; j < per_label; j++) {
            GenerateAndAddVector<TEST_DATA_T>(decider(n_labels, n) ? hnsw : flat, d, n_labels,
                                              n - j);
        }
        VecSimBatchIterator_Reset(batchIterator);
        res = VecSimBatchIterator_Next(batchIterator, 10, BY_SCORE);
        ASSERT_EQ(VecSimQueryResult_Len(res), 1) << decider_name;
        VecSimQueryResult_Free(res);
        ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator)) << decider_name;
        VecSimBatchIterator_Free(batchIterator);

        // Insert vectors to the index and re-create the batch iterator.
        for (size_t i = 1; i < n_labels; i++) {
            auto cur = decider(i, n) ? hnsw : flat;
            for (size_t j = 1; j <= per_label; j++) {
                GenerateAndAddVector<TEST_DATA_T>(cur, d, i, (i - 1) * per_label + j);
            }
        }
        ASSERT_EQ(VecSimIndex_IndexSize(tiered_index), n) << decider_name;
        batchIterator = VecSimBatchIterator_New(tiered_index, query, &query_params);

        // Try to get 0 results.
        res = VecSimBatchIterator_Next(batchIterator, 0, BY_SCORE);
        ASSERT_EQ(VecSimQueryResult_Len(res), 0) << decider_name;
        VecSimQueryResult_Free(res);

        // n_res does not divide into ef or vice versa - expect leftovers between the graph scans.
        size_t n_res = 7;
        size_t iteration_num = 0;

        while (VecSimBatchIterator_HasNext(batchIterator)) {
            iteration_num++;
            std::vector<size_t> expected_ids;
            // We ask to get the results sorted by ID in a specific batch (in ascending order), but
            // in every iteration the ids should be lower than the previous one, according to the
            // distance from the query.
            for (size_t i = 1; i <= n_res; i++) {
                expected_ids.push_back(n_labels - iteration_num * n_res + i);
            }
            auto verify_res = [&](size_t id, double score, size_t index) {
                ASSERT_EQ(expected_ids[index], id) << decider_name;
            };
            if (iteration_num <= n_labels / n_res) {
                runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_ID);
            } else {
                // In the last iteration there are `n_labels % n_res` results left to return.
                size_t n_left = n_labels % n_res;
                // Remove the first `n_res - n_left` ids from the expected ids.
                while (expected_ids.size() > n_left) {
                    expected_ids.erase(expected_ids.begin());
                }
                runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_ID, n_left);
            }
        }
        ASSERT_EQ(iteration_num, n_labels / n_res + 1) << decider_name;
        // Try to get more results even though there are no.
        res = VecSimBatchIterator_Next(batchIterator, 1, BY_SCORE);
        ASSERT_EQ(VecSimQueryResult_Len(res), 0) << decider_name;
        VecSimQueryResult_Free(res);

        VecSimBatchIterator_Free(batchIterator);
    }
}

TYPED_TEST(HNSWTieredIndexTest, BatchIteratorWithOverlaps) {
    size_t d = 4;
    size_t M = 8;
    size_t ef = 20;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams hnsw_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .initialCapacity = n,
        .efConstruction = ef,
        .efRuntime = ef,
    };
    VecSimParams params = CreateParams(hnsw_params);

    // for (auto &[decider_name, decider] : lambdas) { // TODO: not supported by clang < 16
    for (auto &lambda : lambdas) {
        // manually deconstruct the pair to avoid the clang error
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredHNSWIndex(params, mock_thread_pool);
        auto allocator = tiered_index->getAllocator();

        auto *hnsw = tiered_index->backendIndex;
        auto *flat = tiered_index->frontendIndex;

        // For every i, add the vector (i,i,i,i) under the label i.
        size_t flat_count = 0;
        for (size_t i = 0; i < n; i++) {
            auto cur = decider(i, n) ? hnsw : flat;
            GenerateAndAddVector<TEST_DATA_T>(cur, d, i % n_labels, i);
            if (cur == flat) {
                flat_count++;
                // Add 10% of the vectors in FLAT to HNSW as well.
                if (flat_count % 10 == 0) {
                    GenerateAndAddVector<TEST_DATA_T>(hnsw, d, i % n_labels, i);
                }
            }
        }
        // The index size should be 100-110% of n.
        ASSERT_LE(VecSimIndex_IndexSize(tiered_index), n * 1.1) << decider_name;
        ASSERT_GE(VecSimIndex_IndexSize(tiered_index), n) << decider_name;
        // The number of unique labels should be n_labels.
        ASSERT_EQ(tiered_index->indexLabelCount(), n_labels) << decider_name;

        // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
        TEST_DATA_T query[d];
        GenerateVector<TEST_DATA_T>(query, d, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(tiered_index, query, nullptr);
        size_t iteration_num = 0;

        // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet
        // in every iteration. The results order should be sorted by their score (distance from
        // the query vector), which means sorted from the largest id to the lowest.
        size_t n_res = 5;
        size_t n_expected = n_res;
        size_t excessive_iterations = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            if (iteration_num * n_res == n_labels) {
                // in some cases, the batch iterator may report that it has more results to return,
                // but it's actually not true and the next call to `VecSimBatchIterator_Next` will
                // return 0 results. This is safe because we don't guarantee how many results the
                // batch iterator will return, and a similar scenario can happen when checking
                // `VecSimBatchIterator_HasNext` on an empty index for the first time (before the
                // first call to `VecSimBatchIterator_Next`). we check that this scenario doesn't
                // happen more than once.
                ASSERT_EQ(excessive_iterations, 0) << decider_name;
                excessive_iterations = 1;
                n_expected = 0;
            }
            std::vector<size_t> expected_ids(n_expected);
            for (size_t i = 0; i < n_expected; i++) {
                expected_ids[i] = (n - iteration_num * n_expected - i - 1) % n_labels;
            }
            auto verify_res = [&](size_t id, double score, size_t index) {
                ASSERT_EQ(expected_ids[index], id) << decider_name;
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_SCORE, n_expected);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num - excessive_iterations, n_labels / n_res)
            << decider_name << "\nHad excessive iterations: " << (excessive_iterations != 0);
        VecSimBatchIterator_Free(batchIterator);
    }
}

TYPED_TEST(HNSWTieredIndexTestBasic, BatchIteratorWithOverlaps_SpacialMultiCases) {
    size_t d = 4;

    std::shared_ptr<VecSimAllocator> allocator;
    TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *tiered_index;
    VecSimIndex *hnsw, *flat;
    TEST_DATA_T query[d];
    VecSimBatchIterator *iterator;
    VecSimQueryResult_List batch;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams hnsw_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = true,
    };
    VecSimParams params = CreateParams(hnsw_params);
    auto mock_thread_pool = tieredIndexMock();

    auto L2 = [&](size_t element) { return element * element * d; };

    // TEST 1:
    // first batch contains duplicates with different scores.
    tiered_index = this->CreateTieredHNSWIndex(params, mock_thread_pool);
    allocator = tiered_index->getAllocator();
    hnsw = tiered_index->backendIndex;
    flat = tiered_index->frontendIndex;

    GenerateAndAddVector<TEST_DATA_T>(flat, d, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(flat, d, 1, 1);
    GenerateAndAddVector<TEST_DATA_T>(flat, d, 2, 2);

    GenerateAndAddVector<TEST_DATA_T>(hnsw, d, 1, 3);
    GenerateAndAddVector<TEST_DATA_T>(hnsw, d, 0, 4);
    GenerateAndAddVector<TEST_DATA_T>(hnsw, d, 3, 5);

    ASSERT_EQ(tiered_index->indexLabelCount(), 4);

    GenerateVector<TEST_DATA_T>(query, d, 0);
    iterator = VecSimBatchIterator_New(tiered_index, query, nullptr);

    // batch size is 3 (the size of each index). Internally the tiered batch iterator will have to
    // handle the duplicates with different scores.
    ASSERT_TRUE(VecSimBatchIterator_HasNext(iterator));
    batch = VecSimBatchIterator_Next(iterator, 3, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(batch), 3);
    ASSERT_EQ(VecSimQueryResult_GetId(batch.results + 0), 0);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch.results + 0), L2(0));
    ASSERT_EQ(VecSimQueryResult_GetId(batch.results + 1), 1);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch.results + 1), L2(1));
    ASSERT_EQ(VecSimQueryResult_GetId(batch.results + 2), 2);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch.results + 2), L2(2));
    VecSimQueryResult_Free(batch);

    // we have 1 more label in the index. we expect the tiered batch iterator to return it only and
    // filter out the duplicates.
    ASSERT_TRUE(VecSimBatchIterator_HasNext(iterator));
    batch = VecSimBatchIterator_Next(iterator, 2, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(batch), 1);
    ASSERT_EQ(VecSimQueryResult_GetId(batch.results + 0), 3);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch.results + 0), L2(5));
    ASSERT_FALSE(VecSimBatchIterator_HasNext(iterator));
    VecSimQueryResult_Free(batch);
    // TEST 1 clean up.
    VecSimBatchIterator_Free(iterator);

    // TEST 2:
    // second batch contains duplicates (different scores) from the first batch.
    auto *ctx = new tieredIndexMock::IndexExtCtx(&mock_thread_pool);
    mock_thread_pool.reset_ctx(ctx);
    tiered_index = this->CreateTieredHNSWIndex(params, mock_thread_pool);
    allocator = tiered_index->getAllocator();
    hnsw = tiered_index->backendIndex;
    flat = tiered_index->frontendIndex;

    GenerateAndAddVector<TEST_DATA_T>(hnsw, d, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(hnsw, d, 1, 1);
    GenerateAndAddVector<TEST_DATA_T>(hnsw, d, 2, 2);
    GenerateAndAddVector<TEST_DATA_T>(hnsw, d, 3, 3);

    GenerateAndAddVector<TEST_DATA_T>(flat, d, 2, 0);
    GenerateAndAddVector<TEST_DATA_T>(flat, d, 3, 1);
    GenerateAndAddVector<TEST_DATA_T>(flat, d, 0, 2);
    GenerateAndAddVector<TEST_DATA_T>(flat, d, 1, 3);

    ASSERT_EQ(tiered_index->indexLabelCount(), 4);

    iterator = VecSimBatchIterator_New(tiered_index, query, nullptr);

    // ask for 2 results. The internal batch iterators will return 2 results: hnsw - [0, 1], flat -
    // [2, 3] so there are no duplicates.
    ASSERT_TRUE(VecSimBatchIterator_HasNext(iterator));
    batch = VecSimBatchIterator_Next(iterator, 2, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(batch), 2);
    ASSERT_EQ(VecSimQueryResult_GetId(batch.results + 0), 0);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch.results + 0), L2(0));
    ASSERT_EQ(VecSimQueryResult_GetId(batch.results + 1), 2);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch.results + 1), L2(0));
    VecSimQueryResult_Free(batch);

    // first batch contained 1 result from each index, so there is one leftover from each iterator.
    // Asking for 3 results will return additional 2 results from each iterator and the tiered batch
    // iterator will have to handle the duplicates that each iterator returned (both labels that
    // were returned in the first batch and duplicates in the current batch).
    ASSERT_TRUE(VecSimBatchIterator_HasNext(iterator));
    batch = VecSimBatchIterator_Next(iterator, 3, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(batch), 2);
    ASSERT_EQ(VecSimQueryResult_GetId(batch.results + 0), 1);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch.results + 0), L2(1));
    ASSERT_EQ(VecSimQueryResult_GetId(batch.results + 1), 3);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch.results + 1), L2(1));
    ASSERT_FALSE(VecSimBatchIterator_HasNext(iterator));
    VecSimQueryResult_Free(batch);
    // TEST 2 clean up.
    VecSimBatchIterator_Free(iterator);
}

TYPED_TEST(HNSWTieredIndexTest, parallelBatchIteratorSearch) {
    size_t dim = 4;
    size_t ef = 500;
    size_t n = 1000;
    size_t n_res_min = 3;  // minimum number of results to return per batch
    size_t n_res_max = 15; // maximum number of results to return per batch
    bool isMulti = TypeParam::isMulti();

    size_t per_label = isMulti ? 5 : 1;
    size_t n_labels = n / per_label;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = isMulti,
        .efRuntime = ef,
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    std::atomic_int successful_searches(0);
    auto parallel_10_batches = [](AsyncJob *job) {
        auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
        const size_t res_per_batch = search_job->k;
        const size_t dim = search_job->dim;
        const auto query = search_job->query;

        size_t iteration = 0;
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            TEST_DATA_T element = *(TEST_DATA_T *)query;
            res_index += iteration * res_per_batch;
            ASSERT_EQ(std::abs(id - element), (res_index + 1) / 2);
            ASSERT_EQ(score, dim * (id - element) * (id - element));
        };

        // Run 10 batches of search.
        auto tiered_iterator = VecSimBatchIterator_New(search_job->index, query, nullptr);
        do {
            runBatchIteratorSearchTest(tiered_iterator, res_per_batch, verify_res);
        } while (++iteration < 10 && VecSimBatchIterator_HasNext(tiered_iterator));

        VecSimBatchIterator_Free(tiered_iterator);
        (*search_job->successful_searches)++;
        delete job;
    };

    // Fill the job queue with insert and batch-search jobs, while filling the flat index, before
    // initializing the thread pool.
    for (size_t i = 0; i < n; i++) {
        // Insert a vector to the flat index and add a job to insert it to the main index.
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i % n_labels, i);

        // Add a search job.
        size_t cur_res_per_batch = i % (n_res_max - n_res_min) + n_res_min;
        size_t n_res = cur_res_per_batch * 10;
        auto query = (TEST_DATA_T *)allocator->allocate(dim * sizeof(TEST_DATA_T));
        // make sure there are `n_res / 2` vectors in the index in each "side" of the query vector.
        GenerateVector<TEST_DATA_T>(query, dim, (i % (n_labels - n_res)) + (n_res / 2));
        auto search_job = new (allocator)
            tieredIndexMock::SearchJobMock(allocator, parallel_10_batches, tiered_index,
                                           cur_res_per_batch, query, n, dim, &successful_searches);
        tiered_index->submitSingleJob(search_job);
    }

    EXPECT_EQ(tiered_index->indexSize(), n);
    EXPECT_EQ(tiered_index->indexLabelCount(), n_labels);
    EXPECT_EQ(tiered_index->labelToInsertJobs.size(), n_labels);
    for (auto &it : tiered_index->labelToInsertJobs) {
        EXPECT_EQ(it.second.size(), per_label);
    }
    EXPECT_EQ(tiered_index->frontendIndex->indexSize(), n);
    EXPECT_EQ(tiered_index->backendIndex->indexSize(), 0);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    // All the vectors are already in the tiered index, so we expect to find the expected
    // results from the get-go.
    mock_thread_pool.init_threads();
    mock_thread_pool.thread_pool_join();

    EXPECT_EQ(tiered_index->backendIndex->indexSize(), n);
    EXPECT_EQ(tiered_index->backendIndex->indexLabelCount(), n_labels);
    EXPECT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(HNSWTieredIndexTestBasic, overwriteVectorBasic) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool, 1);
    auto allocator = tiered_index->getAllocator();

    TEST_DATA_T val = 1.0;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 0, val);
    // Overwrite label 0 (in the flat buffer) with a different value.
    val = 2.0;
    TEST_DATA_T overwritten_vec[] = {val, val, val, val};
    ASSERT_EQ(tiered_index->addVector(overwritten_vec, 0), 0);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->getDistanceFrom(0, overwritten_vec), 0);

    // Validate that jobs were created properly - first job should be invalid after overwrite,
    // the second should be a pending insert job.
    ASSERT_EQ(tiered_index->labelToInsertJobs.at(0).size(), 1);
    auto *pending_insert_job = tiered_index->labelToInsertJobs.at(0)[0];
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 2);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_INSERT_VECTOR_JOB);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->isValid, false);
    ASSERT_EQ(reinterpret_cast<HNSWInsertJob *>(mock_thread_pool.jobQ.front().job)->label, 0);
    ASSERT_EQ(reinterpret_cast<HNSWInsertJob *>(mock_thread_pool.jobQ.front().job)->id, 0);
    mock_thread_pool.thread_iteration();

    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_INSERT_VECTOR_JOB);
    ASSERT_EQ(reinterpret_cast<HNSWInsertJob *>(mock_thread_pool.jobQ.front().job)->label, 0);
    ASSERT_EQ(reinterpret_cast<HNSWInsertJob *>(mock_thread_pool.jobQ.front().job)->id, 0);
    ASSERT_EQ(reinterpret_cast<HNSWInsertJob *>(mock_thread_pool.jobQ.front().job),
              pending_insert_job);

    // Ingest vector into HNSW, and then overwrite it.
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    val = 3.0;
    overwritten_vec[0] = overwritten_vec[1] = overwritten_vec[2] = overwritten_vec[3] = val;
    ASSERT_EQ(tiered_index->addVector(overwritten_vec, 0), 0);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    // Swap job should be executed for the overwritten vector since limit is 1, and we are calling
    // swap job execution prior to insert jobs.
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->getDistanceFrom(0, overwritten_vec), 0);

    // Ingest the updated vector to HNSW.
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->getDistanceFrom(0, overwritten_vec), 0);
}

TYPED_TEST(HNSWTieredIndexTestBasic, overwriteVectorAsync) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};
    VecSimParams hnsw_params = CreateParams(params);
    for (size_t maxSwapJobs : {(int)n + 1, 1}) {
        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index =
            this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool, maxSwapJobs);
        auto allocator = tiered_index->getAllocator();

        // Launch the BG threads loop that takes jobs from the queue and executes them.

        for (size_t i = 0; i < mock_thread_pool.thread_pool_size; i++) {
            mock_thread_pool.thread_pool.emplace_back(tieredIndexMock::thread_main_loop, i,
                                                      std::ref(mock_thread_pool));
        }

        // Insert vectors and overwrite them multiple times while thread run in the background.
        std::srand(10); // create pseudo random generator with any arbitrary seed.
        for (size_t i = 0; i < n; i++) {
            TEST_DATA_T vector[dim];
            for (size_t j = 0; j < dim; j++) {
                vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
            }
            tiered_index->addVector(vector, i);
        }
        EXPECT_EQ(tiered_index->indexLabelCount(), n);

        size_t num_overwrites = 1000;
        for (size_t i = 0; i < num_overwrites; i++) {
            size_t label_to_overwrite = std::rand() % n;
            TEST_DATA_T vector[dim];
            for (size_t j = 0; j < dim; j++) {
                vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
            }
            EXPECT_EQ(tiered_index->addVector(vector, label_to_overwrite), 0);
        }

        mock_thread_pool.thread_pool_join();

        EXPECT_EQ(tiered_index->indexSize() - tiered_index->getHNSWIndex()->getNumMarkedDeleted(),
                  n);
        EXPECT_EQ(tiered_index->frontendIndex->indexSize(), 0);
        EXPECT_EQ(tiered_index->indexLabelCount(), n);
        auto report = tiered_index->getHNSWIndex()->checkIntegrity();
        EXPECT_EQ(report.connections_to_repair, 0);
        EXPECT_EQ(report.valid_state, true);
    }
}

TYPED_TEST(HNSWTieredIndexTest, testInfo) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti()};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool, 1, 1000);
    auto allocator = tiered_index->getAllocator();

    VecSimIndexInfo info = tiered_index->info();
    EXPECT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_HNSWLIB);
    EXPECT_EQ(info.commonInfo.indexSize, 0);
    EXPECT_EQ(info.commonInfo.indexLabelCount, 0);
    EXPECT_EQ(info.commonInfo.memory, tiered_index->getAllocationSize());
    EXPECT_EQ(info.commonInfo.basicInfo.isMulti, TypeParam::isMulti());
    EXPECT_EQ(info.commonInfo.basicInfo.dim, dim);
    EXPECT_EQ(info.commonInfo.basicInfo.metric, VecSimMetric_L2);
    EXPECT_EQ(info.commonInfo.basicInfo.type, TypeParam::get_index_type());
    EXPECT_EQ(info.commonInfo.basicInfo.blockSize, DEFAULT_BLOCK_SIZE);
    VecSimIndexInfo frontendIndexInfo = tiered_index->frontendIndex->info();
    VecSimIndexInfo backendIndexInfo = tiered_index->backendIndex->info();

    compareCommonInfo(info.tieredInfo.frontendCommonInfo, frontendIndexInfo.commonInfo);
    compareFlatInfo(info.tieredInfo.bfInfo, frontendIndexInfo.bfInfo);
    compareCommonInfo(info.tieredInfo.backendCommonInfo, backendIndexInfo.commonInfo);
    compareHNSWInfo(info.tieredInfo.backendInfo.hnswInfo, backendIndexInfo.hnswInfo);

    EXPECT_EQ(info.commonInfo.memory, info.tieredInfo.management_layer_memory +
                                          backendIndexInfo.commonInfo.memory +
                                          frontendIndexInfo.commonInfo.memory);
    EXPECT_EQ(info.tieredInfo.backgroundIndexing, false);
    EXPECT_EQ(info.tieredInfo.bufferLimit, 1000);
    EXPECT_EQ(info.tieredInfo.specificTieredBackendInfo.hnswTieredInfo.pendingSwapJobsThreshold, 1);

    // Validate that Static info returns the right restricted info as well.
    VecSimIndexBasicInfo s_info = VecSimIndex_BasicInfo(tiered_index);
    ASSERT_EQ(info.commonInfo.basicInfo.algo, s_info.algo);
    ASSERT_EQ(info.commonInfo.basicInfo.dim, s_info.dim);
    ASSERT_EQ(info.commonInfo.basicInfo.blockSize, s_info.blockSize);
    ASSERT_EQ(info.commonInfo.basicInfo.type, s_info.type);
    ASSERT_EQ(info.commonInfo.basicInfo.isMulti, s_info.isMulti);
    ASSERT_EQ(info.commonInfo.basicInfo.type, s_info.type);
    ASSERT_EQ(info.commonInfo.basicInfo.isTiered, s_info.isTiered);

    GenerateAndAddVector(tiered_index, dim, 1, 1);
    info = tiered_index->info();

    EXPECT_EQ(info.commonInfo.indexSize, 1);
    EXPECT_EQ(info.commonInfo.indexLabelCount, 1);
    EXPECT_EQ(info.tieredInfo.backendCommonInfo.indexSize, 0);
    EXPECT_EQ(info.tieredInfo.backendCommonInfo.indexLabelCount, 0);
    EXPECT_EQ(info.tieredInfo.frontendCommonInfo.indexSize, 1);
    EXPECT_EQ(info.tieredInfo.frontendCommonInfo.indexLabelCount, 1);
    EXPECT_EQ(info.commonInfo.memory, info.tieredInfo.management_layer_memory +
                                          info.tieredInfo.backendCommonInfo.memory +
                                          info.tieredInfo.frontendCommonInfo.memory);
    EXPECT_EQ(info.tieredInfo.backgroundIndexing, true);

    mock_thread_pool.thread_iteration();
    info = tiered_index->info();

    EXPECT_EQ(info.commonInfo.indexSize, 1);
    EXPECT_EQ(info.commonInfo.indexLabelCount, 1);
    EXPECT_EQ(info.tieredInfo.backendCommonInfo.indexSize, 1);
    EXPECT_EQ(info.tieredInfo.backendCommonInfo.indexLabelCount, 1);
    EXPECT_EQ(info.tieredInfo.frontendCommonInfo.indexSize, 0);
    EXPECT_EQ(info.tieredInfo.frontendCommonInfo.indexLabelCount, 0);
    EXPECT_EQ(info.commonInfo.memory, info.tieredInfo.management_layer_memory +
                                          info.tieredInfo.backendCommonInfo.memory +
                                          info.tieredInfo.frontendCommonInfo.memory);
    EXPECT_EQ(info.tieredInfo.backgroundIndexing, false);

    if (TypeParam::isMulti()) {
        GenerateAndAddVector(tiered_index, dim, 1, 1);
        info = tiered_index->info();

        EXPECT_EQ(info.commonInfo.indexSize, 2);
        EXPECT_EQ(info.commonInfo.indexLabelCount, 1);
        EXPECT_EQ(info.tieredInfo.backendCommonInfo.indexSize, 1);
        EXPECT_EQ(info.tieredInfo.backendCommonInfo.indexLabelCount, 1);
        EXPECT_EQ(info.tieredInfo.frontendCommonInfo.indexSize, 1);
        EXPECT_EQ(info.tieredInfo.frontendCommonInfo.indexLabelCount, 1);
        EXPECT_EQ(info.commonInfo.memory, info.tieredInfo.management_layer_memory +
                                              info.tieredInfo.backendCommonInfo.memory +
                                              info.tieredInfo.frontendCommonInfo.memory);
        EXPECT_EQ(info.tieredInfo.backgroundIndexing, true);
    }

    VecSimIndex_DeleteVector(tiered_index, 1);
    info = tiered_index->info();

    EXPECT_EQ(info.commonInfo.indexSize, 0);
    EXPECT_EQ(info.commonInfo.indexLabelCount, 0);
    EXPECT_EQ(info.tieredInfo.backendCommonInfo.indexSize, 0);
    EXPECT_EQ(info.tieredInfo.backendCommonInfo.indexLabelCount, 0);
    EXPECT_EQ(info.tieredInfo.frontendCommonInfo.indexSize, 0);
    EXPECT_EQ(info.tieredInfo.frontendCommonInfo.indexLabelCount, 0);
    EXPECT_EQ(info.commonInfo.memory, info.tieredInfo.management_layer_memory +
                                          info.tieredInfo.backendCommonInfo.memory +
                                          info.tieredInfo.frontendCommonInfo.memory);
    EXPECT_EQ(info.tieredInfo.backgroundIndexing, false);
}

TYPED_TEST(HNSWTieredIndexTest, testInfoIterator) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti()};

    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool, 1);
    auto allocator = tiered_index->getAllocator();

    GenerateAndAddVector(tiered_index, dim, 1, 1);
    VecSimIndexInfo info = tiered_index->info();
    VecSimIndexInfo frontendIndexInfo = tiered_index->frontendIndex->info();
    VecSimIndexInfo backendIndexInfo = tiered_index->backendIndex->info();

    VecSimInfoIterator *infoIterator = tiered_index->infoIterator();
    EXPECT_EQ(infoIterator->numberOfFields(), 15);

    while (infoIterator->hasNext()) {
        VecSim_InfoField *infoField = VecSimInfoIterator_NextField(infoIterator);

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
    VecSimInfoIterator_Free(infoIterator);
}

TYPED_TEST(HNSWTieredIndexTest, writeInPlaceMode) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;

    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti()};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    VecSim_SetWriteMode(VecSim_WriteInPlace);
    // Validate that the vector was inserted directly to the HNSW index.
    labelType vec_label = 0;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, 0);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);

    // Overwrite inplace - only in single-value mode
    if (!TypeParam::isMulti()) {
        TEST_DATA_T overwritten_vec[] = {1, 1, 1, 1};
        tiered_index->addVector(overwritten_vec, vec_label);
        ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);
        ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
        ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);
        ASSERT_EQ(tiered_index->getDistanceFrom(vec_label, overwritten_vec), 0);
    }

    // Validate that the vector is removed in place.
    tiered_index->deleteVector(vec_label);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 0);
}

TYPED_TEST(HNSWTieredIndexTest, switchWriteModes) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 500;
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti(),
                         .M = 32,
                         .efRuntime = 2 * n};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();
    VecSim_SetWriteMode(VecSim_WriteAsync);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();

    // Create and insert vectors one by one async.
    size_t per_label = TypeParam::isMulti() ? 5 : 1;
    size_t n_labels = n / per_label;
    std::srand(10); // create pseudo random generator with any arbitrary seed.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        for (size_t j = 0; j < dim; j++) {
            vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
        }
        VecSimIndex_AddVector(tiered_index, vector, i % n_labels);
    }

    // Insert another n more vectors INPLACE, while the previous vectors are still being indexed.
    VecSim_SetWriteMode(VecSim_WriteInPlace);
    EXPECT_LE(tiered_index->backendIndex->indexSize(), n);
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        for (size_t j = 0; j < dim; j++) {
            vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
        }
        VecSimIndex_AddVector(tiered_index, vector, i % n_labels + n_labels);
    }
    mock_thread_pool.thread_pool_join();
    EXPECT_EQ(tiered_index->backendIndex->indexSize(), 2 * n);

    // Now delete the last n inserted vectors of the index using async jobs.
    VecSim_SetWriteMode(VecSim_WriteAsync);
    mock_thread_pool.init_threads();
    for (size_t i = 0; i < n_labels; i++) {
        VecSimIndex_DeleteVector(tiered_index, n_labels + i);
    }
    // At this point, repair jobs should be executed in the background.
    EXPECT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), n);

    // Insert INPLACE another n vector (instead of the ones that were deleted).
    VecSim_SetWriteMode(VecSim_WriteInPlace);
    // Run twice, at first run we insert non-existing labels, in the second run we overwrite them
    // (for single-value index only).
    for (auto overwrite : {0, 1}) {
        for (size_t i = 0; i < n; i++) {
            TEST_DATA_T vector[dim];
            for (size_t j = 0; j < dim; j++) {
                vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
            }
            EXPECT_EQ(tiered_index->addVector(vector, i % n_labels + n_labels),
                      TypeParam::isMulti() ? 1 : 1 - overwrite);
            // Run a query and see that we only receive ids with label < n_labels+i
            // (the label that we just inserted), and the first result should be this vector.
            auto ver_res = [&](size_t label, double score, size_t index) {
                if (index == 0) {
                    EXPECT_EQ(label, i % n_labels + n_labels);
                    EXPECT_DOUBLE_EQ(score, 0);
                }
                if (!overwrite) {
                    ASSERT_LE(label, i + n_labels);
                }
            };
            runTopKSearchTest(tiered_index, vector, 10, ver_res);
        }
    }

    mock_thread_pool.thread_pool_join();
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->backendIndex->indexLabelCount(), 2 * n_labels);
}

TYPED_TEST(HNSWTieredIndexTest, bufferLimit) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti()};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Create tiered index with buffer limit set to 0.
    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool,
                                                     DEFAULT_PENDING_SWAP_JOBS_THRESHOLD, 0);
    auto allocator = tiered_index->getAllocator();

    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 0);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);

    // Set the flat limit to 1 and insert another vector - expect it to go to the flat buffer.
    tiered_index->flatBufferLimit = 1;
    labelType vec_label = 1;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, 0); // vector is [0,0,0,0]
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);

    // Overwrite the vector, expect removing it from the flat buffer and replace it with the new one
    // only in single-value mode
    if (!TypeParam::isMulti()) {
        TEST_DATA_T overwritten_vec[] = {1, 1, 1, 1};
        ASSERT_EQ(tiered_index->addVector(overwritten_vec, vec_label), 0);
        ASSERT_EQ(tiered_index->backendIndex->indexSize(), 1);
        ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);
        ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);
        ASSERT_EQ(tiered_index->getDistanceFrom(vec_label, overwritten_vec), 0);
        // The first job in Q should be the invalid overwritten insert vector job.
        ASSERT_EQ(mock_thread_pool.jobQ.front().job->isValid, false);
        ASSERT_EQ(reinterpret_cast<HNSWInsertJob *>(mock_thread_pool.jobQ.front().job)->id, 0);
        mock_thread_pool.jobQ.pop();
    }

    // Insert another vector, this one should go directly to HNSW index since the buffer limit has
    // reached.
    vec_label = 2;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, 0); // vector is [0,0,0,0]
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), 2);
    ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);
    ASSERT_EQ(tiered_index->indexLabelCount(), 3);

    // Overwrite the vector, expect marking it as deleted in HNSW and insert the new one directly
    // to HNSW as well.
    if (!TypeParam::isMulti()) {
        TEST_DATA_T overwritten_vec[] = {1, 1, 1, 1};
        ASSERT_EQ(tiered_index->addVector(overwritten_vec, vec_label), 0);
        ASSERT_EQ(tiered_index->backendIndex->indexSize(), 3);
        ASSERT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 1);
        ASSERT_EQ(tiered_index->frontendIndex->indexSize(), 1);
        ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);
        ASSERT_EQ(tiered_index->indexLabelCount(), 3);
        ASSERT_EQ(tiered_index->getDistanceFrom(vec_label, overwritten_vec), 0);
    }
}

TYPED_TEST(HNSWTieredIndexTest, bufferLimitAsync) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 500;
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .multi = TypeParam::isMulti(),
                         .M = 64};

    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Create tiered index with buffer limit set to 100.
    size_t flat_buffer_limit = 100;
    auto *tiered_index = this->CreateTieredHNSWIndex(
        hnsw_params, mock_thread_pool, DEFAULT_PENDING_SWAP_JOBS_THRESHOLD, flat_buffer_limit);
    auto allocator = tiered_index->getAllocator();
    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    // Create and insert vectors one by one async. At some point, buffer limit gets full and vectors
    // are inserted directly to HNSW.
    size_t per_label = TypeParam::isMulti() ? 5 : 1;
    size_t n_labels = n / per_label;
    std::srand(10); // create pseudo random generator with any arbitrary seed.
    // Run twice, at first run we insert non-existing labels, in the second run we overwrite them
    // (for single-value index only).
    for (auto overwrite : {0, 1}) {
        for (size_t i = 0; i < n; i++) {
            TEST_DATA_T vector[dim];
            for (size_t j = 0; j < dim; j++) {
                vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
            }
            EXPECT_EQ(tiered_index->addVector(vector, i % n_labels),
                      TypeParam::isMulti() ? 1 : 1 - overwrite);
            EXPECT_LE(tiered_index->frontendIndex->indexSize(), flat_buffer_limit);
        }
    }

    mock_thread_pool.thread_pool_join();
    EXPECT_EQ(tiered_index->backendIndex->indexSize(), 2 * n);
    EXPECT_EQ(tiered_index->indexLabelCount(), n_labels);
}

TYPED_TEST(HNSWTieredIndexTest, RangeSearch) {
    size_t dim = 4;
    size_t k = 11;
    size_t per_label = TypeParam::isMulti() ? 5 : 1;

    size_t n_labels = k * 3;
    size_t n = n_labels * per_label;

    auto edge_delta = (k - 0.8) * per_label;
    auto mid_delta = edge_delta / 2;
    // `range` for querying the "edges" of the index and get k results.
    double range = dim * edge_delta * edge_delta; // L2 distance.
    // `half_range` for querying a point in the "middle" of the index and get k results around it.
    double half_range = dim * mid_delta * mid_delta; // L2 distance.

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .epsilon = 3.0 * per_label,
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    size_t cur_memory_usage;

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();
    ASSERT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    auto hnsw_index = tiered_index->backendIndex;
    auto flat_index = tiered_index->frontendIndex;

    TEST_DATA_T query_0[dim];
    GenerateVector<TEST_DATA_T>(query_0, dim, 0);
    TEST_DATA_T query_1mid[dim];
    GenerateVector<TEST_DATA_T>(query_1mid, dim, n / 3);
    TEST_DATA_T query_2mid[dim];
    GenerateVector<TEST_DATA_T>(query_2mid, dim, n * 2 / 3);
    TEST_DATA_T query_n[dim];
    GenerateVector<TEST_DATA_T>(query_n, dim, n - 1);

    // Search for vectors when the index is empty.
    runRangeQueryTest(tiered_index, query_0, range, nullptr, 0);

    // Define the verification functions.
    auto ver_res_0 = [&](size_t id, double score, size_t index) {
        EXPECT_EQ(id, index);
        // The expected score is the distance to the first vector of `id` label.
        auto element = id * per_label;
        EXPECT_DOUBLE_EQ(score, dim * element * element);
    };

    auto ver_res_1mid_by_id = [&](size_t id, double score, size_t index) {
        size_t q_id = query_1mid[0] / per_label;
        size_t mod = query_1mid[0] - q_id * per_label;
        // In single value mode, `per_label` is always 1 and `mod` is always 0, so the following
        // branchings is simply `expected_score = abs(id - q_id)`.
        // In multi value mode, for ids higher than the query id, the score is the distance to the
        // first vector of `id` label, and for ids lower than the query id, the score is the
        // distance to the last vector of `id` label. `mod` is the distance to the first vector of
        // `q_id` label.
        double expected_score = 0;
        if (id > q_id) {
            expected_score = (id - q_id) * per_label - mod;
        } else if (id < q_id) {
            expected_score = (q_id - id) * per_label - (per_label - mod - 1);
        }
        expected_score = expected_score * expected_score * dim;
        EXPECT_DOUBLE_EQ(score, expected_score);
    };

    auto ver_res_2mid_by_id = [&](size_t id, double score, size_t index) {
        size_t q_id = query_2mid[0] / per_label;
        size_t mod = query_2mid[0] - q_id * per_label;
        // In single value mode, `per_label` is always 1 and `mod` is always 0, so the following
        // branchings is simply `expected_score = abs(id - q_id)`.
        // In multi value mode, for ids higher than the query id, the score is the distance to the
        // first vector of `id` label, and for ids lower than the query id, the score is the
        // distance to the last vector of `id` label. `mod` is the distance to the first vector of
        // `q_id` label.
        double expected_score = 0;
        if (id > q_id) {
            expected_score = (id - q_id) * per_label - mod;
        } else if (id < q_id) {
            expected_score = (q_id - id) * per_label - (per_label - mod - 1);
        }
        expected_score = expected_score * expected_score * dim;
        EXPECT_DOUBLE_EQ(score, expected_score);
    };

    auto ver_res_1mid_by_score = [&](size_t id, double score, size_t index) {
        size_t q_id = query_1mid[0] / per_label;
        EXPECT_EQ(std::abs(int(id - q_id)), (index + 1) / 2);
        ver_res_1mid_by_id(id, score, index);
    };

    auto ver_res_2mid_by_score = [&](size_t id, double score, size_t index) {
        size_t q_id = query_2mid[0] / per_label;
        EXPECT_EQ(std::abs(int(id - q_id)), (index + 1) / 2);
        ver_res_2mid_by_id(id, score, index);
    };

    auto ver_res_n = [&](size_t id, double score, size_t index) {
        EXPECT_EQ(id, n_labels - 1 - index);
        auto element = index * per_label;
        EXPECT_DOUBLE_EQ(score, dim * element * element);
    };

    // Insert n/2 vectors to the main index.
    for (size_t i = 0; i < (n + 1) / 2; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, i / per_label, i);
    }
    ASSERT_EQ(tiered_index->indexSize(), (n + 1) / 2);
    ASSERT_EQ(tiered_index->indexSize(), hnsw_index->indexSize());

    // Search for `range` with the flat index empty.
    cur_memory_usage = allocator->getAllocationSize();
    runRangeQueryTest(tiered_index, query_0, range, ver_res_0, k, BY_ID);
    runRangeQueryTest(tiered_index, query_0, range, ver_res_0, k, BY_SCORE);
    runRangeQueryTest(tiered_index, query_1mid, half_range, ver_res_1mid_by_score, k, BY_SCORE);
    runRangeQueryTest(tiered_index, query_1mid, half_range, ver_res_1mid_by_id, k, BY_ID);
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // Insert n/2 vectors to the flat index.
    for (size_t i = (n + 1) / 2; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, i / per_label, i);
    }
    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->indexSize(), hnsw_index->indexSize() + flat_index->indexSize());

    cur_memory_usage = allocator->getAllocationSize();
    // Search for `range` so all the vectors will be from the HNSW index.
    runRangeQueryTest(tiered_index, query_0, range, ver_res_0, k, BY_ID);
    runRangeQueryTest(tiered_index, query_0, range, ver_res_0, k, BY_SCORE);
    // Search for `range` so all the vectors will be from the flat index.
    runRangeQueryTest(tiered_index, query_n, range, ver_res_n, k, BY_SCORE);
    // Search for `range` so some of the results will be from the main and some from the flat index.
    runRangeQueryTest(tiered_index, query_1mid, half_range, ver_res_1mid_by_score, k, BY_SCORE);
    runRangeQueryTest(tiered_index, query_2mid, half_range, ver_res_2mid_by_score, k, BY_SCORE);
    runRangeQueryTest(tiered_index, query_1mid, half_range, ver_res_1mid_by_id, k, BY_ID);
    runRangeQueryTest(tiered_index, query_2mid, half_range, ver_res_2mid_by_id, k, BY_ID);
    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // Add some overlapping vectors to the main and flat index.
    // adding directly to the underlying indexes to avoid jobs logic.
    // The main index will have vectors 0 - 2n/3 and the flat index will have vectors n/3 - n
    for (size_t i = n / 3; i < n / 2; i++) {
        GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, i / per_label, i);
    }
    for (size_t i = n / 2; i < n * 2 / 3; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, i / per_label, i);
    }

    cur_memory_usage = allocator->getAllocationSize();
    // Search for `range` so all the vectors will be from the main index.
    runRangeQueryTest(tiered_index, query_0, range, ver_res_0, k, BY_ID);
    runRangeQueryTest(tiered_index, query_0, range, ver_res_0, k, BY_SCORE);
    // Search for `range` so all the vectors will be from the flat index.
    runRangeQueryTest(tiered_index, query_n, range, ver_res_n, k, BY_SCORE);
    // Search for `range` so some of the results will be from the main and some from the flat index.
    runRangeQueryTest(tiered_index, query_1mid, half_range, ver_res_1mid_by_score, k, BY_SCORE);
    runRangeQueryTest(tiered_index, query_2mid, half_range, ver_res_2mid_by_score, k, BY_SCORE);
    runRangeQueryTest(tiered_index, query_1mid, half_range, ver_res_1mid_by_id, k, BY_ID);
    runRangeQueryTest(tiered_index, query_2mid, half_range, ver_res_2mid_by_id, k, BY_ID);
    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // // // // // // // // // // // //
    // Check behavior upon timeout.  //
    // // // // // // // // // // // //

    VecSimQueryResult_List res;
    // Add a vector to the HNSW index so there will be a reason to query it.
    GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, n, n);

    // Set timeout callback to always return 1 (will fail while querying the flat buffer).
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out

    res = VecSimIndex_RangeQuery(tiered_index, query_0, range, nullptr, BY_ID);
    ASSERT_EQ(res.code, VecSim_QueryResult_TimedOut);
    VecSimQueryResult_Free(res);

    // Set timeout callback to return 1 after n checks (will fail while querying the HNSW index).
    // Brute-force index checks for timeout after each vector.
    size_t checks_in_flat = flat_index->indexSize();
    VecSimQueryParams qparams = {.timeoutCtx = &checks_in_flat};
    VecSim_SetTimeoutCallbackFunction([](void *ctx) {
        auto count = static_cast<size_t *>(ctx);
        if (*count == 0) {
            return 1;
        }
        (*count)--;
        return 0;
    });
    res = VecSimIndex_RangeQuery(tiered_index, query_0, range, &qparams, BY_SCORE);
    ASSERT_EQ(res.code, VecSim_QueryResult_TimedOut);
    VecSimQueryResult_Free(res);
    // Make sure we didn't get the timeout in the flat index.
    checks_in_flat = flat_index->indexSize(); // Reset the counter.
    res = VecSimIndex_RangeQuery(flat_index, query_0, range, &qparams, BY_SCORE);
    ASSERT_EQ(res.code, VecSim_QueryResult_OK);
    VecSimQueryResult_Free(res);

    // Check again with BY_ID.
    checks_in_flat = flat_index->indexSize(); // Reset the counter.
    res = VecSimIndex_RangeQuery(tiered_index, query_0, range, &qparams, BY_ID);
    ASSERT_EQ(res.code, VecSim_QueryResult_TimedOut);
    VecSimQueryResult_Free(res);
    // Make sure we didn't get the timeout in the flat index.
    checks_in_flat = flat_index->indexSize(); // Reset the counter.
    res = VecSimIndex_RangeQuery(flat_index, query_0, range, &qparams, BY_ID);
    ASSERT_EQ(res.code, VecSim_QueryResult_OK);
    VecSimQueryResult_Free(res);

    // Clean up.
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; });
}

TYPED_TEST(HNSWTieredIndexTest, parallelRangeSearch) {
    size_t dim = 4;
    size_t k = 11;
    size_t n = 1000;
    bool isMulti = TypeParam::isMulti();

    size_t per_label = isMulti ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = isMulti,
        .epsilon = double(dim * k * k),
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();
    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    std::atomic_int successful_searches(0);
    auto parallel_range_search = [](AsyncJob *job) {
        auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
        size_t k = search_job->k;
        size_t dim = search_job->dim;
        // The range that will get us k results.
        double range = dim * ((k - 0.5) / 2) * ((k - 0.5) / 2); // L2 distance.
        auto query = search_job->query;

        auto verify_res = [&](size_t id, double score, size_t res_index) {
            TEST_DATA_T element = *(TEST_DATA_T *)query;
            ASSERT_EQ(std::abs(id - element), (res_index + 1) / 2);
            ASSERT_EQ(score, dim * (id - element) * (id - element));
        };
        runRangeQueryTest(job->index, query, range, verify_res, k, BY_SCORE);
        (*search_job->successful_searches)++;
        delete job;
    };

    // Fill the job queue with insert and search jobs, while filling the flat index, before
    // initializing the thread pool.
    for (size_t i = 0; i < n; i++) {
        // Insert a vector to the flat index and add a job to insert it to the main index.
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i % n_labels, i);

        // Add a search job. Make sure the query element is between k and n_labels - k.
        auto query = (TEST_DATA_T *)allocator->allocate(dim * sizeof(TEST_DATA_T));
        GenerateVector<TEST_DATA_T>(query, dim, ((n - i) % (n_labels - (2 * k))) + k);
        auto search_job = new (allocator) tieredIndexMock::SearchJobMock(
            allocator, parallel_range_search, tiered_index, k, query, n, dim, &successful_searches);
        tiered_index->submitSingleJob(search_job);
    }

    EXPECT_EQ(tiered_index->indexSize(), n);
    EXPECT_EQ(tiered_index->indexLabelCount(), n_labels);
    EXPECT_EQ(tiered_index->labelToInsertJobs.size(), n_labels);
    for (auto &it : tiered_index->labelToInsertJobs) {
        EXPECT_EQ(it.second.size(), per_label);
    }
    EXPECT_EQ(tiered_index->frontendIndex->indexSize(), n);
    EXPECT_EQ(tiered_index->backendIndex->indexSize(), 0);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    // All the vectors are already in the tiered index, so we expect to find the expected
    // results from the get-go.
    mock_thread_pool.init_threads();
    mock_thread_pool.thread_pool_join();

    EXPECT_EQ(tiered_index->backendIndex->indexSize(), n);
    EXPECT_EQ(tiered_index->backendIndex->indexLabelCount(), n_labels);
    EXPECT_EQ(tiered_index->frontendIndex->indexSize(), 0);
    EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(HNSWTieredIndexTestBasic, preferAdHocOptimization) {
    size_t dim = 4;

    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
    };
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Create tiered index with buffer limit set to 0.
    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    auto hnsw = tiered_index->backendIndex;
    auto flat = tiered_index->frontendIndex;

    // Insert 5 vectors to the main index.
    for (size_t i = 0; i < 5; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnsw, dim, i, i);
    }
    // Sanity check. Should choose as HNSW.
    ASSERT_EQ(tiered_index->preferAdHocSearch(5, 5, true), hnsw->preferAdHocSearch(5, 5, true));

    // Insert 6 vectors to the flat index.
    for (size_t i = 0; i < 6; i++) {
        GenerateAndAddVector<TEST_DATA_T>(flat, dim, i, i);
    }
    // Sanity check. Should choose as flat as it has more vectors.
    ASSERT_EQ(tiered_index->preferAdHocSearch(5, 5, true), flat->preferAdHocSearch(5, 5, true));

    // Check for preference of tiered with subset (10) smaller than the tiered index size (11),
    // but larger than any of the underlying indexes.
    ASSERT_NO_THROW(tiered_index->preferAdHocSearch(10, 5, false));
}

TYPED_TEST(HNSWTieredIndexTestBasic, runGCAPI) {
    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2};
    VecSimParams hnsw_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredHNSWIndex(hnsw_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Test initialization of the pendingSwapJobsThreshold value.
    ASSERT_EQ(tiered_index->pendingSwapJobsThreshold, DEFAULT_PENDING_SWAP_JOBS_THRESHOLD);

    // Insert three block of vectors directly to HNSW.
    size_t n = DEFAULT_PENDING_SWAP_JOBS_THRESHOLD * 3;
    std::srand(10); // create pseudo random generator with any arbitrary seed.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        for (size_t j = 0; j < dim; j++) {
            vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
        }
        VecSimIndex_AddVector(tiered_index->backendIndex, vector, i);
    }

    // Delete all the vectors and wait for the thread pool to finish running the repair jobs.
    for (size_t i = 0; i < n; i++) {
        tiered_index->deleteVector(i);
    }
    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    mock_thread_pool.thread_pool_join();

    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->backendIndex->indexSize(), n);
    ASSERT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), n);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Run the GC API call, expect that we will clean the defined threshold number of vectors
    // each time we call the GC.
    while (tiered_index->indexSize() > 0) {
        size_t cur_size = tiered_index->indexSize();
        VecSimTieredIndex_GC(tiered_index);
        ASSERT_EQ(tiered_index->indexSize(), cur_size - DEFAULT_PENDING_SWAP_JOBS_THRESHOLD);
    }
}
