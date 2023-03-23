#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"

#include "test_utils.h"

#include <thread>

using namespace tiered_index_mock;

template <typename index_type_t>
class HNSWTieredIndexTest : public ::testing::Test {};

TYPED_TEST_SUITE(HNSWTieredIndexTest, DataTypeSet);

TYPED_TEST(HNSWTieredIndexTest, CreateIndexInstance) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Create TieredHNSW index instance with a mock queue.
    for (auto is_multi : {true, false}) {
        HNSWParams params = {.type = TypeParam::get_index_type(),
                             .dim = 4,
                             .metric = VecSimMetric_L2,
                             .multi = is_multi};
        auto jobQ = JobQueue();
        auto jobQueueCtx = IndexExtCtx();
        size_t memory_ctx = 0;
        TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                           .jobQueueCtx = &jobQueueCtx,
                                           .submitCb = submit_callback,
                                           .memoryCtx = &memory_ctx,
                                           .UpdateMemCb = update_mem_callback};
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
        // Set the created tiered index in the index external context.
        jobQueueCtx.index_strong_ref.reset(tiered_index);

        // Add a vector to the flat index.
        TEST_DATA_T vector[tiered_index->index->getDim()];
        GenerateVector<TEST_DATA_T>(vector, tiered_index->index->getDim());
        labelType vector_label = 1;
        VecSimIndex_AddVector(tiered_index->flatBuffer, vector, vector_label);

        // Create a mock job that inserts some vector into the HNSW index.
        auto insert_to_index = [](AsyncJob *job) {
            auto *my_insert_job = reinterpret_cast<HNSWInsertJob *>(job);
            auto my_index =
                reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(my_insert_job->index);

            // Move the vector from the temp flat index into the HNSW index.
            // Note that we access the vector via its internal id since in index of type MULTI,
            // this is the only way to do so (knowing the label is not enough...)
            VecSimIndex_AddVector(my_index->index,
                                  my_index->flatBuffer->getDataByInternalId(my_insert_job->id),
                                  my_insert_job->label);
            // TODO: enable deleting vectors by internal id for the case of moving a single vector
            //  from the flat buffer in MULTI.
            VecSimIndex_DeleteVector(my_index->flatBuffer, my_insert_job->label);
            auto it = my_index->labelToInsertJobs.at(my_insert_job->label).begin();
            ASSERT_EQ(job, *it); // Assert pointers equation
            // Here we update labelToInsertJobs mapping, as we except that for every insert job
            // there will be a corresponding item in the map.
            my_index->labelToInsertJobs.at(my_insert_job->label).erase(it);
            my_index->UpdateIndexMemory(my_index->memoryCtx,
                                        my_index->getAllocator()->getAllocationSize());
        };

        HNSWInsertJob job(tiered_index->allocator, vector_label, 0, insert_to_index, tiered_index);
        auto jobs_vec = vecsim_stl::vector<HNSWInsertJob *>(1, &job, allocator);
        tiered_index->labelToInsertJobs.insert({vector_label, jobs_vec});

        // Wrap this job with an array and submit the jobs to the queue.
        // TODO: in the future this should be part of the tiered index "add_vector" flow, and
        //  we can replace this to avoid the breaking of the abstraction.
        tiered_index->submitSingleJob((AsyncJob *)&job);
        ASSERT_EQ(jobQ.size(), 1);

        // Execute the job from the queue and validate that the index was updated properly.
        reinterpret_cast<AsyncJob *>(jobQ.front().job)->Execute(jobQ.front().job);
        ASSERT_EQ(tiered_index->indexSize(), 1);
        ASSERT_EQ(tiered_index->getDistanceFrom(1, vector), 0);
        ASSERT_EQ(memory_ctx, tiered_index->getAllocator()->getAllocationSize());
        ASSERT_EQ(tiered_index->flatBuffer->indexSize(), 0);
        ASSERT_EQ(tiered_index->labelToInsertJobs.at(vector_label).size(), 0);
    }
}

template <typename index_type_t>
class HNSWTieredIndexTest2 : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        auto tiered_index = reinterpret_cast<TieredHNSWIndex<data_t, dist_t> *>(index);
        return tiered_index->getHNSWIndex();
    }
};

TYPED_TEST_SUITE(HNSWTieredIndexTest2, DataTypeSetExtended);

TYPED_TEST(HNSWTieredIndexTest2, testSizeEstimation) {
    size_t dim = 128;
    size_t n = 1000;
    size_t M = 32;
    size_t bs = DEFAULT_BLOCK_SIZE;
    bool isMulti = TypeParam::get_index_label_type();

    HNSWParams hnsw_params = {.type = TypeParam::get_index_type(),
                            .dim = dim,
                            .metric = VecSimMetric_L2,
                            .multi = isMulti,
                            .initialCapacity = n, 
                            .M = M
                            };
    auto jobQ = JobQueue();
    auto index_ctx = IndexExtCtx();
    size_t memory_ctx = 0;
    TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                        .jobQueueCtx = &index_ctx,
                                        .submitCb = submit_callback,
                                        .memoryCtx = &memory_ctx,
                                        .UpdateMemCb = update_mem_callback};
    TieredHNSWParams tiered_hnsw_params = {.hnswParams = hnsw_params, .tieredParams = tiered_params};
    VecSimParams params = CreateParams(tiered_hnsw_params);
    auto *index = VecSimIndex_New(&params);

    size_t initial_size_estimation = VecSimIndex_EstimateInitialSize(&params);

    // labels_lookup hash table has additional memory, since STL implementation chooses "an
    // appropriate prime number" higher than n as the number of allocated buckets (for n=1000, 1031
    // buckets are created)
    auto hnsw_index = this->CastToHNSW(index);
    if(isMulti == false) {
        auto hnsw = reinterpret_cast<HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T> *>(hnsw_index);
        initial_size_estimation += (hnsw->label_lookup_.bucket_count() - n) * sizeof(size_t);
    } else { // if its a multi value index cast to HNSW_Multi
        auto hnsw = reinterpret_cast<HNSWIndex_Multi<TEST_DATA_T, TEST_DIST_T> *>(hnsw_index);
        initial_size_estimation += (hnsw->label_lookup_.bucket_count() - n) * sizeof(size_t);
    }

    ASSERT_EQ(memory_ctx, index->getAllocationSize());
    ASSERT_EQ(initial_size_estimation, memory_ctx);

    // Add vectors up to initial capacity (initial capacity < block size).
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
        jobQ.front().job->Execute(jobQ.front().job);
        jobQ.pop();
    }

    // Estimate memory delta for filling up the first block and adding another block.
    size_t estimation = EstimateElementSize(tiered_hnsw_params) * (bs % n + bs);
    
    size_t memory_before = memory_ctx;

    // Note we are adding vectors with ascending values. This causes the numbers of
    // double connections, which are not taking into account in EstimateElementSize,
    // to be zero
    for (size_t i = 0; i < bs % n + bs; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i + n, i + n);
        jobQ.front().job->Execute(jobQ.front().job);
        jobQ.pop();
    }

    size_t delta = memory_ctx - memory_before;

    // Flat index should be empty, hence the index size includes only hnsw size.
    ASSERT_EQ(index->indexSize(), hnsw_index->indexSize());
    // We added 2 * bs vectors
    ASSERT_EQ(index->indexSize(), 2 * bs);
    // Which is the current maximum capacity.
    ASSERT_EQ(index->indexSize(), index->indexCapacity());
    ASSERT_EQ(index->indexSize(), hnsw_index->indexCapacity());

    ASSERT_GE(estimation * 1.02, delta);
    ASSERT_LE(estimation * 0.99, delta);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTieredIndexTest, addVector) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    for (auto is_multi : {false, true}) {
        HNSWParams params = {.type = TypeParam::get_index_type(),
                             .dim = dim,
                             .metric = VecSimMetric_L2,
                             .multi = is_multi};
        auto jobQ = JobQueue();
        auto index_ctx = IndexExtCtx();
        size_t memory_ctx = 0;
        TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                           .jobQueueCtx = &index_ctx,
                                           .submitCb = submit_callback,
                                           .memoryCtx = &memory_ctx,
                                           .UpdateMemCb = update_mem_callback};
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
        // Set the created tiered index in the index external context.
        index_ctx.index_strong_ref.reset(tiered_index);

        BFParams bf_params = {.type = TypeParam::get_index_type(),
                              .dim = dim,
                              .metric = VecSimMetric_L2,
                              .multi = is_multi};


        // Create a vector and add it to the tiered index.
        labelType vec_label = 1;
        TEST_DATA_T vector[dim];
        GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
        VecSimIndex_AddVector(tiered_index, vector, vec_label);
        // Validate that the vector was inserted to the flat buffer properly.
        ASSERT_EQ(tiered_index->indexSize(), 1);
        ASSERT_EQ(tiered_index->index->indexSize(), 0);
        ASSERT_EQ(tiered_index->flatBuffer->indexSize(), 1);
        ASSERT_EQ(tiered_index->flatBuffer->indexCapacity(), DEFAULT_BLOCK_SIZE);
        ASSERT_EQ(tiered_index->indexCapacity(), DEFAULT_BLOCK_SIZE);
        ASSERT_EQ(tiered_index->flatBuffer->getDistanceFrom(vec_label, vector), 0);
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
            sizeof(vecsim_stl::unordered_map<labelType,
                                             vecsim_stl::vector<HNSWInsertJob *>>::value_type) +
            sizeof(void *) + sizeof(size_t);
        // Account for the inner buffer of the std::vector<HNSWInsertJob *> in the map.
        expected_mem += sizeof(void *) + sizeof(size_t);
        // Account for the insert job that was created.
        expected_mem += sizeof(HNSWInsertJob) + sizeof(size_t);
        ASSERT_GE(expected_mem * 1.02, memory_ctx);
        ASSERT_LE(expected_mem, memory_ctx);

        if (is_multi) {
            // Add another vector under the same label (create another insert job)
            VecSimIndex_AddVector(tiered_index, vector, vec_label);
            ASSERT_EQ(tiered_index->indexSize(), 2);
            ASSERT_EQ(tiered_index->indexLabelCount(), 1);
            ASSERT_EQ(tiered_index->index->indexSize(), 0);
            ASSERT_EQ(tiered_index->flatBuffer->indexSize(), 2);
            // Validate that the second job was created properly
            ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label).size(), 2);
            ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label)[1]->label, vec_label);
            ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label)[1]->id, 1);
        }
    }
}

TYPED_TEST(HNSWTieredIndexTest, manageIndexOwnership) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Create TieredHNSW index instance with a mock queue.
    for (auto is_multi : {true, false}) {
        size_t dim = 4;
        HNSWParams params = {.type = TypeParam::get_index_type(),
                             .dim = dim,
                             .metric = VecSimMetric_L2,
                             .multi = is_multi};
        auto jobQ = JobQueue();
        auto *index_ctx = new IndexExtCtx();
        size_t memory_ctx = 0;
        TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                           .jobQueueCtx = index_ctx,
                                           .submitCb = submit_callback,
                                           .memoryCtx = &memory_ctx,
                                           .UpdateMemCb = update_mem_callback};
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
        // Set the created tiered index in the index external context.
        index_ctx->index_strong_ref.reset(tiered_index);
        EXPECT_EQ(index_ctx->index_strong_ref.use_count(), 1);
        size_t initial_mem = memory_ctx;

        // Create a dummy job callback that insert one vector to the underline HNSW index.
        auto dummy_job = [](AsyncJob *job) {
            auto *my_index =
                reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(job->index);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            size_t dim = 4;
            TEST_DATA_T vector[dim];
            GenerateVector<TEST_DATA_T>(vector, dim);
            if (my_index->index->indexCapacity() == my_index->index->indexSize()) {
                my_index->index->increaseCapacity();
            }
            my_index->index->addVector(vector, my_index->index->indexSize());
        };

        AsyncJob job(tiered_index->allocator, HNSW_INSERT_VECTOR_JOB, dummy_job, tiered_index);

        // Wrap this job with an array and submit the jobs to the queue.
        tiered_index->submitSingleJob((AsyncJob *)&job);
        tiered_index->submitSingleJob((AsyncJob *)&job);
        ASSERT_EQ(jobQ.size(), 2);

        // Execute the job from the queue asynchronously, delete the index in the meantime.
        auto run_fn = [&jobQ]() {
            // Create a temporary strong reference of the index from the weak reference that the
            // job holds, to ensure that the index is not deleted while the job is running.
            if (auto temp_ref = jobQ.front().index_weak_ref.lock()) {
                // At this point we wish to validate that we have both the index strong ref (stored
                // in index_ctx) and the weak ref owned by the job (that we currently promoted).
                EXPECT_EQ(jobQ.front().index_weak_ref.use_count(), 2);

                jobQ.front().job->Execute(jobQ.front().job);
            }
            jobQ.pop();
        };
        std::thread t1(run_fn);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // Delete the index while the job is still running, to ensure that the weak ref protects
        // the index.
        delete index_ctx;
        EXPECT_EQ(jobQ.front().index_weak_ref.use_count(), 1);
        t1.join();
        // Expect that the first job will succeed.
        ASSERT_GE(memory_ctx, initial_mem);
        size_t cur_mem = memory_ctx;

        // The second job should not run, since the weak reference is not supposed to become a
        // strong references now.
        ASSERT_EQ(jobQ.front().index_weak_ref.use_count(), 0);
        std::thread t2(run_fn);
        t2.join();
        ASSERT_EQ(memory_ctx, cur_mem);
    }
}

TYPED_TEST(HNSWTieredIndexTest, insertJob) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    for (auto is_multi : {false, true}) {
        HNSWParams params = {.type = TypeParam::get_index_type(),
                             .dim = dim,
                             .metric = VecSimMetric_L2,
                             .multi = is_multi};
        auto jobQ = JobQueue();
        auto index_ctx = IndexExtCtx();
        size_t memory_ctx = 0;
        TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                           .jobQueueCtx = &index_ctx,
                                           .submitCb = submit_callback,
                                           .memoryCtx = &memory_ctx,
                                           .UpdateMemCb = update_mem_callback};
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
        index_ctx.index_strong_ref.reset(tiered_index);

        // Create a vector and add it to the tiered index.
        labelType vec_label = 1;
        TEST_DATA_T vector[dim];
        GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
        VecSimIndex_AddVector(tiered_index, vector, vec_label);
        ASSERT_EQ(tiered_index->indexSize(), 1);
        ASSERT_EQ(tiered_index->flatBuffer->indexSize(), 1);

        // Execute the insert job manually (in a synchronous manner).
        ASSERT_EQ(jobQ.size(), 1);
        auto *insertion_job = reinterpret_cast<HNSWInsertJob *>(jobQ.front().job);
        ASSERT_EQ(insertion_job->label, vec_label);
        ASSERT_EQ(insertion_job->id, 0);
        ASSERT_EQ(insertion_job->jobType, HNSW_INSERT_VECTOR_JOB);

        insertion_job->Execute(insertion_job);
        ASSERT_EQ(tiered_index->indexSize(), 1);
        ASSERT_EQ(tiered_index->flatBuffer->indexSize(), 0);
        ASSERT_EQ(tiered_index->index->indexSize(), 1);
        // HNSW index should have allocated a single block, while flat index should remove the
        // block.
        ASSERT_EQ(tiered_index->index->indexCapacity(), DEFAULT_BLOCK_SIZE);
        ASSERT_EQ(tiered_index->indexCapacity(), DEFAULT_BLOCK_SIZE);
        ASSERT_EQ(tiered_index->flatBuffer->indexCapacity(), 0);
        ASSERT_EQ(tiered_index->index->getDistanceFrom(vec_label, vector), 0);
        // After the execution, the job should be removed from the labelToInsertJobs mapping.
        ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    }
}

TYPED_TEST(HNSWTieredIndexTest, insertJobAsync) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 5000;
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};
    auto jobQ = JobQueue();
    auto index_ctx = IndexExtCtx();

    size_t memory_ctx = 0;
    TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                       .jobQueueCtx = &index_ctx,
                                       .submitCb = submit_callback,
                                       .memoryCtx = &memory_ctx,
                                       .UpdateMemCb = update_mem_callback};
    TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
    auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
        HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
    index_ctx.index_strong_ref.reset(tiered_index);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    bool run_thread = true;
    for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
        thread_pool.emplace_back(thread_main_loop, std::ref(jobQ), std::ref(run_thread));
    }

    // Insert vectors
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i, i);
    }
    ASSERT_GE(tiered_index->labelToInsertJobs.size(), 0);

    // Check every 10 ms if queue is empty, and if so, terminate the threads loop.
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::unique_lock<std::mutex> lock(queue_guard);
        if (jobQ.empty()) {
            run_thread = false;
            queue_cond.notify_all();
            break;
        }
    }
    for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
        thread_pool[i].join();
    }
    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->index->indexSize(), n);
    ASSERT_EQ(tiered_index->flatBuffer->indexSize(), 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    ASSERT_EQ(jobQ.size(), 0);
    // Verify that the vectors were inserted to HNSW as expected
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T expected_vector[dim];
        GenerateVector<TEST_DATA_T>(expected_vector, dim, i);
        ASSERT_EQ(tiered_index->index->getDistanceFrom(i, expected_vector), 0);
    }

    thread_pool.clear();
}

TYPED_TEST(HNSWTieredIndexTest, insertJobAsyncMulti) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 5000;
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = true};
    size_t per_label = 5;
    auto jobQ = JobQueue();
    auto index_ctx = IndexExtCtx();
    size_t memory_ctx = 0;
    TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                       .jobQueueCtx = &index_ctx,
                                       .submitCb = submit_callback,
                                       .memoryCtx = &memory_ctx,
                                       .UpdateMemCb = update_mem_callback};
    TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
    auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
        HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
    index_ctx.index_strong_ref.reset(tiered_index);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    bool run_thread = true;
    for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
        thread_pool.emplace_back(thread_main_loop, std::ref(jobQ), std::ref(run_thread));
    }

    // Create and insert vectors, store them in this continuous array.
    TEST_DATA_T vectors[n * dim];
    for (size_t i = 0; i < n / per_label; i++) {
        for (size_t j = 0; j < per_label; j++) {
            GenerateVector<TEST_DATA_T>(vectors + i * dim * per_label + j * dim, dim,
                                        i * per_label + j);
            tiered_index->addVector(vectors + i * dim * per_label + j * dim, i);
        }
    }
    ASSERT_GE(tiered_index->labelToInsertJobs.size(), 0);

    // Check every 10 ms if queue is empty, and if so, terminate the threads loop.
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::unique_lock<std::mutex> lock(queue_guard);
        if (jobQ.empty()) {
            run_thread = false;
            queue_cond.notify_all();
            break;
        }
    }
    for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
        thread_pool[i].join();
    }
    EXPECT_EQ(tiered_index->index->indexSize(), n);
    EXPECT_EQ(tiered_index->indexLabelCount(), n / per_label);
    EXPECT_EQ(tiered_index->flatBuffer->indexSize(), 0);
    EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0);
    EXPECT_EQ(jobQ.size(), 0);
    // Verify that the vectors were inserted to HNSW as expected
    for (size_t i = 0; i < n / per_label; i++) {
        for (size_t j = 0; j < per_label; j++) {
            // The distance from every vector that is stored under the label i should be zero
            EXPECT_EQ(
                tiered_index->index->getDistanceFrom(i, vectors + i * per_label * dim + j * dim),
                0);
        }
    }

    // Cleanup.
    thread_pool.clear();
}

TYPED_TEST(HNSWTieredIndexTest, KNNSearch) {
    size_t dim = 4;
    size_t k = 10;

    size_t n = k * 3;

    // Create TieredHNSW index instance with a mock queue.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
    };
    auto jobQ = JobQueue();
    auto index_ctx = IndexExtCtx();
    size_t memory_ctx = 0;
    TieredIndexParams tiered_params = {
        .jobQueue = &jobQ,
        .jobQueueCtx = &index_ctx,
        .submitCb = submit_callback,
        .memoryCtx = &memory_ctx,
        .UpdateMemCb = update_mem_callback,
    };
    TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
    auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
        HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
    // Set the created tiered index in the index external context.
    index_ctx.index_strong_ref.reset(tiered_index);
    EXPECT_EQ(index_ctx.index_strong_ref.use_count(), 1);

    auto hnsw_index = tiered_index->index;
    auto flat_index = tiered_index->flatBuffer;

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
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);

    // Insert n/2 vectors to the flat index.
    for (size_t i = n / 2; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, i, i);
    }
    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->indexSize(), hnsw_index->indexSize() + flat_index->indexSize());

    // Search for k vectors so all the vectors will be from the flat index.
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    // Search for k vectors so all the vectors will be from the main index.
    runTopKSearchTest(tiered_index, query_n, k, ver_res_n);
    // Search for k so some of the results will be from the main and some from the flat index.
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    runTopKSearchTest(tiered_index, query_2mid, k, ver_res_2mid);

    // Add some overlapping vectors to the main and flat index.
    // adding directly to the underlying indexes to avoid jobs logic.
    // The main index will have vectors 0 - 2n/3 and the flat index will have vectors n/3 - n
    for (size_t i = n / 3; i < n / 2; i++) {
        GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, i, i);
    }
    for (size_t i = n / 2; i < n * 2 / 3; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, i, i);
    }

    // Search for k vectors so all the vectors will be from the main index.
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    // Search for k vectors so all the vectors will be from the flat index.
    runTopKSearchTest(tiered_index, query_n, k, ver_res_n);
    // Search for k so some of the results will be from the main and some from the flat index.
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    runTopKSearchTest(tiered_index, query_2mid, k, ver_res_2mid);

    // More edge cases:

    // Search for more vectors than the index size.
    k = n + 1;
    runTopKSearchTest(tiered_index, query_0, k, n, ver_res_0);
    runTopKSearchTest(tiered_index, query_n, k, n, ver_res_n);

    // Search for less vectors than the index size, but more than the flat and main index sizes.
    k = n * 5 / 6;
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_n, k, ver_res_n);

    // Search for more vectors than the main index size, but less than the flat index size.
    for (size_t i = n / 2; i < n * 2 / 3; i++) {
        VecSimIndex_DeleteVector(hnsw_index, i);
    }
    ASSERT_EQ(flat_index->indexSize(), n * 2 / 3);
    ASSERT_EQ(hnsw_index->indexSize(), n / 2);
    k = n * 2 / 3;
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_n, k, ver_res_n);
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    runTopKSearchTest(tiered_index, query_2mid, k, ver_res_2mid);

    // Search for more vectors than the flat index size, but less than the main index size.
    for (size_t i = n / 2; i < n; i++) {
        VecSimIndex_DeleteVector(flat_index, i);
    }
    ASSERT_EQ(flat_index->indexSize(), n / 6);
    ASSERT_EQ(hnsw_index->indexSize(), n / 2);
    k = n / 4;
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);

    // Search for vectors when the flat index is not empty but the main index is empty.
    for (size_t i = 0; i < n * 2 / 3; i++) {
        VecSimIndex_DeleteVector(hnsw_index, i);
        GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, i, i);
    }
    ASSERT_EQ(flat_index->indexSize(), n * 2 / 3);
    ASSERT_EQ(hnsw_index->indexSize(), 0);
    k = n / 3;
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);

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

    // Create TieredHNSW index instance with a mock queue.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    for (auto isMulti : {false, true}) {
        HNSWParams params = {
            .type = TypeParam::get_index_type(),
            .dim = dim,
            .metric = VecSimMetric_L2,
            .multi = isMulti,
            .efRuntime = 100,
        };
        auto jobQ = JobQueue();
        auto index_ctx = IndexExtCtx();
        size_t memory_ctx = 0;
        TieredIndexParams tiered_params = {
            .jobQueue = &jobQ,
            .jobQueueCtx = &index_ctx,
            .submitCb = submit_callback,
            .memoryCtx = &memory_ctx,
            .UpdateMemCb = update_mem_callback,
        };
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
        // Set the created tiered index in the index external context.
        index_ctx.index_strong_ref.reset(tiered_index);
        EXPECT_EQ(index_ctx.index_strong_ref.use_count(), 1);

        std::atomic_int successful_searches(0);
        auto parallel_knn_search = [](AsyncJob *job) {
            auto *search_job = reinterpret_cast<SearchJobMock *>(job);
            size_t k = search_job->k;
            size_t dim = search_job->dim;
            auto query = search_job->query;

            auto verify_res = [&](size_t id, double score, size_t res_index) {
                TEST_DATA_T element = *(TEST_DATA_T *)query;
                ASSERT_EQ(std::abs(id - element), (res_index + 1) / 2);
                ASSERT_EQ(score, dim * (id - element) * (id - element));
            };
            runTopKSearchTest(job->index, query, k, verify_res);
            search_job->successful_searches++;

            delete search_job;
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
            auto search_job =
                new (allocator) SearchJobMock(allocator, parallel_knn_search, tiered_index, query,
                                              k, n, dim, successful_searches);
            tiered_index->submitSingleJob(search_job);
        }

        EXPECT_EQ(tiered_index->indexSize(), n) << (isMulti ? "multi" : "single");
        EXPECT_EQ(tiered_index->indexLabelCount(), n_labels) << (isMulti ? "multi" : "single");
        EXPECT_EQ(tiered_index->labelToInsertJobs.size(), n_labels)
            << (isMulti ? "multi" : "single");
        for (auto &it : tiered_index->labelToInsertJobs) {
            EXPECT_EQ(it.second.size(), per_label) << (isMulti ? "multi" : "single");
        }
        EXPECT_EQ(tiered_index->flatBuffer->indexSize(), n) << (isMulti ? "multi" : "single");
        EXPECT_EQ(tiered_index->index->indexSize(), 0) << (isMulti ? "multi" : "single");

        // Launch the BG threads loop that takes jobs from the queue and executes them.
        // All the vectors are already in the tiered index, so we expect to find the expected
        // results from the get-go.
        bool run_thread = true;
        for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
            thread_pool.emplace_back(thread_main_loop, std::ref(jobQ), std::ref(run_thread));
        }

        // Check every 10 ms if queue is empty, and if so, terminate the threads loop.
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::unique_lock<std::mutex> lock(queue_guard);
            if (jobQ.empty()) {
                run_thread = false;
                queue_cond.notify_all();
                break;
            }
        }
        for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
            thread_pool[i].join();
        }

        EXPECT_EQ(tiered_index->index->indexSize(), n) << (isMulti ? "multi" : "single");
        EXPECT_EQ(tiered_index->index->indexLabelCount(), n_labels)
            << (isMulti ? "multi" : "single");
        EXPECT_EQ(tiered_index->flatBuffer->indexSize(), 0) << (isMulti ? "multi" : "single");
        EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0) << (isMulti ? "multi" : "single");
        EXPECT_EQ(successful_searches, n) << (isMulti ? "multi" : "single");
        EXPECT_EQ(jobQ.size(), 0) << (isMulti ? "multi" : "single");

        // Cleanup.
        thread_pool.clear();
    }
}

TYPED_TEST(HNSWTieredIndexTest, parallelInsertSearch) {
    size_t dim = 4;
    size_t k = 10;
    size_t n = 3000;

    size_t block_size = n / 100;

    // Create TieredHNSW index instance with a mock queue.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    for (auto isMulti : {false, true}) {
        size_t n_labels = isMulti ? n / 25 : n;
        HNSWParams params = {
            .type = TypeParam::get_index_type(),
            .dim = dim,
            .metric = VecSimMetric_L2,
            .multi = isMulti,
            .blockSize = block_size,
        };
        auto jobQ = JobQueue();
        auto index_ctx = IndexExtCtx();
        size_t memory_ctx = 0;
        TieredIndexParams tiered_params = {
            .jobQueue = &jobQ,
            .jobQueueCtx = &index_ctx,
            .submitCb = submit_callback,
            .memoryCtx = &memory_ctx,
            .UpdateMemCb = update_mem_callback,
        };
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
        // Set the created tiered index in the index external context.
        index_ctx.index_strong_ref.reset(tiered_index);
        EXPECT_EQ(index_ctx.index_strong_ref.use_count(), 1);

        // Launch the BG threads loop that takes jobs from the queue and executes them.
        // Save the number fo tasks done by thread i in the i-th entry.
        std::vector<size_t> completed_tasks(THREAD_POOL_SIZE, 0);
        bool run_thread = true;
        for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
            thread_pool.emplace_back(thread_main_loop, std::ref(jobQ), std::ref(run_thread));
        }
        std::atomic_int successful_searches(0);

        auto parallel_knn_search = [](AsyncJob *job) {
            auto *search_job = reinterpret_cast<SearchJobMock *>(job);
            size_t k = search_job->k;
            auto query = search_job->query;
            // In this test we don't care about the results, just that the search doesn't crash
            // and returns the correct number of valid results.
            auto verify_res = [&](size_t id, double score, size_t res_index) {};
            runTopKSearchTest(job->index, query, k, verify_res);
            search_job->successful_searches++;

            delete search_job;
        };

        // Insert vectors in parallel to search.
        for (size_t i = 0; i < n; i++) {
            GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i % n_labels, i);
            auto query = (TEST_DATA_T *)allocator->allocate(dim * sizeof(TEST_DATA_T));
            GenerateVector<TEST_DATA_T>(query, dim, (TEST_DATA_T)n / 4 + (i % 1000) * M_PI);
            auto search_job =
                new (allocator) SearchJobMock(allocator, parallel_knn_search, tiered_index, query,
                                              k, n, dim, successful_searches);
            tiered_index->submitSingleJob(search_job);
        }

        // Check every 10 ms if queue is empty, and if so, terminate the threads loop.
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::unique_lock<std::mutex> lock(queue_guard);
            if (jobQ.empty()) {
                run_thread = false;
                queue_cond.notify_all();
                break;
            }
        }
        for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
            thread_pool[i].join();
        }
        EXPECT_EQ(successful_searches, n) << (isMulti ? "multi" : "single");
        EXPECT_EQ(tiered_index->index->indexSize(), n) << (isMulti ? "multi" : "single");
        EXPECT_EQ(tiered_index->index->indexLabelCount(), n_labels)
            << (isMulti ? "multi" : "single");
        EXPECT_EQ(tiered_index->flatBuffer->indexSize(), 0) << (isMulti ? "multi" : "single");
        EXPECT_EQ(tiered_index->labelToInsertJobs.size(), 0) << (isMulti ? "multi" : "single");
        EXPECT_EQ(jobQ.size(), 0);

        // Cleanup.
        thread_pool.clear();
    }
}

TYPED_TEST(HNSWTieredIndexTest, MergeMulti) {
    size_t dim = 4;

    // Create TieredHNSW index instance with a mock queue.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    HNSWParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = true,
    };
    auto jobQ = JobQueue();
    auto index_ctx = IndexExtCtx();
    size_t memory_ctx = 0;
    TieredIndexParams tiered_params = {
        .jobQueue = &jobQ,
        .jobQueueCtx = &index_ctx,
        .submitCb = submit_callback,
        .memoryCtx = &memory_ctx,
        .UpdateMemCb = update_mem_callback,
    };
    TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
    auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
        HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
    // Set the created tiered index in the index external context.
    index_ctx.index_strong_ref.reset(tiered_index);
    EXPECT_EQ(index_ctx.index_strong_ref.use_count(), 1);

    auto hnsw_index = tiered_index->index;
    auto flat_index = tiered_index->flatBuffer;

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
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t dim = 4;

    for (auto isMulti : {false, true}) {
        HNSWParams params = {.type = TypeParam::get_index_type(),
                             .dim = dim,
                             .metric = VecSimMetric_L2,
                             .multi = isMulti};
        auto jobQ = JobQueue();
        size_t memory_ctx = 0;
        auto index_ctx = IndexExtCtx();
        TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                           .jobQueueCtx = &index_ctx,
                                           .submitCb = submit_callback,
                                           .memoryCtx = &memory_ctx,
                                           .UpdateMemCb = update_mem_callback};
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));

        // Delete a non existing label.
        ASSERT_EQ(tiered_index->deleteLabelFromHNSW(0), 0);
        ASSERT_EQ(jobQ.size(), 0);

        // Insert one vector to HNSW and then delete it (it should have no neighbors to repair).
        GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, 0);
        ASSERT_EQ(tiered_index->deleteLabelFromHNSW(0), 1);
        ASSERT_EQ(jobQ.size(), 0);

        // Add another vector and remove it. Since the other vector in the index has marked deleted,
        // this vector should have no neighbors, and again, no neighbors to repair.
        GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, 1, 1);
        ASSERT_EQ(tiered_index->deleteLabelFromHNSW(1), 1);
        ASSERT_EQ(jobQ.size(), 0);

        // Add two vectors and delete one, expect that at least one repair job will be created.
        GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, 2, 2);
        GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, 3, 3);
        ASSERT_EQ(tiered_index->deleteLabelFromHNSW(3), 1);

        // The first job should be a repair job of the first inserted non-deleted node id (2)
        // in level 0.
        ASSERT_EQ(jobQ.size(), 1);
        ASSERT_EQ(jobQ.front().job->jobType, HNSW_REPAIR_NODE_CONNECTIONS_JOB);
        ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->node_id, 2);
        ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->level, 0);
        ASSERT_EQ(tiered_index->idToRepairJobs.size(), 1);
        ASSERT_GE(tiered_index->idToRepairJobs.at(2).size(), 1);
        ASSERT_EQ(tiered_index->idToRepairJobs.at(2)[0]->associatedSwapJobs.size(), 1);
        ASSERT_EQ(tiered_index->idToRepairJobs.at(2)[0]->associatedSwapJobs[0]->deleted_id, 3);

        ASSERT_EQ(tiered_index->indexSize(), 4);
        ASSERT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 3);
        ASSERT_EQ(tiered_index->idToSwapJob.size(), 3);
        jobQ.pop();

        delete tiered_index;
    }
}

TYPED_TEST(HNSWTieredIndexTest, deleteFromHNSWMulti) {
    // Create TieredHNSW index instance with a mock queue.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t dim = 4;

    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = true};
    auto jobQ = JobQueue();
    size_t memory_ctx = 0;
    auto index_ctx = IndexExtCtx();
    TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                       .jobQueueCtx = &index_ctx,
                                       .submitCb = submit_callback,
                                       .memoryCtx = &memory_ctx,
                                       .UpdateMemCb = update_mem_callback};
    TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
    auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
        HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));

    // Add two vectors and delete one, expect that at least one repair job will be created.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, 1, 1);
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(0), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(1).size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(1)[0]->associatedSwapJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(1)[0]->associatedSwapJobs[0]->deleted_id, 0);
    ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->node_id, 1);
    ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->level, 0);
    jobQ.pop();

    // Insert another vector under the label (1) that has not been deleted.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, 1, 2);

    // Expect to see both ids stored under this label being deleted (1 and 2), and have both
    // ids need repair (as the connection between the two vectors is mutual). However, 1 has
    // also an outgoing edge to his other (deleted) neighbor (0), so there will be no new
    // repair job created for 1, since the previous repair job is expected to have both 0 and 2 in
    // its associated swap jobs. Also, there is an edge 0->1 whose going to be repaired as well.
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(1), 2);
    ASSERT_EQ(jobQ.size(), 2);
    ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->node_id, 0);
    ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->level, 0);
    jobQ.pop();
    ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->node_id, 2);
    ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->level, 0);
    jobQ.pop();
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
    delete tiered_index;
}

TYPED_TEST(HNSWTieredIndexTest, deleteFromHNSWMultiLevels) {
    // Create TieredHNSW index instance with a mock queue.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t dim = 4;

    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};
    auto jobQ = JobQueue();
    size_t memory_ctx = 0;
    auto index_ctx = IndexExtCtx();
    TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                       .jobQueueCtx = &index_ctx,
                                       .submitCb = submit_callback,
                                       .memoryCtx = &memory_ctx,
                                       .UpdateMemCb = update_mem_callback};
    TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
    auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
        HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));

    // Test that repair jobs are created for multiple levels.
    size_t num_elements_with_multiple_levels = 0;
    int vec_id = -1;
    do {
        vec_id++;
        GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, vec_id, vec_id);
        if (tiered_index->getHNSWIndex()->element_levels_[vec_id] > 0) {
            num_elements_with_multiple_levels++;
        }
    } while (num_elements_with_multiple_levels < 2);

    // Delete the last inserted vector, which is in level 1.
    ASSERT_EQ(tiered_index->deleteLabelFromHNSW(vec_id), 1);
    ASSERT_EQ(tiered_index->getHNSWIndex()->element_levels_[vec_id], 1);
    // This should be an array of length 1.
    auto *level_one_neighbors = tiered_index->getHNSWIndex()->getNodeNeighborsAtLevel(vec_id, 1);
    ASSERT_EQ(tiered_index->getHNSWIndex()->getNodeNeighborsCount(level_one_neighbors), 1);

    size_t num_repair_jobs = jobQ.size();
    // There should be at least two nodes to repair, the neighbors of next_id in levels 0 and 1
    ASSERT_GE(num_repair_jobs, 2);
    while (jobQ.size() > 1) {
        // First we should have jobs for repairing nodes in level 0.
        ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->level, 0);
        jobQ.pop();
    }

    // The last job should be repairing the single neighbor in level 1.
    ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->level, 1);
    ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->node_id, *level_one_neighbors);

    delete tiered_index;
}

TYPED_TEST(HNSWTieredIndexTest, deleteFromHNSWWithRepairJobExec) {
    // Create TieredHNSW index instance with a mock queue.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t n = 1000;
    size_t dim = 4;

    for (auto isMulti : {false, true}) {
        HNSWParams params = {.type = TypeParam::get_index_type(),
                             .dim = dim,
                             .metric = VecSimMetric_L2,
                             .multi = isMulti,
                             .M = 4};
        auto jobQ = JobQueue();
        size_t memory_ctx = 0;
        auto index_ctx = IndexExtCtx();
        TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                           .jobQueueCtx = &index_ctx,
                                           .submitCb = submit_callback,
                                           .memoryCtx = &memory_ctx,
                                           .UpdateMemCb = update_mem_callback};
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));

        for (size_t i = 0; i < n; i++) {
            GenerateAndAddVector(tiered_index->index, dim, i, i);
        }

        // Delete vectors one by one and run the resulted repair jobs.
        while (tiered_index->getHNSWIndex()->getNumMarkedDeleted() < n) {
            // Choose the current entry point each time (it should be modified after the deletion).
            idType ep = tiered_index->getHNSWIndex()->safeGetEntryPointCopy();
            auto ep_level = tiered_index->getHNSWIndex()->getMaxLevel();
            auto incoming_neighbors =
                tiered_index->getHNSWIndex()->safeCollectAllNodeIncomingNeighbors(ep, ep_level);
            ASSERT_EQ(tiered_index->deleteLabelFromHNSW(ep), 1);
            ASSERT_EQ(jobQ.size(), incoming_neighbors.size());
            ASSERT_EQ(tiered_index->getHNSWIndex()->checkIntegrity().connections_to_repair,
                      jobQ.size());
            ASSERT_NE(tiered_index->getHNSWIndex()->safeGetEntryPointCopy(), ep);

            // Execute synchronously all the repair jobs for the current deletion.
            while (!jobQ.empty()) {
                idType repair_node_id = ((HNSWRepairJob *)(jobQ.front().job))->node_id;
                auto repair_node_level = ((HNSWRepairJob *)(jobQ.front().job))->level;
                auto orig_neighbors = tiered_index->getHNSWIndex()->getNodeNeighborsAtLevel(
                    repair_node_id, repair_node_level);

                tiered_index->getHNSWIndex()->repairNodeConnections(repair_node_id,
                                                                    repair_node_level);
                auto new_neighbors = tiered_index->getHNSWIndex()->getNodeNeighborsAtLevel(
                    repair_node_id, repair_node_level);
                size_t new_neighbors_count =
                    tiered_index->getHNSWIndex()->getNodeNeighborsCount(new_neighbors);
                // This makes sure that the deleted node is no longer in the neighbors set of the
                // repaired node.
                ASSERT_TRUE(std::find(new_neighbors, new_neighbors + new_neighbors_count, ep) ==
                            new_neighbors + new_neighbors_count);
                // Remove the job from the id -> repair_jobs lookup, so we won't think that it is
                // still pending and avoid creating new jobs for nodes that already been repaired
                // as they were pointing to deleted elements.
                tiered_index->idToRepairJobs.erase(repair_node_id);
                delete jobQ.front().job;
                jobQ.pop();
            }
            ASSERT_EQ(tiered_index->getHNSWIndex()->checkIntegrity().connections_to_repair, 0);
        }
        delete tiered_index;
    }
}

TYPED_TEST(HNSWTieredIndexTest, manageIndexOwnershipWithPendingJobs) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Create TieredHNSW index instance with a mock queue.
    for (auto is_multi : {true, false}) {
        size_t dim = 4;
        HNSWParams params = {.type = TypeParam::get_index_type(),
                             .dim = dim,
                             .metric = VecSimMetric_L2,
                             .multi = is_multi};
        auto jobQ = JobQueue();
        auto *index_ctx = new IndexExtCtx();
        size_t memory_ctx = 0;
        TieredIndexParams tiered_params = {.jobQueue = &jobQ,
                                           .jobQueueCtx = index_ctx,
                                           .submitCb = submit_callback,
                                           .memoryCtx = &memory_ctx,
                                           .UpdateMemCb = update_mem_callback};
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
        // Set the created tiered index in the index external context.
        index_ctx->index_strong_ref.reset(tiered_index);
        EXPECT_EQ(index_ctx->index_strong_ref.use_count(), 1);

        // Add a vector and create a pending insert job.
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 0);
        ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);

        // Delete the index before the job was executed.
        EXPECT_EQ(jobQ.size(), 1);
        EXPECT_EQ(jobQ.front().index_weak_ref.use_count(), 1);
        delete index_ctx;
        EXPECT_EQ(jobQ.size(), 1);
        EXPECT_EQ(jobQ.front().index_weak_ref.use_count(), 0);
        jobQ.pop();

        // Recreate the index with a new ctx.
        tiered_params.jobQueueCtx = index_ctx = new IndexExtCtx();
        tiered_hnsw_params.tieredParams = tiered_params;
        tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::tiered::NewIndex(&tiered_hnsw_params, allocator));
        index_ctx->index_strong_ref.reset(tiered_index);
        EXPECT_EQ(index_ctx->index_strong_ref.use_count(), 1);

        // Add two vectors directly to HNSW, and remove one vector to create a repair job.
        GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, 0, 0);
        GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, 1, 1);
        ASSERT_EQ(tiered_index->deleteLabelFromHNSW(0), 1);
        ASSERT_EQ(tiered_index->idToRepairJobs.size(), 1);

        // Delete the index before the job was executed.
        EXPECT_EQ(jobQ.size(), 1);
        EXPECT_EQ(jobQ.front().index_weak_ref.use_count(), 1);
        delete index_ctx;
        EXPECT_EQ(jobQ.size(), 1);
        EXPECT_EQ(jobQ.front().index_weak_ref.use_count(), 0);
    }
}
