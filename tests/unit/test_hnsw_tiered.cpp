#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "test_utils.h"

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
        auto *jobQ = new JobQueue();
        size_t memory_ctx = 0;
        TieredIndexParams tiered_params = {.jobQueue = jobQ,
                                           .submitCb = submit_callback,
                                           .memoryCtx = &memory_ctx,
                                           .UpdateMemCb = update_mem_callback};
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));

        // Add a vector to the flat index.
        TEST_DATA_T vector[tiered_index->index->getDim()];
        GenerateVector<TEST_DATA_T>(vector, tiered_index->index->getDim());
        labelType vector_label = 1;
        VecSimIndex_AddVector(tiered_index->flatBuffer, vector, vector_label);

        // Create a mock job that inserts some vector into the HNSW index.
        auto insert_to_index = [](void *job) {
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
        ASSERT_EQ(jobQ->size(), 1);

        // Execute the job from the queue and validate that the index was updated properly.
        reinterpret_cast<AsyncJob *>(jobQ->front())->Execute(jobQ->front());
        ASSERT_EQ(tiered_index->indexSize(), 1);
        ASSERT_EQ(tiered_index->getDistanceFrom(1, vector), 0);
        ASSERT_EQ(memory_ctx, tiered_index->getAllocator()->getAllocationSize());
        ASSERT_EQ(tiered_index->flatBuffer->indexSize(), 0);
        ASSERT_EQ(tiered_index->labelToInsertJobs.at(vector_label).size(), 0);

        // Cleanup.
        delete jobQ;
        VecSimIndex_Free(tiered_index);
    }
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
        auto *jobQ = new JobQueue();
        size_t memory_ctx = 0;
        TieredIndexParams tiered_params = {.jobQueue = jobQ,
                                           .submitCb = submit_callback,
                                           .memoryCtx = &memory_ctx,
                                           .UpdateMemCb = update_mem_callback};
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));

        BFParams bf_params = {.type = TypeParam::get_index_type(),
                              .dim = dim,
                              .metric = VecSimMetric_L2,
                              .multi = is_multi};

        // Validate that memory upon creating the tiered index is as expected.
        size_t expected_mem = HNSWFactory::EstimateInitialSize(&params) +
                              BruteForceFactory::EstimateInitialSize(&bf_params) +
                              sizeof(*tiered_index);
        ASSERT_EQ(expected_mem, memory_ctx);

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

        // Cleanup.
        delete jobQ;
        VecSimIndex_Free(tiered_index);
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
        auto *jobQ = new JobQueue();
        size_t memory_ctx = 0;
        TieredIndexParams tiered_params = {.jobQueue = jobQ,
                                           .submitCb = submit_callback,
                                           .memoryCtx = &memory_ctx,
                                           .UpdateMemCb = update_mem_callback};
        TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
        auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
            HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));

        // Create a vector and add it to the tiered index.
        labelType vec_label = 1;
        TEST_DATA_T vector[dim];
        GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
        VecSimIndex_AddVector(tiered_index, vector, vec_label);
        ASSERT_EQ(tiered_index->indexSize(), 1);
        ASSERT_EQ(tiered_index->flatBuffer->indexSize(), 1);

        // Execute the insert job manually (in a synchronous manner).
        ASSERT_EQ(jobQ->size(), 1);
        auto *insert_job = reinterpret_cast<HNSWInsertJob *>(jobQ->front());
        ASSERT_EQ(insert_job->label, vec_label);
        ASSERT_EQ(insert_job->id, 0);
        ASSERT_EQ(insert_job->jobType, HNSW_INSERT_VECTOR_JOB);

        insert_job->Execute(insert_job);
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

        // Cleanup.
        delete jobQ;
        VecSimIndex_Free(tiered_index);
    }
}

TYPED_TEST(HNSWTieredIndexTest, insertJobAsync) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Create TieredHNSW index instance with a mock queue.
    size_t dim = 4;
    size_t n = 5000;
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};
    auto *jobQ = new JobQueue();
    size_t memory_ctx = 0;
    TieredIndexParams tiered_params = {.jobQueue = jobQ,
                                       .submitCb = submit_callback,
                                       .memoryCtx = &memory_ctx,
                                       .UpdateMemCb = update_mem_callback};
    TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
    auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
        HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));

    // Main loop for background worker threads that execute the jobs (ingest from flat to HNSW).
    bool run_thread = true;
    auto thread_fn = [jobQ, &run_thread]() {
        while (run_thread) {
            std::unique_lock<std::mutex> lock(queue_guard);
            queue_cond.wait(lock, [jobQ, &run_thread]() { return !jobQ->empty() || !run_thread; });
            if (!run_thread)
                return;
            auto *job = jobQ->front();
            jobQ->pop();
            lock.unlock();
            job->Execute(job);
        }
    };

    for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
        thread_pool.push_back(std::thread(thread_fn));
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
        if (jobQ->empty()) {
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
    ASSERT_EQ(jobQ->size(), 0);
    // Verify that the vectors were inserted to HNSW as expected
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T expected_vector[dim];
        GenerateVector<TEST_DATA_T>(expected_vector, dim, i);
        ASSERT_EQ(tiered_index->index->getDistanceFrom(i, expected_vector), 0);
    }

    // Cleanup.
    delete jobQ;
    VecSimIndex_Free(tiered_index);
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
    auto *jobQ = new JobQueue();
    size_t memory_ctx = 0;
    TieredIndexParams tiered_params = {.jobQueue = jobQ,
                                       .submitCb = submit_callback,
                                       .memoryCtx = &memory_ctx,
                                       .UpdateMemCb = update_mem_callback};
    TieredHNSWParams tiered_hnsw_params = {.hnswParams = params, .tieredParams = tiered_params};
    auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
        HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));

    // Main loop for background worker threads that execute the jobs (ingest from flat to HNSW).
    bool run_thread = true;
    auto thread_fn = [jobQ, &run_thread]() {
        while (run_thread) {
            std::unique_lock<std::mutex> lock(queue_guard);
            queue_cond.wait(lock, [jobQ, &run_thread]() { return !jobQ->empty() || !run_thread; });
            if (!run_thread)
                return;
            auto *job = jobQ->front();
            jobQ->pop();
            lock.unlock();
            job->Execute(job);
        }
    };

    for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
        thread_pool.push_back(std::thread(thread_fn));
    }

    // Insert vectors
    for (size_t i = 0; i < n / per_label; i++) {
        for (size_t j = 0; j < per_label; j++) {
            GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i, i * per_label + j);
        }
    }
    ASSERT_GE(tiered_index->labelToInsertJobs.size(), 0);

    // Check every 10 ms if queue is empty, and if so, terminate the threads loop.
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::unique_lock<std::mutex> lock(queue_guard);
        if (jobQ->empty()) {
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
    EXPECT_EQ(jobQ->size(), 0);
    // Verify that the vectors were inserted to HNSW as expected
    for (size_t i = 0; i < n / per_label; i++) {
        for (size_t j = 0; j < per_label; j++) {
            // The distance from every vector that is stored under the label i should be zero
            TEST_DATA_T expected_vector[dim];
            GenerateVector<TEST_DATA_T>(expected_vector, dim, i * per_label + j);
            EXPECT_EQ(tiered_index->index->getDistanceFrom(i, expected_vector), 0);
        }
    }

    // Cleanup.
    delete jobQ;
    VecSimIndex_Free(tiered_index);
    thread_pool.clear();
}
