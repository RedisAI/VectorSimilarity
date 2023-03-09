#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
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
            HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));
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
            HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));
        // Set the created tiered index in the index external context.
        index_ctx.index_strong_ref.reset(tiered_index);

        BFParams bf_params = {.type = TypeParam::get_index_type(),
                              .dim = dim,
                              .metric = VecSimMetric_L2,
                              .multi = is_multi};

        // Validate that memory upon creating the tiered index is as expected (no more than 2%
        // above te expected, since in different platforms there are some minor additional
        // allocations).
        size_t expected_mem = HNSWFactory::EstimateInitialSize(&params) +
                              BruteForceFactory::EstimateInitialSize(&bf_params) +
                              sizeof(*tiered_index);
        ASSERT_LE(expected_mem, memory_ctx);
        ASSERT_GE(expected_mem * 1.02, memory_ctx);

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
            HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));
        // Set the created tiered index in the index external context.
        index_ctx->index_strong_ref.reset(tiered_index);
        EXPECT_EQ(index_ctx->index_strong_ref.use_count(), 1);
        size_t initial_mem = memory_ctx;

        // Create a dummy job callback that insert one vector to the underline HNSW index.
        auto dummy_job = [](AsyncJob *job) {
            auto *my_index =
                reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(job->index);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
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
            HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));
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
        HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));
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
        HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));
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

TYPED_TEST(HNSWTieredIndexTest, deleteFromHNSWBasic) {
    // Create TieredHNSW index instance with a mock queue.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t dim = 4;

    for (auto is_multi : {false, true}) {
        HNSWParams params = {.type = TypeParam::get_index_type(),
                             .dim = dim,
                             .metric = VecSimMetric_L2,
                             .multi = is_multi};
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
            HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));

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

        ASSERT_EQ(tiered_index->indexSize(), 4);
        ASSERT_EQ(tiered_index->getHNSWIndex()->getNumMarkedDeleted(), 3);
        ASSERT_EQ(tiered_index->idToSwapJob.size(), 3);
        jobQ.pop();

        if (is_multi) {
            // Insert another vector under the label (2) that has not been deleted.
            GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, 2, 4);
            // Expect to see both ids stored under this label being deleted (2 and 4), and have both
            // ids need repair (as the connection between the two vectors is mutual). Also, 2 has an
            // incoming edge from his other (deleted) neighbor (3)
            ASSERT_EQ(tiered_index->deleteLabelFromHNSW(2), 2);
            ASSERT_EQ(jobQ.size(), 3);
            ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->node_id, 3);
            ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->level, 0);
            jobQ.pop();
            ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->node_id, 4);
            ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->level, 0);
            jobQ.pop();
            // Repair for node id 4.
            ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->node_id, 2);
            ASSERT_EQ(((HNSWRepairJob *)(jobQ.front().job))->level, 0);
            jobQ.pop();
            ASSERT_EQ(tiered_index->idToSwapJob.size(), 5);
            ASSERT_EQ(tiered_index->idToRepairJobs.size(), 3);
            ASSERT_EQ(tiered_index->idToRepairJobs.at(2).size(), 2);
            ASSERT_EQ(tiered_index->idToRepairJobs.at(3).size(), 1);
            ASSERT_EQ(tiered_index->idToRepairJobs.at(4).size(), 1);
        }

        // Test that repair jobs are created for multiple levels.
        size_t num_elements_with_multiple_levels = 0;
        size_t vec_id = tiered_index->index->indexSize();
        while (num_elements_with_multiple_levels < 2) {
            GenerateAndAddVector<TEST_DATA_T>(tiered_index->index, dim, vec_id, vec_id);
            if (tiered_index->getHNSWIndex()->element_levels_[vec_id] > 0) {
                num_elements_with_multiple_levels++;
            }
            vec_id++;
        }

        // Delete the last inserted vector, which is in level 1.
        ASSERT_EQ(tiered_index->deleteLabelFromHNSW(--vec_id), 1);
        ASSERT_EQ(tiered_index->getHNSWIndex()->element_levels_[vec_id], 1);
        auto *level_one_neighbors =
            tiered_index->getHNSWIndex()->getNodeNeighborsAtLevel(vec_id, 1);
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
    }
}

TYPED_TEST(HNSWTieredIndexTest, deleteFromHNSWWithRepairJobExec) {
    // Create TieredHNSW index instance with a mock queue.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t n = 100;
    size_t dim = 4;

    for (auto is_multi : {false, true}) {
        HNSWParams params = {.type = TypeParam::get_index_type(),
                             .dim = dim,
                             .metric = VecSimMetric_L2,
                             .multi = is_multi,
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
            HNSWFactory::NewTieredIndex(&tiered_hnsw_params, allocator));

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
            // ASSERT_EQ(tiered_index->getHNSWIndex()->checkIntegrity().connection_to_repair,
            // jobQ.size());

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
                jobQ.pop();
            }
            // ASSERT_EQ(tiered_index->getHNSWIndex()->checkIntegrity().connection_to_repair, 0);
        }
        delete tiered_index;
    }
}
