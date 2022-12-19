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
        VecSimIndex_AddVector(tiered_index->tempFlat, vector, vector_label);

        // Create a mock job that inserts some vector into the HNSW index.
        auto insert_to_index = [](void *job) {
            auto *my_insert_job = reinterpret_cast<HNSWInsertJob *>(job);
            auto my_index =
                reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(my_insert_job->index);

            // Move the vector from the temp flat index into the HNSW index.
            my_index->addVector(my_index->tempFlat->getDataByInternalId(my_insert_job->id),
                                my_insert_job->label);
            VecSimIndex_DeleteVector(my_index->tempFlat, my_insert_job->label);
            auto it = my_index->labelToInsertJobs[my_insert_job->label].begin();
            ASSERT_EQ(job, *it); // Assert pointers equation
            my_index->labelToInsertJobs[my_insert_job->label].erase(it);
            my_index->UpdateIndexMemory(my_index->memoryCtx,
                                        my_index->getAllocator()->getAllocationSize());
        };

        HNSWInsertJob job = {
            .base = AsyncJob{.jobType = HNSW_INSERT_VECTOR_JOB, .Execute = insert_to_index},
            .index = tiered_index,
            .label = vector_label};
        tiered_index->labelToInsertJobs[vector_label].push_back(&job);

        // Wrap this job with an array and submit the jobs to the queue.
        auto **jobs = array_new<AsyncJob *>(1);
        jobs = array_append(jobs, (AsyncJob *)&job);
        tiered_index->SubmitJobsToQueue(tiered_index->jobQueue, (void **)jobs, 1);
        ASSERT_EQ(jobQ->size(), 1);

        // Execute the job from the queue and validate that the index was updated properly.
        reinterpret_cast<AsyncJob *>(jobQ->front())->Execute(jobQ->front());
        ASSERT_EQ(tiered_index->indexSize(), 1);
        ASSERT_EQ(tiered_index->getDistanceFrom(1, vector), 0);
        ASSERT_EQ(memory_ctx, tiered_index->getAllocator()->getAllocationSize());
        ASSERT_EQ(tiered_index->tempFlat->indexSize(), 0);
        ASSERT_EQ(tiered_index->labelToInsertJobs[vector_label].size(), 0);

        // Cleanup.
        delete jobQ;
        array_free(jobs);
        VecSimIndex_Free(tiered_index);
    }
}
