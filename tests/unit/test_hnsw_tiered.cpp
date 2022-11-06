#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "test_utils.h"

using namespace tiered_index_mock;

template <typename index_type_t>
class HNSWTieredIndexTest : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;
};

TYPED_TEST_SUITE(HNSWTieredIndexTest, DataTypeSet);

TYPED_TEST(HNSWTieredIndexTest, CreateIndexInstance) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {
        .type = TypeParam::get_index_type(), .dim = 4, .metric = VecSimMetric_L2, .multi = false};
    auto *jobQ = new JobQueue();
    size_t memory_ctx = 0;
    TieredHNSWParams tiered_params = {.hnswParams = params,
                                      .jobQueue = jobQ,
                                      .submitCb = submit_callback,
                                      .memoryCtx = &memory_ctx,
                                      .UpdateMemCb = update_mem_callback};
    auto *tiered_index = reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(
        HNSWFactory::NewTieredIndex(&tiered_params, allocator));

    // Create a mock job that inserts a vector into the HNSW index.
    auto insert_to_index = [](void *job) {
        auto *my_insert_job = reinterpret_cast<HNSWJob *>(job);
        auto my_index =
            reinterpret_cast<TieredHNSWIndex<TEST_DATA_T, TEST_DIST_T> *>(my_insert_job->index);
        my_index->addVector(my_insert_job->blob, my_insert_job->label);
        my_index->UpdateIndexMemory(my_index->memoryCtx,
                                    my_index->getAllocator()->getAllocationSize());
    };
    TEST_DATA_T vector[tiered_index->index->getDim()];
    GenerateVector<TEST_DATA_T>(vector, tiered_index->index->getDim());
    HNSWJob job = {.jobType = HNSW_INSERT_JOB,
                   .label = 1,
                   .blob = vector,
                   .index = tiered_index,
                   .Execute = insert_to_index};

    // Wrap this job with an array and submit the jobs to the queue.
    auto **jobs = array_new<HNSWJob *>(1);
    jobs = array_append(jobs, &job);
    tiered_index->SubmitJobsToQueue(tiered_index->jobQueue, (void **)jobs, 1);
    ASSERT_EQ(jobQ->size(), 1);

    // Execute the job from the queue and validate that the index was updated properly.
    reinterpret_cast<HNSWJob *>(jobQ->front())->Execute(jobQ->front());
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(memory_ctx, tiered_index->getAllocator()->getAllocationSize());

    // Cleanup.
    delete jobQ;
    array_free(jobs);
    VecSimIndex_Free(tiered_index);
}
