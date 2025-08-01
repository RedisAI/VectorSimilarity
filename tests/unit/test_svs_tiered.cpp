#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/vec_sim_debug.h"
#include <string>
#include <array>

#include "unit_test_utils.h"
#include "mock_thread_pool.h"

#if HAVE_SVS
#include <thread>
// For getAvailableCPUs():
#include <sched.h>

#include "VecSim/algorithms/svs/svs.h"
#include "VecSim/algorithms/svs/svs_tiered.h"

// There are possible cases when SVS Index cannot be created with the requested quantization mode
// due to platform and/or hardware limitations or combination of requested 'compression' modes.
// This assert handle those cases and skip a test if the mode is not supported.
// Elsewhere, test will fail if the index creation failed with no reason explained above.
#define ASSERT_INDEX(index)                                                                        \
    if (index == nullptr) {                                                                        \
        if (std::get<1>(svs_details::isSVSQuantBitsSupported(TypeParam::get_quant_bits()))) {      \
            GTEST_FAIL() << "Failed to create SVS index";                                          \
        } else {                                                                                   \
            GTEST_SKIP() << "SVS LVQ is not supported.";                                           \
        }                                                                                          \
    }

// Get available number of CPUs
// Returns the number of logical processors on the process
// Returns std::thread::hardware_concurrency() if the number of logical processors is not available
static unsigned int getAvailableCPUs() {
#ifdef __linux__
    // On Linux, use sched_getaffinity to get the number of CPUs available to the current process.
    cpu_set_t cpu_set;
    if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) == 0) {
        return CPU_COUNT(&cpu_set);
    }
#endif
    // Fallback.
    return std::thread::hardware_concurrency();
}

// Log callback function to print non-debug log messages
static void svsTestLogCallBackNoDebug(void *ctx, const char *level, const char *message) {
    if (level == nullptr || message == nullptr) {
        return; // Skip null messages
    }
    if (std::string_view{level} == VecSimCommonStrings::LOG_DEBUG_STRING) {
        return; // Skip debug messages
    }
    // Print other log levels
    std::cout << level << ": " << message << std::endl;
}

// Runs the test for combination of data type and quantization mode.
// TODO: Add support for label type combination(single/multi)
template <typename index_type_t>
class SVSTieredIndexTest : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    static const size_t defaultTrainingThreshold = 1024;
    static const size_t defaultUpdateThreshold = 16;

protected:
    TieredSVSIndex<data_t> *CastToTieredSVS(VecSimIndex *index) {
        return reinterpret_cast<TieredSVSIndex<data_t> *>(index);
    }

    TieredIndexParams CreateTieredSVSParams(VecSimParams &svs_params,
                                            tieredIndexMock &mock_thread_pool,
                                            size_t training_threshold = defaultTrainingThreshold,
                                            size_t update_threshold = defaultUpdateThreshold) {
        trainingThreshold = training_threshold;
        updateThreshold = update_threshold;
        svs_params.algoParams.svsParams.quantBits = index_type_t::get_quant_bits();
        if (svs_params.algoParams.svsParams.num_threads == 0) {
            svs_params.algoParams.svsParams.num_threads = mock_thread_pool.thread_pool_size;
        }
        return TieredIndexParams{
            .jobQueue = &mock_thread_pool.jobQ,
            .jobQueueCtx = mock_thread_pool.ctx,
            .submitCb = tieredIndexMock::submit_callback,
            .primaryIndexParams = &svs_params,
            .specificParams = {.tieredSVSParams =
                                   TieredSVSParams{.trainingTriggerThreshold = training_threshold,
                                                   .updateTriggerThreshold = update_threshold}}};
    }

    TieredSVSIndex<data_t> *CreateTieredSVSIndex(const TieredIndexParams &tiered_params,
                                                 tieredIndexMock &mock_thread_pool) {
        auto *tiered_index =
            reinterpret_cast<TieredSVSIndex<data_t> *>(TieredFactory::NewIndex(&tiered_params));

        // Set the created tiered index in the index external context (it will take ownership over
        // the index, and we'll need to release the ctx at the end of the test.
        mock_thread_pool.ctx->index_strong_ref.reset(tiered_index);
        return tiered_index;
    }

    TieredSVSIndex<data_t> *
    CreateTieredSVSIndex(VecSimParams &svs_params, tieredIndexMock &mock_thread_pool,
                         size_t training_threshold = defaultTrainingThreshold,
                         size_t update_threshold = defaultUpdateThreshold) {
        svs_params.algoParams.svsParams.quantBits = index_type_t::get_quant_bits();
        TieredIndexParams tiered_params = CreateTieredSVSParams(
            svs_params, mock_thread_pool, training_threshold, update_threshold);
        return CreateTieredSVSIndex(tiered_params, mock_thread_pool);
    }

    void SetUp() override {
        // Restore the write mode to default.
        VecSim_SetWriteMode(VecSim_WriteAsync);
        // Limit VecSim log level to avoid printing too much information
        VecSimIndexInterface::setLogCallbackFunction(svsTestLogCallBackNoDebug);
    }

    // Check if the test is running in fallback mode to scalar quantization.
    bool isFallbackToSQ() const {
        // Get the fallback quantization mode and compare it to the scalar quantization mode.
        return VecSimSvsQuant_Scalar ==
               std::get<0>(svs_details::isSVSQuantBitsSupported(index_type_t::get_quant_bits()));
    }

    size_t getTrainingThreshold() const { return trainingThreshold; }

    size_t getUpdateThreshold() const { return updateThreshold; }

private:
    size_t trainingThreshold = defaultTrainingThreshold;
    size_t updateThreshold = defaultUpdateThreshold;
};

// TEST_DATA_T and TEST_DIST_T are defined in test_utils.h

template <VecSimType type, typename DataType, VecSimSvsQuantBits quantBits, bool IsMulti>
struct SVSIndexType {
    static constexpr VecSimType get_index_type() { return type; }
    static constexpr VecSimSvsQuantBits get_quant_bits() { return quantBits; }
    static constexpr bool isMulti() { return IsMulti; }
    typedef DataType data_t;
};

// clang-format off
using SVSDataTypeSet = ::testing::Types<SVSIndexType<VecSimType_FLOAT32, float, VecSimSvsQuant_NONE, false>
                                       ,SVSIndexType<VecSimType_FLOAT32, float, VecSimSvsQuant_NONE, true>
                                       ,SVSIndexType<VecSimType_FLOAT32, float, VecSimSvsQuant_8, false>
                                        >;
// clang-format on

TYPED_TEST_SUITE(SVSTieredIndexTest, SVSDataTypeSet);

// Runs the test for each data type(float/double). The label type should be explicitly
// set in the test.

template <typename index_type_t>
class SVSTieredIndexTestBasic : public SVSTieredIndexTest<index_type_t> {};

template <VecSimType type, typename DataType, VecSimSvsQuantBits quantBits>
struct SVSIndexTypeBasic {
    static constexpr VecSimType get_index_type() { return type; }
    static constexpr VecSimSvsQuantBits get_quant_bits() { return quantBits; }
    typedef DataType data_t;
};
using SVSBasicDataTypeSet =
    ::testing::Types<SVSIndexTypeBasic<VecSimType_FLOAT32, float, VecSimSvsQuant_NONE>>;

TYPED_TEST_SUITE(SVSTieredIndexTestBasic, SVSBasicDataTypeSet);

TYPED_TEST(SVSTieredIndexTest, ThreadsReservation) {
    // Set thread_pool_size to 4 or actual number of available CPUs
    const auto num_threads = std::min(4U, getAvailableCPUs());
    if (num_threads < 2) {
        // If the number of threads is less than 2, we can't run the test
        GTEST_SKIP() << "No threads available";
    }

    std::chrono::milliseconds timeout{1000}; // long enough to reserve all threads
    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = 4, .metric = VecSimMetric_L2, .num_threads = 1};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();
    mock_thread_pool.thread_pool_size = num_threads;

    // Create TieredSVS index instance with a mock queue.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    ASSERT_INDEX(tiered_index);

    // Get the allocator from the tiered index.
    auto allocator = tiered_index->getAllocator();

    // Counter of reserved threads
    // This is set in the update_job_mock callback only
    std::atomic<size_t> num_reserved_threads = 0;

    auto update_job_mock = [&num_reserved_threads](VecSimIndex * /*unused*/, size_t num_threads) {
        num_reserved_threads = num_threads;
    };

    SVSMultiThreadJob::JobsRegistry registry(allocator);
    // Request 4 threads but just 1 thread is available
    auto jobs = SVSMultiThreadJob::createJobs(allocator, SVS_BATCH_UPDATE_JOB, update_job_mock,
                                              tiered_index, 4, timeout, &registry);
    ASSERT_EQ(jobs.size(), 4);
    tiered_index->submitJobs(jobs);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 4);
    // emulate 1 thread availability
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 3);
    ASSERT_EQ(num_reserved_threads, 1);

    // Complete rest of wait jobs
    mock_thread_pool.init_threads();
    mock_thread_pool.thread_pool_wait();
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Request and run exact number of available threads
    jobs = SVSMultiThreadJob::createJobs(allocator, SVS_BATCH_UPDATE_JOB, update_job_mock,
                                         tiered_index, num_threads, timeout, &registry);
    tiered_index->submitJobs(jobs);
    mock_thread_pool.thread_pool_wait();
    ASSERT_EQ(num_reserved_threads, num_threads);

    // Request and run 1 thread
    jobs = SVSMultiThreadJob::createJobs(allocator, SVS_BATCH_UPDATE_JOB, update_job_mock,
                                         tiered_index, 1, timeout, &registry);
    tiered_index->submitJobs(jobs);
    mock_thread_pool.thread_pool_wait();
    ASSERT_EQ(num_reserved_threads, 1);

    // Request and run less threads than available
    jobs = SVSMultiThreadJob::createJobs(allocator, SVS_BATCH_UPDATE_JOB, update_job_mock,
                                         tiered_index, num_threads - 1, timeout, &registry);
    tiered_index->submitJobs(jobs);
    mock_thread_pool.thread_pool_wait();
    // The number of reserved threads should be equal to requested
    ASSERT_EQ(num_reserved_threads, num_threads - 1);

    // Request more threads than available
    jobs = SVSMultiThreadJob::createJobs(allocator, SVS_BATCH_UPDATE_JOB, update_job_mock,
                                         tiered_index, num_threads + 1, timeout, &registry);
    tiered_index->submitJobs(jobs);
    mock_thread_pool.thread_pool_wait();
    // The number of reserved threads should be equal to the number of available threads
    ASSERT_EQ(num_reserved_threads, num_threads);
    mock_thread_pool.thread_pool_join();
}

TYPED_TEST(SVSTieredIndexTest, CreateIndexInstance) {
    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = 4,
                        .metric = VecSimMetric_L2,
                        .multi = TypeParam::isMulti()};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    ASSERT_INDEX(tiered_index);

    // Get the allocator from the tiered index.
    auto allocator = tiered_index->getAllocator();

    // Add a vector to the flat index.
    TEST_DATA_T vector[params.dim];
    GenerateVector<TEST_DATA_T>(vector, params.dim);
    labelType vector_label = 1;
    VecSimIndex_AddVector(tiered_index, vector, vector_label);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 0);

    // Submit the index update job.
    tiered_index->scheduleSVSIndexUpdate();
    ASSERT_EQ(mock_thread_pool.jobQ.size(), mock_thread_pool.thread_pool_size);

    // Execute the job from the queue and validate that the index was updated properly.
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(1, vector), 0);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
}

TYPED_TEST(SVSTieredIndexTest, addVector) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .multi = TypeParam::isMulti()};
    VecSimParams svs_params = CreateParams(params);

    auto mock_thread_pool = tieredIndexMock();

    auto tiered_params = this->CreateTieredSVSParams(svs_params, mock_thread_pool, 1, 1);
    auto *tiered_index = this->CreateTieredSVSIndex(tiered_params, mock_thread_pool);
    ASSERT_INDEX(tiered_index);

    // Get the allocator from the tiered index.
    auto allocator = tiered_index->getAllocator();

    BFParams bf_params = {.type = TypeParam::get_index_type(),
                          .dim = dim,
                          .metric = VecSimMetric_L2,
                          .multi = TypeParam::isMulti()};

    size_t expected_mem = TieredFactory::EstimateInitialSize(&tiered_params);
    ASSERT_LE(expected_mem, tiered_index->getAllocationSize());
    ASSERT_GE(expected_mem * 1.02, tiered_index->getAllocationSize());
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Create a vector and add it to the tiered index.
    labelType vec_label = 1;
    TEST_DATA_T vector[dim];
    GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
    VecSimIndex_AddVector(tiered_index, vector, vec_label);
    // Validate that the vector was inserted to the flat buffer properly.
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexCapacity(), DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(tiered_index->indexCapacity(), DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(tiered_index->GetFlatIndex()->getDistanceFrom_Unsafe(vec_label, vector), 0);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), mock_thread_pool.thread_pool_size);

    // Account for the allocation of a new block due to the vector insertion.
    expected_mem += (BruteForceFactory::EstimateElementSize(&bf_params)) * DEFAULT_BLOCK_SIZE;
    // Account for the memory that was allocated in the labelToId map (approx.)
    expected_mem += sizeof(vecsim_stl::unordered_map<labelType, idType>::value_type) +
                    sizeof(void *) + sizeof(size_t);
    // Account for the insert job that was created.
    expected_mem +=
        SVSMultiThreadJob::estimateSize(mock_thread_pool.thread_pool_size) + sizeof(size_t);
    auto actual_mem = tiered_index->getAllocationSize();
    ASSERT_GE(expected_mem * 1.02, tiered_index->getAllocationSize());
    ASSERT_LE(expected_mem, tiered_index->getAllocationSize());

    if constexpr (TypeParam::isMulti()) {
        // Add another vector under the same label
        VecSimIndex_AddVector(tiered_index, vector, vec_label);
        ASSERT_EQ(tiered_index->indexSize(), 2);
        ASSERT_EQ(tiered_index->indexLabelCount(), 1);
        ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 0);
        ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 2);
        // Validate that there still 1 update jobs set
        ASSERT_EQ(mock_thread_pool.jobQ.size(), mock_thread_pool.thread_pool_size);
    }
}

TYPED_TEST(SVSTieredIndexTest, insertJob) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .multi = TypeParam::isMulti()};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Force thired index to submit the update job on every insert.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1, 1);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    // Create a vector and add it to the tiered index.
    labelType vec_label = 1;
    TEST_DATA_T vector[dim];
    GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
    VecSimIndex_AddVector(tiered_index, vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);

    // Execute the insert job manually (in a synchronous manner).
    ASSERT_EQ(mock_thread_pool.jobQ.size(), mock_thread_pool.thread_pool_size);
    auto *insertion_job = mock_thread_pool.jobQ.front().job;
    ASSERT_EQ(insertion_job->jobType, SVS_BATCH_UPDATE_JOB);

    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
    // SVS index should have allocated a single record, while flat index should remove the
    // block.
    // Compression datasets do not provide a capacity method
    const size_t expected_capacity = TypeParam::get_quant_bits() == VecSimSvsQuant_NONE
                                         ? DEFAULT_BLOCK_SIZE
                                         : tiered_index->GetBackendIndex()->indexCapacity();
    ASSERT_EQ(tiered_index->indexCapacity(), expected_capacity);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexCapacity(), 0);
    ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(vec_label, vector), 0);
}

TYPED_TEST(SVSTieredIndexTestBasic, insertJobAsync) {
    size_t dim = 4;
    size_t n = 1000;
    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Forcibly set the training threshold to a value that will trigger the update job
    // after inserting a half of vectors.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, n / 2);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    // Insert vectors
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i, i / (TEST_DATA_T)n);
    }

    mock_thread_pool.thread_pool_join();
    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);
    // Verify that vectors were moved to SVS as expected
    auto sz_f = tiered_index->GetFlatIndex()->indexSize();
    auto sz_b = tiered_index->GetBackendIndex()->indexSize();
    EXPECT_LE(sz_f, this->getUpdateThreshold());
    EXPECT_EQ(sz_f + sz_b, n);

    // Quantization has limited accuracy, so we need to check the relative error.
    // If quantization is enabled, we allow a larger relative error.
    double abs_err = TypeParam::get_quant_bits() != VecSimSvsQuant_NONE ? 1e-2 : 1e-6;

    // Verify that the vectors were inserted to Flat/SVS as expected
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T expected_vector[dim];
        GenerateVector<TEST_DATA_T>(expected_vector, dim, i / (TEST_DATA_T)n);
        ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(i, expected_vector), 0, abs_err)
            << "Vector label: " << i;
    }
}

TYPED_TEST(SVSTieredIndexTestBasic, insertJobAsyncMulti) {
    // Create TieredSVS index instance with a mock queue.
    const size_t dim = 4;
    const size_t n = 1000;
    const size_t per_label = 5;

    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = true};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Forcibly set the training threshold to a value that will trigger the update job
    // after inserting a half of vectors.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, n / 2);
    ASSERT_INDEX(tiered_index);
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
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
    EXPECT_EQ(tiered_index->indexLabelCount(), n / per_label);
    // Verify that vectors were moved to SVS as expected
    auto sz_f = tiered_index->GetFlatIndex()->indexSize();
    auto sz_b = tiered_index->GetBackendIndex()->indexSize();
    EXPECT_LE(sz_f, this->getUpdateThreshold());
    EXPECT_EQ(sz_f + sz_b, n);

    // Quantization has limited accuaracy, so we need to check the relative error.
    // If quantization is enabled, we allow a larger relative error.
    double abs_err = TypeParam::get_quant_bits() != VecSimSvsQuant_NONE ? 1e-2 : 1e-6;

    // Verify that the vectors were inserted to SVS as expected
    for (size_t i = 0; i < n / per_label; i++) {
        for (size_t j = 0; j < per_label; j++) {
            // The distance from every vector that is stored under the label i should be zero
            auto expected_vector = vectors + i * per_label * dim + j * dim;
            ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(i, expected_vector), 0, abs_err)
                << "Vector label: " << i;
        }
    }
}

TYPED_TEST(SVSTieredIndexTestBasic, KNNSearch) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }

    size_t dim = 4;
    size_t k = 10;

    size_t n = k * 3;

    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    size_t cur_memory_usage;

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, k, 1);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();
    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    auto svs_index = tiered_index->GetBackendIndex();
    auto flat_index = tiered_index->GetFlatIndex();

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
        GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, i, i);
    }
    ASSERT_EQ(tiered_index->indexSize(), n / 2);
    ASSERT_EQ(tiered_index->indexSize(), svs_index->indexSize());

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
    ASSERT_EQ(tiered_index->indexSize(), svs_index->indexSize() + flat_index->indexSize());

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
        GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, i, i);
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
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_n, k, ver_res_n);

    // Search for less vectors than the index size, but more than the flat and main index sizes.
    k = n * 5 / 6;
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_n, k, ver_res_n);

    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // Search for more vectors than the main index size, but less than the flat index size.
    for (size_t i = n / 2; i < n * 2 / 3; i++) {
        VecSimIndex_DeleteVector(svs_index, i);
    }
    ASSERT_EQ(flat_index->indexSize(), n * 2 / 3);
    ASSERT_EQ(svs_index->indexSize(), n / 2);
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
    ASSERT_EQ(svs_index->indexSize(), n / 2);
    k = n / 4;
    cur_memory_usage = allocator->getAllocationSize();
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // Search for vectors when the flat index is not empty but the main index is empty.
    for (size_t i = 0; i < n * 2 / 3; i++) {
        VecSimIndex_DeleteVector(svs_index, i);
        GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, i, i);
    }
    ASSERT_EQ(flat_index->indexSize(), n * 2 / 3);
    ASSERT_EQ(svs_index->indexSize(), 0);
    k = n / 3;
    cur_memory_usage = allocator->getAllocationSize();
    runTopKSearchTest(tiered_index, query_0, k, ver_res_0);
    runTopKSearchTest(tiered_index, query_1mid, k, ver_res_1mid);
    // Memory usage should not change.
    ASSERT_EQ(allocator->getAllocationSize(), cur_memory_usage);

    // // // // // // // // // // // //
    // Check behavior upon timeout.  //
    // // // // // // // // // // // //

    VecSimQueryReply *res;
    // Add a vector to the SVS index so there will be a reason to query it.
    GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, n, n);

    // Set timeout callback to always return 1 (will fail while querying the flat buffer).
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out

    res = VecSimIndex_TopKQuery(tiered_index, query_0, k, nullptr, BY_SCORE);
    ASSERT_TRUE(res->results.empty());
    ASSERT_EQ(VecSimQueryReply_GetCode(res), VecSim_QueryReply_TimedOut);
    VecSimQueryReply_Free(res);

    // Set timeout callback to return 1 after n checks (will fail while querying the SVS index).
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
    ASSERT_TRUE(res->results.empty());
    ASSERT_EQ(VecSimQueryReply_GetCode(res), VecSim_QueryReply_TimedOut);
    VecSimQueryReply_Free(res);
    // Make sure we didn't get the timeout in the flat index.
    checks_in_flat = flat_index->indexSize(); // Reset the counter.
    res = VecSimIndex_TopKQuery(flat_index, query_0, k, &qparams, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_GetCode(res), VecSim_QueryReply_OK);
    VecSimQueryReply_Free(res);

    // Clean up.
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; });
}

TYPED_TEST(SVSTieredIndexTestBasic, KNNSearchCosine) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    const size_t dim = 128;
    const size_t n = 100;

    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_Cosine};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, n);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();

    for (size_t i = 1; i <= n; i++) {
        TEST_DATA_T f[dim];
        f[0] = (TEST_DATA_T)i / n;
        for (size_t j = 1; j < dim; j++) {
            f[j] = 1.0;
        }
        VecSimIndex_AddVector(tiered_index, f, i);
    }

    mock_thread_pool.thread_pool_join();

    ASSERT_EQ(VecSimIndex_IndexSize(tiered_index), n);
    // Verify that vectors were moved to SVS as expected
    auto sz_f = tiered_index->GetFlatIndex()->indexSize();
    auto sz_b = tiered_index->GetBackendIndex()->indexSize();
    EXPECT_LE(sz_f, this->getUpdateThreshold());
    EXPECT_EQ(sz_f + sz_b, n);

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, 1.0);

    // topK search will normalize the query so we keep the original data to
    // avoid normalizing twice.
    TEST_DATA_T normalized_query[dim];
    memcpy(normalized_query, query, dim * sizeof(TEST_DATA_T));
    VecSim_Normalize(normalized_query, dim, params.type);

    auto verify_res = [&](size_t id, double score, size_t result_rank) {
        ASSERT_EQ(id, (n - result_rank));
        TEST_DATA_T expected_score = tiered_index->getDistanceFrom_Unsafe(id, normalized_query);
        // Verify that abs difference between the actual and expected score is at most 1/10^5.
        ASSERT_NEAR((TEST_DATA_T)score, expected_score, 1e-5f);
    };
    runTopKSearchTest(tiered_index, query, 10, verify_res);
}

TYPED_TEST(SVSTieredIndexTest, deleteVector) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .multi = TypeParam::isMulti(),
                        .num_threads = 1};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1, 1);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    labelType vec_label = 0;
    // Delete from an empty index.
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 0);

    // Create a vector and add it to the tiered index (expect it to go into the flat buffer).
    TEST_DATA_T vector[dim];
    GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
    VecSimIndex_AddVector(tiered_index, vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);

    // Remove vector from flat buffer.
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 1);
    ASSERT_EQ(tiered_index->indexSize(), 0);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);

    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 0);

    // Create a vector and add it to SVS in the tiered index.
    VecSimIndex_AddVector(tiered_index->GetBackendIndex(), vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);

    // Remove from main index.
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 1);
    ASSERT_EQ(tiered_index->indexLabelCount(), 0);
    ASSERT_EQ(tiered_index->indexSize(), 0);

    // Re-insert a deleted label with a different vector.
    TEST_DATA_T new_vec_val = 2.0;
    GenerateVector<TEST_DATA_T>(vector, dim, new_vec_val);
    VecSimIndex_AddVector(tiered_index, vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);

    // Move the vector to SVS by executing the insert job.
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
    // Scalar quantization accuracy is insufficient for this check.
    if (!this->isFallbackToSQ()) {
        // Check that the distance from the deleted vector (of zeros) to the label is the distance
        // to the new vector (L2 distance).
        TEST_DATA_T deleted_vector[dim];
        GenerateVector<TEST_DATA_T>(deleted_vector, dim, 0);
        ASSERT_EQ(
            tiered_index->GetBackendIndex()->getDistanceFrom_Unsafe(vec_label, deleted_vector),
            dim * pow(new_vec_val, 2));
    }
}

TYPED_TEST(SVSTieredIndexTestBasic, deleteVectorMulti) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .multi = true,
                        .num_threads = 1};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1, 1);
    auto allocator = tiered_index->getAllocator();

    // Test some more scenarios that are relevant only for multi value index.
    labelType vec_label = 0;
    labelType other_vec_val = 2.0;
    idType invalidJobsCounter = 0;
    // Create a vector and add it to SVS in the tiered index.
    TEST_DATA_T vector[dim];
    GenerateVector<TEST_DATA_T>(vector, dim, vec_label);
    VecSimIndex_AddVector(tiered_index->GetBackendIndex(), vector, vec_label);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);

    // Test deleting a label for which one of its vector's is in the flat index while the
    // second one is in SVS.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, other_vec_val);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->indexSize(), 2);
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 2);
    ASSERT_EQ(tiered_index->indexSize(), 0);
    ASSERT_EQ(tiered_index->indexLabelCount(), 0);
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Test deleting a label for which both of its vector's is in the flat index.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, vec_label);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, other_vec_val);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 2);
    ASSERT_EQ(tiered_index->indexSize(), 2);
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 2);
    ASSERT_EQ(tiered_index->indexLabelCount(), 0);
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Test deleting a label for which both of its vector's is in SVS index.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, vec_label);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, other_vec_val);
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 2);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->deleteVector(vec_label), 2);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexLabelCount(), 0);
}

TYPED_TEST(SVSTieredIndexTest, manageIndexOwnership) {

    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .multi = TypeParam::isMulti()};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    ASSERT_INDEX(tiered_index);

    // Get the allocator from the tiered index.
    auto allocator = tiered_index->getAllocator();

    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);
    size_t initial_mem = allocator->getAllocationSize();

    // Create a dummy job callback that insert one vector to the underline SVS index.
    auto dummy_job = [](AsyncJob *job) {
        auto *my_index = reinterpret_cast<TieredSVSIndex<TEST_DATA_T> *>(job->index);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        size_t dim = 4;
        TEST_DATA_T vector[dim];
        GenerateVector<TEST_DATA_T>(vector, dim);
        my_index->GetBackendIndex()->addVector(vector, my_index->GetBackendIndex()->indexSize());
    };

    std::atomic_int successful_executions(0);
    auto job1 = new (allocator) AsyncJob(allocator, SVS_BATCH_UPDATE_JOB, dummy_job, tiered_index);
    auto job2 = new (allocator) AsyncJob(allocator, SVS_BATCH_UPDATE_JOB, dummy_job, tiered_index);

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

TYPED_TEST(SVSTieredIndexTest, parallelSearch) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t dim = 4;
    size_t k = 10;
    size_t n = 1000;

    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .search_window_size = n,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Forcibly set the training threshold to a value that will trigger the update job
    // after inserting a half of vectors.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, n / 2);
    ASSERT_INDEX(tiered_index);

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

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
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
    EXPECT_EQ(tiered_index->GetFlatIndex()->indexSize(), n);
    EXPECT_EQ(tiered_index->GetBackendIndex()->indexSize(), 0);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    // All the vectors are already in the tiered index, so we expect to find the expected
    // results from the get-go.
    mock_thread_pool.init_threads();
    mock_thread_pool.thread_pool_join();

    EXPECT_EQ(tiered_index->GetBackendIndex()->indexSize(), n);
    EXPECT_EQ(tiered_index->GetBackendIndex()->indexLabelCount(), n_labels);
    EXPECT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(SVSTieredIndexTest, parallelInsertSearch) {
    size_t dim = 4;
    size_t k = 10;
    size_t n = 1000;

    size_t block_size = n / 100;

    // Create TieredSVS index instance with a mock queue.
    size_t n_labels = TypeParam::isMulti() ? n / 25 : n;
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .blockSize = block_size,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Forcibly set the training threshold to a value that will trigger the update job
    // after inserting a half of vectors.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, n / 2);
    ASSERT_INDEX(tiered_index);
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

    // Verify that vectors were moved to SVS as expected
    auto sz_f = tiered_index->GetFlatIndex()->indexSize();
    auto sz_b = tiered_index->GetBackendIndex()->indexSize();
    EXPECT_LE(sz_f, this->getUpdateThreshold());
    EXPECT_EQ(sz_f + sz_b, n);
    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(SVSTieredIndexTestBasic, MergeMulti) {
    size_t dim = 4;

    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = true,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    auto svs_index = tiered_index->GetBackendIndex();
    auto flat_index = tiered_index->GetFlatIndex();

    // Insert vectors with label 0 to SVS only.
    GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, 0, 1);
    GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, 0, 2);
    // Insert vectors with label 1 to flat buffer only.
    GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, 1, 0);
    GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, 1, 1);
    GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, 1, 2);
    // Insert DIFFERENT vectors with label 2 to both SVS and flat buffer.
    GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, 2, 0);
    GenerateAndAddVector<TEST_DATA_T>(flat_index, dim, 2, 1);

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, 0);

    // Search in the tiered index for more vectors than it has. Merging the results from the two
    // indexes should result in a list of unique vectors, even if the scores of the duplicates are
    // different.
    runTopKSearchTest(tiered_index, query, 5, [](size_t _, double __, size_t ___) {});
}

TYPED_TEST(SVSTieredIndexTest, testSizeEstimation) {
    size_t dim = 128;
#if HAVE_SVS_LVQ
    // SVS block sizes always rounded to a power of 2
    // This why, in case of quantization, actual block size can be differ than requested
    // In addition, block size to be passed to graph and dataset counted in bytes,
    // converted then to a number of elements.
    // IMHO, would be better to always interpret block size to a number of elements
    // rather than conversion to-from number of bytes
    auto quantBits = TypeParam::get_quant_bits();
    if (quantBits != VecSimSvsQuant_NONE && !this->isFallbackToSQ()) {
        // Extra data in LVQ vector
        const auto lvq_vector_extra = sizeof(svs::quantization::lvq::ScalarBundle);
        dim -= (lvq_vector_extra * 8) / TypeParam::get_quant_bits();
    }
#endif
    size_t n = DEFAULT_BLOCK_SIZE;
    size_t graph_degree = 31; // power of 2 - 1
    size_t bs = DEFAULT_BLOCK_SIZE;

    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .blockSize = bs,
        .graph_max_degree = graph_degree,
    };
    VecSimParams vecsim_svs_params = CreateParams(svs_params);

    auto mock_thread_pool = tieredIndexMock();
    // Forcibly trigger first update job after inserting n vectors, then for every one vector.
    auto tiered_params = this->CreateTieredSVSParams(vecsim_svs_params, mock_thread_pool, n, 1);
    VecSimParams params = CreateParams(tiered_params);
    auto *index = VecSimIndex_New(&params);
    mock_thread_pool.ctx->index_strong_ref.reset(index);
    mock_thread_pool.init_threads();

    auto allocator = index->getAllocator();

    size_t initial_size_estimation = VecSimIndex_EstimateInitialSize(&params);

    ASSERT_EQ(initial_size_estimation, index->getAllocationSize());

    // Add vectors up to initial capacity (initial capacity == block size).
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    mock_thread_pool.thread_pool_wait();

    // Estimate memory delta for filling up the first block and adding another block.
    size_t estimation = VecSimIndex_EstimateElementSize(&params) * bs;

    size_t before = index->getAllocationSize();
    GenerateAndAddVector<TEST_DATA_T>(index, dim, bs + n, bs + n);
    mock_thread_pool.thread_pool_join();
    size_t actual = index->getAllocationSize() - before;

    auto tiered_index = this->CastToTieredSVS(index);

    // Flat index should be empty, hence the index size includes only svs size.
    EXPECT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    EXPECT_EQ(tiered_index->GetBackendIndex()->indexSize(), n + 1);

    // We added n + 1 vectors
    EXPECT_EQ(index->indexSize(), n + 1);

    EXPECT_EQ(index->indexCapacity(), tiered_index->GetBackendIndex()->indexCapacity());

    // We check that the actual size is within 1% of the estimation.
    EXPECT_GE(estimation, actual * 0.99);
    EXPECT_LE(estimation, actual * 1.01);
}

TYPED_TEST(SVSTieredIndexTest, parallelInsertAdHoc) {
    size_t dim = 4;
    size_t n = 1000;

    size_t block_size = n / 100;

    // Create TieredSVS index instance with a mock queue.
    size_t n_labels = n;
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .blockSize = block_size,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    ASSERT_INDEX(tiered_index);
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
        VecSimTieredIndex_AcquireSharedLocks(search_job->index);
        ASSERT_EQ(0, VecSimIndex_GetDistanceFrom_Unsafe(search_job->index, label, query));
        VecSimTieredIndex_ReleaseSharedLocks(search_job->index);
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

    tiered_index->scheduleSVSIndexUpdate();
    mock_thread_pool.thread_pool_join();

    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(tiered_index->GetBackendIndex()->indexSize(), n);
    EXPECT_EQ(tiered_index->GetBackendIndex()->indexLabelCount(), n_labels);
    EXPECT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

// A set of lambdas that determine whether a vector should be inserted to the
// SVS index (returns true) or to the flat index (returns false).
inline constexpr std::array<std::pair<std::string_view, bool (*)(size_t, size_t)>, 11> lambdas = {{
    {"100% SVS,   0% FLAT ", [](size_t idx, size_t n) -> bool { return 1; }},
    {" 50% SVS,  50% FLAT ", [](size_t idx, size_t n) -> bool { return idx % 2; }},
    {"  0% SVS, 100% FLAT ", [](size_t idx, size_t n) -> bool { return 0; }},
    {" 90% SVS,  10% FLAT ", [](size_t idx, size_t n) -> bool { return idx % 10; }},
    {" 10% SVS,  90% FLAT ", [](size_t idx, size_t n) -> bool { return !(idx % 10); }},
    {" 99% SVS,   1% FLAT ", [](size_t idx, size_t n) -> bool { return idx % 100; }},
    {"  1% SVS,  99% FLAT ", [](size_t idx, size_t n) -> bool { return !(idx % 100); }},
    {"first 10% are in SVS", [](size_t idx, size_t n) -> bool { return idx < (n / 10); }},
    {"first 10% are in FLAT", [](size_t idx, size_t n) -> bool { return idx >= (n / 10); }},
    {" last 10% are in FLAT", [](size_t idx, size_t n) -> bool { return idx < (9 * n / 10); }},
    {" last 10% are in SVS", [](size_t idx, size_t n) -> bool { return idx >= (9 * n / 10); }},
}};

TYPED_TEST(SVSTieredIndexTest, BatchIterator) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t d = 4;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
    };
    VecSimParams params = CreateParams(svs_params);

    for (auto &lambda : lambdas) {
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredSVSIndex(params, mock_thread_pool);
        ASSERT_INDEX(tiered_index);
        auto allocator = tiered_index->getAllocator();

        auto *svs = tiered_index->GetBackendIndex();
        auto *flat = tiered_index->GetFlatIndex();

        // For every i, add the vector (i,i,i,i) under the label i.
        for (size_t i = 0; i < n; i++) {
            auto cur = decider(i, n) ? svs : flat;
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

TYPED_TEST(SVSTieredIndexTest, BatchIteratorReset) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t d = 4;
    size_t M = 8;
    size_t sws = 20;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .construction_window_size = sws,
        .search_window_size = sws,
    };
    VecSimParams params = CreateParams(svs_params);

    for (auto &lambda : lambdas) {
        // manually deconstruct the pair to avoid the clang error
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredSVSIndex(params, mock_thread_pool);
        ASSERT_INDEX(tiered_index);
        auto allocator = tiered_index->getAllocator();

        auto *svs = tiered_index->GetBackendIndex();
        auto *flat = tiered_index->GetFlatIndex();

        // For every i, add the vector (i,i,i,i) under the label i.
        for (size_t i = 0; i < n; i++) {
            auto cur = decider(i, n) ? svs : flat;
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

TYPED_TEST(SVSTieredIndexTest, BatchIteratorSize1) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t d = 4;
    size_t M = 8;
    size_t sws = 20;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .construction_window_size = sws,
        .search_window_size = sws,
    };
    VecSimParams params = CreateParams(svs_params);

    // for (auto &[decider_name, decider] : lambdas) { // TODO: not supported by clang < 16
    for (auto &lambda : lambdas) {
        // manually deconstruct the pair to avoid the clang error
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredSVSIndex(params, mock_thread_pool);
        ASSERT_INDEX(tiered_index);
        auto allocator = tiered_index->getAllocator();

        auto *svs = tiered_index->GetBackendIndex();
        auto *flat = tiered_index->GetFlatIndex();

        // For every i, add the vector (i,i,i,i) under the label `n_labels - (i % n_labels)`.
        for (size_t i = 0; i < n; i++) {
            auto cur = decider(i, n) ? svs : flat;
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

TYPED_TEST(SVSTieredIndexTest, BatchIteratorAdvanced) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t d = 4;
    size_t M = 8;
    size_t sws = 20;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .construction_window_size = sws,
    };
    VecSimParams params = CreateParams(svs_params);
    SVSRuntimeParams svsRuntimeParams = {.windowSize = sws};
    VecSimQueryParams query_params = CreateQueryParams(svsRuntimeParams);

    // for (auto &[decider_name, decider] : lambdas) { // TODO: not supported by clang < 16
    for (auto &lambda : lambdas) {
        // manually deconstruct the pair to avoid the clang error
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredSVSIndex(params, mock_thread_pool);
        ASSERT_INDEX(tiered_index);
        auto allocator = tiered_index->getAllocator();

        auto *svs = tiered_index->GetBackendIndex();
        auto *flat = tiered_index->GetFlatIndex();

        TEST_DATA_T query[d];
        GenerateVector<TEST_DATA_T>(query, d, n);

        VecSimBatchIterator *batchIterator =
            VecSimBatchIterator_New(tiered_index, query, &query_params);

        // Try to get results even though there are no vectors in the index.
        VecSimQueryReply *res = VecSimBatchIterator_Next(batchIterator, 10, BY_SCORE);
        ASSERT_EQ(VecSimQueryReply_Len(res), 0) << decider_name;
        VecSimQueryReply_Free(res);
        ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator)) << decider_name;

        // Insert one label and query again. The internal id will be 0.
        for (size_t j = 0; j < per_label; j++) {
            GenerateAndAddVector<TEST_DATA_T>(decider(n_labels, n) ? svs : flat, d, n_labels,
                                              n - j);
        }
        VecSimBatchIterator_Reset(batchIterator);
        res = VecSimBatchIterator_Next(batchIterator, 10, BY_SCORE);
        ASSERT_EQ(VecSimQueryReply_Len(res), 1) << decider_name;
        VecSimQueryReply_Free(res);
        ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator)) << decider_name;
        VecSimBatchIterator_Free(batchIterator);

        // Insert vectors to the index and re-create the batch iterator.
        for (size_t i = 1; i < n_labels; i++) {
            auto cur = decider(i, n) ? svs : flat;
            for (size_t j = 1; j <= per_label; j++) {
                GenerateAndAddVector<TEST_DATA_T>(cur, d, i, (i - 1) * per_label + j);
            }
        }
        ASSERT_EQ(VecSimIndex_IndexSize(tiered_index), n) << decider_name;
        batchIterator = VecSimBatchIterator_New(tiered_index, query, &query_params);

        // Try to get 0 results.
        res = VecSimBatchIterator_Next(batchIterator, 0, BY_SCORE);
        ASSERT_EQ(VecSimQueryReply_Len(res), 0) << decider_name;
        VecSimQueryReply_Free(res);

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
        ASSERT_EQ(VecSimQueryReply_Len(res), 0) << decider_name;
        VecSimQueryReply_Free(res);

        VecSimBatchIterator_Free(batchIterator);
    }
}

TYPED_TEST(SVSTieredIndexTest, BatchIteratorWithOverlaps) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t d = 4;
    size_t M = 8;
    size_t sws = 20;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .construction_window_size = sws,
        .search_window_size = sws,
    };
    VecSimParams params = CreateParams(svs_params);

    // for (auto &[decider_name, decider] : lambdas) { // TODO: not supported by clang < 16
    for (auto &lambda : lambdas) {
        // manually deconstruct the pair to avoid the clang error
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredSVSIndex(params, mock_thread_pool);
        ASSERT_INDEX(tiered_index);
        auto allocator = tiered_index->getAllocator();

        auto *svs = tiered_index->GetBackendIndex();
        auto *flat = tiered_index->GetFlatIndex();

        // For every i, add the vector (i,i,i,i) under the label i.
        size_t flat_count = 0;
        for (size_t i = 0; i < n; i++) {
            auto cur = decider(i, n) ? svs : flat;
            GenerateAndAddVector<TEST_DATA_T>(cur, d, i % n_labels, i);
            if (cur == flat) {
                flat_count++;
                // Add 10% of the vectors in FLAT to SVS as well.
                if (flat_count % 10 == 0) {
                    GenerateAndAddVector<TEST_DATA_T>(svs, d, i % n_labels, i);
                }
            }
        }
        // The index size should be 100-110% of n.
        ASSERT_LE(VecSimIndex_IndexSize(tiered_index), n * 1.1) << decider_name;
        ASSERT_GE(VecSimIndex_IndexSize(tiered_index), n) << decider_name;
        // The number of unique labels should be n_labels.
        // ASSERT_EQ(tiered_index->indexLabelCount(), n_labels) << decider_name;

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

TYPED_TEST(SVSTieredIndexTestBasic, BatchIteratorWithOverlaps_SpacialMultiCases) {
    size_t d = 4;

    std::shared_ptr<VecSimAllocator> allocator;
    VecSimIndex *svs, *flat;
    TEST_DATA_T query[d];
    VecSimBatchIterator *iterator;
    VecSimQueryReply *batch;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .multi = true,
    };
    VecSimParams params = CreateParams(svs_params);
    auto mock_thread_pool = tieredIndexMock();

    auto L2 = [&](size_t element) { return element * element * d; };

    // TEST 1:
    // first batch contains duplicates with different scores.
    auto tiered_index = this->CreateTieredSVSIndex(params, mock_thread_pool);
    allocator = tiered_index->getAllocator();
    svs = tiered_index->GetBackendIndex();
    flat = tiered_index->GetFlatIndex();

    GenerateAndAddVector<TEST_DATA_T>(flat, d, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(flat, d, 1, 1);
    GenerateAndAddVector<TEST_DATA_T>(flat, d, 2, 2);

    GenerateAndAddVector<TEST_DATA_T>(svs, d, 1, 3);
    GenerateAndAddVector<TEST_DATA_T>(svs, d, 0, 4);
    GenerateAndAddVector<TEST_DATA_T>(svs, d, 3, 5);

    ASSERT_EQ(tiered_index->indexLabelCount(), 4);

    GenerateVector<TEST_DATA_T>(query, d, 0);
    iterator = VecSimBatchIterator_New(tiered_index, query, nullptr);

    // batch size is 3 (the size of each index). Internally the tiered batch iterator will have to
    // handle the duplicates with different scores.
    ASSERT_TRUE(VecSimBatchIterator_HasNext(iterator));
    batch = VecSimBatchIterator_Next(iterator, 3, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(batch), 3);
    ASSERT_EQ(VecSimQueryResult_GetId(batch->results.data() + 0), 0);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch->results.data() + 0), L2(0));
    ASSERT_EQ(VecSimQueryResult_GetId(batch->results.data() + 1), 1);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch->results.data() + 1), L2(1));
    ASSERT_EQ(VecSimQueryResult_GetId(batch->results.data() + 2), 2);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch->results.data() + 2), L2(2));
    VecSimQueryReply_Free(batch);

    // we have 1 more label in the index. we expect the tiered batch iterator to return it only and
    // filter out the duplicates.
    ASSERT_TRUE(VecSimBatchIterator_HasNext(iterator));
    batch = VecSimBatchIterator_Next(iterator, 2, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(batch), 1);
    ASSERT_EQ(VecSimQueryResult_GetId(batch->results.data() + 0), 3);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch->results.data() + 0), L2(5));
    ASSERT_FALSE(VecSimBatchIterator_HasNext(iterator));
    VecSimQueryReply_Free(batch);
    // TEST 1 clean up.
    VecSimBatchIterator_Free(iterator);

    // TEST 2:
    // second batch contains duplicates (different scores) from the first batch.
    auto *ctx = new tieredIndexMock::IndexExtCtx(&mock_thread_pool);
    mock_thread_pool.reset_ctx(ctx);
    tiered_index = this->CreateTieredSVSIndex(params, mock_thread_pool);
    allocator = tiered_index->getAllocator();
    svs = tiered_index->GetBackendIndex();
    flat = tiered_index->GetFlatIndex();

    GenerateAndAddVector<TEST_DATA_T>(svs, d, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(svs, d, 1, 1);
    GenerateAndAddVector<TEST_DATA_T>(svs, d, 2, 2);
    GenerateAndAddVector<TEST_DATA_T>(svs, d, 3, 3);

    GenerateAndAddVector<TEST_DATA_T>(flat, d, 2, 0);
    GenerateAndAddVector<TEST_DATA_T>(flat, d, 3, 1);
    GenerateAndAddVector<TEST_DATA_T>(flat, d, 0, 2);
    GenerateAndAddVector<TEST_DATA_T>(flat, d, 1, 3);

    ASSERT_EQ(tiered_index->indexLabelCount(), 4);

    iterator = VecSimBatchIterator_New(tiered_index, query, nullptr);

    // ask for 2 results. The internal batch iterators will return 2 results: svs - [0, 1], flat -
    // [2, 3] so there are no duplicates.
    ASSERT_TRUE(VecSimBatchIterator_HasNext(iterator));
    batch = VecSimBatchIterator_Next(iterator, 2, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(batch), 2);
    ASSERT_EQ(VecSimQueryResult_GetId(batch->results.data() + 0), 0);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch->results.data() + 0), L2(0));
    ASSERT_EQ(VecSimQueryResult_GetId(batch->results.data() + 1), 2);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch->results.data() + 1), L2(0));
    VecSimQueryReply_Free(batch);

    // first batch contained 1 result from each index, so there is one leftover from each iterator.
    // Asking for 3 results will return additional 2 results from each iterator and the tiered batch
    // iterator will have to handle the duplicates that each iterator returned (both labels that
    // were returned in the first batch and duplicates in the current batch).
    ASSERT_TRUE(VecSimBatchIterator_HasNext(iterator));
    batch = VecSimBatchIterator_Next(iterator, 3, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(batch), 2);
    ASSERT_EQ(VecSimQueryResult_GetId(batch->results.data() + 0), 1);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch->results.data() + 0), L2(1));
    ASSERT_EQ(VecSimQueryResult_GetId(batch->results.data() + 1), 3);
    ASSERT_EQ(VecSimQueryResult_GetScore(batch->results.data() + 1), L2(1));
    ASSERT_FALSE(VecSimBatchIterator_HasNext(iterator));
    VecSimQueryReply_Free(batch);
    // TEST 2 clean up.
    VecSimBatchIterator_Free(iterator);
}

TYPED_TEST(SVSTieredIndexTest, parallelBatchIteratorSearch) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t dim = 4;
    size_t sws = 500;
    size_t n = 1000;
    size_t n_res_min = 3;  // minimum number of results to return per batch
    size_t n_res_max = 15; // maximum number of results to return per batch

    size_t per_label = TypeParam::isMulti() ? 5 : 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .search_window_size = sws,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    auto *svs = tiered_index->GetBackendIndex();
    auto *flat = tiered_index->GetFlatIndex();

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

    for (size_t i = 0; i < n; i++) {
        auto cur = i % 2 ? svs : flat;
        GenerateAndAddVector<TEST_DATA_T>(cur, dim, i % n_labels, i);

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
    EXPECT_EQ(svs->indexSize(), n / 2);
    EXPECT_EQ(flat->indexSize(), n / 2);

    mock_thread_pool.init_threads();
    mock_thread_pool.thread_pool_join();

    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(SVSTieredIndexTest, RangeSearch) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
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

    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .epsilon = 3.0 * per_label,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    size_t cur_memory_usage;

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();
    ASSERT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    auto svs_index = tiered_index->GetBackendIndex();
    auto flat_index = tiered_index->GetFlatIndex();

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
        GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, i / per_label, i);
    }
    ASSERT_EQ(tiered_index->indexSize(), (n + 1) / 2);
    ASSERT_EQ(tiered_index->indexSize(), svs_index->indexSize());

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
    ASSERT_EQ(tiered_index->indexSize(), svs_index->indexSize() + flat_index->indexSize());

    cur_memory_usage = allocator->getAllocationSize();
    // Search for `range` so all the vectors will be from the SVS index.
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
        GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, i / per_label, i);
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

    VecSimQueryReply *res;
    // Add a vector to the SVS index so there will be a reason to query it.
    GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, n, n);

    // Set timeout callback to always return 1 (will fail while querying the flat buffer).
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out

    res = VecSimIndex_RangeQuery(tiered_index, query_0, range, nullptr, BY_ID);
    ASSERT_EQ(VecSimQueryReply_GetCode(res), VecSim_QueryReply_TimedOut);
    VecSimQueryReply_Free(res);

    // Set timeout callback to return 1 after n checks (will fail while querying the SVS index).
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
    ASSERT_EQ(VecSimQueryReply_GetCode(res), VecSim_QueryReply_TimedOut);
    VecSimQueryReply_Free(res);
    // Make sure we didn't get the timeout in the flat index.
    checks_in_flat = flat_index->indexSize(); // Reset the counter.
    res = VecSimIndex_RangeQuery(flat_index, query_0, range, &qparams, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_GetCode(res), VecSim_QueryReply_OK);
    VecSimQueryReply_Free(res);

    // Check again with BY_ID.
    checks_in_flat = flat_index->indexSize(); // Reset the counter.
    res = VecSimIndex_RangeQuery(tiered_index, query_0, range, &qparams, BY_ID);
    ASSERT_EQ(VecSimQueryReply_GetCode(res), VecSim_QueryReply_TimedOut);
    VecSimQueryReply_Free(res);
    // Make sure we didn't get the timeout in the flat index.
    checks_in_flat = flat_index->indexSize(); // Reset the counter.
    res = VecSimIndex_RangeQuery(flat_index, query_0, range, &qparams, BY_ID);
    ASSERT_EQ(VecSimQueryReply_GetCode(res), VecSim_QueryReply_OK);
    VecSimQueryReply_Free(res);

    // Clean up.
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; });
}

TYPED_TEST(SVSTieredIndexTest, parallelRangeSearch) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t dim = 4;
    size_t k = 11;
    size_t n = 1000;

    size_t per_label = TypeParam::isMulti() ? 10 : 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .multi = TypeParam::isMulti(),
        .epsilon = double(dim * k * k),
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();
    EXPECT_EQ(mock_thread_pool.ctx->index_strong_ref.use_count(), 1);

    auto *svs = tiered_index->GetBackendIndex();
    auto *flat = tiered_index->GetFlatIndex();

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

    for (size_t i = 0; i < n; i++) {
        auto cur = i % 2 ? svs : flat;
        GenerateAndAddVector<TEST_DATA_T>(cur, dim, i % n_labels, i);

        // Add a search job. Make sure the query element is between k and n_labels - k.
        auto query = (TEST_DATA_T *)allocator->allocate(dim * sizeof(TEST_DATA_T));
        GenerateVector<TEST_DATA_T>(query, dim, ((n - i) % (n_labels - (2 * k))) + k);
        auto search_job = new (allocator) tieredIndexMock::SearchJobMock(
            allocator, parallel_range_search, tiered_index, k, query, n, dim, &successful_searches);
        tiered_index->submitSingleJob(search_job);
    }

    EXPECT_EQ(tiered_index->indexSize(), n);
    EXPECT_EQ(tiered_index->indexLabelCount(), n_labels);
    EXPECT_EQ(svs->indexSize(), n / 2);
    EXPECT_EQ(flat->indexSize(), n / 2);

    mock_thread_pool.init_threads();
    mock_thread_pool.thread_pool_join();

    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(SVSTieredIndexTestBasic, overwriteVectorBasic) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .multi = false,
                        .num_threads = 1};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1, 1);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    TEST_DATA_T val = 1.0;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 0, val);
    // Overwrite label 0 (in the flat buffer) with a different value.
    val = 2.0;
    TEST_DATA_T overwritten_vec[] = {val, val, val, val};
    ASSERT_EQ(tiered_index->addVector(overwritten_vec, 0), 0);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(0, overwritten_vec), 0);

    // Validate that jobs were created properly - first job should be invalid after overwrite,
    // the second should be a pending insert job.
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 1);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, SVS_BATCH_UPDATE_JOB);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->isValid, true);
    mock_thread_pool.thread_iteration();

    // Ingest vector into SVS, and then overwrite it.
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    val = 3.0;
    overwritten_vec[0] = overwritten_vec[1] = overwritten_vec[2] = overwritten_vec[3] = val;
    ASSERT_EQ(tiered_index->addVector(overwritten_vec, 0), 0);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    // Overriding vector in tiered index should remove the vector from SVS to avoid duplicates.
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(0, overwritten_vec), 0);

    // Ingest the updated vector to SVS.
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(0, overwritten_vec), 0);
}

TYPED_TEST(SVSTieredIndexTestBasic, overwriteVectorAsync) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};
    VecSimParams svs_params = CreateParams(params);
    for (size_t updateThreshold : {n, size_t{1}}) {
        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool,
                                                        updateThreshold, updateThreshold);
        ASSERT_INDEX(tiered_index);
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

        EXPECT_LE(tiered_index->GetFlatIndex()->indexSize(), updateThreshold);
        EXPECT_EQ(tiered_index->indexLabelCount(), n);
    }
}

TYPED_TEST(SVSTieredIndexTest, testInfo) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .multi = TypeParam::isMulti()};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1, 1);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    VecSimIndexDebugInfo info = tiered_index->debugInfo();
    EXPECT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVS);
    EXPECT_EQ(info.commonInfo.indexSize, 0);
    EXPECT_EQ(info.commonInfo.indexLabelCount, 0);
    EXPECT_EQ(info.commonInfo.memory, tiered_index->getAllocationSize());
    EXPECT_EQ(info.commonInfo.basicInfo.isMulti, TypeParam::isMulti());
    EXPECT_EQ(info.commonInfo.basicInfo.dim, dim);
    EXPECT_EQ(info.commonInfo.basicInfo.metric, VecSimMetric_L2);
    EXPECT_EQ(info.commonInfo.basicInfo.type, TypeParam::get_index_type());
    EXPECT_EQ(info.commonInfo.basicInfo.blockSize, DEFAULT_BLOCK_SIZE);
    VecSimIndexDebugInfo frontendIndexInfo = tiered_index->GetFlatIndex()->debugInfo();
    VecSimIndexDebugInfo backendIndexInfo = tiered_index->GetBackendIndex()->debugInfo();

    compareCommonInfo(info.tieredInfo.frontendCommonInfo, frontendIndexInfo.commonInfo);
    compareFlatInfo(info.tieredInfo.bfInfo, frontendIndexInfo.bfInfo);
    compareCommonInfo(info.tieredInfo.backendCommonInfo, backendIndexInfo.commonInfo);
    compareSVSInfo(info.tieredInfo.backendInfo.svsInfo, backendIndexInfo.svsInfo);

    EXPECT_EQ(info.commonInfo.memory, info.tieredInfo.management_layer_memory +
                                          backendIndexInfo.commonInfo.memory +
                                          frontendIndexInfo.commonInfo.memory);
    EXPECT_EQ(info.tieredInfo.backgroundIndexing, false);
    // Validate tiered svs info fields
    EXPECT_EQ(info.tieredInfo.specificTieredBackendInfo.svsTieredInfo.trainingTriggerThreshold, 1);
    EXPECT_EQ(info.tieredInfo.specificTieredBackendInfo.svsTieredInfo.updateTriggerThreshold, 1);
    EXPECT_EQ(info.tieredInfo.specificTieredBackendInfo.svsTieredInfo.updateJobWaitTime,
              SVS_DEFAULT_UPDATE_JOB_WAIT_TIME);
    EXPECT_EQ(info.tieredInfo.specificTieredBackendInfo.svsTieredInfo.indexUpdateScheduled, false);

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
    info = tiered_index->debugInfo();

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
    EXPECT_EQ(info.tieredInfo.specificTieredBackendInfo.svsTieredInfo.indexUpdateScheduled, true);

    mock_thread_pool.thread_iteration();
    info = tiered_index->debugInfo();

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
        info = tiered_index->debugInfo();

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
    info = tiered_index->debugInfo();

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

TYPED_TEST(SVSTieredIndexTest, testInfoIterator) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .multi = TypeParam::isMulti()};

    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1, 1);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    GenerateAndAddVector(tiered_index, dim, 1, 1);
    VecSimIndexDebugInfo info = tiered_index->debugInfo();
    VecSimIndexDebugInfo frontendIndexInfo = tiered_index->GetFlatIndex()->debugInfo();
    VecSimIndexDebugInfo backendIndexInfo = tiered_index->GetBackendIndex()->debugInfo();

    VecSimDebugInfoIterator *infoIterator = tiered_index->debugInfoIterator();
    EXPECT_EQ(infoIterator->numberOfFields(), 14);

    while (infoIterator->hasNext()) {
        VecSim_InfoField *infoField = VecSimDebugInfoIterator_NextField(infoIterator);

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
            compareSVSIndexInfoToIterator(backendIndexInfo, infoField->fieldValue.iteratorValue);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::TIERED_BUFFER_LIMIT_STRING)) {
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.tieredInfo.bufferLimit);
        } else {
            FAIL();
        }
    }
    VecSimDebugInfoIterator_Free(infoIterator);
}

TYPED_TEST(SVSTieredIndexTest, writeInPlaceMode) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    const size_t updateThreshold = 2;

    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .multi = TypeParam::isMulti()};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index =
        this->CreateTieredSVSIndex(svs_params, mock_thread_pool, updateThreshold, updateThreshold);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    VecSim_SetWriteMode(VecSim_WriteInPlace);
    // Validate that the first vector was buffered in flat index.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 10, 10);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);

    // Validate that the second vector causes moving to the SVS index.
    labelType vec_label = 0;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, 0);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 2);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);

    // Overwrite inplace - only in single-value mode
    if (!TypeParam::isMulti()) {
        TEST_DATA_T overwritten_vec[] = {1, 1, 1, 1};
        tiered_index->addVector(overwritten_vec, vec_label);
        ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 2);
        ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
        ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(vec_label, overwritten_vec), 0);
    }
    // Validate that the vector is removed in place.
    tiered_index->deleteVector(vec_label);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
}

TYPED_TEST(SVSTieredIndexTest, switchWriteModes) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 500;
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .multi = TypeParam::isMulti()};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Forcibly trigger the first update job for at least half of the vectors.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, n / 2);
    ASSERT_INDEX(tiered_index);
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
    EXPECT_LE(tiered_index->GetBackendIndex()->indexSize(), n);
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        for (size_t j = 0; j < dim; j++) {
            vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
        }
        VecSimIndex_AddVector(tiered_index, vector, i % n_labels + n_labels);
    }
    mock_thread_pool.thread_pool_join();
    EXPECT_GE(tiered_index->GetBackendIndex()->indexSize(), 2 * n - this->getUpdateThreshold());

    // Now delete the last n inserted vectors of the index using async jobs.
    VecSim_SetWriteMode(VecSim_WriteAsync);
    mock_thread_pool.init_threads();
    for (size_t i = 0; i < n_labels; i++) {
        VecSimIndex_DeleteVector(tiered_index, n_labels + i);
    }

    // Insert INPLACE another n vector (instead of the ones that were deleted).
    VecSim_SetWriteMode(VecSim_WriteInPlace);
    auto svs_index = tiered_index->GetBackendIndex();
    // Run twice, at first run we insert non-existing labels, in the second run we overwrite them
    // (for single-value index only).
    for (auto overwrite : {0, 1}) {
        for (size_t i = 0; i < n; i++) {
            TEST_DATA_T vector[dim];
            for (size_t j = 0; j < dim; j++) {
                vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
            }
            labelType cur_label = i % n_labels + n_labels;
            EXPECT_EQ(tiered_index->addVector(vector, cur_label),
                      TypeParam::isMulti() ? 1 : 1 - overwrite);
            // Quantization has limited accuaracy, so we need to check the relative error.
            // If quantization is enabled, we allow a larger relative error.
            double abs_err = TypeParam::get_quant_bits() != VecSimSvsQuant_NONE ? 1e-2 : 1.e-6;
            // Run a query over svs index and see that we only receive ids with label < n_labels+i
            // (the label that we just inserted), and the first result should be this vector
            // (unless it is unreachable)
            auto ver_res = [&](size_t res_label, double score, size_t index) {
                if (index == 0) {
                    if (res_label == cur_label) {
                        EXPECT_NEAR(score, 0, abs_err);
                    } else {
                        tiered_index->acquireSharedLocks();
                        ASSERT_EQ(svs_index->getDistanceFrom_Unsafe(cur_label, vector), 0);
                        tiered_index->releaseSharedLocks();
                    }
                }
                if (!overwrite) {
                    ASSERT_LE(res_label, i + n_labels);
                }
            };
            runTopKSearchTest(svs_index, vector, 10, ver_res);
        }
    }

    mock_thread_pool.thread_pool_join();
    // Verify that vectors were moved to SVS as expected
    auto sz_f = tiered_index->GetFlatIndex()->indexSize();
    auto sz_b = tiered_index->GetBackendIndex()->indexSize();
    EXPECT_LE(sz_f, this->getUpdateThreshold());
    if (TypeParam::isMulti()) {
        ASSERT_EQ(tiered_index->indexLabelCount(), 2 * n_labels);
    } else {
        EXPECT_EQ(sz_f + sz_b, 2 * n_labels);
    }
}

TYPED_TEST(SVSTieredIndexTestBasic, preferAdHocOptimization) {
    size_t dim = 4;

    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Create tiered index with buffer limit set to 0.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    auto svs = tiered_index->GetBackendIndex();
    auto flat = tiered_index->GetFlatIndex();

    // Insert 5 vectors to the main index.
    for (size_t i = 0; i < 5; i++) {
        GenerateAndAddVector<TEST_DATA_T>(svs, dim, i, i);
    }
    // Sanity check. Should choose as SVS.
    ASSERT_EQ(tiered_index->preferAdHocSearch(5, 5, true), svs->preferAdHocSearch(5, 5, true));

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

TYPED_TEST(SVSTieredIndexTestBasic, runGCAPI) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t threshold = 1024;
    const size_t n = threshold * 3;
    SVSParams params = {.type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, threshold);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    // Insert three block of vectors directly to SVS.
    std::srand(10); // create pseudo random generator with any arbitrary seed.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        for (size_t j = 0; j < dim; j++) {
            vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
        }
        VecSimIndex_AddVector(tiered_index->GetBackendIndex(), vector, i);
    }

    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), n);

    // Delete all the vectors and wait for the thread pool to finish running the update jobs.
    for (size_t i = 0; i < threshold; i++) {
        tiered_index->deleteVector(i);
    }
    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    mock_thread_pool.thread_pool_join();

    ASSERT_EQ(tiered_index->indexSize(), n - threshold);
    ASSERT_EQ(tiered_index->GetSVSIndex()->indexStorageSize(), n);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);
    auto size_before_gc = tiered_index->getAllocationSize();

    // Run the GC API call, expect that we will clean up the SVS index.
    VecSimTieredIndex_GC(tiered_index);
    ASSERT_EQ(tiered_index->indexSize(), n - threshold);
    ASSERT_EQ(tiered_index->GetSVSIndex()->indexStorageSize(), n - threshold);
    auto size_after_gc = tiered_index->getAllocationSize();
    // Expect that the size of the index was reduced.
    ASSERT_LT(size_after_gc, size_before_gc);
}

TYPED_TEST(SVSTieredIndexTestBasic, switchDeleteModes) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 16;
    size_t n = 1000;
    size_t update_threshold = 10;
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = 100,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, update_threshold,
                                                    update_threshold);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();

    // Create and insert vectors one by one inplace.
    VecSim_SetWriteMode(VecSim_WriteInPlace);
    std::srand(10); // create pseudo random generator with any arbitrary seed.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        for (size_t j = 0; j < dim; j++) {
            vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
        }
        VecSimIndex_AddVector(tiered_index, vector, i);
    }
    EXPECT_EQ(tiered_index->GetBackendIndex()->indexSize(), n);

    // Update vectors while changing the write mode.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        for (size_t j = 0; j < dim; j++) {
            vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
        }
        EXPECT_EQ(tiered_index->addVector(vector, i), 0);
        if (i % 10 == 0) {
            // Change mode every 10 vectors.
            auto next_mode = tiered_index->getWriteMode() == VecSim_WriteInPlace
                                 ? VecSim_WriteAsync
                                 : VecSim_WriteInPlace;
            VecSim_SetWriteMode(next_mode);
        }
    }

    mock_thread_pool.thread_pool_join();
    // Verify that vectors were moved to SVS as expected
    auto sz_f = tiered_index->GetFlatIndex()->indexSize();
    auto sz_b = tiered_index->GetBackendIndex()->indexSize();
    EXPECT_LE(sz_f, update_threshold);
    EXPECT_EQ(sz_f + sz_b, n);
}

TYPED_TEST(SVSTieredIndexTestBasic, testSwapJournalSingle) {
    // Create TieredSVS index instance with a mock queue.
    const size_t dim = 4;
    const size_t n = 15;

    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Forcibly set the training threshold to a value that will trigger the update job
    // for first vector only.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1, n * 100);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    // Add n vectors to the index.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        GenerateVector<TEST_DATA_T>(vector, dim, i);
        tiered_index->addVector(vector, i);
    }

    // Pause the update job after svs index update
    std::mutex mtx;
    bool added_to_svs = false;
    bool continue_job = false;
    std::condition_variable cv;

    auto tracing_callback = [&]() {
        {
            std::unique_lock lock(mtx);
            added_to_svs = true; // Indicate that we are waiting for the update job to start.
        }
        cv.notify_one(); // Notify that the update job has started.
        {
            std::unique_lock lock(mtx);
            cv.wait(lock, [&] { return continue_job; }); // Wait until we continue.
        }
    };
    tiered_index->registerTracingCallback("UpdateJob::after_add_to_svs", tracing_callback);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();

    {
        // IMPORTANT!: Do not use ASSERT here, as it will not release the mutex and will cause a
        // deadlock. Use EXPECT instead, so we can continue the test even if the condition is not
        // met.

        // Wait for the update job to start.
        std::unique_lock lock(mtx);
        EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(100), [&] { return added_to_svs; }));

        // update job paused, we have vectors 0-(n-1) in the index, let's do index modifications

        // Remove vector label=n-2, it is copied to backend index.
        EXPECT_EQ(VecSimIndex_DeleteVector(tiered_index, n - 2), 2);
        // Update vector label=1.
        EXPECT_EQ(GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 1, 10), 0);
        // Add a new vector
        EXPECT_EQ(GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, n, n), 1);
        // Add another one
        EXPECT_EQ(GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, n + 1, n + 1), 1);
        // Remove vector label=0, it is copied to backend index.
        EXPECT_EQ(VecSimIndex_DeleteVector(tiered_index, 0), 2);
        // Remove the last vector copied to backend index
        EXPECT_EQ(VecSimIndex_DeleteVector(tiered_index, n - 1), 2);
        // Update vector label=2.
        EXPECT_EQ(GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 2, 20), 0);
        // Remove vector label=2.
        EXPECT_EQ(VecSimIndex_DeleteVector(tiered_index, 2), 1);
        // Remove vector label=n - in flat only
        EXPECT_EQ(VecSimIndex_DeleteVector(tiered_index, n), 1);
        // Add vector (n-1) again
        EXPECT_EQ(GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, n - 1, (n - 1) * 10), 1);

        continue_job = true; // Indicate that we can continue the update job.
    }

    // continue the update job
    cv.notify_all(); // Notify that we can continue.

    mock_thread_pool.thread_pool_join();

    // For single-value index, following vectors should be in the index:
    // 0:deleted, 1: 10, 2: deleted, 3:3, ..., n-2:deleted n-1: 10(n-1), n+1: n+1;
    // total: n-2 vectors and labels
    ASSERT_EQ(tiered_index->indexSize(), n - 2);
    ASSERT_EQ(tiered_index->indexLabelCount(), n - 2);

    // Backend index: 0:deleted, 1:deleted, 2:deleted, 3:3, ..., n-2:deleted, n-1:deleted;
    // total: n-5
    EXPECT_EQ(tiered_index->GetBackendIndex()->indexSize(), n - 5);
    // Frontend index: 1:10, n-1:10(n-1), n+1:n+1
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 3);

    double abs_err = 1e-2; // Allow a larger relative error for quantization.
    TEST_DATA_T expected_vector[dim];

    // Vector label 0 - deleted
    GenerateVector<TEST_DATA_T>(expected_vector, dim, 0);
    ASSERT_TRUE(std::isnan(tiered_index->getDistanceFrom_Unsafe(0, expected_vector)));

    // Vector label n-2 - deleted
    GenerateVector<TEST_DATA_T>(expected_vector, dim, 0);
    ASSERT_TRUE(std::isnan(tiered_index->getDistanceFrom_Unsafe(n - 2, expected_vector)));

    // Vector label=1, with value 10 should be in the flat index.
    GenerateVector<TEST_DATA_T>(expected_vector, dim, 10);
    ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(1, expected_vector), 0, abs_err);

    // Vector label 2 - deleted
    GenerateVector<TEST_DATA_T>(expected_vector, dim, 2);
    ASSERT_TRUE(std::isnan(tiered_index->getDistanceFrom_Unsafe(2, expected_vector)));

    // Vectors labels [3,n-3] - unchanged
    for (size_t i = 3; i < n - 2; i++) {
        GenerateVector<TEST_DATA_T>(expected_vector, dim, i);
        ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(i, expected_vector), 0, abs_err);
    }

    // Vector label n-1 - deleted and re-added
    GenerateVector<TEST_DATA_T>(expected_vector, dim, (n - 1) * 10);
    ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(n - 1, expected_vector), 0, abs_err);

    // Vector label n+1 - added
    GenerateVector<TEST_DATA_T>(expected_vector, dim, (n + 1));
    ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(n + 1, expected_vector), 0, abs_err);
}

TYPED_TEST(SVSTieredIndexTestBasic, testSwapJournalMulti) {
    // Create TieredSVS index instance with a mock queue.
    const size_t dim = 4;
    const size_t n = 15;

    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = true};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Forcibly set the training threshold to a value that will trigger the update job
    // for first vector only.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1, n * 100);
    ASSERT_INDEX(tiered_index);
    auto allocator = tiered_index->getAllocator();

    // Add n vectors to the index.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        GenerateVector<TEST_DATA_T>(vector, dim, i);
        tiered_index->addVector(vector, i);
    }

    // Pause the update job after svs index update
    std::mutex mtx;
    bool added_to_svs = false;
    bool continue_job = false;
    std::condition_variable cv;

    auto tracing_callback = [&]() {
        {
            std::unique_lock lock(mtx);
            added_to_svs = true; // Indicate that we are waiting for the update job to start.
        }
        cv.notify_one(); // Notify that the update job has started.
        {
            std::unique_lock lock(mtx);
            cv.wait(lock, [&] { return continue_job; }); // Wait until we continue.
        }
    };
    tiered_index->registerTracingCallback("UpdateJob::after_add_to_svs", tracing_callback);

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();

    {
        // IMPORTANT!: Do not use ASSERT here, as it will not release the mutex and will cause a
        // deadlock. Use EXPECT instead, so we can continue the test even if the condition is not
        // met.

        // Wait for the update job to start.
        std::unique_lock lock(mtx);
        EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(100), [&] { return added_to_svs; }));

        // update job paused, we have vectors 0-(n-1) in the index, let's do index modifications

        // Remove vector label=n-2, it is copied to backend index.
        EXPECT_EQ(VecSimIndex_DeleteVector(tiered_index, n - 2), 2);
        // Add one more vector label=1.
        EXPECT_EQ(GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 1, 10), 1);
        // Add a new vector
        EXPECT_EQ(GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, n, n), 1);
        // Add another one
        EXPECT_EQ(GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, n + 1, n + 1), 1);
        // Remove vector label=0, it is copied to backend index.
        EXPECT_EQ(VecSimIndex_DeleteVector(tiered_index, 0), 2);
        // Remove the last vector copied to backend index
        EXPECT_EQ(VecSimIndex_DeleteVector(tiered_index, n - 1), 2);
        // Add one more vector label=2.
        EXPECT_EQ(GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 2, 20), 1);
        // Remove vector label=2: for multi: old is copied to backend , old + new are in flat
        EXPECT_EQ(VecSimIndex_DeleteVector(tiered_index, 2), 3);
        // Remove vector label=n - in flat only
        EXPECT_EQ(VecSimIndex_DeleteVector(tiered_index, n), 1);
        // Add vector (n-1) again
        EXPECT_EQ(GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, n - 1, (n - 1) * 10), 1);

        continue_job = true; // Indicate that we can continue the update job.
    }

    // continue the update job
    cv.notify_all(); // Notify that we can continue.

    mock_thread_pool.thread_pool_join();

    // For multi-value index, following vectors should be in the index:
    // 0: deleted, 1: (1,10), 2: deleted, 3:3, ..., n-2: deleted n-1: 10(n-1), n+1: n+1;
    // total: n-2 labels, n-1 vectors
    ASSERT_EQ(tiered_index->indexSize(), n - 1);
    ASSERT_EQ(tiered_index->indexLabelCount(), n - 2);

    // Backend index: 0:deleted, 1:1, 2:deleted, 3:3, ..., n-2:deleted, n-1:deleted; total: n-4
    EXPECT_EQ(tiered_index->GetBackendIndex()->indexSize(), n - 4);
    // Frontend index: 1:10, n-1:10(n-1), n+1:n+1
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 3);

    double abs_err = 1e-2; // Allow a larger relative error for quantization.
    TEST_DATA_T expected_vector[dim];

    // Vector label 0 - deleted
    GenerateVector<TEST_DATA_T>(expected_vector, dim, 0);
    ASSERT_TRUE(std::isnan(tiered_index->getDistanceFrom_Unsafe(0, expected_vector)));

    // Vector label n-2 - deleted
    GenerateVector<TEST_DATA_T>(expected_vector, dim, 0);
    ASSERT_TRUE(std::isnan(tiered_index->getDistanceFrom_Unsafe(n - 2, expected_vector)));

    // There are 2 vectors labeled "1" with values 1 in backend and 10 in flat.
    // We expect the minimal distance for the query 10 to be taken from flat index.
    GenerateVector<TEST_DATA_T>(expected_vector, dim, 10);
    ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(1, expected_vector), 0, abs_err);
    // And the minimal distance for the query 1.0 to be taken from backend
    GenerateVector<TEST_DATA_T>(expected_vector, dim, 1);
    ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(1, expected_vector), 0, abs_err);

    // Vector label 2 - deleted
    GenerateVector<TEST_DATA_T>(expected_vector, dim, 2);
    ASSERT_TRUE(std::isnan(tiered_index->getDistanceFrom_Unsafe(2, expected_vector)));

    // Vectors labels [3,n-3] - unchanged
    for (size_t i = 3; i < n - 2; i++) {
        GenerateVector<TEST_DATA_T>(expected_vector, dim, i);
        ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(i, expected_vector), 0, abs_err);
    }

    // Vector label n-1 - deleted and re-added
    GenerateVector<TEST_DATA_T>(expected_vector, dim, (n - 1) * 10);
    ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(n - 1, expected_vector), 0, abs_err);

    // Vector label n+1 - added
    GenerateVector<TEST_DATA_T>(expected_vector, dim, (n + 1));
    ASSERT_NEAR(tiered_index->getDistanceFrom_Unsafe(n + 1, expected_vector), 0, abs_err);
}

TEST(SVSTieredIndexTest, testThreadPool) {
    // Test VecSimSVSThreadPool
    const size_t num_threads = 4;
    auto pool = VecSimSVSThreadPool(num_threads);
    ASSERT_EQ(pool.capacity(), num_threads);
    ASSERT_EQ(pool.size(), num_threads);

    std::atomic_int counter(0);
    auto task = [&counter](size_t i) { counter += i + 1; };

    // exeed the number of threads
    ASSERT_THROW(pool.parallel_for(task, 10), svs::threads::ThreadingException);

    counter = 0;
    pool.parallel_for(task, 4);
    ASSERT_EQ(counter, 10); // 1+2+3+4 = 10

    pool.resize(1);
    ASSERT_EQ(pool.capacity(), 4);
    ASSERT_EQ(pool.size(), 1);
    // exeed the new pool size
    ASSERT_THROW(pool.parallel_for(task, 4), svs::threads::ThreadingException);

    counter = 0;
    pool.parallel_for(task, 1);
    ASSERT_EQ(counter, 1); // 0+1 = 1

    pool.resize(0);
    ASSERT_EQ(pool.capacity(), 4);
    ASSERT_EQ(pool.size(), 1);

    pool.resize(5);
    ASSERT_EQ(pool.capacity(), 4);
    ASSERT_EQ(pool.size(), 4);

    // Test VecSimSVSThreadPool for exception handling
    auto err_task = [](size_t) { throw std::runtime_error("Test exception"); };

    ASSERT_NO_THROW(pool.parallel_for(err_task, 0)); // no task - no err
    ASSERT_THROW(pool.parallel_for(err_task, 1), svs::threads::ThreadingException);
    ASSERT_THROW(pool.parallel_for(err_task, 4), svs::threads::ThreadingException);
}

#else // HAVE_SVS

VecSimIndex *CreateTieredSVSIndex(VecSimParams &svs_params) {
    TieredIndexParams tiered_params{.primaryIndexParams = &svs_params};
    auto *tiered_index = TieredFactory::NewIndex(&tiered_params);
    return tiered_index;
}

TEST(SVSTieredIndexTest, svs_not_supported) {
    SVSParams params = {
        .type = VecSimType_FLOAT32,
        .dim = 16,
        .metric = VecSimMetric_IP,
    };
    auto svs_params = CreateParams(params);

    TieredIndexParams tiered_params{.primaryIndexParams = &svs_params};
    auto index_params = CreateParams(tiered_params);
    auto index = VecSimIndex_New(&index_params);

    ASSERT_EQ(index, nullptr);

    // Although nothing is actually been allocated we calculate a brute force index size to align
    // with the logic of the tiered index function, which currently doesn’t have a verification of
    // the backend index algorithm. This to be changed once a proper verification is introduced.
    auto bf_params = TieredFactory::TieredSVSFactory::NewBFParams(&tiered_params);
    auto expected_size = BruteForceFactory::EstimateInitialSize(&bf_params, false);
    auto size = VecSimIndex_EstimateInitialSize(&index_params);
    ASSERT_EQ(size, expected_size);

    auto size2 = VecSimIndex_EstimateElementSize(&index_params);
    ASSERT_EQ(size2, -1);
}

#endif
