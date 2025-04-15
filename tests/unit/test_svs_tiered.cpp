#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/algorithms/svs/svs_tiered.h"
#include "VecSim/algorithms/svs/svs.h"
#include "VecSim/vec_sim_debug.h"
#include <string>
#include <array>

#include "unit_test_utils.h"
#include "mock_thread_pool.h"

#include <thread>
#include <cpuid.h>

// For getAvailableCPUs():
#if defined(__linux__)
#include <sched.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// System helpers
static bool checkCPU() {
    uint32_t eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);
    std::string vendor_id = std::string((const char *)&ebx, 4) +
                            std::string((const char *)&edx, 4) + std::string((const char *)&ecx, 4);
    return (vendor_id == "GenuineIntel");
}

// Get available number of CPUs
// Returns the number of logical processors on the process
// Returns std::thread::hardware_concurrency() if the number of logical processors is not available
static unsigned int getAvailableCPUs() {
#if defined(__linux__)
    // On Linux, use sched_getaffinity to get the number of CPUs available to the current process.
    cpu_set_t cpu_set;
    if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) == 0) {
        return CPU_COUNT(&cpu_set);
    }

#elif defined(_WIN32)
    // On Windows, use GetProcessAffinityMask to get the number of CPUs available to the current
    // process.
    DWORD_PTR process_affinity, system_affinity;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_affinity, &system_affinity)) {
        return std::bitset<sizeof(DWORD_PTR) * 8>(process_affinity).count();
    }

#elif defined(__APPLE__)
    // On macOS, use sysctl to get the number of CPUs available to the current process.
    int num_cpus;
    size_t size = sizeof(num_cpus);
    if (sysctlbyname("hw.logicalcpu", &num_cpus, &size, nullptr, 0) == 0) {
        return num_cpus;
    }
#endif
    // Fallback.
    return std::thread::hardware_concurrency();
}

// Runs the test for all combination of data type(float/double) - label type (single/multi)

template <typename index_type_t>
class SVSTieredIndexTest : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;

protected:
    TieredSVSIndex<data_t> *CastToTieredSVS(VecSimIndex *index) {
        return reinterpret_cast<TieredSVSIndex<data_t> *>(index);
    }

    TieredIndexParams CreateTieredSVSParams(VecSimParams &svs_params,
                                            tieredIndexMock &mock_thread_pool,
                                            size_t update_job_threshold = 1024,
                                            size_t flat_buffer_limit = SIZE_MAX) {
        svs_params.algoParams.svsParams.quantBits = index_type_t::get_quant_bits();
        if (svs_params.algoParams.svsParams.num_threads == 0) {
            svs_params.algoParams.svsParams.num_threads = mock_thread_pool.thread_pool_size;
        }
        return TieredIndexParams{
            .jobQueue = &mock_thread_pool.jobQ,
            .jobQueueCtx = mock_thread_pool.ctx,
            .submitCb = tieredIndexMock::submit_callback,
            .flatBufferLimit = flat_buffer_limit,
            .primaryIndexParams = &svs_params,
            .specificParams = {.tieredSVSParams =
                                   TieredSVSParams{.updateJobThreshold = update_job_threshold}}};
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

    TieredSVSIndex<data_t> *CreateTieredSVSIndex(VecSimParams &svs_params,
                                                 tieredIndexMock &mock_thread_pool,
                                                 size_t update_job_threshold = 1024,
                                                 size_t flat_buffer_limit = SIZE_MAX) {
        svs_params.algoParams.svsParams.quantBits = index_type_t::get_quant_bits();
        TieredIndexParams tiered_params = CreateTieredSVSParams(
            svs_params, mock_thread_pool, update_job_threshold, flat_buffer_limit);
        return CreateTieredSVSIndex(tiered_params, mock_thread_pool);
    }

    void SetUp() override {
        if constexpr (index_type_t::get_quant_bits() != VecSimSvsQuant_NONE)
            if (!checkCPU()) {
                GTEST_SKIP() << "SVS LVQ is not supported on non-Intel hardware.";
            }
    }
};

// TEST_DATA_T and TEST_DIST_T are defined in test_utils.h

template <VecSimType type, typename DataType, VecSimSvsQuantBits quantBits>
struct SVSIndexType {
    static constexpr VecSimType get_index_type() { return type; }
    static constexpr VecSimSvsQuantBits get_quant_bits() { return quantBits; }
    typedef DataType data_t;
};

// clang-format off
using SVSDataTypeSet = ::testing::Types<SVSIndexType<VecSimType_FLOAT32, float, VecSimSvsQuant_NONE>
#if 0
                                       ,SVSIndexType<VecSimType_FLOAT32, float, VecSimSvsQuant_8>
#endif
                                        >;
// clang-format on

TYPED_TEST_SUITE(SVSTieredIndexTest, SVSDataTypeSet);

// Runs the test for each data type(float/double). The label type should be explicitly
// set in the test.

template <typename index_type_t>
class SVSTieredIndexTestBasic : public SVSTieredIndexTest<index_type_t> {};

using SVSBasicDataTypeSet =
    ::testing::Types<SVSIndexType<VecSimType_FLOAT32, float, VecSimSvsQuant_NONE>>;

TYPED_TEST_SUITE(SVSTieredIndexTestBasic, SVSBasicDataTypeSet);

TYPED_TEST(SVSTieredIndexTest, ThreadsReservation) {
    // Set thread_pool_size to 4 or actual number of available CPUs
    const auto num_threads = std::min(4U, getAvailableCPUs());
    if (num_threads < 2) {
        // If the number of threads is less than 2, we can't run the test
        GTEST_SKIP() << "No threads available";
    }

    std::chrono::milliseconds timeout{1};
    SVSParams params = {.type = TypeParam::get_index_type(), .dim = 4, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();
    mock_thread_pool.thread_pool_size = num_threads;

    // Create TieredSVS index instance with a mock queue.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);

    // Get the allocator from the tiered index.
    auto allocator = tiered_index->getAllocator();

    // Counter of reserved threads
    // This is set in the update_job_mock callback only
    std::atomic<size_t> num_reserved_threads = 0;

    auto update_job_mock = [&num_reserved_threads](VecSimIndex * /*unused*/, size_t num_threads) {
        num_reserved_threads = num_threads;
    };

    // Request 4 threads but just 1 thread is available
    auto jobs = SVSMultiThreadJob::createJobs(allocator, HNSW_INSERT_VECTOR_JOB, update_job_mock,
                                              tiered_index, 4, timeout);
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
    jobs = SVSMultiThreadJob::createJobs(allocator, HNSW_INSERT_VECTOR_JOB, update_job_mock,
                                         tiered_index, num_threads, timeout);
    tiered_index->submitJobs(jobs);
    mock_thread_pool.thread_pool_wait();
    ASSERT_EQ(num_reserved_threads, num_threads);

    // Request and run 1 thread
    jobs = SVSMultiThreadJob::createJobs(allocator, HNSW_INSERT_VECTOR_JOB, update_job_mock,
                                         tiered_index, 1, timeout);
    tiered_index->submitJobs(jobs);
    mock_thread_pool.thread_pool_wait();
    ASSERT_EQ(num_reserved_threads, 1);

    // Request and run less threads than available
    jobs = SVSMultiThreadJob::createJobs(allocator, HNSW_INSERT_VECTOR_JOB, update_job_mock,
                                         tiered_index, num_threads - 1, timeout);
    tiered_index->submitJobs(jobs);
    mock_thread_pool.thread_pool_wait();
    // The number of reserved threads should be equal to requested
    ASSERT_EQ(num_reserved_threads, num_threads - 1);

    // Request more threads than available
    jobs = SVSMultiThreadJob::createJobs(allocator, HNSW_INSERT_VECTOR_JOB, update_job_mock,
                                         tiered_index, num_threads + 1, timeout);
    tiered_index->submitJobs(jobs);
    mock_thread_pool.thread_pool_wait();
    // The number of reserved threads should be equal to the number of available threads
    ASSERT_EQ(num_reserved_threads, num_threads);
    mock_thread_pool.thread_pool_join();
}

TYPED_TEST(SVSTieredIndexTest, CreateIndexInstance) {
    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {.type = TypeParam::get_index_type(), .dim = 4, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);

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
    SVSParams params = {.type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);

    auto mock_thread_pool = tieredIndexMock();

    auto tiered_params = this->CreateTieredSVSParams(svs_params, mock_thread_pool, 1);
    auto *tiered_index = this->CreateTieredSVSIndex(tiered_params, mock_thread_pool);

    // Get the allocator from the tiered index.
    auto allocator = tiered_index->getAllocator();

    BFParams bf_params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .multi = false};

    // Validate that memory upon creating the tiered index is as expected (no more than 2%
    // above te expected, since in different platforms there are some minor additional
    // allocations).
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
    // Validate that the job was created properly
    // ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label).size(), 1);
    // ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label)[0]->label, vec_label);
    // ASSERT_EQ(tiered_index->labelToInsertJobs.at(vec_label)[0]->id, 0);

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
}

TYPED_TEST(SVSTieredIndexTest, insertJob) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    SVSParams params = {.type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1);
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
    ASSERT_EQ(insertion_job->jobType, HNSW_INSERT_VECTOR_JOB);

    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
    // SVS index should have allocated a single record, while flat index should remove the
    // block.
    // LVQDataset does not provide a capacity method
    const size_t expected_capacity =
        TypeParam::get_quant_bits() > 0 ? tiered_index->indexSize() : DEFAULT_BLOCK_SIZE;
    ASSERT_EQ(tiered_index->indexCapacity(), expected_capacity);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexCapacity(), 0);
    ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(vec_label, vector), 0);
}

TYPED_TEST(SVSTieredIndexTest, insertJobAsync) {
    // LVQ vectors representation do not allow to always get exact distance==0 for big range of
    // values
    if (TypeParam::get_quant_bits() != VecSimSvsQuant_NONE) {
        GTEST_SKIP() << "LVQ is not precise enough";
    }
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 5000;
    SVSParams params = {.type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    // Insert vectors
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, i, i);
    }

    mock_thread_pool.thread_pool_wait();
    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);
    auto sz_f = tiered_index->GetFlatIndex()->indexSize();
    auto sz_b = tiered_index->GetBackendIndex()->indexSize();
    EXPECT_EQ(sz_f + sz_b, n);

    // Verify that the vectors were inserted to SVS as expected
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T expected_vector[dim];
        GenerateVector<TEST_DATA_T>(expected_vector, dim, i);
        ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(i, expected_vector), 0)
            << "Vector label: " << i;
    }

    mock_thread_pool.thread_pool_join();
    // Verify that all vectors were moved to SVS as expected
    sz_f = tiered_index->GetFlatIndex()->indexSize();
    sz_b = tiered_index->GetBackendIndex()->indexSize();
    EXPECT_EQ(sz_f, 0);
    EXPECT_EQ(sz_b, n);

    // Verify that the vectors were inserted to SVS as expected
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T expected_vector[dim];
        GenerateVector<TEST_DATA_T>(expected_vector, dim, i);
        ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(i, expected_vector), 0)
            << "Vector label: " << i;
    }
}

TYPED_TEST(SVSTieredIndexTest, KNNSearch) {
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

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, k);
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

TYPED_TEST(SVSTieredIndexTest, deleteVector) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    SVSParams params = {.type = TypeParam::get_index_type(),
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .num_threads = 1};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1);
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
    // ASSERT_EQ(tiered_index->getSVSIndex()->getNumMarkedDeleted(), 1);

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
    // Check that the distance from the deleted vector (of zeros) to the label is the distance
    // to the new vector (L2 distance).
    TEST_DATA_T deleted_vector[dim];
    GenerateVector<TEST_DATA_T>(deleted_vector, dim, 0);
    ASSERT_EQ(tiered_index->GetBackendIndex()->getDistanceFrom_Unsafe(vec_label, deleted_vector),
              dim * pow(new_vec_val, 2));
}

TYPED_TEST(SVSTieredIndexTest, manageIndexOwnership) {

    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    SVSParams params = {.type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);

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

TYPED_TEST(SVSTieredIndexTest, parallelSearch) {
    size_t dim = 4;
    size_t k = 10;
    size_t n = 2000;

    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .search_window_size = n,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
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

    size_t n_labels = n;

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
    size_t n = 3000;

    size_t block_size = n / 100;

    // Create TieredSVS index instance with a mock queue.
    size_t n_labels = n;
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = block_size,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
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

    auto sz_f = tiered_index->GetFlatIndex()->indexSize();
    auto sz_b = tiered_index->GetBackendIndex()->indexSize();
    EXPECT_EQ(sz_f + sz_b, n);
    EXPECT_EQ(successful_searches, n);
    EXPECT_EQ(mock_thread_pool.jobQ.size(), 0);
}

TYPED_TEST(SVSTieredIndexTest, testSizeEstimation) {
    size_t dim = 128;
    size_t n = DEFAULT_BLOCK_SIZE;
    size_t graph_degree = 31; // power of 2 - 1
    size_t bs = DEFAULT_BLOCK_SIZE;

    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .graph_max_degree = graph_degree,
    };
    VecSimParams vecsim_svs_params = CreateParams(svs_params);

    auto mock_thread_pool = tieredIndexMock();
    TieredIndexParams tiered_params = {.jobQueue = &mock_thread_pool.jobQ,
                                       .jobQueueCtx = mock_thread_pool.ctx,
                                       .submitCb = tieredIndexMock::submit_callback,
                                       .flatBufferLimit = SIZE_MAX,
                                       .primaryIndexParams = &vecsim_svs_params};
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
        .blockSize = block_size,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
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
    size_t d = 4;
    size_t n = 1000;

    size_t block_size = n / 100;
    size_t n_labels = n;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
        .blockSize = block_size,
    };
    VecSimParams params = CreateParams(svs_params);

    for (auto &lambda : lambdas) {
        auto &decider_name = lambda.first;
        auto &decider = lambda.second;

        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index = this->CreateTieredSVSIndex(params, mock_thread_pool);
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
        auto query = (TEST_DATA_T *)allocator->allocate(d * sizeof(TEST_DATA_T));
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
    size_t d = 4;
    size_t M = 8;
    size_t sws = 20;
    size_t n = 1000;

    size_t per_label = 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
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
        auto query = (TEST_DATA_T *)allocator->allocate(d * sizeof(TEST_DATA_T));
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
    size_t d = 4;
    size_t M = 8;
    size_t sws = 20;
    size_t n = 1000;

    size_t per_label = 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
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
    size_t d = 4;
    size_t M = 8;
    size_t sws = 20;
    size_t n = 1000;

    size_t per_label = 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
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
    size_t d = 4;
    size_t M = 8;
    size_t sws = 20;
    size_t n = 1000;

    size_t per_label = 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams svs_params = {
        .type = TypeParam::get_index_type(),
        .dim = d,
        .metric = VecSimMetric_L2,
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

TYPED_TEST(SVSTieredIndexTest, parallelBatchIteratorSearch) {
    size_t dim = 4;
    size_t sws = 500;
    size_t n = 1000;
    size_t n_res_min = 3;  // minimum number of results to return per batch
    size_t n_res_max = 15; // maximum number of results to return per batch

    size_t per_label = 1;
    size_t n_labels = n / per_label;

    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .search_window_size = sws,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
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
    size_t dim = 4;
    size_t k = 11;
    size_t per_label = 1;

    size_t n_labels = k * 3;
    size_t n = n_labels * per_label;
    size_t block_size = 10;

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
        .blockSize = block_size,
        .epsilon = 3.0 * per_label,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    size_t cur_memory_usage;

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
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
    size_t dim = 4;
    size_t k = 11;
    size_t n = 1000;

    size_t per_label = 1;
    size_t n_labels = n / per_label;
    size_t block_size = n / 100;

    // Create TieredSVS index instance with a mock queue.
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = block_size,
        .epsilon = double(dim * k * k),
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
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
                        .num_threads = 1};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1);
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
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, HNSW_INSERT_VECTOR_JOB);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->isValid, true);
    mock_thread_pool.thread_iteration();

    // Ingest vector into SVS, and then overwrite it.
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    val = 3.0;
    overwritten_vec[0] = overwritten_vec[1] = overwritten_vec[2] = overwritten_vec[3] = val;
    ASSERT_EQ(tiered_index->addVector(overwritten_vec, 0), 0);
    ASSERT_EQ(tiered_index->indexLabelCount(), 1);
    // Swap job should be executed for the overwritten vector since limit is 1, and we are calling
    // swap job execution prior to insert jobs.
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
    SVSParams params = {.type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    for (size_t updateThreshold : {n, size_t{1}}) {
        auto mock_thread_pool = tieredIndexMock();

        auto *tiered_index =
            this->CreateTieredSVSIndex(svs_params, mock_thread_pool, updateThreshold);
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

        EXPECT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
        EXPECT_EQ(tiered_index->indexLabelCount(), n);
    }
}

// TODO: Uncomment tests below or remove if not relevant.
/*
TYPED_TEST(SVSTieredIndexTest, testInfo) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    SVSParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1, 1000);
    auto allocator = tiered_index->getAllocator();

    VecSimIndexInfo info = tiered_index->info();
    EXPECT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVSLIB);
    EXPECT_EQ(info.commonInfo.indexSize, 0);
    EXPECT_EQ(info.commonInfo.indexLabelCount, 0);
    EXPECT_EQ(info.commonInfo.memory, tiered_index->getAllocationSize());
    EXPECT_EQ(info.commonInfo.basicInfo.isMulti, TypeParam::isMulti());
    EXPECT_EQ(info.commonInfo.basicInfo.dim, dim);
    EXPECT_EQ(info.commonInfo.basicInfo.metric, VecSimMetric_L2);
    EXPECT_EQ(info.commonInfo.basicInfo.type, TypeParam::get_index_type());
    EXPECT_EQ(info.commonInfo.basicInfo.blockSize, DEFAULT_BLOCK_SIZE);
    VecSimIndexInfo frontendIndexInfo = tiered_index->GetFlatIndex()->info();
    VecSimIndexInfo backendIndexInfo = tiered_index->GetBackendIndex()->info();

    compareCommonInfo(info.tieredInfo.frontendCommonInfo, frontendIndexInfo.commonInfo);
    compareFlatInfo(info.tieredInfo.bfInfo, frontendIndexInfo.bfInfo);
    compareCommonInfo(info.tieredInfo.backendCommonInfo, backendIndexInfo.commonInfo);
    compareSVSInfo(info.tieredInfo.backendInfo.svsInfo, backendIndexInfo.svsInfo);

    EXPECT_EQ(info.commonInfo.memory, info.tieredInfo.management_layer_memory +
                                          backendIndexInfo.commonInfo.memory +
                                          frontendIndexInfo.commonInfo.memory);
    EXPECT_EQ(info.tieredInfo.backgroundIndexing, false);
    EXPECT_EQ(info.tieredInfo.bufferLimit, 1000);
    EXPECT_EQ(info.tieredInfo.specificTieredBackendInfo.svsTieredInfo.pendingSwapJobsThreshold, 1);

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

TYPED_TEST(SVSTieredIndexTest, testInfoIterator) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 1000;
    SVSParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2};

    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool, 1);
    auto allocator = tiered_index->getAllocator();

    GenerateAndAddVector(tiered_index, dim, 1, 1);
    VecSimIndexInfo info = tiered_index->info();
    VecSimIndexInfo frontendIndexInfo = tiered_index->GetFlatIndex()->info();
    VecSimIndexInfo backendIndexInfo = tiered_index->GetBackendIndex()->info();

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
            compareSVSIndexInfoToIterator(backendIndexInfo, infoField->fieldValue.iteratorValue);
        } else if (!strcmp(infoField->fieldName, VecSimCommonStrings::TIERED_BUFFER_LIMIT_STRING)) {
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(infoField->fieldValue.uintegerValue, info.tieredInfo.bufferLimit);
        } else if (!strcmp(infoField->fieldName,
                           VecSimCommonStrings::TIERED_SVS_SWAP_JOBS_THRESHOLD_STRING)) {
            ASSERT_EQ(infoField->fieldType, INFOFIELD_UINT64);
            ASSERT_EQ(
                infoField->fieldValue.uintegerValue,
                info.tieredInfo.specificTieredBackendInfo.svsTieredInfo.pendingSwapJobsThreshold);
        } else {
            FAIL();
        }
    }
    VecSimInfoIterator_Free(infoIterator);
}

TYPED_TEST(SVSTieredIndexTest, writeInPlaceMode) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;

    SVSParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    VecSim_SetWriteMode(VecSim_WriteInPlace);
    // Validate that the vector was inserted directly to the SVS index.
    labelType vec_label = 0;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, 0);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);

    // Overwrite inplace - only in single-value mode
    if (!TypeParam::isMulti()) {
        TEST_DATA_T overwritten_vec[] = {1, 1, 1, 1};
        tiered_index->addVector(overwritten_vec, vec_label);
        ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
        ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
        ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);
        ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(vec_label, overwritten_vec), 0);
    }

    // Validate that the vector is removed in place.
    tiered_index->deleteVector(vec_label);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->getSVSIndex()->getNumMarkedDeleted(), 0);
}

TYPED_TEST(SVSTieredIndexTest, switchWriteModes) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 500;
    SVSParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .M = 32,
                         .efRuntime = 3 * n};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
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
    EXPECT_EQ(tiered_index->GetBackendIndex()->indexSize(), 2 * n);

    // Now delete the last n inserted vectors of the index using async jobs.
    VecSim_SetWriteMode(VecSim_WriteAsync);
    mock_thread_pool.init_threads();
    for (size_t i = 0; i < n_labels; i++) {
        VecSimIndex_DeleteVector(tiered_index, n_labels + i);
    }
    // At this point, repair jobs should be executed in the background.
    EXPECT_EQ(tiered_index->getSVSIndex()->getNumMarkedDeleted(), n);

    // Insert INPLACE another n vector (instead of the ones that were deleted).
    VecSim_SetWriteMode(VecSim_WriteInPlace);
    auto svs_index = tiered_index->getSVSIndex();
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
            // Run a query over svs index and see that we only receive ids with label < n_labels+i
            // (the label that we just inserted), and the first result should be this vector
            // (unless it is unreachable)
            auto ver_res = [&](size_t res_label, double score, size_t index) {
                if (index == 0) {
                    if (res_label == cur_label) {
                        EXPECT_DOUBLE_EQ(score, 0);
                    } else {
                        svs_index->lockSharedIndexDataGuard();
                        ASSERT_EQ(svs_index->getDistanceFrom_Unsafe(cur_label, vector), 0);
                        svs_index->unlockSharedIndexDataGuard();
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
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexLabelCount(), 2 * n_labels);
}

TYPED_TEST(SVSTieredIndexTest, bufferLimit) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    SVSParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Create tiered index with buffer limit set to 0.
    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool,
                                                     DEFAULT_PENDING_SWAP_JOBS_THRESHOLD, 0);
    auto allocator = tiered_index->getAllocator();

    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, 0);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 0);

    // Set the flat limit to 1 and insert another vector - expect it to go to the flat buffer.
    tiered_index->flatBufferLimit = 1;
    labelType vec_label = 1;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, 0); // vector is [0,0,0,0]
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);

    // Overwrite the vector, expect removing it from the flat buffer and replace it with the new one
    // only in single-value mode
    if (!TypeParam::isMulti()) {
        TEST_DATA_T overwritten_vec[] = {1, 1, 1, 1};
        ASSERT_EQ(tiered_index->addVector(overwritten_vec, vec_label), 0);
        ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 1);
        ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);
        ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);
        ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(vec_label, overwritten_vec), 0);
        // The first job in Q should be the invalid overwritten insert vector job.
        ASSERT_EQ(mock_thread_pool.jobQ.front().job->isValid, false);
        ASSERT_EQ(reinterpret_cast<SVSInsertJob *>(mock_thread_pool.jobQ.front().job)->id, 0);
        mock_thread_pool.jobQ.pop();
    }

    // Insert another vector, this one should go directly to SVS index since the buffer limit has
    // reached.
    vec_label = 2;
    GenerateAndAddVector<TEST_DATA_T>(tiered_index, dim, vec_label, 0); // vector is [0,0,0,0]
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 2);
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);
    ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);
    ASSERT_EQ(tiered_index->indexLabelCount(), 3);

    // Overwrite the vector, expect marking it as deleted in SVS and insert the new one directly
    // to SVS as well.
    if (!TypeParam::isMulti()) {
        TEST_DATA_T overwritten_vec[] = {1, 1, 1, 1};
        ASSERT_EQ(tiered_index->addVector(overwritten_vec, vec_label), 0);
        ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), 3);
        ASSERT_EQ(tiered_index->getSVSIndex()->getNumMarkedDeleted(), 1);
        ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 1);
        ASSERT_EQ(tiered_index->labelToInsertJobs.size(), 1);
        ASSERT_EQ(tiered_index->indexLabelCount(), 3);
        ASSERT_EQ(tiered_index->getDistanceFrom_Unsafe(vec_label, overwritten_vec), 0);
    }
}

TYPED_TEST(SVSTieredIndexTest, bufferLimitAsync) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 500;
    SVSParams params = {.type = TypeParam::get_index_type(),
                         .dim = dim,
                         .metric = VecSimMetric_L2,
                         .M = 64};

    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    // Create tiered index with buffer limit set to 100.
    size_t flat_buffer_limit = 100;
    auto *tiered_index = this->CreateTieredSVSIndex(
        svs_params, mock_thread_pool, DEFAULT_PENDING_SWAP_JOBS_THRESHOLD, flat_buffer_limit);
    auto allocator = tiered_index->getAllocator();
    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    // Create and insert vectors one by one async. At some point, buffer limit gets full and vectors
    // are inserted directly to SVS.
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
            EXPECT_LE(tiered_index->GetFlatIndex()->indexSize(), flat_buffer_limit);
        }
        // In first run, wait until all vectors are moved from flat index to SVS backend index.
        while (tiered_index->GetBackendIndex()->indexSize() < n) {
            //do nothing
        }
    }
    mock_thread_pool.thread_pool_join();
    EXPECT_EQ(tiered_index->GetBackendIndex()->indexSize(), 2 * n);
    EXPECT_EQ(tiered_index->indexLabelCount(), n_labels);
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
    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Test initialization of the pendingSwapJobsThreshold value.
    ASSERT_EQ(tiered_index->pendingSwapJobsThreshold, DEFAULT_PENDING_SWAP_JOBS_THRESHOLD);

    // Insert three block of vectors directly to SVS.
    size_t n = DEFAULT_PENDING_SWAP_JOBS_THRESHOLD * 3;
    std::srand(10); // create pseudo random generator with any arbitrary seed.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        for (size_t j = 0; j < dim; j++) {
            vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
        }
        VecSimIndex_AddVector(tiered_index->GetBackendIndex(), vector, i);
    }

    // Delete all the vectors and wait for the thread pool to finish running the repair jobs.
    for (size_t i = 0; i < n; i++) {
        tiered_index->deleteVector(i);
    }
    // Launch the BG threads loop that takes jobs from the queue and executes them.
    mock_thread_pool.init_threads();
    mock_thread_pool.thread_pool_join();

    ASSERT_EQ(tiered_index->indexSize(), n);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexSize(), n);
    ASSERT_EQ(tiered_index->getSVSIndex()->getNumMarkedDeleted(), n);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 0);

    // Run the GC API call, expect that we will clean the defined threshold number of vectors
    // each time we call the GC.
    while (tiered_index->indexSize() > 0) {
        size_t cur_size = tiered_index->indexSize();
        VecSimTieredIndex_GC(tiered_index);
        ASSERT_EQ(tiered_index->indexSize(), cur_size - DEFAULT_PENDING_SWAP_JOBS_THRESHOLD);
    }
}

TYPED_TEST(SVSTieredIndexTestBasic, getElementNeighbors) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    size_t n = 0;
    size_t M = 20;
    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2, .M = M};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();
    auto *svs_index = this->CastToSVS(tiered_index);

    // Add vectors directly to SVS index until we have at least 2 vectors at level 1.
    size_t vectors_in_higher_levels = 0;
    while (vectors_in_higher_levels < 2) {
        GenerateAndAddVector<TEST_DATA_T>(svs_index, dim, n, n);
        if (svs_index->getGraphDataByInternalId(n)->toplevel > 0) {
            vectors_in_higher_levels++;
        }
        n++;
    }
    // Go over all vectors and validate that the getElementNeighbors debug command returns the
    // neighbors properly.
    for (size_t id = 0; id < n; id++) {
        ElementLevelData &cur = svs_index->getElementLevelData(id, 0);
        int **neighbors_output;
        VecSimDebug_GetElementNeighborsInSVSGraph(tiered_index, id, &neighbors_output);
        auto graph_data = svs_index->getGraphDataByInternalId(id);
        for (size_t l = 0; l <= graph_data->toplevel; l++) {
            auto &level_data = svs_index->getElementLevelData(graph_data, l);
            auto &neighbours = neighbors_output[l];
            ASSERT_EQ(neighbours[0], level_data.numLinks);
            for (size_t j = 1; j <= neighbours[0]; j++) {
                ASSERT_EQ(neighbours[j], level_data.links[j - 1]);
            }
        }
        VecSimDebug_ReleaseElementNeighborsInSVSGraph(neighbors_output);
    }
}

TYPED_TEST(SVSTieredIndexTestBasic, FitMemoryTest) {
    size_t dim = 4;
    SVSParams params = {.dim = dim, .blockSize = DEFAULT_BLOCK_SIZE};
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);

    // Add vector
    GenerateAndAddVector<TEST_DATA_T>(index->GetFlatIndex(), dim, 0);
    GenerateAndAddVector<TEST_DATA_T>(index->GetBackendIndex(), dim, 0);
    size_t initial_memory = index->getAllocationSize();
    index->fitMemory();
    // Tired backendIndex index doesn't have initial capacity, so adding the first vector triggers
    // allocation.
    ASSERT_EQ(index->getAllocationSize(), initial_memory) << "fitMemory() after adding 1 vector";
}

TYPED_TEST(SVSTieredIndexTestBasic, deleteBothAsyncAndInplace) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;

    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);

    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Insert one vector to SVS.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->GetBackendIndex(), dim, 0);
    // Add another vector and remove it. Expect that at SVS index one repair job will be created.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->GetBackendIndex(), dim, 1, 1);
    ASSERT_EQ(tiered_index->deleteLabelFromSVS(1), 1);
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 1);

    // The first job should be a repair job of the first inserted node id (0) in level 0.
    ASSERT_EQ(mock_thread_pool.jobQ.size(), 1);
    ASSERT_EQ(mock_thread_pool.jobQ.front().job->jobType, SVS_REPAIR_NODE_CONNECTIONS_JOB);
    ASSERT_TRUE(mock_thread_pool.jobQ.front().job->isValid);
    ASSERT_EQ(((SVSRepairJob *)(mock_thread_pool.jobQ.front().job))->node_id, 0);
    ASSERT_EQ(((SVSRepairJob *)(mock_thread_pool.jobQ.front().job))->level, 0);
    ASSERT_EQ(tiered_index->idToRepairJobs.size(), 1);
    ASSERT_GE(tiered_index->idToRepairJobs.at(0).size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(0)[0]->associatedSwapJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(0)[0]->associatedSwapJobs[0]->deleted_id, 1);

    // Add one more vector and remove it, expect that the same repair job for 0 would be created
    // for repairing 0->2.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->GetBackendIndex(), dim, 2, 2);
    ASSERT_EQ(tiered_index->deleteLabelFromSVS(2), 1);
    ASSERT_TRUE(tiered_index->idToSwapJob.contains(2));
    ASSERT_EQ(tiered_index->idToRepairJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(0)[0]->associatedSwapJobs.size(), 2);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(0)[0]->associatedSwapJobs[1]->deleted_id, 2);
    ASSERT_EQ(tiered_index->readySwapJobs, 0);

    tiered_index->setWriteMode(VecSim_WriteInPlace);
    // Delete inplace, expect that the repair job for 0->1 and 0->2 will not be valid anymore.
    ASSERT_EQ(tiered_index->deleteVector(0), 1);
    ASSERT_EQ(tiered_index->indexSize(), 2);
    ASSERT_EQ(((SVSRepairJob *)(mock_thread_pool.jobQ.front().job))->node_id, 0);
    ASSERT_FALSE(mock_thread_pool.jobQ.front().job->isValid);

    // Also expect that the swap job for 2 will not exist anymore, as 2 swapped with 0
    ASSERT_EQ(tiered_index->getSVSIndex()->getNumMarkedDeleted(), 2);
    ASSERT_EQ(tiered_index->idToSwapJob.size(), 2);
    ASSERT_TRUE(tiered_index->idToSwapJob.contains(0));
    ASSERT_FALSE(tiered_index->idToSwapJob.contains(2));
    // Both ids 1 and 0 (previously was 2) are now ready due to the deletion of 0 and its associated
    // jobs.
    ASSERT_EQ(tiered_index->readySwapJobs, 2);
}

TYPED_TEST(SVSTieredIndexTestBasic, deleteInplaceAvoidUpdatedMarkedDeleted) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 4;
    SVSParams params = {
        .type = TypeParam::get_index_type(), .dim = dim, .metric = VecSimMetric_L2};
    VecSimParams svs_params = CreateParams(params);

    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index = this->CreateTieredSVSIndex(svs_params, mock_thread_pool);
    auto allocator = tiered_index->getAllocator();

    // Insert three vector to SVS, expect a full graph to be created
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->GetBackendIndex(), dim, 0, 0);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->GetBackendIndex(), dim, 1, 1);
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->GetBackendIndex(), dim, 2, 2);

    // Delete vector with id=0 asynchronously, expect to have a repair job for the other vectors.
    ASSERT_EQ(tiered_index->deleteVector(0), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.size(), 2);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(1)[0]->associatedSwapJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(2)[0]->associatedSwapJobs.size(), 1);

    // Execute the repair job for 1->0. Now, 0->1 is unidirectional edge
    ASSERT_TRUE(mock_thread_pool.jobQ.front().job->isValid);
    mock_thread_pool.thread_iteration();
    ASSERT_EQ(tiered_index->idToRepairJobs.size(), 1);
    ASSERT_EQ(tiered_index->idToRepairJobs.at(2)[0]->associatedSwapJobs.size(), 1);

    // Insert another vector with id=3, that should be connected to both 1 and 2.
    GenerateAndAddVector<TEST_DATA_T>(tiered_index->GetBackendIndex(), dim, 3, -1);

    // Delete in-place id=2, expect that upon repairing inplace 1 due to 1->2, there will *not* be
    // a new edge 1->0 since 0 is deleted. Also the other repair job 2->0 should be invalidated.
    // Also, expect that repairing 3 in-place will not create a new edge to marked deleted 0 and
    // vice versa.
    tiered_index->setWriteMode(VecSim_WriteInPlace);
    ASSERT_EQ(tiered_index->deleteVector(2), 1);
    ASSERT_FALSE(mock_thread_pool.jobQ.front().job->isValid);
    int **neighbours;
    ASSERT_EQ(tiered_index->getSVSIndex()->getSVSElementNeighbors(1, &neighbours),
              VecSimDebugCommandCode_OK);
    // Expect 1 neighbors at level 0 (id=3) and that 0 is NOT a new neighbor for 1.
    ASSERT_EQ(neighbours[0][0], 1);
    ASSERT_EQ(neighbours[0][1], 3);
    VecSimDebug_ReleaseElementNeighborsInSVSGraph(neighbours);

    ASSERT_EQ(tiered_index->getSVSIndex()->getSVSElementNeighbors(3, &neighbours),
              VecSimDebugCommandCode_OK);
    ASSERT_EQ(neighbours[0][0], 1);
    // Expect 1 neighbors at level 0 (id=1) and that 0 is NOT a new neighbor for 3.
    ASSERT_EQ(neighbours[0][1], 1);
    VecSimDebug_ReleaseElementNeighborsInSVSGraph(neighbours);

    auto &level_data = tiered_index->getSVSIndex()->getElementLevelData((idType)0, 0);
    // Expect 1 neighbors at level 0 (id=1) and that 3 is NOT a new neighbor for 0.
    ASSERT_EQ(level_data.getNumLinks(), 1);
    ASSERT_EQ(level_data.getLinkAtPos(0), 1);

    // Expect that id=0 is a ready swap job and execute it.
    ASSERT_EQ(tiered_index->readySwapJobs, 1);
    ASSERT_TRUE(tiered_index->idToSwapJob.contains(0));
    tiered_index->runGC();
}

TYPED_TEST(SVSTieredIndexTestBasic, switchDeleteModes) {
    // Create TieredSVS index instance with a mock queue.
    size_t dim = 16;
    size_t n = 1000;
    size_t swap_job_threshold = 10;
    SVSParams params = {
        .type = TypeParam::get_index_type(),
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = 100,
    };
    VecSimParams svs_params = CreateParams(params);
    auto mock_thread_pool = tieredIndexMock();

    auto *tiered_index =
        this->CreateTieredSVSIndex(svs_params, mock_thread_pool, swap_job_threshold);
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
    ASSERT_EQ(tiered_index->GetFlatIndex()->indexSize(), 0);
    ASSERT_EQ(tiered_index->GetBackendIndex()->indexLabelCount(), n);
    auto state = tiered_index->getSVSIndex()->checkIntegrity();
    ASSERT_EQ(state.valid_state, 1);
    ASSERT_EQ(state.connections_to_repair, 0);
}

*/
