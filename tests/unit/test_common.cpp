/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_debug.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/utils/updatable_heap.h"
#include "VecSim/utils/vec_utils.h"
#include "unit_test_utils.h"
#include "VecSim/containers/vecsim_results_container.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "mock_thread_pool.h"
#include "tests_utils.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "VecSim/spaces/normalize/compute_norm.h"

#include <cstdlib>
#include <limits>
#include <cmath>
#include <random>
#include <cstdarg>

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

template <typename index_type_t>
class CommonIndexTest : public ::testing::Test {};

// DataTypeSet are defined in unit_test_utils.h

TYPED_TEST_SUITE(CommonIndexTest, DataTypeSet);

TYPED_TEST(CommonIndexTest, ResolveQueryRuntimeParams) {
    size_t dim = 4;

    BFParams params = {.dim = dim, .metric = VecSimMetric_L2, .blockSize = 5};
    VecSimIndex *index = test_utils::CreateNewIndex(params, TypeParam::get_index_type());

    VecSimQueryParams qparams, zero;
    bzero(&zero, sizeof(VecSimQueryParams));

    std::vector<VecSimRawParam> rparams;

    // Empty raw params array, nothing should change in query params.
    for (VecsimQueryType query_type : test_utils::query_types) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, query_type),
            VecSim_OK);
    }
    ASSERT_EQ(memcmp(&qparams, &zero, sizeof(VecSimQueryParams)), 0);

    for (VecsimQueryType query_type : test_utils::query_types) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), nullptr, query_type),
            VecSimParamResolverErr_NullParam);
    }

    /** Testing with common hybrid query params. **/
    rparams.push_back(VecSimRawParam{"batch_size", strlen("batch_size"), "100", strlen("100")});

    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSim_OK);
    ASSERT_EQ(qparams.batchSize, 100);

    // Both params are "batch_size".
    rparams.push_back(VecSimRawParam{"batch_size", strlen("batch_size"), "200", strlen("200")});

    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSimParamResolverErr_AlreadySet);

    rparams[1] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "batches_wrong",
                                  .valLen = strlen("batches_wrong")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSimParamResolverErr_InvalidPolicy_NExits);

    rparams[1].value = "batches";
    rparams[1].valLen = strlen("batches");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 100);

    // Both params are "hybrid policy".
    rparams[0] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "ADhOC_bf",
                                  .valLen = strlen("ADhOC_bf")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSimParamResolverErr_AlreadySet);

    // Sending HYBRID_POLICY=adhoc as the single parameter is valid.
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), 1, &qparams, QUERY_TYPE_HYBRID),
              VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_ADHOC_BF);

    // Cannot set batch_size param with "hybrid_policy" which is "ADHOC_BF"
    rparams[1] = (VecSimRawParam){.name = "batch_size",
                                  .nameLen = strlen("batch_size"),
                                  .value = "100",
                                  .valLen = strlen("100")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSimParamResolverErr_InvalidPolicy_AdHoc_With_BatchSize);

    rparams[0] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "batches",
                                  .valLen = strlen("batches")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 100);

    // Trying to set hybrid policy for non-hybrid query.
    for (VecsimQueryType query_type : {QUERY_TYPE_NONE, QUERY_TYPE_KNN, QUERY_TYPE_RANGE}) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, query_type),
            VecSimParamResolverErr_InvalidPolicy_NHybrid);
        ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data() + 1, 1, &qparams, query_type),
                  VecSimParamResolverErr_InvalidPolicy_NHybrid);
    }

    // Check for invalid batch sizes params.
    rparams[1].value = "not_a_number";
    rparams[1].valLen = strlen("not_a_number");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "9223372036854775808"; // LLONG_MAX+1
    rparams[1].valLen = strlen("9223372036854775808");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "-5";
    rparams[1].valLen = strlen("-5");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "0";
    rparams[1].valLen = strlen("0");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "10f";
    rparams[1].valLen = strlen("10f");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSimParamResolverErr_BadValue);

    VecSimIndex_Free(index);
}

TYPED_TEST(CommonIndexTest, DumpHNSWNeighborsDebugEdgeCases) {
    size_t dim = 4;
    size_t top_level;
    int **neighbors_data;

    BFParams params = {.dim = dim, .metric = VecSimMetric_L2};
    VecSimIndex *index = test_utils::CreateNewIndex(params, TypeParam::get_index_type());

    auto res = VecSimDebug_GetElementNeighborsInHNSWGraph(index, 0, &neighbors_data);
    ASSERT_EQ(res, VecSimDebugCommandCode_BadIndex);
    ASSERT_EQ(neighbors_data, nullptr);
    VecSimIndex_Free(index);

    HNSWParams hnsw_params = {.dim = dim, .metric = VecSimMetric_L2};
    index = test_utils::CreateNewIndex(hnsw_params, TypeParam::get_index_type(), true);

    res = VecSimDebug_GetElementNeighborsInHNSWGraph(index, 0, &neighbors_data);
    ASSERT_EQ(res, VecSimDebugCommandCode_MultiNotSupported);
    ASSERT_EQ(neighbors_data, nullptr);
    VecSimIndex_Free(index);

    hnsw_params = {.dim = dim, .metric = VecSimMetric_L2};
    index = test_utils::CreateNewIndex(hnsw_params, TypeParam::get_index_type());

    res = VecSimDebug_GetElementNeighborsInHNSWGraph(index, 0, &neighbors_data);
    ASSERT_EQ(res, VecSimDebugCommandCode_LabelNotExists);
    ASSERT_EQ(neighbors_data, nullptr);

    // Add one vector, then try to get its neighbors (an array with a single array of zero).
    GenerateAndAddVector<TEST_DATA_T>(index, dim, 0, 0);
    res = VecSimDebug_GetElementNeighborsInHNSWGraph(index, 0, &neighbors_data);
    ASSERT_EQ(res, VecSimDebugCommandCode_OK);
    ASSERT_EQ(neighbors_data[0][0], 0);
    VecSimDebug_ReleaseElementNeighborsInHNSWGraph(neighbors_data);

    // Try to get non-existing label again.
    res = VecSimDebug_GetElementNeighborsInHNSWGraph(index, 1, &neighbors_data);
    ASSERT_EQ(res, VecSimDebugCommandCode_LabelNotExists);
    ASSERT_EQ(neighbors_data, nullptr);

    VecSimIndex_Free(index);
}

template <typename DataType>
class UtilsTests : public ::testing::Test {};

using DataTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UtilsTests, DataTypes);

TYPED_TEST(UtilsTests, Max_Updatable_Heap) {
    std::pair<TypeParam, size_t> p;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    vecsim_stl::updatable_max_heap<TypeParam, size_t> heap(allocator);

    // Initial state checks
    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());
    ASSERT_NO_THROW(heap.top());

    // Insert some data in random order
    size_t riders[] = {46, 16, 99, 93};
    const size_t n_riders = sizeof(riders) / sizeof(riders[0]);
    enum Priority { FIRST = 0, SECOND = 1, THIRD = 2, FOURTH = 3 };
    const TypeParam priorities[] = {M_PI, M_E, M_SQRT2, -M_SQRT2 * M_E};

    heap.emplace(priorities[THIRD], riders[1]);
    heap.emplace(priorities[FIRST], riders[3]);
    heap.emplace(priorities[SECOND], riders[2]);
    heap.emplace(priorities[FOURTH], riders[0]);

    for (size_t i = 0; i < n_riders; ++i) {
        ASSERT_EQ(heap.size(), n_riders - i);
        p = {priorities[i], riders[n_riders - 1 - i]};
        ASSERT_TRUE(heap.top() == p);
        ASSERT_FALSE(heap.empty());
        heap.pop();
    }

    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());

    // Inserting data with the same priority
    heap.emplace(priorities[SECOND], 2);
    heap.emplace(priorities[FIRST], 1);
    heap.emplace(priorities[SECOND], 4);
    heap.emplace(priorities[SECOND], 3);

    ASSERT_EQ(heap.size(), 4);
    ASSERT_FALSE(heap.empty());
    p = {priorities[FIRST], 1};
    ASSERT_TRUE(heap.top() == p);

    heap.emplace(priorities[THIRD], 1); // Update priority

    ASSERT_EQ(heap.size(), 4); // Same size after update
    ASSERT_FALSE(heap.empty());

    // Make sure each pop deletes a single element, even if some have the same priority.
    // Also, make sure the elements are popped in the correct order (highest priority first, and on
    // a tie - the element with the highest value).
    size_t len = heap.size();
    for (size_t i = len; i > 0; i--) {
        ASSERT_EQ(heap.size(), i);
        ASSERT_EQ(heap.top().second, i);
        ASSERT_EQ(heap.top().first, i == 1 ? priorities[THIRD] : priorities[SECOND]);
        ASSERT_FALSE(heap.empty());
        heap.pop();
    }
    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());

    // Update a priority of an element that share its priority with many others.
    size_t last = 10;
    for (size_t i = 0; i <= last; i++) {
        heap.emplace(priorities[SECOND], i);
    }
    // Bound the existing elements with higher and lower priorities.
    heap.emplace(priorities[THIRD], 42);
    heap.emplace(priorities[FIRST], 46);
    size_t size = heap.size();

    // Update to the lowest priority
    heap.emplace(-priorities[THIRD], last);
    ASSERT_EQ(heap.size(), size);

    while (heap.size() > 1) {
        heap.pop();
    }
    ASSERT_EQ(heap.size(), 1);
    ASSERT_FALSE(heap.empty());
    p = {-priorities[THIRD], last};
    ASSERT_TRUE(heap.top() == p);
    heap.pop();
    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());
}

TYPED_TEST(UtilsTests, VecSim_Normalize_Vector) {
    const size_t dim = 1000;
    TypeParam v[dim];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> dis(0.0, (TypeParam)std::numeric_limits<int>::max());

    // generate random values - always generates the same values
    for (size_t i = 0; i < dim; ++i) {
        v[i] = dis(rng);
    }

    // Change some of the vector's values so that the sum of the squared vector's
    // values will overflow for floats but not for doubles.
    v[dim - 3] = exp(44);
    v[dim - 2] = exp(44);
    v[dim - 1] = exp(44);

    // Normalize the vector
    spaces::GetNormalizeFunc<TypeParam>()(v, dim);

    // Check that the normelized vector norm is 1
    TypeParam norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        norm += v[i] * v[i];
    }

    TypeParam one = 1.0;
    ASSERT_NEAR(one, norm, 0.0000001);
}

TYPED_TEST(UtilsTests, results_containers) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    auto res1 = new VecSimQueryReply(allocator);
    auto res2 = new VecSimQueryReply(allocator);
    {
        vecsim_stl::default_results_container drc(allocator);
        vecsim_stl::unique_results_container urc(allocator);
        // Checks for leaks if `get_results()` is not invoked
        vecsim_stl::default_results_container dummy1(allocator);
        vecsim_stl::unique_results_container dummy2(allocator);

        for (size_t i = 0; i < 10; i++) {
            drc.emplace(i, i);
            urc.emplace(i, i + 10);

            dummy1.emplace(i, i);
            dummy2.emplace(i, i + 10);
        }
        for (size_t i = 0; i < 10; i++) {
            urc.emplace(i, i);
            dummy2.emplace(i, i);
        }
        ASSERT_EQ(drc.size(), 10);
        ASSERT_EQ(urc.size(), 10);
        ASSERT_EQ(dummy1.size(), 10);
        ASSERT_EQ(dummy2.size(), 10);

        res1->results = drc.get_results();
        res2->results = urc.get_results();
    }
    sort_results_by_id(res1);
    sort_results_by_score(res2);

    for (size_t i = 0; i < VecSimQueryReply_Len(res1); i++) {
        ASSERT_EQ(i, VecSimQueryResult_GetId(res1->results.data() + i));
    }
    for (size_t i = 0; i < VecSimQueryReply_Len(res2); i++) {
        ASSERT_EQ(i, VecSimQueryResult_GetId(res2->results.data() + i));
    }

    VecSimQueryReply_Free(res1);
    VecSimQueryReply_Free(res2);
}

TYPED_TEST(UtilsTests, data_blocks_container) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    // Create a simple data blocks container of chars with block of size 1.
    auto chars_container = DataBlocksContainer(1, 1, allocator, 64);
    ASSERT_EQ(chars_container.size(), 0);
    ASSERT_EQ(chars_container.addElement(std::string("a").c_str(), 0),
              RawDataContainer::Status::OK);
    ASSERT_EQ(chars_container.size(), 1);
    ASSERT_EQ(chars_container.addElement(std::string("b").c_str(), 1),
              RawDataContainer::Status::OK);
    ASSERT_EQ(chars_container.size(), 2);
    ASSERT_EQ(chars_container.updateElement(0, std::string("c").c_str()),
              RawDataContainer::Status::OK);
    ASSERT_EQ(*chars_container.getElement(0), *std::string("c").c_str());
    ASSERT_EQ(chars_container.removeElement(1), RawDataContainer::Status::OK);
    ASSERT_EQ(chars_container.size(), 1);
    ASSERT_EQ(chars_container.removeElement(0), RawDataContainer::Status::OK);
    ASSERT_EQ(chars_container.size(), 0);
    ASSERT_EQ(chars_container.addElement(std::string("b").c_str(), 0),
              RawDataContainer::Status::OK);
    ASSERT_EQ(chars_container.size(), 1);
    ASSERT_EQ(*chars_container.getElement(0), *std::string("b").c_str());
}

class CommonAPITest : public ::testing::Test {};

TEST(CommonAPITest, VecSim_QueryResult_Iterator) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    auto res_list = new VecSimQueryReply(allocator);
    res_list->results.push_back(VecSimQueryResult{.id = 0, .score = 0.0});
    res_list->results.push_back(VecSimQueryResult{.id = 1, .score = 1.0});
    res_list->results.push_back(VecSimQueryResult{.id = 2, .score = 2.0});

    ASSERT_EQ(3, VecSimQueryReply_Len(res_list));

    // Go over the list result with the iterator. Reset the iterator and re-iterate several times.
    VecSimQueryReply_Iterator *it = VecSimQueryReply_GetIterator(res_list);
    for (size_t rep = 0; rep < 3; rep++) {
        for (size_t i = 0; i < VecSimQueryReply_Len(res_list); i++) {
            ASSERT_TRUE(VecSimQueryReply_IteratorHasNext(it));
            VecSimQueryResult *res = VecSimQueryReply_IteratorNext(it);
            ASSERT_EQ(i, VecSimQueryResult_GetId(res));
            ASSERT_EQ((double)i, VecSimQueryResult_GetScore(res));
        }
        ASSERT_FALSE(VecSimQueryReply_IteratorHasNext(it));
        VecSimQueryReply_IteratorReset(it);
    }

    // Destroying the iterator without destroying the list.
    VecSimQueryReply_IteratorFree(it);
    ASSERT_EQ(3, VecSimQueryReply_Len(res_list));
    VecSimQueryReply_Free(res_list);
}

class SerializerTest : public ::testing::Test {
protected:
    ~SerializerTest() { remove(file_name.c_str()); }

    std::streampos GetFileSize() {
        std::ifstream file(file_name, std::ios::binary);
        const auto begin = file.tellg();
        file.seekg(0, std::ios::end);
        const auto end = file.tellg();
        file.close();

        return end - begin;
    }

    std::string file_name;
};

TEST_F(SerializerTest, HNSWSerialzer) {

    this->file_name = std::string(getenv("ROOT")) + "/tests/unit/bad_index.hnsw";

    // Try to load an index from a file that doesnt exist.
    ASSERT_EXCEPTION_MESSAGE(HNSWFactory::NewIndex(this->file_name), std::runtime_error,
                             "Cannot open file");

    std::ofstream output(this->file_name, std::ios::binary);
    // Write invalid encoding version
    Serializer::writeBinaryPOD(output, 0);
    output.flush();
    ASSERT_EXCEPTION_MESSAGE(HNSWFactory::NewIndex(this->file_name), std::runtime_error,
                             "Cannot load index: deprecated encoding version: 0");

    output.seekp(0, std::ios_base::beg);
    Serializer::writeBinaryPOD(output, 42);
    output.flush();
    ASSERT_EXCEPTION_MESSAGE(HNSWFactory::NewIndex(this->file_name), std::runtime_error,
                             "Cannot load index: bad encoding version: 42");

    // Test WRONG index algorithm exception
    // Use a valid version
    output.seekp(0, std::ios_base::beg);

    Serializer::writeBinaryPOD(output, Serializer::EncodingVersion_V3);
    Serializer::writeBinaryPOD(output, 42);
    output.flush();

    ASSERT_EXCEPTION_MESSAGE(
        HNSWFactory::NewIndex(this->file_name), std::runtime_error,
        "Cannot load index: Expected HNSW file but got algorithm type: Unknown (corrupted file?)");

    // Test WRONG index data type
    // Use a valid version
    output.seekp(0, std::ios_base::beg);

    Serializer::writeBinaryPOD(output, Serializer::EncodingVersion_V3);
    Serializer::writeBinaryPOD(output, VecSimAlgo_HNSWLIB);
    Serializer::writeBinaryPOD(output, size_t(128));

    Serializer::writeBinaryPOD(output, 42);
    output.flush();

    ASSERT_EXCEPTION_MESSAGE(HNSWFactory::NewIndex(this->file_name), std::runtime_error,
                             "Cannot load index: bad index data type: Unknown (corrupted file?)");

    output.close();
}

struct logCtx {
public:
    std::vector<std::string> logBuffer;
    std::string prefix;
};

void test_log_impl(void *ctx, const char *level, const char *message) {
    logCtx *log = (logCtx *)ctx;
    std::string msg = std::string(level) + ": " + log->prefix + message;
    log->logBuffer.push_back(msg);
}

TEST(CommonAPITest, testlogBasic) {

    logCtx log;
    log.prefix = "test log prefix: ";

    BFParams bfParams = {.dim = 1, .metric = VecSimMetric_L2, .blockSize = 5};
    VecSimParams params = {
        .algo = VecSimAlgo_BF, .algoParams = {.bfParams = BFParams{bfParams}}, .logCtx = &log};
    auto *index =
        dynamic_cast<BruteForceIndex<float, float> *>(BruteForceFactory::NewIndex(&params));
    VecSim_SetLogCallbackFunction(test_log_impl);

    index->log(VecSimCommonStrings::LOG_NOTICE_STRING, "test log message no fmt");
    index->log(VecSimCommonStrings::LOG_WARNING_STRING, "test log message %s %s", "with", "args");

    ASSERT_EQ(log.logBuffer.size(), 2);
    ASSERT_EQ(log.logBuffer[0], "notice: test log prefix: test log message no fmt");
    ASSERT_EQ(log.logBuffer[1], "warning: test log prefix: test log message with args");

    VecSimIndex_Free(index);
}

TEST(CommonAPITest, testlogTieredIndex) {
    logCtx log;
    log.prefix = "tiered prefix: ";
    VecSim_SetLogCallbackFunction(test_log_impl);

    HNSWParams params_raw = {.type = VecSimType_FLOAT32, .dim = 4, .metric = VecSimMetric_L2};
    VecSimParams hnsw_params = {.algo = VecSimAlgo_HNSWLIB,
                                .algoParams = {.hnswParams = HNSWParams{params_raw}},
                                .logCtx = &log};
    auto mock_thread_pool = tieredIndexMock();
    TieredIndexParams tiered_params = {.jobQueue = &mock_thread_pool.jobQ,
                                       .jobQueueCtx = mock_thread_pool.ctx,
                                       .submitCb = tieredIndexMock::submit_callback,
                                       .flatBufferLimit = DEFAULT_BLOCK_SIZE,
                                       .primaryIndexParams = &hnsw_params,
                                       .specificParams = {TieredHNSWParams{.swapJobThreshold = 1}}};

    auto *tiered_index =
        reinterpret_cast<TieredHNSWIndex<float, float> *>(TieredFactory::NewIndex(&tiered_params));
    mock_thread_pool.ctx->index_strong_ref.reset(tiered_index);

    GenerateAndAddVector<float>(tiered_index, 4, 1);
    mock_thread_pool.thread_iteration();
    tiered_index->deleteVector(1);
    ASSERT_EQ(log.logBuffer.size(), 4);
    ASSERT_EQ(log.logBuffer[0],
              "verbose: " + log.prefix + "Updating HNSW index capacity from 0 to 1024");
    ASSERT_EQ(log.logBuffer[1],
              "verbose: " + log.prefix +
                  "Tiered HNSW index GC: there are 1 ready swap jobs. Start executing 1 swap jobs");
    ASSERT_EQ(log.logBuffer[2],
              "verbose: " + log.prefix + "Updating HNSW index capacity from 1024 to 0");
    ASSERT_EQ(log.logBuffer[3],
              "verbose: " + log.prefix + "Tiered HNSW index GC: done executing 1 swap jobs");
}

TEST(CommonAPITest, NormalizeBfloat16) {
    size_t dim = 20;
    bfloat16 v[dim];

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-5.0, -5.0);

    for (size_t i = 0; i < dim; i++) {
        float random_number = dis(gen);
        v[i] = vecsim_types::float_to_bf16(random_number);
    }

    VecSim_Normalize(v, dim, VecSimType_BFLOAT16);

    // Check that the normalized vector norm is 1.
    float norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        // Convert assuming little endian system.
        float val = vecsim_types::bfloat16_to_float32(v[i]);
        norm += val * val;
    }

    ASSERT_NEAR(1.0, norm, 0.001);
}

TEST(CommonAPITest, NormalizeFloat16) {
    size_t dim = 20;
    float16 v[dim];

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-5.0, -5.0);

    for (size_t i = 0; i < dim; i++) {
        float random_number = dis(gen);
        v[i] = vecsim_types::FP32_to_FP16(random_number);
    }

    VecSim_Normalize(v, dim, VecSimType_FLOAT16);

    // Check that the normalized vector norm is 1.
    float norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        // Convert assuming little endian system.
        float val = vecsim_types::FP16_to_FP32(v[i]);
        norm += val * val;
    }

    ASSERT_NEAR(1.0, norm, 0.001);
}

TEST(CommonAPITest, NormalizeInt8) {
    size_t dim = 20;
    int8_t v[dim + sizeof(float)];

    test_utils::populate_int8_vec(v, dim);

    VecSim_Normalize(v, dim, VecSimType_INT8);

    float res_norm = *(reinterpret_cast<float *>(v + dim));
    // Check that the normalized vector norm is 1.
    float norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        float val = v[i] / res_norm;
        norm += val * val;
    }

    ASSERT_FLOAT_EQ(norm, 1.0);
}

class CommonTypeMetricTests : public testing::TestWithParam<std::tuple<VecSimType, VecSimMetric>> {
protected:
    template <typename algo_params>
    void test_datasize();

    template <typename algo_params>
    void test_initial_size_estimation();

    virtual void TearDown() { VecSimIndex_Free(index); }

    VecSimIndex *index;
};

template <typename algo_params>
void CommonTypeMetricTests::test_datasize() {
    size_t dim = 4;
    VecSimType type = std::get<0>(GetParam());
    VecSimMetric metric = std::get<1>(GetParam());
    algo_params params = {.dim = dim, .metric = metric};
    this->index = test_utils::CreateNewIndex(params, type);
    size_t actual = test_utils::CalcIndexDataSize(index, type);
    size_t expected = dim * VecSimType_sizeof(type);
    if (type == VecSimType_INT8 && metric == VecSimMetric_Cosine) {
        expected += sizeof(float);
    }
    ASSERT_EQ(actual, expected);
}

TEST_P(CommonTypeMetricTests, TestDataSizeBF) { this->test_datasize<BFParams>(); }
TEST_P(CommonTypeMetricTests, TestDataSizeHNSW) { this->test_datasize<HNSWParams>(); }

template <typename algo_params>
void CommonTypeMetricTests::test_initial_size_estimation() {
    size_t dim = 4;
    VecSimType type = std::get<0>(GetParam());
    VecSimMetric metric = std::get<1>(GetParam());
    algo_params params = {.dim = dim, .metric = metric};
    this->index = test_utils::CreateNewIndex(params, type);

    size_t estimation = EstimateInitialSize(params);
    size_t actual = index->getAllocationSize();

    ASSERT_EQ(estimation, actual);
}

TEST_P(CommonTypeMetricTests, TestInitialSizeEstimationBF) {
    this->test_initial_size_estimation<BFParams>();
}
TEST_P(CommonTypeMetricTests, TestInitialSizeEstimationHNSW) {
    this->test_initial_size_estimation<HNSWParams>();
}

class CommonTypeMetricTieredTests : public CommonTypeMetricTests {
protected:
    virtual void TearDown() override {}

    tieredIndexMock mock_thread_pool;
};

TEST_P(CommonTypeMetricTieredTests, TestDataSizeTieredHNSW) {
    size_t dim = 4;
    VecSimType type = std::get<0>(GetParam());
    VecSimMetric metric = std::get<1>(GetParam());

    HNSWParams hnsw_params = {.type = type, .dim = 4, .metric = metric};
    VecSimIndex *index = test_utils::CreateNewTieredHNSWIndex(hnsw_params, this->mock_thread_pool);

    auto verify_data_size = [&](const auto &tiered_index) {
        auto hnsw_index = tiered_index->getHNSWIndex();
        auto bf_index = tiered_index->getFlatBufferIndex();
        size_t expected = dim * VecSimType_sizeof(type);
        if (type == VecSimType_INT8 && metric == VecSimMetric_Cosine) {
            expected += sizeof(float);
        }
        size_t actual_hnsw = hnsw_index->getDataSize();
        ASSERT_EQ(actual_hnsw, expected);
        size_t actual_bf = bf_index->getDataSize();
        ASSERT_EQ(actual_bf, expected);
    };

    switch (type) {
    case VecSimType_FLOAT32: {
        auto tiered_index = test_utils::cast_to_tiered_index<float, float>(index);
        verify_data_size(tiered_index);
        break;
    }
    case VecSimType_FLOAT64: {
        auto tiered_index = test_utils::cast_to_tiered_index<double, double>(index);
        verify_data_size(tiered_index);
        break;
    }
    case VecSimType_BFLOAT16: {
        auto tiered_index = test_utils::cast_to_tiered_index<vecsim_types::bfloat16, float>(index);
        verify_data_size(tiered_index);
        break;
    }
    case VecSimType_FLOAT16: {
        auto tiered_index = test_utils::cast_to_tiered_index<vecsim_types::float16, float>(index);
        verify_data_size(tiered_index);
        break;
    }
    case VecSimType_INT8: {
        auto tiered_index = test_utils::cast_to_tiered_index<int8_t, float>(index);
        verify_data_size(tiered_index);
        break;
    }
    default:
        FAIL() << "Unsupported data type";
    }
}

TEST_P(CommonTypeMetricTieredTests, TestInitialSizeEstimationTieredHNSW) {
    size_t dim = 4;
    VecSimType type = std::get<0>(GetParam());
    VecSimMetric metric = std::get<1>(GetParam());
    HNSWParams hnsw_params = {.type = type, .dim = dim, .metric = metric};
    VecSimParams vecsim_hnsw_params = CreateParams(hnsw_params);
    TieredIndexParams tiered_params =
        test_utils::CreateTieredParams(vecsim_hnsw_params, this->mock_thread_pool);
    VecSimParams params = CreateParams(tiered_params);
    auto *index = VecSimIndex_New(&params);
    mock_thread_pool.ctx->index_strong_ref.reset(index);

    size_t estimation = VecSimIndex_EstimateInitialSize(&params);
    size_t actual = index->getAllocationSize();

    ASSERT_EQ(estimation, actual);
}

constexpr VecSimType vecsim_datatypes[] = {VecSimType_FLOAT32, VecSimType_FLOAT64,
                                           VecSimType_BFLOAT16, VecSimType_FLOAT16,
                                           VecSimType_INT8};

INSTANTIATE_TEST_SUITE_P(CommonTest, CommonTypeMetricTests,
                         testing::Combine(testing::ValuesIn(vecsim_datatypes),
                                          testing::Values(VecSimMetric_L2, VecSimMetric_IP,
                                                          VecSimMetric_Cosine)),
                         [](const testing::TestParamInfo<CommonTypeMetricTests::ParamType> &info) {
                             const char *type = VecSimType_ToString(std::get<0>(info.param));
                             const char *metric = VecSimMetric_ToString(std::get<1>(info.param));
                             std::string test_name(type);
                             return test_name + "_" + metric;
                         });

INSTANTIATE_TEST_SUITE_P(
    CommonTieredTest, CommonTypeMetricTieredTests,
    testing::Combine(testing::ValuesIn(vecsim_datatypes),
                     testing::Values(VecSimMetric_L2, VecSimMetric_IP, VecSimMetric_Cosine)),
    [](const testing::TestParamInfo<CommonTypeMetricTieredTests::ParamType> &info) {
        const char *type = VecSimType_ToString(std::get<0>(info.param));
        const char *metric = VecSimMetric_ToString(std::get<1>(info.param));
        std::string test_name(type);
        return test_name + "_" + metric;
    });

class IndexCalculatorTest : public ::testing::Test {};
namespace dummyCalcultor {

using DummyType = int;
using dummy_dist_func_t = DummyType (*)(int);

int dummyDistFunc(int value) { return value; }

template <typename DistType>
class DistanceCalculatorDummy : public DistanceCalculatorInterface<DistType, dummy_dist_func_t> {
public:
    DistanceCalculatorDummy(std::shared_ptr<VecSimAllocator> allocator, dummy_dist_func_t dist_func)
        : DistanceCalculatorInterface<DistType, dummy_dist_func_t>(allocator, dist_func) {}

    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const {
        return this->dist_func(7);
    }
};

} // namespace dummyCalcultor

TEST(IndexCalculatorTest, TestIndexCalculator) {

    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Test computer with a distance function signature different from dim(v1, v2, dim()).
    using namespace dummyCalcultor;
    auto distance_calculator = DistanceCalculatorDummy<DummyType>(allocator, dummyDistFunc);

    ASSERT_EQ(distance_calculator.calcDistance(nullptr, nullptr, 0), 7);
}

class PreprocessorsTest : public ::testing::Test {};

namespace dummyPreprocessors {

using DummyType = int;

enum pp_mode { STORAGE_ONLY, QUERY_ONLY, BOTH, EMPTY };

// Dummy storage preprocessor
template <typename DataType>
class DummyStoragePreprocessor : public PreprocessorInterface {
public:
    DummyStoragePreprocessor(std::shared_ptr<VecSimAllocator> allocator, int value_to_add_storage,
                             int value_to_add_query = 0)
        : PreprocessorInterface(allocator), value_to_add_storage(value_to_add_storage),
          value_to_add_query(value_to_add_query) {
        if (!value_to_add_query)
            value_to_add_query = value_to_add_storage;
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t processed_bytes_count, unsigned char alignment) const override {

        this->preprocessForStorage(original_blob, storage_blob, processed_bytes_count);
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t processed_bytes_count) const override {
        // If the blob was not allocated yet, allocate it.
        if (blob == nullptr) {
            blob = this->allocator->allocate(processed_bytes_count);
            memcpy(blob, original_blob, processed_bytes_count);
        }
        static_cast<DataType *>(blob)[0] += value_to_add_storage;
    }
    void preprocessQueryInPlace(void *blob, size_t processed_bytes_count,
                                unsigned char alignment) const override {}
    void preprocessQuery(const void *original_blob, void *&blob, size_t processed_bytes_count,
                         unsigned char alignment) const override {
        /* do nothing*/
    }

private:
    int value_to_add_storage;
    int value_to_add_query;
};

// Dummy query preprocessor
template <typename DataType>
class DummyQueryPreprocessor : public PreprocessorInterface {
public:
    DummyQueryPreprocessor(std::shared_ptr<VecSimAllocator> allocator, int value_to_add_storage,
                           int _value_to_add_query = 0)
        : PreprocessorInterface(allocator), value_to_add_storage(value_to_add_storage),
          value_to_add_query(_value_to_add_query) {
        if (!_value_to_add_query)
            value_to_add_query = value_to_add_storage;
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t processed_bytes_count, unsigned char alignment) const override {
        this->preprocessQuery(original_blob, query_blob, processed_bytes_count, alignment);
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t processed_bytes_count) const override {
        /* do nothing*/
    }
    void preprocessQueryInPlace(void *blob, size_t processed_bytes_count,
                                unsigned char alignment) const override {
        static_cast<DataType *>(blob)[0] += value_to_add_query;
    }
    void preprocessQuery(const void *original_blob, void *&blob, size_t processed_bytes_count,
                         unsigned char alignment) const override {
        // If the blob was not allocated yet, allocate it.
        if (blob == nullptr) {
            blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
            memcpy(blob, original_blob, processed_bytes_count);
        }
        static_cast<DataType *>(blob)[0] += value_to_add_query;
    }

private:
    int value_to_add_storage;
    int value_to_add_query;
};

// Dummy mixed preprocessor (precesses the blobs  differently)
template <typename DataType>
class DummyMixedPreprocessor : public PreprocessorInterface {
public:
    DummyMixedPreprocessor(std::shared_ptr<VecSimAllocator> allocator, int value_to_add_storage,
                           int value_to_add_query)
        : PreprocessorInterface(allocator), value_to_add_storage(value_to_add_storage),
          value_to_add_query(value_to_add_query) {}
    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t processed_bytes_count, unsigned char alignment) const override {

        // One blob was already allocated by a previous preprocessor(s) that process both blobs the
        // same. The blobs are pointing to the same memory, we need to allocate another memory slot
        // to split them.
        if ((storage_blob == query_blob) && (query_blob != nullptr)) {
            storage_blob = this->allocator->allocate(processed_bytes_count);
            memcpy(storage_blob, query_blob, processed_bytes_count);
        }

        // Either both are nullptr or they are pointing to different memory slots. Both cases are
        // handled by the designated functions.
        this->preprocessForStorage(original_blob, storage_blob, processed_bytes_count);
        this->preprocessQuery(original_blob, query_blob, processed_bytes_count, alignment);
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t processed_bytes_count) const override {
        // If the blob was not allocated yet, allocate it.
        if (blob == nullptr) {
            blob = this->allocator->allocate(processed_bytes_count);
            memcpy(blob, original_blob, processed_bytes_count);
        }
        static_cast<DataType *>(blob)[0] += value_to_add_storage;
    }
    void preprocessQueryInPlace(void *blob, size_t processed_bytes_count,
                                unsigned char alignment) const override {}
    void preprocessQuery(const void *original_blob, void *&blob, size_t processed_bytes_count,
                         unsigned char alignment) const override {
        // If the blob was not allocated yet, allocate it.
        if (blob == nullptr) {
            blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
            memcpy(blob, original_blob, processed_bytes_count);
        }
        static_cast<DataType *>(blob)[0] += value_to_add_query;
    }

private:
    int value_to_add_storage;
    int value_to_add_query;
};
} // namespace dummyPreprocessors

TEST(PreprocessorsTest, PreprocessorsTestBasicAlignmentTest) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    unsigned char alignment = 5;
    auto preprocessor = PreprocessorsContainerAbstract(allocator, alignment);
    const int original_blob[4] = {1, 1, 1, 1};
    size_t processed_bytes_count = sizeof(original_blob);

    {
        auto aligned_query = preprocessor.preprocessQuery(original_blob, processed_bytes_count);
        unsigned char address_alignment = (uintptr_t)(aligned_query.get()) % alignment;
        ASSERT_EQ(address_alignment, 0);
    }

    // The index computer is responsible for releasing the distance calculator.
}

template <unsigned char alignment>
void MultiPPContainerEmpty() {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    constexpr size_t dim = 4;
    const int original_blob[dim] = {1, 2, 3, 4};
    const int original_blob_cpy[dim] = {1, 2, 3, 4};

    constexpr size_t n_preprocessors = 3;

    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        // Original blob should not be changed
        CompareVectors(original_blob, original_blob_cpy, dim);

        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();

        // Storage blob should not be reallocated or changed
        ASSERT_EQ(storage_blob, (const int *)original_blob);
        CompareVectors(original_blob, (const int *)storage_blob, dim);

        // query blob *values* should not be changed
        CompareVectors(original_blob, (const int *)query_blob, dim);

        // If alignment is set the query blob address should be aligned to the specified alignment.
        if constexpr (alignment) {
            unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
            ASSERT_EQ(address_alignment, 0);
        }
    }
}

TEST(PreprocessorsTest, MultiPPContainerEmptyNoAlignment) {
    using namespace dummyPreprocessors;
    MultiPPContainerEmpty<0>();
}

TEST(PreprocessorsTest, MultiPPContainerEmptyAlignment) {
    using namespace dummyPreprocessors;
    MultiPPContainerEmpty<5>();
}

template <typename PreprocessorType>
void MultiPreprocessorsContainerNoAlignment(dummyPreprocessors::pp_mode MODE) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    unsigned char alignment = 0;
    int initial_value = 1;
    int value_to_add = 7;
    const int original_blob[4] = {initial_value, initial_value, initial_value, initial_value};
    size_t processed_bytes_count = sizeof(original_blob);

    // Test computer with multiple preprocessors of the same type.
    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    auto verify_preprocess = [&](int expected_processed_value) {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, processed_bytes_count);
        // Original blob should not be changed
        ASSERT_EQ(original_blob[0], initial_value);

        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        if (MODE == STORAGE_ONLY) {
            // New storage blob should be allocated
            ASSERT_NE(storage_blob, original_blob);
            // query blob should be unprocessed
            ASSERT_EQ(query_blob, original_blob);
            ASSERT_EQ(((const int *)storage_blob)[0], expected_processed_value);
        } else if (MODE == QUERY_ONLY) {
            // New query blob should be allocated
            ASSERT_NE(query_blob, original_blob);
            // Storage blob should be unprocessed
            ASSERT_EQ(storage_blob, original_blob);
            ASSERT_EQ(((const int *)query_blob)[0], expected_processed_value);
        }
    };

    /* ==== Add the first preprocessor ==== */
    auto preprocessor0 = new (allocator) PreprocessorType(allocator, value_to_add);
    // add preprocessor returns next free spot in its preprocessors array.
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor0), 1);
    verify_preprocess(initial_value + value_to_add);

    /* ==== Add the second preprocessor ==== */
    auto preprocessor1 = new (allocator) PreprocessorType(allocator, value_to_add);
    // add preprocessor returns 0 when adding the last preprocessor.
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor1), 0);
    ASSERT_NO_FATAL_FAILURE(verify_preprocess(initial_value + 2 * value_to_add));
}

TEST(PreprocessorsTest, MultiPreprocessorsContainerStorageNoAlignment) {
    using namespace dummyPreprocessors;
    MultiPreprocessorsContainerNoAlignment<DummyStoragePreprocessor<DummyType>>(
        pp_mode::STORAGE_ONLY);
}

TEST(PreprocessorsTest, MultiPreprocessorsContainerQueryNoAlignment) {
    using namespace dummyPreprocessors;
    MultiPreprocessorsContainerNoAlignment<DummyQueryPreprocessor<DummyType>>(pp_mode::QUERY_ONLY);
}

template <typename FirstPreprocessorType, typename SecondPreprocessorType>
void multiPPContainerMixedPreprocessorNoAlignment() {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 3;
    unsigned char alignment = 0;
    int initial_value = 1;
    int value_to_add_storage = 7;
    int value_to_add_query = 2;
    const int original_blob[4] = {initial_value, initial_value, initial_value, initial_value};
    size_t processed_bytes_count = sizeof(original_blob);

    // Test multiple preprocessors of the same type.
    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    /* ==== Add one preprocessor of each type ==== */
    auto preprocessor0 =
        new (allocator) FirstPreprocessorType(allocator, value_to_add_storage, value_to_add_query);
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor0), 1);
    auto preprocessor1 =
        new (allocator) SecondPreprocessorType(allocator, value_to_add_storage, value_to_add_query);
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor1), 2);

    // scope this section so the blobs are released before the allocator.
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, processed_bytes_count);
        // Original blob should not be changed
        ASSERT_EQ(original_blob[0], initial_value);

        // Both blobs should be allocated
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();

        // Ensure the computer process returns a new allocation of the expected processed blob with
        // the new value.
        ASSERT_NE(storage_blob, original_blob);
        ASSERT_NE(query_blob, original_blob);
        ASSERT_NE(query_blob, storage_blob);

        ASSERT_EQ(((const int *)storage_blob)[0], initial_value + value_to_add_storage);
        ASSERT_EQ(((const int *)query_blob)[0], initial_value + value_to_add_query);
    }

    /* ==== Add a preprocessor that processes both storage and query ==== */
    auto preprocessor2 = new (allocator)
        DummyMixedPreprocessor<DummyType>(allocator, value_to_add_storage, value_to_add_query);
    // add preprocessor returns 0 when adding the last preprocessor.
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor2), 0);
    {
        ProcessedBlobs mixed_processed_blobs =
            multiPPContainer.preprocess(original_blob, processed_bytes_count);

        const void *mixed_pp_storage_blob = mixed_processed_blobs.getStorageBlob();
        const void *mixed_pp_query_blob = mixed_processed_blobs.getQueryBlob();

        // Ensure the computer process both blobs.
        ASSERT_EQ(((const int *)mixed_pp_storage_blob)[0],
                  initial_value + 2 * value_to_add_storage);
        ASSERT_EQ(((const int *)mixed_pp_query_blob)[0], initial_value + 2 * value_to_add_query);
    }

    // try adding another preprocessor and fail.
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor2), -1);
}

TEST(PreprocessorsTest, multiPPContainerMixedPreprocessorQueryFirst) {
    using namespace dummyPreprocessors;
    multiPPContainerMixedPreprocessorNoAlignment<DummyQueryPreprocessor<DummyType>,
                                                 DummyStoragePreprocessor<DummyType>>();
}

TEST(PreprocessorsTest, multiPPContainerMixedPreprocessorStorageFirst) {
    using namespace dummyPreprocessors;
    multiPPContainerMixedPreprocessorNoAlignment<DummyStoragePreprocessor<DummyType>,
                                                 DummyQueryPreprocessor<DummyType>>();
}

template <typename PreprocessorType>
void multiPPContainerAlignment(dummyPreprocessors::pp_mode MODE) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    unsigned char alignment = 5;
    constexpr size_t n_preprocessors = 1;
    int initial_value = 1;
    int value_to_add = 7;
    const int original_blob[4] = {initial_value, initial_value, initial_value, initial_value};
    size_t processed_bytes_count = sizeof(original_blob);

    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    auto verify_preprocess = [&](int expected_processed_value) {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, processed_bytes_count);

        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        if (MODE == STORAGE_ONLY) {
            // New storage blob should be allocated and processed
            ASSERT_NE(storage_blob, original_blob);
            ASSERT_EQ(((const int *)storage_blob)[0], expected_processed_value);
            // query blob *values* should be unprocessed, however, it might be allocated if the
            // original blob is not aligned.
            ASSERT_EQ(((const int *)query_blob)[0], original_blob[0]);
        } else if (MODE == QUERY_ONLY) {
            // New query blob should be allocated
            ASSERT_NE(query_blob, original_blob);
            // Storage blob should be unprocessed and not allocated.
            ASSERT_EQ(storage_blob, original_blob);
            ASSERT_EQ(((const int *)query_blob)[0], expected_processed_value);
        }

        // anyway the query blob should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);
    };

    auto preprocessor0 = new (allocator) PreprocessorType(allocator, value_to_add);
    // add preprocessor returns next free spot in its preprocessors array.
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor0), 0);
    verify_preprocess(initial_value + value_to_add);
}

TEST(PreprocessorsTest, StoragePreprocessorWithAlignment) {
    using namespace dummyPreprocessors;
    multiPPContainerAlignment<DummyStoragePreprocessor<DummyType>>(pp_mode::STORAGE_ONLY);
}

TEST(PreprocessorsTest, QueryPreprocessorWithAlignment) {
    using namespace dummyPreprocessors;
    multiPPContainerAlignment<DummyQueryPreprocessor<DummyType>>(pp_mode::QUERY_ONLY);
}

TEST(PreprocessorsTest, multiPPContainerCosineThenMixedPreprocess) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    constexpr size_t dim = 4;
    unsigned char alignment = 5;

    float initial_value = 1.0f;
    float normalized_value = 0.5f;
    float value_to_add_storage = 7.0f;
    float value_to_add_query = 2.0f;
    const float original_blob[dim] = {initial_value, initial_value, initial_value, initial_value};

    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    // adding cosine preprocessor
    auto cosine_preprocessor = new (allocator) CosinePreprocessor<float>(allocator, dim);
    multiPPContainer.addPreprocessor(cosine_preprocessor);
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to the same memory slot
        ASSERT_EQ(storage_blob, query_blob);
        // memory should be aligned
        unsigned char address_alignment = (uintptr_t)(storage_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);
        // They need to be allocated and processed
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_EQ(((const float *)storage_blob)[0], normalized_value);
        // the original blob should not change
        ASSERT_NE(storage_blob, original_blob);
    }
    // adding mixed preprocessor
    auto mixed_preprocessor = new (allocator)
        DummyMixedPreprocessor<float>(allocator, value_to_add_storage, value_to_add_query);
    multiPPContainer.addPreprocessor(mixed_preprocessor);
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to a different memory slot
        ASSERT_NE(storage_blob, query_blob);
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(query_blob, nullptr);

        // query blob should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);

        // They need to be processed by both processors.
        ASSERT_EQ(((const float *)storage_blob)[0], normalized_value + value_to_add_storage);
        ASSERT_EQ(((const float *)query_blob)[0], normalized_value + value_to_add_query);

        // the original blob should not change
        ASSERT_NE(storage_blob, original_blob);
        ASSERT_NE(query_blob, original_blob);
    }
    // The preprocessors should be released by the preprocessors container.
}

TEST(PreprocessorsTest, multiPPContainerMixedThenCosinePreprocess) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    constexpr size_t dim = 4;
    unsigned char alignment = 5;

    float initial_value = 1.0f;
    float normalized_value = 0.5f;
    float value_to_add_storage = 7.0f;
    float value_to_add_query = 2.0f;
    const float original_blob[dim] = {initial_value, initial_value, initial_value, initial_value};

    // Creating multi preprocessors container
    auto mixed_preprocessor = new (allocator)
        DummyMixedPreprocessor<float>(allocator, value_to_add_storage, value_to_add_query);
    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);
    multiPPContainer.addPreprocessor(mixed_preprocessor);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to a different memory slot
        ASSERT_NE(storage_blob, query_blob);
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(query_blob, nullptr);

        // query blob should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);

        // They need to be processed by both processors.
        ASSERT_EQ(((const float *)storage_blob)[0], initial_value + value_to_add_storage);
        ASSERT_EQ(((const float *)query_blob)[0], initial_value + value_to_add_query);

        // the original blob should not change
        ASSERT_NE(storage_blob, original_blob);
        ASSERT_NE(query_blob, original_blob);
    }

    // adding cosine preprocessor
    auto cosine_preprocessor = new (allocator) CosinePreprocessor<float>(allocator, dim);
    multiPPContainer.addPreprocessor(cosine_preprocessor);
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to a different memory slot
        ASSERT_NE(storage_blob, query_blob);
        // query memory should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);
        // They need to be allocated and processed
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(query_blob, nullptr);
        float expected_processed_storage[dim] = {initial_value + value_to_add_storage,
                                                 initial_value, initial_value, initial_value};
        float expected_processed_query[dim] = {initial_value + value_to_add_query, initial_value,
                                               initial_value, initial_value};
        VecSim_Normalize(expected_processed_storage, dim, VecSimType_FLOAT32);
        VecSim_Normalize(expected_processed_query, dim, VecSimType_FLOAT32);
        ASSERT_EQ(((const float *)storage_blob)[0], expected_processed_storage[0]);
        ASSERT_EQ(((const float *)query_blob)[0], expected_processed_query[0]);
        // the original blob should not change
        ASSERT_NE(storage_blob, original_blob);
        ASSERT_NE(query_blob, original_blob);
    }
    // The preprocessors should be released by the preprocessors container.
}
