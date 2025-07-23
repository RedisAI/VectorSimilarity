/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
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
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "mock_thread_pool.h"
#include "tests_utils.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"

#include <cstdlib>
#include <limits>
#include <cmath>
#include <random>
#include <cstdarg>
#include <filesystem>

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
    Serializer::writeBinaryPOD(output, VecSimMetric_Cosine);
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

TEST(CommonAPITest, NormalizeUint8) {
    size_t dim = 20;
    uint8_t v[dim + sizeof(float)];

    test_utils::populate_uint8_vec(v, dim);

    VecSim_Normalize(v, dim, VecSimType_UINT8);

    float res_norm = *(reinterpret_cast<float *>(v + dim));
    // Check that the normalized vector norm is 1.
    float norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        float val = v[i] / res_norm;
        norm += val * val;
    }

    ASSERT_FLOAT_EQ(norm, 1.0);
}

/**
 * This test verifies that a tiered index correctly returns the closest vectors when querying data
 * distributed across both the flat and the backend indices, specifically when duplicate labels
 * exist in both indices with different distances. It adds vectors with known scores, including such
 * duplicates, and ensures that only the closer instance is returned. The test covers both top-K and
 * range queries, validating result ordering by score and by ID.
 */
TEST(CommonAPITest, SearchDifferentScores) {
    size_t dim = 4;
    size_t constexpr k = 3;

    // Create TieredHNSW index instance with a mock queue.
    HNSWParams params = {
        .type = VecSimType_FLOAT32,
        .dim = dim,
        .metric = VecSimMetric_L2,
    };
    auto mock_thread_pool = tieredIndexMock();
    auto *tiered_index = dynamic_cast<TieredHNSWIndex<float, float> *>(
        test_utils::CreateNewTieredHNSWIndex(params, mock_thread_pool));
    ASSERT_NE(tiered_index, nullptr);

    auto hnsw_index = tiered_index->getHNSWIndex();
    auto flat_index = tiered_index->frontendIndex;

    // Define IDs and distance values for test vectors
    // ids are intentionally in random order to verify sorting works correctly
    size_t constexpr ids[k] = {54, 4, 15};
    double constexpr res_values[k] = {2, 3, 100};
    // Define a type for our result pair
    using ResultPair = std::pair<size_t, double>; // (id, score)

    // Create a vector of expected results - these are the scores we expect
    // when querying with a zero vector (L2 distance = value²*dim)
    std::vector<ResultPair> expected_results_by_score(k);

    for (size_t i = 0; i < k; i++) {
        expected_results_by_score[i] = {ids[i], res_values[i] * res_values[i] * dim};
    }

    // Insert duplicate vectors with same ID but different distances across the two indices.
    // The index should return only the closer of the two.

    // ID 54: closer in HNSW, farther in flat — expect to return HNSW version
    GenerateAndAddVector<float>(hnsw_index, dim, ids[0], res_values[0]);
    GenerateAndAddVector<float>(flat_index, dim, ids[0], res_values[0] + 1);

    // ID 4: closer in flat, farther in HNSW — expect to return flat version
    GenerateAndAddVector<float>(flat_index, dim, ids[1], res_values[1]);
    GenerateAndAddVector<float>(hnsw_index, dim, ids[1], res_values[1] + 1);

    // ID 15: identical in both indices — distance is large, should still return one instance
    GenerateAndAddVector<float>(hnsw_index, dim, ids[2], res_values[2]);
    GenerateAndAddVector<float>(flat_index, dim, ids[2], res_values[2]);

    // Create a zero vector for querying - this makes scores directly proportional to vector values
    float query_0[dim];
    GenerateVector<float>(query_0, dim, 0);

    // Verify results ordered by increasing score (distance).
    double prev_score = 0; // all scores are positive
    auto verify_by_score = [&](size_t id, double score, size_t res_index) {
        ASSERT_LT(prev_score, score); // prev_score < score
        prev_score = score;
        ASSERT_EQ(id, expected_results_by_score[res_index].first);
        ASSERT_EQ(score, expected_results_by_score[res_index].second);
    };

    runTopKTieredIndexSearchTest<true>(tiered_index, query_0, k, verify_by_score, nullptr);
    // Reset score tracking for range query
    prev_score = 0;
    // Use the largest score as the range to include all vectors
    double range = expected_results_by_score.back().second;
    runRangeTieredIndexSearchTest<true>(tiered_index, query_0, range, verify_by_score, k, BY_SCORE);
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
    size_t actual = test_utils::CalcVectorDataSize(index, type);
    size_t expected = dim * VecSimType_sizeof(type);
    if (metric == VecSimMetric_Cosine && (type == VecSimType_INT8 || type == VecSimType_UINT8)) {
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
        if (metric == VecSimMetric_Cosine &&
            (type == VecSimType_INT8 || type == VecSimType_UINT8)) {
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
    case VecSimType_UINT8: {
        auto tiered_index = test_utils::cast_to_tiered_index<uint8_t, float>(index);
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

constexpr VecSimType vecsim_datatypes[] = {VecSimType_FLOAT32,  VecSimType_FLOAT64,
                                           VecSimType_BFLOAT16, VecSimType_FLOAT16,
                                           VecSimType_INT8,     VecSimType_UINT8};

/** Run all CommonTypeMetricTests tests for each {VecSimType, VecSimMetric} combination */
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

/** Run all CommonTypeMetricTieredTests tests for each {VecSimType, VecSimMetric} combination */
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

TEST(CommonAPITest, testSetTestLogContext) {
    // Create an index with the log context
    BFParams bfParams = {.dim = 1, .metric = VecSimMetric_L2, .blockSize = 5};
    VecSimIndex *index = test_utils::CreateNewIndex(bfParams, VecSimType_FLOAT32);
    auto *bf_index = dynamic_cast<BruteForceIndex<float, float> *>(index);

    std::string log_dir = "logs/tests/unit";
    std::cout << "Log directory: " << log_dir << std::endl;
    if (!std::filesystem::exists(log_dir)) {
        std::filesystem::create_directories(log_dir);
    }
    bf_index->log(VecSimCommonStrings::LOG_VERBOSE_STRING, "%s", "printed before setting context");
    // Set the log context
    const char *testContext = "test_context";
    VecSim_SetTestLogContext(testContext, "unit");
    std::string msg = "Test message with context";
    // Trigger a log message
    bf_index->log(VecSimCommonStrings::LOG_VERBOSE_STRING, "%s", msg.c_str());

    // check if the log message was written to the log file
    std::string log_file = log_dir + "/test_context.log";
    std::ifstream file(log_file);
    ASSERT_TRUE(file.is_open()) << "Log file not found: " << log_file;
    std::string line;
    bool found = false;
    while (std::getline(file, line)) {
        if (line.find(msg) != std::string::npos) {
            found = true;
            break;
        }
    }

    ASSERT_TRUE(found) << "Log message not found in log file: " << log_file;
    VecSimIndex_Free(index);
}

TEST(UtilsTests, testMockThreadPool) {
    const size_t num_repeats = 2;
    const size_t num_submissions = 200;
    // 100 seconds timeout for the test should be enough for CI MemoryChecks
    std::chrono::seconds test_timeout(100);

    auto TestBody = [=]() {
        // Protection against test deadlock is implemented by a thread which exits process if
        // condition variable is not notified within a timeout.
        std::mutex mtx;
        std::condition_variable cv;
        auto guard_thread = std::thread([&]() {
            std::unique_lock<std::mutex> lock(mtx);
            if (cv.wait_for(lock, test_timeout) == std::cv_status::timeout) {
                std::cerr << "Test timeout! Exiting..." << std::endl;
                std::exit(-1);
            }
        });

        // Create and test a mock thread pool several times
        for (size_t i = 0; i < num_repeats; i++) {
            // Create a mock thread pool and verify its properties
            tieredIndexMock mock_thread_pool;
            ASSERT_EQ(mock_thread_pool.ctx->index_strong_ref, nullptr);
            ASSERT_TRUE(mock_thread_pool.jobQ.empty());

            // Create a new stub index to add to the mock thread pool
            BFParams params = {.dim = 4, .metric = VecSimMetric_L2};
            auto index = test_utils::CreateNewIndex(params, VecSimType_FLOAT32);
            mock_thread_pool.ctx->index_strong_ref.reset(index);
            auto allocator = index->getAllocator();

            // Very fast and simple job routine that increments a counter
            // This is just to simulate a job that does some work.
            std::atomic_int32_t job_counter = 0;
            auto job_mock = [&job_counter](AsyncJob * /*unused*/) { job_counter++; };

            // Define a mock job just to convert lambda with capture to a function pointer
            class LambdaJob : public AsyncJob {
            public:
                LambdaJob(std::shared_ptr<VecSimAllocator> allocator, JobType type,
                          std::function<void(AsyncJob *)> execute, VecSimIndex *index)
                    : AsyncJob(allocator, type, executeJob, index), impl_(execute) {}

                static void executeJob(AsyncJob *job) {
                    static_cast<LambdaJob *>(job)->impl_(job);
                    delete job; // Clean up the job after execution
                }
                std::function<void(AsyncJob *)> impl_;
            };

            mock_thread_pool.init_threads();
            // Verify the job queue is empty
            ASSERT_TRUE(mock_thread_pool.jobQ.empty());

            // Create a vector of jobs to submit to the mock thread pool
            // The number of jobs is equal to the thread pool size, so they will all be executed in
            // parallel
            std::vector<AsyncJob *> jobs(mock_thread_pool.thread_pool_size);

            // Submit jobs to the mock thread pool and wait several times
            for (size_t j = 0; j < num_submissions; j++) {
                job_counter.store(0); // Reset the counter for each iteration
                // Generate jobs and submit them to the mock thread pool
                std::generate(jobs.begin(), jobs.end(), [&]() {
                    return new (allocator) LambdaJob(allocator, HNSW_SEARCH_JOB, job_mock, index);
                });
                mock_thread_pool.submit_callback_internal(jobs.data(), nullptr /*unused*/,
                                                          jobs.size());
                mock_thread_pool.thread_pool_wait();
                // Verify the job queue is empty
                ASSERT_TRUE(mock_thread_pool.jobQ.empty());
                // Verify counter was incremented
                ASSERT_EQ(job_counter.load(), mock_thread_pool.thread_pool_size);
            }
            mock_thread_pool.thread_pool_join();
        }

        // Notify the guard thread that the test is done
        cv.notify_one();
        guard_thread.join();
        std::cerr << "Success" << std::endl;
        std::exit(testing::Test::HasFailure() ? -1 : 0); // Exit with failure if any test failed
    };

    EXPECT_EXIT(TestBody(), ::testing::ExitedWithCode(0), "Success");
}
