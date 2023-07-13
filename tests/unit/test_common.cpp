/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/updatable_heap.h"
#include "VecSim/utils/vec_utils.h"
#include "test_utils.h"
#include "VecSim/utils/vecsim_results_container.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/index_factories/hnsw_factory.h"

#include <cstdlib>
#include <limits>
#include <cmath>
#include <random>
#include <cstdarg>

template <typename index_type_t>
class CommonIndexTest : public ::testing::Test {};

// DataTypeSet are defined in test_utils.h

TYPED_TEST_SUITE(CommonIndexTest, DataTypeSet);

TYPED_TEST(CommonIndexTest, ResolveQueryRuntimeParams) {
    size_t dim = 4;

    BFParams params = {.dim = dim, .metric = VecSimMetric_L2, .initialCapacity = 0, .blockSize = 5};
    VecSimIndex *index = test_utils::CreateNewIndex(params, TypeParam::get_index_type());

    VecSimQueryParams qparams, zero;
    bzero(&zero, sizeof(VecSimQueryParams));

    auto *rparams = array_new<VecSimRawParam>(2);

    // Empty raw params array, nothing should change in query params.
    for (VecsimQueryType query_type : test_utils::query_types) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, query_type),
            VecSim_OK);
    }
    ASSERT_EQ(memcmp(&qparams, &zero, sizeof(VecSimQueryParams)), 0);

    for (VecsimQueryType query_type : test_utils::query_types) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams, array_len(rparams), nullptr, query_type),
            VecSimParamResolverErr_NullParam);
    }

    /** Testing with common hybrid query params. **/
    array_append(rparams, (VecSimRawParam){.name = "batch_size",
                                           .nameLen = strlen("batch_size"),
                                           .value = "100",
                                           .valLen = strlen("100")});
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSim_OK);
    ASSERT_EQ(qparams.batchSize, 100);

    // Both params are "batch_size".
    array_append(rparams, (VecSimRawParam){.name = "batch_size",
                                           .nameLen = strlen("batch_size"),
                                           .value = "200",
                                           .valLen = strlen("200")});
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSimParamResolverErr_AlreadySet);

    rparams[1] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "batches_wrong",
                                  .valLen = strlen("batches_wrong")};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSimParamResolverErr_InvalidPolicy_NExits);

    rparams[1].value = "batches";
    rparams[1].valLen = strlen("batches");
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 100);

    // Both params are "hybrid policy".
    rparams[0] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "ADhOC_bf",
                                  .valLen = strlen("ADhOC_bf")};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSimParamResolverErr_AlreadySet);

    // Sending HYBRID_POLICY=adhoc as the single parameter is valid.
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, 1, &qparams, QUERY_TYPE_HYBRID), VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_ADHOC_BF);

    // Cannot set batch_size param with "hybrid_policy" which is "ADHOC_BF"
    rparams[1] = (VecSimRawParam){.name = "batch_size",
                                  .nameLen = strlen("batch_size"),
                                  .value = "100",
                                  .valLen = strlen("100")};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSimParamResolverErr_InvalidPolicy_AdHoc_With_BatchSize);

    rparams[0] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "batches",
                                  .valLen = strlen("batches")};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 100);

    // Trying to set hybrid policy for non-hybrid query.
    for (VecsimQueryType query_type : {QUERY_TYPE_NONE, QUERY_TYPE_KNN, QUERY_TYPE_RANGE}) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, query_type),
            VecSimParamResolverErr_InvalidPolicy_NHybrid);
        ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams + 1, 1, &qparams, query_type),
                  VecSimParamResolverErr_InvalidPolicy_NHybrid);
    }

    // Check for invalid batch sizes params.
    rparams[1].value = "not_a_number";
    rparams[1].valLen = strlen("not_a_number");
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSimParamResolverErr_BadValue);

    rparams[1].value = "9223372036854775808"; // LLONG_MAX+1
    rparams[1].valLen = strlen("9223372036854775808");
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSimParamResolverErr_BadValue);

    rparams[1].value = "-5";
    rparams[1].valLen = strlen("-5");
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSimParamResolverErr_BadValue);

    rparams[1].value = "0";
    rparams[1].valLen = strlen("0");
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSimParamResolverErr_BadValue);

    rparams[1].value = "10f";
    rparams[1].valLen = strlen("10f");
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSimParamResolverErr_BadValue);

    VecSimIndex_Free(index);
    array_free(rparams);
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
    normalizeVector(v, dim);

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

    auto res1 = VecSimQueryResult_List{0};
    auto res2 = VecSimQueryResult_List{0};
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

        res1.results = drc.get_results();
        res2.results = urc.get_results();
    }
    sort_results_by_id(res1);
    sort_results_by_score(res2);

    for (size_t i = 0; i < VecSimQueryResult_Len(res1); i++) {
        ASSERT_EQ(i, VecSimQueryResult_GetId(res1.results + i));
    }
    for (size_t i = 0; i < VecSimQueryResult_Len(res2); i++) {
        ASSERT_EQ(i, VecSimQueryResult_GetId(res2.results + i));
    }

    VecSimQueryResult_Free(res1);
    VecSimQueryResult_Free(res2);
}

class CommonAPITest : public ::testing::Test {};

TEST(CommonAPITest, VecSim_QueryResult_Iterator) {
    auto *res_array = array_new<VecSimQueryResult>(3);
    array_append(res_array, VecSimQueryResult{.id = 0, .score = 0.0});
    array_append(res_array, VecSimQueryResult{.id = 1, .score = 1.0});
    array_append(res_array, VecSimQueryResult{.id = 2, .score = 2.0});

    VecSimQueryResult_List res_list = {.results = res_array, .code = VecSim_QueryResult_OK};
    VecSimQueryResult *res_inner_array = VecSimQueryResult_GetArray(res_list);
    ASSERT_EQ(res_list.results, res_inner_array);

    ASSERT_EQ(3, VecSimQueryResult_Len(res_list));
    ASSERT_EQ(3, VecSimQueryResult_ArrayLen(res_inner_array));

    // Go over the list result with the iterator. Reset the iterator and re-iterate several times.
    VecSimQueryResult_Iterator *it = VecSimQueryResult_List_GetIterator(res_list);
    for (size_t rep = 0; rep < 3; rep++) {
        for (size_t i = 0; i < VecSimQueryResult_Len(res_list); i++) {
            ASSERT_TRUE(VecSimQueryResult_IteratorHasNext(it));
            VecSimQueryResult *res = VecSimQueryResult_IteratorNext(it);
            ASSERT_EQ(res, res_inner_array + i);
            ASSERT_EQ(i, VecSimQueryResult_GetId(res));
            ASSERT_EQ((double)i, VecSimQueryResult_GetScore(res));
        }
        ASSERT_FALSE(VecSimQueryResult_IteratorHasNext(it));
        VecSimQueryResult_IteratorReset(it);
    }

    // Destroying the iterator without destroying the list.
    VecSimQueryResult_IteratorFree(it);
    ASSERT_EQ(3, VecSimQueryResult_Len(res_list));

    // Free the internal array pointer - the validation for success is by having no leaks.
    VecSimQueryResult_FreeArray(res_inner_array);
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

    this->file_name = std::string(getenv("ROOT")) + "/tests/unit/data/bad_index.hnsw";

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

void test_log_impl(void *ctx, const char *message) {
    logCtx *log = (logCtx *)ctx;
    std::string msg = log->prefix + message;
    log->logBuffer.push_back(msg);
}

TEST(CommonAPITest, testlog) {

    logCtx log;
    log.prefix = "test log prefix: ";

    BFParams bfParams = {.dim = 1, .metric = VecSimMetric_L2, .initialCapacity = 0, .blockSize = 5};
    VecSimParams params = {
        .algo = VecSimAlgo_BF, .algoParams = {.bfParams = BFParams{bfParams}}, .logCtx = &log};
    auto *index =
        dynamic_cast<BruteForceIndex<float, float> *>(BruteForceFactory::NewIndex(&params));
    VecSim_SetLogCallbackFunction(test_log_impl);

    index->log("test log message no fmt");
    index->log("test log message %s %s", "with", "args");

    ASSERT_EQ(log.logBuffer.size(), 2);
    ASSERT_EQ(log.logBuffer[0], "test log prefix: test log message no fmt");
    ASSERT_EQ(log.logBuffer[1], "test log prefix: test log message with args");

    VecSimIndex_Free(index);
}
