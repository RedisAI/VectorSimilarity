#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/updatable_heap.h"

#include <cstdlib> // rand, RAND_MAX
#include <limits>  //numeric_limits
#include <cmath>   // exp

class CommonTest : public ::testing::Test {
protected:
    CommonTest() {}

    ~CommonTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CommonTest, ResolveQueryRuntimeParams) {
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 0,
                                             .blockSize = 5}};
    VecSimIndex *index = VecSimIndex_New(&params);

    VecSimQueryParams qparams, zero;
    bzero(&zero, sizeof(VecSimQueryParams));

    auto *rparams = array_new<VecSimRawParam>(2);

    // Empty raw params array, nothing should change in query params.
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSim_OK);
    ASSERT_EQ(memcmp(&qparams, &zero, sizeof(VecSimQueryParams)), 0);

    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), nullptr, false),
              VecSimParamResolverErr_NullParam);

    /** Testing with common hybrid query params. **/
    array_append(rparams, (VecSimRawParam){.name = "batch_size",
                                           .nameLen = strlen("batch_size"),
                                           .value = "100",
                                           .valLen = strlen("100")});
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSim_OK);
    ASSERT_EQ(qparams.batchSize, 100);

    // Both params are "batch_size".
    array_append(rparams, (VecSimRawParam){.name = "batch_size",
                                           .nameLen = strlen("batch_size"),
                                           .value = "200",
                                           .valLen = strlen("200")});
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_AlreadySet);

    rparams[1] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "batches_wrong",
                                  .valLen = strlen("batches_wrong")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_InvalidPolicy_NExits);

    rparams[1].value = "batches";
    rparams[1].valLen = strlen("batches");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 100);

    // Both params are "hybrid policy".
    rparams[0] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "ADhOC_bf",
                                  .valLen = strlen("ADhOC_bf")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_AlreadySet);

    // Sending HYBRID_POLICY=adhoc as the single parameter is valid.
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, 1, &qparams, true), VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_ADHOC_BF);

    // Cannot set batch_size param with "hybrid_policy" which is "ADHOC_BF"
    rparams[1] = (VecSimRawParam){.name = "batch_size",
                                  .nameLen = strlen("batch_size"),
                                  .value = "100",
                                  .valLen = strlen("100")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_InvalidPolicy_AdHoc_With_BatchSize);

    rparams[0] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "batches",
                                  .valLen = strlen("batches")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 100);

    // Trying to set hybrid policy for non-hybrid query.
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSimParamResolverErr_InvalidPolicy_NHybrid);
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams + 1, 1, &qparams, false),
              VecSimParamResolverErr_InvalidPolicy_NHybrid);

    // Check for invalid batch sizes params.
    rparams[1].value = "not_a_number";
    rparams[1].valLen = strlen("not_a_number");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "9223372036854775808"; // LLONG_MAX+1
    rparams[1].valLen = strlen("9223372036854775808");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "-5";
    rparams[1].valLen = strlen("-5");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "0";
    rparams[1].valLen = strlen("0");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "10f";
    rparams[1].valLen = strlen("10f");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_BadValue);

    VecSimIndex_Free(index);
    array_free(rparams);
}

TEST_F(CommonTest, SetTimeoutCallbackFunction) {
    size_t dim = 4;
    float vec[] = {1.0f, 1.0f, 1.0f, 1.0f};
    VecSimQueryResult_List rl;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 1,
                                             .blockSize = 5}};
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndex_AddVector(index, vec, 0);

    rl = VecSimIndex_TopKQuery(index, vec, 1, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_OK);
    VecSimQueryResult_Free(rl);

    // Actual test: sets the vecsim index timeout callback to always return timeout
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; });

    rl = VecSimIndex_TopKQuery(index, vec, 1, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    VecSimQueryResult_Free(rl);

    VecSimIndex_Free(index);
}

TEST_F(CommonTest, Max_Updatable_Heap) {
    std::pair<int, size_t> p;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    vecsim_stl::updatable_max_heap<int, size_t> heap(allocator);

    // Initial state checks
    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());
    ASSERT_NO_THROW(heap.top());

    // Insert some data in random order
    size_t riders[] = {46, 16, 99, 93};
    const size_t n_riders = sizeof(riders) / sizeof(riders[0]);
    heap.emplace(1, 46);
    heap.emplace(4, 93);
    heap.emplace(3, 99);
    heap.emplace(2, 16);

    for (int i = n_riders; i > 0; i--) {
        ASSERT_EQ(heap.size(), i);
        p = {i, riders[i - 1]};
        ASSERT_TRUE(heap.top() == p);
        ASSERT_FALSE(heap.empty());
        heap.pop();
    }

    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());

    // Inserting data with the same priority
    heap.emplace(1, 1);
    heap.emplace(11, 55);
    heap.emplace(1, 3);
    heap.emplace(1, 2);

    ASSERT_EQ(heap.size(), 4);
    ASSERT_FALSE(heap.empty());
    p = {11, 55};
    ASSERT_TRUE(heap.top() == p);

    heap.emplace(0, 55); // Update priority

    ASSERT_EQ(heap.size(), 4); // Same size after update
    ASSERT_FALSE(heap.empty());

    // Make sure each pop deletes a single element, even if some have the same priority.
    size_t len = heap.size();
    for (size_t i = len; i > 0; i--) {
        ASSERT_EQ(heap.size(), i);
        ASSERT_FALSE(heap.empty());
        heap.pop();
    }
    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());

    // Update a priority of an element that share its priority with many others.
    size_t last = 10;
    for (size_t i = 0; i <= last; i++) {
        heap.emplace(2, i);
    }
    // Bound the existing elements with higher and lower priorities.
    heap.emplace(1, 42);
    heap.emplace(3, 46);
    size_t size = heap.size();

    // Update to the lowest priority
    heap.emplace(0, last);
    ASSERT_EQ(heap.size(), size);

    while (heap.size() > 1) {
        heap.pop();
    }
    ASSERT_EQ(heap.size(), 1);
    ASSERT_FALSE(heap.empty());
    p = {0, last};
    ASSERT_TRUE(heap.top() == p);
    heap.pop();
    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());
}

TEST_F(CommonTest, VecSim_Normalize_FP32_Vector) {
    const size_t dim = 1000;
    float v[dim];

    // generate random values - always generates the same values
    for (size_t i = 0; i < dim - 3; ++i) {
        v[i] = (float)rand() // generate an integral number between 0 and RAND_MAX.
               + (float)rand() / (float)(RAND_MAX); // add a fp32 number between 0 and 1;
    }

    // Change some of the vector's values so that the sum of the squared vector's
    // values will overflow for floats but not for doubles.
    v[dim - 3] = exp(44);
    v[dim - 2] = exp(44);
    v[dim - 1] = exp(44);

    // Normalize the vector
    VecSim_Normalize(v, dim, VecSimType_FLOAT32);

    // Check that the normelized vector norm is 1
    float norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        norm += v[i] * v[i];
    }

    ASSERT_FLOAT_EQ(1.0f, norm);
}

TEST_F(CommonTest, VecSim_Normalize_FP64_Vector) {
    const size_t dim = 1000;
    double v[dim];

    // generate random values - always generates the same values
    for (size_t i = 0; i < dim; ++i) {
        v[i] = (double)rand() // generate an integral number between 0 and RAND_MAX.
               + (double)rand() / (double)(RAND_MAX); // add a fp64 number between 0 and 1;
    }

    // Normalize the vector
    VecSim_Normalize(v, dim, VecSimType_FLOAT64);

    // Check that the normelized vector norm is 1.
    double norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        norm += v[i] * v[i];
    }
    ASSERT_LE(1.0 * 0.99, norm);
}
