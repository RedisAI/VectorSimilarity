#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/utils/arr_cpp.h"

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

    // Actual test: sets the vecsimindex timeout callback to always return timeout
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; });

    rl = VecSimIndex_TopKQuery(index, vec, 1, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    VecSimQueryResult_Free(rl);

    VecSimIndex_Free(index);
}