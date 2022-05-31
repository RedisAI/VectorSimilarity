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
