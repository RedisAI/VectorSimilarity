#include "gtest/gtest.h"
#include "VecSim/vecsim.h"

class BruteForceTest : public ::testing::Test {
  protected:
    BruteForceTest() {}

    ~BruteForceTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(BruteForceTest, brute_force_vector_add_test) {
    VecSimParams params = {
        hnswParams : {initialCapacity : 200, M : 16, efConstruction : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_IP,
        algo : VecSimAlgo_BF
    };
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[4] = {1.0, 1.0, 1.0, 1.0};
    VecSimIndex_AddVector(index, (const void *)a, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    VecSimIndex_Free(index);
}