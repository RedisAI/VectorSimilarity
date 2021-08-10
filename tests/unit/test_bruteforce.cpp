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
        bfParams : {initialCapacity : 200},
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

TEST_F(BruteForceTest, brute_force_vector_search_test_ip) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_IP,
        algo : VecSimAlgo_BF
    };
    size_t n = 100;
    size_t k = 11;
    VecSimIndex *index = VecSimIndex_New(&params);

    for (float i = 0; i < n; i++) {
        float f[4] = {i, i, i, i};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[4] = {50, 50, 50, 50};
    size_t ids[100] = {0};
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    for (int i = 0; i < k; i++) {
        ids[res[i].id] = res[i].id;
        std::cout<<res[i].id<<std::endl;
    }
    for(size_t i = 89; i < 99; i++) {
        ASSERT_EQ(i, ids[i]);
    }
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}