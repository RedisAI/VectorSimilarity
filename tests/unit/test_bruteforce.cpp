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
    }
    for (size_t i = n - 1; i > n - 1 - k; i--) {
        ASSERT_EQ(i, ids[i]);
    }
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_vector_search_test_l2) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
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
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    for (int i = 0; i < k; i++) {
        int diff_id = ((int)(res[i].id - 50) > 0) ? (res[i].id - 50) : (50 - res[i].id);
        int dist = res[i].score;
        ASSERT_TRUE(((diff_id == (i + 1) / 2)) && (dist == (4 * ((i + 1) / 2) * ((i + 1) / 2))));
    }
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_vector_search_by_id_test) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_BF
    };
    size_t k = 11;
    VecSimIndex *index = VecSimIndex_New(&params);

    for (float i = 0; i < 100; i++) {
        float f[4] = {i, i, i, i};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 100);

    float query[4] = {50, 50, 50, 50};
    VecSimQueryResult *res = VecSimIndex_TopKQueryByID(index, (const void *)query, k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    for (int i = 0; i < k; i++) {
        ASSERT_EQ(res[i].id, (i + 45));
    }
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_indexing_same_vector) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_BF
    };

    size_t n = 100;
    size_t k = 10;

    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float num = i / 10;
        float f[4] = {num, num, num, num};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float query[4] = {4.9, 4.95, 5.05, 5.1};
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(res[i].id >= 50 && res[i].id < 60 && res[i].score <= 1);
    }
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_reindexing_same_vector) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_BF
    };

    size_t n = 100;
    size_t k = 10;

    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float num = i / 10;
        float f[4] = {num, num, num, num};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float query[4] = {4.9, 4.95, 5.05, 5.1};
    size_t ids[n] = {0};
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(res[i].id >= 50 && res[i].id < 60 && res[i].score <= 1);
        ids[res[i].id] = res[i].id;
    }
    VecSimQueryResult_Free(res);
    for (size_t i = 50; i < 60; i++) {
        ASSERT_EQ(ids[i], i);
        ids[i] = 0;
    }

    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    for (size_t i = 0; i < n; i++) {
        float num = i / 10;
        float f[4] = {num, num, num, num};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    res = VecSimIndex_TopKQuery(index, (const void *)query, 10, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(res[i].id >= 50 && res[i].id < 60 && res[i].score <= 1);

        ids[res[i].id] = res[i].id;
    }
    for (size_t i = 50; i < 60; i++) {
        ASSERT_EQ(ids[i], i);
        ids[i] = 0;
    }
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_reindexing_same_vector_different_id) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_BF
    };
    size_t n = 100;
    size_t k = 10;

    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float num = i / 10;
        float f[4] = {num, num, num, num};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float query[4] = {4.9, 4.95, 5.05, 5.1};
    size_t ids[100] = {0};
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index, (const void *)query, 10, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(res[i].id >= 50 && res[i].id < 60 && res[i].score <= 1);
        ids[res[i].id] = res[i].id;
    }
    VecSimQueryResult_Free(res);
    for (size_t i = 50; i < 60; i++) {
        ASSERT_EQ(ids[i], i);
        ids[i] = 0;
    }

    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }

    for (size_t i = 0; i < n; i++) {
        float num = i / 10;
        float f[4] = {num, num, num, num};
        VecSimIndex_AddVector(index, (const void *)f, i + 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    res = VecSimIndex_TopKQuery(index, (const void *)query, 10, NULL);
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(res[i].id >= 60 && res[i].id < 70 && res[i].score <= 1);
        ids[res[i].id] = res[i].id;
    }
    for (size_t i = 60; i < 70; i++) {
        ASSERT_EQ(ids[i], i);
        ids[i] = 0;
    }
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, sanity_rinsert_1280) {
    size_t n = 5;
    size_t d = 1280;
    size_t k = 5;

    VecSimParams params = {
        bfParams : {initialCapacity : n},
        type : VecSimType_FLOAT32,
        size : d,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_BF,
    };
    VecSimIndex *index = VecSimIndex_New(&params);

    float *vectors = (float *)malloc(n * d * sizeof(float));
    for (size_t iter = 1; iter <= 3; iter++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                (vectors + i * d)[j] = (float)rand() / (float)(RAND_MAX / 100);
            }
        }
        for (size_t i = 0; i < n; i++) {
            VecSimIndex_AddVector(index, (const void *)(vectors + i * d), i * iter);
        }
        VecSimQueryResult *res =
            VecSimIndex_TopKQuery(index, (const void *)(vectors + 3 * d), k, NULL);
        ASSERT_EQ(VecSimQueryResult_Len(res), k);
        size_t ids[5] = {0};
        for (int i = 0; i < k; i++) {
            ids[res[i].id / iter] = res[i].id / iter;
        }
        for (size_t i = 0; i < k; i++) {
            ASSERT_EQ(ids[i], i);
            ids[i] = 0;
        }
        for (size_t i = 0; i < n; i++) {
            VecSimIndex_DeleteVector(index, i * iter);
        }
        VecSimQueryResult_Free(res);
    }
    free(vectors);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, test_bf_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args
    VecSimParams params = {
        bfParams : {
            initialCapacity : n,
        },
        type : VecSimType_FLOAT32,
        size : d,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_BF,
    };
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.d, d);
    // Default args
    ASSERT_EQ(info.bfInfo.blockSize, BF_DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(info.bfInfo.indexSize, 0);
    VecSimIndex_Free(index);

    d = 1280;
    params = {
        bfParams : {
            initialCapacity : n,
            blockSize : 1

        },
        type : VecSimType_FLOAT32,
        size : d,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_BF,
    };
    index = VecSimIndex_New(&params);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.d, d);
    // User args
    ASSERT_EQ(info.bfInfo.blockSize, 1);
    ASSERT_EQ(info.bfInfo.indexSize, 0);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_vector_search_test_ip_blocksize_1) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200, blockSize : 1},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_IP,
        algo : VecSimAlgo_BF
    };
    size_t n = 100;
    size_t k = 11;
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.bfInfo.blockSize, 1);

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
    }
    for (size_t i = n - 1; i > n - 1 - k; i--) {
        ASSERT_EQ(i, ids[i]);
    }
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_vector_search_test_blocksize_1) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200, blockSize : 1},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_BF
    };
    size_t n = 100;
    size_t k = 11;
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.bfInfo.blockSize, 1);
    for (float i = 0; i < n; i++) {
        float f[4] = {i, i, i, i};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[4] = {50, 50, 50, 50};
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    for (int i = 0; i < k; i++) {
        int diff_id = ((int)(res[i].id - 50) > 0) ? (res[i].id - 50) : (50 - res[i].id);
        int dist = res[i].score;
        ASSERT_TRUE(((diff_id == (i + 1) / 2)) && (dist == (4 * ((i + 1) / 2) * ((i + 1) / 2))));
    }
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_search_empty_index) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_BF
    };
    size_t n = 100;
    size_t k = 11;
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float query[4] = {50, 50, 50, 50};
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);

    for (float i = 0; i < n; i++) {
        float f[4] = {i, i, i, i};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    for (float i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_test_inf_score) {
    /* In this test the 32 bits of "efgh" and "efgg", and the 32 bits of "abcd" and "abbd" will
     * yield "inf" result.
     */
    VecSimParams params = {
        bfParams : {initialCapacity : 4},
        type : VecSimType_FLOAT32,
        size : 2,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_BF
    };
    size_t n = 4;
    size_t k = 4;
    VecSimIndex *index = VecSimIndex_New(&params);

    VecSimIndex_AddVector(index, "abcdefgh", 1);
    VecSimIndex_AddVector(index, "abcdefgg", 2);
    VecSimIndex_AddVector(index, "aacdefgh", 3);
    VecSimIndex_AddVector(index, "abbdefgh", 4);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 4);

    VecSimQueryResult *res = VecSimIndex_TopKQuery(index, "abcdefgh", k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), 4);
    ASSERT_EQ(1, res[0].id);
    ASSERT_EQ(3, res[1].id);
    ASSERT_TRUE((2 == res[2].id && 4 == res[3].id) || (4 == res[2].id && 2 == res[3].id));

    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}
