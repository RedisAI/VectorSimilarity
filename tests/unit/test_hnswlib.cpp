#include "gtest/gtest.h"
#include "VecSim/vecsim.h"

class HNSWLibTest : public ::testing::Test {
  protected:
    HNSWLibTest() {}

    ~HNSWLibTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(HNSWLibTest, hnswlib_vector_add_test) {
    VecSimParams params = {.hnswParams = {.initialCapacity = 200, .M = 16, .efConstruction = 200},
                           .type = VecSimType_FLOAT32,
                           .size = 4,
                           .metric = VecSimMetric_L2,
                           .algo = VecSimAlgo_HNSWLIB};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[4] = {1.0, 1.0, 1.0, 1.0};
    VecSimIndex_AddVector(index, (const void *)a, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, hnswlib_vector_search_test) {
    VecSimParams params = {
        .hnswParams = {.initialCapacity = 200, .M = 16, .efConstruction = 200},
        .type = VecSimType_FLOAT32,
        .size = 4,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
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

TEST_F(HNSWLibTest, hnswlib_vector_search_by_id_test) {
    VecSimParams params = {
        hnswParams : {initialCapacity : 200, M : 16, efConstruction : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_HNSWLIB,
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

TEST_F(HNSWLibTest, hnswlib_indexing_same_vector) {
    VecSimParams params = {
        hnswParams : {initialCapacity : 200, M : 16, efConstruction : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_HNSWLIB,
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

TEST_F(HNSWLibTest, hnswlib_reindexing_same_vector) {
    VecSimParams params = {
        hnswParams : {initialCapacity : 200, M : 16, efConstruction : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_HNSWLIB,
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

TEST_F(HNSWLibTest, hnswlib_reindexing_same_vector_different_id) {
    VecSimParams params = {
        hnswParams : {initialCapacity : 200, M : 16, efConstruction : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_HNSWLIB,
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

TEST_F(HNSWLibTest, sanity_rinsert_1280) {
    size_t n = 5;
    size_t d = 1280;
    size_t k = 5;

    VecSimParams params = {
        .hnswParams = {.initialCapacity = n, .M = 16, .efConstruction = 200},
        .type = VecSimType_FLOAT32,
        .size = d,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
    };
    VecSimIndex *index = VecSimIndex_New(&params);

    auto *vectors = (float *)malloc(n * d * sizeof(float));
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
            VecSimIndex_TopKQuery(index, (const void *)(vectors + 3 * d), k, nullptr);
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

TEST_F(HNSWLibTest, test_hnsw_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args
    VecSimParams params = {
        hnswParams : {
            initialCapacity : n,

        },
        type : VecSimType_FLOAT32,
        size : d,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_HNSWLIB,
    };
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_HNSWLIB);
    ASSERT_EQ(info.d, d);
    // Default args
    ASSERT_EQ(info.hnswInfo.M, HNSW_DEFAULT_M);
    ASSERT_EQ(info.hnswInfo.efConstruction, HNSW_DEFAULT_EF_C);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);
    VecSimIndex_Free(index);

    d = 1280;
    params = {
        hnswParams : {
            initialCapacity : n,
            M : 200,
            efConstruction : 1000,
            efRuntime : 500

        },
        type : VecSimType_FLOAT32,
        size : d,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_HNSWLIB,
    };
    index = VecSimIndex_New(&params);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_HNSWLIB);
    ASSERT_EQ(info.d, d);
    // User args
    ASSERT_EQ(info.hnswInfo.efConstruction, 1000);
    ASSERT_EQ(info.hnswInfo.M, 200);
    ASSERT_EQ(info.hnswInfo.efRuntime, 500);
    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, test_query_runtime_params_default_build_args) {
    size_t n = 100;
    size_t d = 4;

    // Build with default args
    VecSimParams params = {
        hnswParams : {
            initialCapacity : n,

        },
        type : VecSimType_FLOAT32,
        size : d,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_HNSWLIB,
    };

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
    VecSimIndexInfo info = VecSimIndex_Info(index);
    // Check that default args did not change
    ASSERT_EQ(info.hnswInfo.M, HNSW_DEFAULT_M);
    ASSERT_EQ(info.hnswInfo.efConstruction, HNSW_DEFAULT_EF_C);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);

    // Run same query again, set efRuntime to 300
    VecSimQueryParams queryParams = {hnswRuntimeParams : {efRuntime : 300}};
    res = VecSimIndex_TopKQuery(index, (const void *)query, k, &queryParams);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    for (int i = 0; i < k; i++) {
        int diff_id = ((int)(res[i].id - 50) > 0) ? (res[i].id - 50) : (50 - res[i].id);
        int dist = res[i].score;
        ASSERT_TRUE(((diff_id == (i + 1) / 2)) && (dist == (4 * ((i + 1) / 2) * ((i + 1) / 2))));
    }
    VecSimQueryResult_Free(res);
    info = VecSimIndex_Info(index);
    // Check that default args did not change
    ASSERT_EQ(info.hnswInfo.M, HNSW_DEFAULT_M);
    ASSERT_EQ(info.hnswInfo.efConstruction, HNSW_DEFAULT_EF_C);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);

    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, test_query_runtime_params_user_build_args) {
    size_t n = 100;
    size_t d = 4;
    size_t M = 100;
    size_t efConstruction = 300;
    size_t efRuntime = 500;
    // Build with default args
    VecSimParams params = {
        .hnswParams = {.initialCapacity = n,
                       .M = M,
                       .efConstruction = efConstruction,
                       .efRuntime = efRuntime},
        .type = VecSimType_FLOAT32,
        .size = d,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
    };

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
    VecSimIndexInfo info = VecSimIndex_Info(index);
    // Check that user args did not change
    ASSERT_EQ(info.hnswInfo.M, M);
    ASSERT_EQ(info.hnswInfo.efConstruction, efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    // Run same query again, set efRuntime to 300
    VecSimQueryParams queryParams = {.hnswRuntimeParams = {.efRuntime = 300}};
    res = VecSimIndex_TopKQuery(index, (const void *)query, k, &queryParams);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    for (int i = 0; i < k; i++) {
        int diff_id = ((int)(res[i].id - 50) > 0) ? (res[i].id - 50) : (50 - res[i].id);
        int dist = res[i].score;
        ASSERT_TRUE(((diff_id == (i + 1) / 2)) && (dist == (4 * ((i + 1) / 2) * ((i + 1) / 2))));
    }
    VecSimQueryResult_Free(res);
    info = VecSimIndex_Info(index);
    // Check that user args did not change
    ASSERT_EQ(info.hnswInfo.M, M);
    ASSERT_EQ(info.hnswInfo.efConstruction, efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, hnsw_search_emptied_index) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_HNSWLIB
    };
    size_t n = 100;
    size_t k = 11;
    VecSimIndex *index = VecSimIndex_New(&params);

    for (float i = 0; i < n; i++) {
        float f[4] = {i, i, i, i};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    for (float i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float query[4] = {50, 50, 50, 50};
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, hnsw_force_search_empty_index) {
    VecSimParams params = {
        bfParams : {initialCapacity : 200},
        type : VecSimType_FLOAT32,
        size : 4,
        metric : VecSimMetric_L2,
        algo : VecSimAlgo_HNSWLIB
    };
    size_t n = 100;
    size_t k = 11;
    VecSimIndex *index = VecSimIndex_New(&params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float query[4] = {50, 50, 50, 50};
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}
