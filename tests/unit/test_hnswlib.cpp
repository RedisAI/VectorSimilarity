#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "test_utils.h"
#include <climits>

class HNSWLibTest : public ::testing::Test {
protected:
    HNSWLibTest() {}

    ~HNSWLibTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(HNSWLibTest, hnswlib_vector_add_test) {
    size_t dim = 4;
    VecSimParams params = {.hnswParams = {.initialCapacity = 200, .M = 16, .efConstruction = 200},
                           .type = VecSimType_FLOAT32,
                           .size = 4,
                           .metric = VecSimMetric_L2,
                           .algo = VecSimAlgo_HNSWLIB};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (float)i;
    }
    VecSimIndex_AddVector(index, (const void *)a, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, hnswlib_vector_search_test) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;
    VecSimParams params = {
        .hnswParams = {.initialCapacity = 200, .M = 16, .efConstruction = 200},
        .type = VecSimType_FLOAT32,
        .size = 4,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
    };
    VecSimIndex *index = VecSimIndex_New(&params);

    for (int i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, (size_t)i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[] = {50, 50, 50, 50};
    auto verify_res = [&](int id, float score, size_t index) {
        int diff_id = ((id - 50) > 0) ? (id - 50) : (50 - id);
        ASSERT_TRUE(((diff_id == (index + 1) / 2)) &&
                    (score == (4 * ((index + 1) / 2) * ((index + 1) / 2))));
    };
    runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, hnswlib_vector_search_by_id_test) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 11;

    VecSimParams params = {
        .hnswParams = {.initialCapacity = 200, .M = 16, .efConstruction = 200},
        .type = VecSimType_FLOAT32,
        .size = dim,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
    };
    VecSimIndex *index = VecSimIndex_New(&params);

    for (int i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, (int)i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[] = {50, 50, 50, 50};
    auto verify_res = [&](int id, float score, size_t index) { ASSERT_EQ(id, (index + 45)); };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, hnswlib_indexing_same_vector) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 10;

    VecSimParams params = {
        .hnswParams = {.initialCapacity = 200, .M = 16, .efConstruction = 200},
        .type = VecSimType_FLOAT32,
        .size = dim,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
    };
    VecSimIndex *index = VecSimIndex_New(&params);

    for (int i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i / 10);
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float query[] = {4.9, 4.95, 5.05, 5.1};
    auto verify_res = [&](int id, float score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, hnswlib_reindexing_same_vector) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 10;

    VecSimParams params = {
        .hnswParams = {.initialCapacity = 200, .M = 16, .efConstruction = 200},
        .type = VecSimType_FLOAT32,
        .size = dim,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
    };
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value)
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float query[] = {4.9, 4.95, 5.05, 5.1};
    auto verify_res = [&](int id, float score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Reinsert the same vectors under the same ids
    for (size_t i = 0; i < n; i++) {
        float num = (float)(i / 10);
        float f[] = {num, num, num, num};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run the same query again
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, hnswlib_reindexing_same_vector_different_id) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 10;

    VecSimParams params = {
        .hnswParams = {.initialCapacity = 200, .M = 16, .efConstruction = 200},
        .type = VecSimType_FLOAT32,
        .size = dim,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
    };
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value)
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float query[] = {4.9, 4.95, 5.05, 5.1};
    auto verify_res = [&](int id, float score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Reinsert the same vectors under different ids than before
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value)
        }
        VecSimIndex_AddVector(index, (const void *)f, i + 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run the same query again
    auto verify_res_different_id = [&](int id, float score, size_t index) {
        ASSERT_TRUE(id >= 60 && id < 70 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res_different_id);

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

    // Generate random vectors in every iteration and inert them under different ids
    for (size_t iter = 1; iter <= 3; iter++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                (vectors + i * d)[j] = (float)rand() / (float)(RAND_MAX) / 100;
            }
        }
        auto expected_ids = std::set<size_t>();
        for (size_t i = 0; i < n; i++) {
            VecSimIndex_AddVector(index, (const void *)(vectors + i * d), i * iter);
            expected_ids.insert(i * iter);
        }
        auto verify_res = [&](int id, float score, size_t index) {
            ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
            expected_ids.erase(id);
        };

        // Send arbitrary vector (the first) and search for top k. This should return all the
        // vectors that were inserted in this iteration - verify their ids.
        runTopKSearchTest(index, (const void *)vectors, k, verify_res);

        // Remove vectors form current iteration.
        for (size_t i = 0; i < n; i++) {
            VecSimIndex_DeleteVector(index, i * iter);
        }
    }
    free(vectors);
    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, test_hnsw_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args
    VecSimParams params = {
        .hnswParams =
            {
                .initialCapacity = n,

            },
        .type = VecSimType_FLOAT32,
        .size = d,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
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
        .hnswParams = {.initialCapacity = n, .M = 200, .efConstruction = 1000, .efRuntime = 500

        },
        .type = VecSimType_FLOAT32,
        .size = d,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
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
    size_t k = 11;

    // Build with default args
    VecSimParams params = {
        .hnswParams =
            {
                .initialCapacity = n,
            },
        .type = VecSimType_FLOAT32,
        .size = d,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
    };

    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[d];
        for (size_t j = 0; j < d; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    auto verify_res = [&](int id, float score, size_t index) {
        int diff_id = ((id - 50) > 0) ? (id - 50) : (50 - id);
        ASSERT_TRUE(((diff_id == (index + 1) / 2)) &&
                    (score == (4 * ((index + 1) / 2) * ((index + 1) / 2))));
    };
    float query[] = {50, 50, 50, 50};
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    // Check that default args did not change
    ASSERT_EQ(info.hnswInfo.M, HNSW_DEFAULT_M);
    ASSERT_EQ(info.hnswInfo.efConstruction, HNSW_DEFAULT_EF_C);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);

    // Run same query again, set efRuntime to 300
    VecSimQueryParams queryParams = {.hnswRuntimeParams = {.efRuntime = 300}};
    runTopKSearchTest(index, query, k, verify_res, &queryParams);

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

    for (size_t i = 0; i < n; i++) {
        float f[d];
        for (size_t j = 0; j < d; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    auto verify_res = [&](int id, float score, size_t index) {
        int diff_id = ((id - 50) > 0) ? (id - 50) : (50 - id);
        ASSERT_TRUE(((diff_id == (index + 1) / 2)) &&
                    (score == (4 * ((index + 1) / 2) * ((index + 1) / 2))));
    };
    float query[] = {50, 50, 50, 50};
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    // Check that user args did not change
    ASSERT_EQ(info.hnswInfo.M, M);
    ASSERT_EQ(info.hnswInfo.efConstruction, efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    // Run same query again, set efRuntime to 300
    VecSimQueryParams queryParams = {.hnswRuntimeParams = {.efRuntime = 300}};
    runTopKSearchTest(index, query, k, verify_res, &queryParams);

    info = VecSimIndex_Info(index);
    // Check that user args did not change
    ASSERT_EQ(info.hnswInfo.M, M);
    ASSERT_EQ(info.hnswInfo.efConstruction, efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, hnsw_search_empty_index) {
    size_t n = 100;
    size_t k = 11;
    size_t d = 4;
    VecSimParams params = {.hnswParams = {.initialCapacity = 0},
                           .type = VecSimType_FLOAT32,
                           .size = d,
                           .metric = VecSimMetric_L2,
                           .algo = VecSimAlgo_HNSWLIB};

    VecSimIndex *index = VecSimIndex_New(&params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float query[] = {50, 50, 50, 50};

    // We do not expect any results
    VecSimQueryResult_List res =
        VecSimIndex_TopKQuery(index, (const void *)query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Iterator *it = VecSimQueryResult_List_GetIterator(res);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);

    // Add some vectors and remove them all from index, so it will be empty again.
    for (size_t i = 0; i < n; i++) {
        float f[d];
        for (size_t j = 0; j < d; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Again - we do not expect any results
    res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    it = VecSimQueryResult_List_GetIterator(res);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);

    VecSimIndex_Free(index);
}

TEST_F(HNSWLibTest, hnsw_inf_score) {
    size_t n = 4;
    size_t k = 4;
    size_t dim = 2;
    VecSimParams params = {.hnswParams = {.initialCapacity = n},
                           .type = VecSimType_FLOAT32,
                           .size = dim,
                           .metric = VecSimMetric_L2,
                           .algo = VecSimAlgo_HNSWLIB};
    VecSimIndex *index = VecSimIndex_New(&params);

    // The 32 bits of "efgh" and "efgg", and the 32 bits of "abcd" and "abbd" will
    // yield "inf" result when we calculate distance between the vectors.
    VecSimIndex_AddVector(index, "abcdefgh", 1);
    VecSimIndex_AddVector(index, "abcdefgg", 2);
    VecSimIndex_AddVector(index, "aacdefgh", 3);
    VecSimIndex_AddVector(index, "abbdefgh", 4);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 4);

    auto verify_res = [&](int id, float score, size_t index) {
        if (index == 0) {
            ASSERT_EQ(1, id);
        } else if (index == 1) {
            ASSERT_EQ(3, id);
        } else {
            ASSERT_TRUE(id == 2 || id == 4);
        }
    };
    runTopKSearchTest(index, "abcdefgh", k, verify_res);
    VecSimIndex_Free(index);
}

// Tests VecSimIndex_New failure on bad M parameter. Should return null.
TEST_F(HNSWLibTest, hnsw_bad_params) {
    size_t n = 1000000;
    size_t dim = 2;
    size_t bad_M[] = {
        1,        // Will fail because 1/log(M).
        10000000, // Will fail on this->allocator->allocate(max_elements_ * size_data_per_element_)
        SIZE_MAX, // Will fail on M * 2 overflow.
        SIZE_MAX / 2, // Will fail on M * 2 overflow.
        SIZE_MAX / 4  // Will fail on size_links_level0_ calculation:
                      // sizeof(linklistsizeint) + M * 2 * sizeof(tableint) + sizeof(void *)
    };
    unsigned long len = sizeof(bad_M) / sizeof(size_t);

    for (unsigned long i = 0; i < len; i++) {
        VecSimParams params = {.hnswParams = {.initialCapacity = n},
                               .type = VecSimType_FLOAT32,
                               .size = dim,
                               .metric = VecSimMetric_L2,
                               .algo = VecSimAlgo_HNSWLIB};

        params.hnswParams.M = bad_M[i];
        VecSimIndex *index = VecSimIndex_New(&params);
        ASSERT_TRUE(index == NULL);
    }
}

TEST_F(HNSWLibTest, hnsw_delete_enter_point) {
    size_t n = 10000;
    size_t dim = 2;
    size_t M = 2;

    VecSimParams params = {
        .hnswParams = {.initialCapacity = n, .M = M, .efConstruction = 0, .efRuntime = 0},
        .type = VecSimType_FLOAT32,
        .size = dim,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB};

    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_TRUE(index != NULL);

    int64_t vec[dim];
    for (int i = 0; i < dim; i++)
        vec[i] = i;
    for (size_t j = 0; j < n; j++)
        VecSimIndex_AddVector(index, vec, j);

    VecSimIndexInfo info = VecSimIndex_Info(index);

    while (info.hnswInfo.indexSize > 0) {
        ASSERT_NO_THROW(VecSimIndex_DeleteVector(index, info.hnswInfo.entry));
        info = VecSimIndex_Info(index);
    }
    VecSimIndex_Free(index);
}
