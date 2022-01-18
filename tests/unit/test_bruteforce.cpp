#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "test_utils.h"
#include "VecSim/utils/arr_cpp.h"
#include <cmath>

class BruteForceTest : public ::testing::Test {
protected:
    BruteForceTest() {}

    ~BruteForceTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(BruteForceTest, brute_force_vector_add_test) {
    size_t dim = 4;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_IP,
                                             .initialCapacity = 200}};
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

TEST_F(BruteForceTest, brute_force_vector_search_test_ip) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_IP,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, (size_t)i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[] = {50, 50, 50, 50};
    std::set<size_t> expected_ids;
    for (size_t i = n - 1; i > n - 1 - k; i--) {
        expected_ids.insert(i);
    }
    auto verify_res = [&](size_t id, float score, size_t index) {
        ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
        expected_ids.erase(id);
    };
    runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_vector_search_test_l2) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, (int)i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    auto verify_res = [&](size_t id, float score, size_t index) {
        size_t diff_id = ((int)(id - 50) > 0) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    float query[] = {50, 50, 50, 50};
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_vector_search_by_id_test) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, (int)i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[] = {50, 50, 50, 50};
    auto verify_res = [&](size_t id, float score, size_t index) { ASSERT_EQ(id, (index + 45)); };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_indexing_same_vector) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200}};
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
    auto verify_res = [&](size_t id, float score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_reindexing_same_vector) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200}};
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
    auto verify_res = [&](size_t id, float score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Reinsert the same vectors under the same ids
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value)
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run the same query again
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_reindexing_same_vector_different_id) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200}};
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
    auto verify_res = [&](size_t id, float score, size_t index) {
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

TEST_F(BruteForceTest, sanity_rinsert_1280) {
    size_t n = 5;
    size_t d = 1280;
    size_t k = 5;

    VecSimParams params{
        .algo = VecSimAlgo_BF,
        .bfParams = BFParams{
            .type = VecSimType_FLOAT32, .dim = d, .metric = VecSimMetric_L2, .initialCapacity = n}};
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
        auto verify_res = [&](size_t id, float score, size_t index) {
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

TEST_F(BruteForceTest, test_bf_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args
    VecSimParams params = {
        .algo = VecSimAlgo_BF,
        .bfParams = BFParams{
            .type = VecSimType_FLOAT32, .dim = d, .metric = VecSimMetric_L2, .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.bfInfo.dim, d);
    // Default args
    ASSERT_EQ(info.bfInfo.blockSize, BF_DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(info.bfInfo.indexSize, 0);
    VecSimIndex_Free(index);

    d = 1280;
    params = VecSimParams{.algo = VecSimAlgo_BF,
                          .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                               .dim = d,
                                               .metric = VecSimMetric_L2,
                                               .initialCapacity = n,
                                               .blockSize = 1}};
    index = VecSimIndex_New(&params);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.bfInfo.dim, d);
    // User args
    ASSERT_EQ(info.bfInfo.blockSize, 1);
    ASSERT_EQ(info.bfInfo.indexSize, 0);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, test_basic_bf_info_iterator) {
    size_t n = 100;
    size_t d = 128;
    VecSimMetric metrics[3] = {VecSimMetric_Cosine, VecSimMetric_IP, VecSimMetric_L2};

    for (size_t i = 0; i < 3; i++) {
        // Build with default args
        VecSimParams params{
            .algo = VecSimAlgo_BF,
            .bfParams = BFParams{
                .type = VecSimType_FLOAT32, .dim = d, .metric = metrics[i], .initialCapacity = n}};
        VecSimIndex *index = VecSimIndex_New(&params);
        VecSimIndexInfo info = VecSimIndex_Info(index);
        VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
        compareFlatIndexInfoToIterator(info, infoIter);
        VecSimInfoIterator_Free(infoIter);
        VecSimIndex_Free(index);
    }
}

TEST_F(BruteForceTest, test_dynamic_bf_info_iterator) {
    size_t d = 128;
    VecSimParams params{
        .algo = VecSimAlgo_BF,
        .bfParams = BFParams{
            .type = VecSimType_FLOAT32, .dim = d, .metric = VecSimMetric_L2, .blockSize = 1}};
    float v[d];
    for (size_t i = 0; i < d; i++) {
        v[i] = (float)i;
    }
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(1, info.bfInfo.blockSize);
    ASSERT_EQ(0, info.bfInfo.indexSize);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Add vector.
    VecSimIndex_AddVector(index, v, 0);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(1, info.bfInfo.indexSize);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Delete vector.
    VecSimIndex_DeleteVector(index, 0);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(0, info.bfInfo.indexSize);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_vector_search_test_ip_blocksize_1) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_IP,
                                             .initialCapacity = 200,
                                             .blockSize = 1}};
    VecSimIndex *index = VecSimIndex_New(&params);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.bfInfo.blockSize, 1);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[] = {50, 50, 50, 50};
    std::set<size_t> expected_ids;
    for (size_t i = n - 1; i > n - 1 - k; i--) {
        expected_ids.insert(i);
    }
    auto verify_res = [&](size_t id, float score, size_t index) {
        ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
        expected_ids.erase(id);
    };
    runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_vector_search_test_l2_blocksize_1) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200,
                                             .blockSize = 1}};
    VecSimIndex *index = VecSimIndex_New(&params);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.bfInfo.blockSize, 1);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    auto verify_res = [&](size_t id, float score, size_t index) {
        size_t diff_id = ((int)(id - 50) > 0) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    float query[] = {50, 50, 50, 50};
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_search_empty_index) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200}};
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
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
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

TEST_F(BruteForceTest, brute_force_test_inf_score) {
    size_t n = 4;
    size_t k = 4;
    size_t dim = 2;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // The 32 bits of "efgh" and "efgg", and the 32 bits of "abcd" and "abbd" will
    // yield "inf" result when we calculate distance between the vectors.
    VecSimIndex_AddVector(index, "abcdefgh", 1);
    VecSimIndex_AddVector(index, "abcdefgg", 2);
    VecSimIndex_AddVector(index, "aacdefgh", 3);
    VecSimIndex_AddVector(index, "abbdefgh", 4);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 4);

    auto verify_res = [&](size_t id, float score, size_t index) {
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

TEST_F(BruteForceTest, brute_force_remove_vector_after_replacing_block) {
    size_t dim = 4;
    size_t n = 2;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200,
                                             .blockSize = 1}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Add 2 vectors, into 2 separated blocks.
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // After deleting the first vector, the second one will be moved to the first block
    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_zero_minimal_capacity) {
    size_t dim = 4;
    size_t n = 2;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 0,
                                             .blockSize = 1}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float vec[dim];
    // Add 2 vectors, into 2 separated blocks.
    for (size_t i = 0; i < n; i++) {
        VecSimIndex_AddVector(index, vec, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // After deleting the first vector, the second one will be moved to the first block
    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_batch_iterator) {
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200,
                                             .blockSize = 5}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // run the test twice - for index of size 100, every iteration will run select-based search,
    // as the number of results is 5, which is more than 0.1% of the index size. for index of size
    // 10000, we will run the heap-based search until we return 5000 results, and then switch to
    // select-based search.
    for (size_t n : {100, 10000}) {
        for (size_t i = 0; i < n; i++) {
            float f[dim];
            for (size_t j = 0; j < dim; j++) {
                f[j] = (float)i;
            }
            VecSimIndex_AddVector(index, (const void *)f, i);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), n);

        // query for (n,n,...,n) vector (recall that n is the largest id in te index)
        float query[dim];
        for (size_t j = 0; j < dim; j++) {
            query[j] = (float)n;
        }
        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query);
        size_t iteration_num = 0;

        // get the 10 vectors whose ids are the maximal among those that hasn't been returned yet,
        // in every iteration. The order should be from the largest to the lowest id.
        size_t n_res = 5;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            std::vector<size_t> expected_ids(n_res);
            for (size_t i = 0; i < n_res; i++) {
                expected_ids[i] = (n - iteration_num * n_res - i - 1);
            }
            auto verify_res = [&](size_t id, float score, size_t index) {
                ASSERT_TRUE(expected_ids[index] == id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num, n / n_res);
        VecSimBatchIterator_Free(batchIterator);
    }
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_batch_iterator_non_unique_scores) {
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200,
                                             .blockSize = 5}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // run the test twice - for index of size 100, every iteration will run select-based search,
    // as the number of results is 5, which is more than 0.1% of the index size. for index of size
    // 10000, we will run the heap-based search until we return 5000 results, and then switch to
    // select-based search.
    for (size_t n : {100, 10000}) {
        for (size_t i = 0; i < n; i++) {
            float f[dim];
            for (size_t j = 0; j < dim; j++) {
                f[j] = (float)(i / 10);
            }
            VecSimIndex_AddVector(index, (const void *)f, i);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), n);

        // query for (n,n,...,n) vector (recall that n is the largest id in te index)
        float query[dim];
        for (size_t j = 0; j < dim; j++) {
            query[j] = (float)n;
        }
        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query);
        size_t iteration_num = 0;

        // get the 5 vectors whose ids are the maximal among those that hasn't been returned yet, in
        // every iteration. there are n/10 groups of 10 different vectors with the same score.
        size_t n_res = 5;
        bool even_iteration = false;
        std::set<size_t> expected_ids;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            // insert the maximal 10 ids in every odd iteration
            if (!even_iteration) {
                for (size_t i = 1; i <= 2 * n_res; i++) {
                    expected_ids.insert(n - iteration_num * n_res - i);
                }
            }
            auto verify_res = [&](size_t id, float score, size_t index) {
                ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
                expected_ids.erase(id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            // make sure that the expected ids set is empty after two iterations.
            if (even_iteration) {
                ASSERT_TRUE(expected_ids.empty());
            }
            iteration_num++;
            even_iteration = !even_iteration;
        }
        ASSERT_EQ(iteration_num, n / n_res);
        VecSimBatchIterator_Free(batchIterator);
    }
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_batch_iterator_reset) {
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 100000,
                                             .blockSize = 100000}};
    VecSimIndex *index = VecSimIndex_New(&params);

    size_t n = 10000;
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // query for (n,n,...,n) vector (recall that n is the largest id in te index)
    float query[dim];
    for (size_t j = 0; j < dim; j++) {
        query[j] = (float)n;
    }
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query);

    // get the 100 vectors whose ids are the maximal among those that hasn't been returned yet, in
    // every iteration. run this flow for 5 times, each time for 10 iteration, and reset the
    // iterator.
    size_t n_res = 100;
    size_t total_iteration = 5;
    size_t re_runs = 3;

    for (size_t take = 0; take < re_runs; take++) {
        size_t iteration_num = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            std::set<size_t> expected_ids;
            for (size_t i = 1; i <= n_res; i++) {
                expected_ids.insert(n - iteration_num * n_res - i);
            }
            auto verify_res = [&](size_t id, float score, size_t index) {
                ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
                expected_ids.erase(id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            iteration_num++;
            if (iteration_num == total_iteration) {
                break;
            }
        }
        VecSimBatchIterator_Reset(batchIterator);
    }
    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_batch_iterator_corner_cases) {
    size_t dim = 4;
    size_t n = 1000;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // query for (n,n,...,n) vector (recall that n is the largest id in te index)
    float query[dim];
    for (size_t j = 0; j < dim; j++) {
        query[j] = (float)n;
    }
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query);

    // get all in first iteration, expect to use select search
    size_t n_res = n;
    auto verify_res = [&](size_t id, float score, size_t index) {
        ASSERT_TRUE(id == n - 1 - index);
    };
    runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
    ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));

    // try to get more results even though there are no.
    VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, n_res, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);

    // Reset, and run in batches, but the final batch is partial.
    VecSimBatchIterator_Reset(batchIterator);
    res = VecSimBatchIterator_Next(batchIterator, n_res / 2, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), n / 2);
    VecSimQueryResult_Free(res);
    res = VecSimBatchIterator_Next(batchIterator, n_res / 2 + 1, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), n / 2);
    VecSimQueryResult_Free(res);
    ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));

    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

TEST_F(BruteForceTest, brute_force_resolve_params) {
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

    VecSimRawParam *rparams = array_new<VecSimRawParam>(1);

    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams), VecSim_OK);
    ASSERT_EQ(memcmp(&qparams, &zero, sizeof(VecSimQueryParams)), 0);

    array_append(rparams, (VecSimRawParam){
                              .name = "ef_runtime", .nameLen = 10, .value = "100", .valLen = 3});
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams),
              VecSimParamResolverErr_UnknownParam);

    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), NULL),
              VecSimParamResolverErr_NullParam);

    VecSimIndex_Free(index);
    array_free(rparams);
}

TEST_F(BruteForceTest, brute_get_distance) {
    size_t n = 4;
    size_t dim = 2;
    size_t numIndex = 3;
    VecSimIndex *index[numIndex];
    std::vector<double> distances;

    VecSimParams params{
        .algo = VecSimAlgo_BF,
        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32, .dim = dim, .initialCapacity = n}};

    for (size_t i = 0; i < numIndex; i++) {
        params.bfParams.metric = (VecSimMetric)i;
        index[i] = VecSimIndex_New(&params);
        VecSimIndex_AddVector(index[i], "\100\055\370\124\100\055\370\124", 1); //  pi, pi
        VecSimIndex_AddVector(index[i], "\100\111\017\333\100\111\017\333", 2); //   e,  e
        VecSimIndex_AddVector(index[i], "\100\055\370\124\100\111\017\333", 3); //  pi,  e
        VecSimIndex_AddVector(index[i], "\277\317\033\275\077\317\033\275", 4); // -gr, gr
        ASSERT_EQ(VecSimIndex_IndexSize(index[i]), 4);
    }

    const char *query = "\100\055\370\124\100\055\370\124";
    double dist;

    // VecSimMetric_L2
    distances = {0, 3.254628852688946e+33, 1.627314426344473e+33, 1.4542929723017879e+26};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_L2], i + 1, query);
        ASSERT_FLOAT_EQ(dist, distances[i]);
    }

    // VecSimMetric_IP
    distances = {-1.4542929723017879e+26, 6.8783602560985513e+29, 3.4384528836109137e+29,
                 648750694400};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_IP], i + 1, query);
        ASSERT_FLOAT_EQ(dist, distances[i]);
    }

    // VecSimMetric_Cosine
    distances = {0, 2, 1.7069573402404785, 2};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_Cosine], i + 1, query);
        ASSERT_FLOAT_EQ(dist, distances[i]);
    }

    // Bad values
    dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_Cosine], 0, query);
    ASSERT_TRUE(std::isnan(dist));
    dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_L2], 46, query);
    ASSERT_TRUE(std::isnan(dist));

    // Clean-up.
    for (size_t i = 0; i < numIndex; i++) {
        VecSimIndex_Free(index[i]);
    }
}
