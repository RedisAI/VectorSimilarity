#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "test_utils.h"
#include "VecSim/algorithms/hnsw/serialization.h"
#include "VecSim/query_result_struct.h"
#include <climits>
#include <unistd.h>

class HNSWTest : public ::testing::Test {
protected:
    HNSWTest() {}

    ~HNSWTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(HNSWTest, hnsw_vector_add_test) {
    size_t dim = 4;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = 200,
                                                 .M = 16,
                                                 .efConstruction = 200}};
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

TEST_F(HNSWTest, hnsw_blob_sanity_test) {
    size_t dim = 4;
    size_t bs = 1;
#define ASSERT_HNSW_BLOB_EQ(id, blob)                                                              \
    do {                                                                                           \
        void *v = hnsw_index->getDataByInternalId(id);                                             \
        ASSERT_FALSE(memcmp(v, blob, sizeof(blob)));                                               \
    } while (0)

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{
                            .type = VecSimType_FLOAT32,
                            .dim = dim,
                            .metric = VecSimMetric_L2,
                            .blockSize = bs,
                        }};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[dim], b[dim], c[dim], d[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (float)0;
        b[i] = (float)1;
        c[i] = (float)2;
        d[i] = (float)3;
    }
    HNSWIndex<float, float> *hnsw_index = reinterpret_cast<HNSWIndex<float, float> *>(index);

    VecSimIndex_AddVector(index, (const void *)a, 42);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    ASSERT_HNSW_BLOB_EQ(0, a);
    ASSERT_EQ(hnsw_index->getExternalLabel(0), 42);

    VecSimIndex_AddVector(index, (const void *)b, 46);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 2);
    ASSERT_HNSW_BLOB_EQ(1, b);
    ASSERT_EQ(hnsw_index->getExternalLabel(1), 46);

    // After inserting c with label 46, we first delete id 1 from the index.
    // we expect id 0 to not change
    VecSimIndex_AddVector(index, (const void *)c, 46);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 2);
    ASSERT_HNSW_BLOB_EQ(0, a);
    ASSERT_HNSW_BLOB_EQ(1, c);
    ASSERT_EQ(hnsw_index->getExternalLabel(0), 42);
    ASSERT_EQ(hnsw_index->getExternalLabel(1), 46);

    // After inserting d with label 42, we first delete id 0 and move the last id (1) to be 0.
    // Then we add the new vector d under the internal id 1.
    VecSimIndex_AddVector(index, (const void *)d, 42);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 2);
    ASSERT_HNSW_BLOB_EQ(0, c);
    ASSERT_HNSW_BLOB_EQ(1, d);
    ASSERT_EQ(hnsw_index->getExternalLabel(0), 46);
    ASSERT_EQ(hnsw_index->getExternalLabel(1), 42);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, search_for_nothing) {
    size_t n = 100;
    size_t k = 0;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n}};
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
    auto verify_res = [&](size_t id, float score, size_t index) { return; };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

/**** resizing cases ****/

// Add up to capacity.
TEST_F(HNSWTest, resizeNAlignIndex) {
    size_t dim = 4;
    size_t n = 10;
    size_t bs = 3;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[dim];

    // Add up to n.
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            a[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)a, i);
    }
    // The size and the capacity should be equal.
    HNSWIndex<float, float> *hnswIndex = reinterpret_cast<HNSWIndex<float, float> *>(index);
    ASSERT_EQ(hnswIndex->getIndexCapacity(), VecSimIndex_IndexSize(index));
    // The capacity shouldn't be changed.
    ASSERT_EQ(hnswIndex->getIndexCapacity(), n);

    // Add another vector to exceed the initial capacity.
    VecSimIndex_AddVector(index, (const void *)a, n);

    // The capacity should be now aligned with the block size.
    // bs = 3, size = 11 -> capacity = 12
    // New capacity = initial capacity + blockSize - initial capacity % blockSize.
    ASSERT_EQ(hnswIndex->getIndexCapacity(), n + bs - n % bs);
    VecSimIndex_Free(index);
}

// Case 1: initial capacity is larger than block size, and it is not aligned.
TEST_F(HNSWTest, resizeNAlignIndex_largeInitialCapacity) {
    size_t dim = 4;
    size_t n = 10;
    size_t bs = 3;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[dim];

    // add up to blocksize + 1 = 3 + 1 = 4
    for (size_t i = 0; i < bs + 1; i++) {
        for (size_t j = 0; j < dim; j++) {
            a[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)a, i);
    }

    // The capacity shouldn't change, should remain n.
    HNSWIndex<float, float> *hnswIndex = reinterpret_cast<HNSWIndex<float, float> *>(index);
    ASSERT_EQ(hnswIndex->getIndexCapacity(), n);

    // Delete last vector, to get size % block_size == 0. size = 3
    VecSimIndex_DeleteVector(index, bs);

    // Index size = bs = 3.
    ASSERT_EQ(VecSimIndex_IndexSize(index), bs);

    // New capacity = initial capacity - block_size - number_of_vectors_to_align =
    // 10  - 3 - 10 % 3 (1) = 6
    size_t curr_capacity = hnswIndex->getIndexCapacity();
    ASSERT_EQ(curr_capacity, n - bs - n % bs);

    // Delete all the vectors to decrease capacity by another bs.
    size_t i = 0;
    while (VecSimIndex_IndexSize(index) > 0) {
        VecSimIndex_DeleteVector(index, i);
        ++i;
    }
    ASSERT_EQ(hnswIndex->getIndexCapacity(), bs);
    // Add and delete a vector to achieve:
    // size % block_size == 0 && size + bs <= capacity(3).
    // the capacity should be resized to zero
    VecSimIndex_AddVector(index, (const void *)a, 0);
    VecSimIndex_DeleteVector(index, 0);
    ASSERT_EQ(hnswIndex->getIndexCapacity(), 0);

    // Do it again. This time after adding a vector the capacity is increased by bs.
    // Upon deletion it will be resized to zero again.
    VecSimIndex_AddVector(index, (const void *)a, 0);
    ASSERT_EQ(hnswIndex->getIndexCapacity(), bs);
    VecSimIndex_DeleteVector(index, 0);
    ASSERT_EQ(hnswIndex->getIndexCapacity(), 0);

    VecSimIndex_Free(index);
}

// Case 2: initial capacity is smaller than block_size.
TEST_F(HNSWTest, resizeNAlignIndex_largerBlockSize) {
    size_t dim = 4;
    size_t n = 4;
    size_t bs = 6;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[dim];

    // Add up to initial capacity.
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            a[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)a, i);
    }

    HNSWIndex<float, float> *hnswIndex = reinterpret_cast<HNSWIndex<float, float> *>(index);
    // The capacity shouldn't change.
    ASSERT_EQ(hnswIndex->getIndexCapacity(), n);

    // Size equals capacity.
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Add another vector - > the capacity is increased to a multiplication of block_size.
    VecSimIndex_AddVector(index, (const void *)a, n);
    ASSERT_EQ(hnswIndex->getIndexCapacity(), bs);

    // Size increased by 1.
    ASSERT_EQ(VecSimIndex_IndexSize(index), n + 1);

    // Delete random vector.
    VecSimIndex_DeleteVector(index, 1);

    // The capacity should remain the same.
    ASSERT_EQ(hnswIndex->getIndexCapacity(), bs);

    VecSimIndex_Free(index);
}

// Test empty index edge cases.
TEST_F(HNSWTest, emptyIndex) {
    size_t dim = 4;
    size_t n = 20;
    size_t bs = 6;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove from an empty index - should fail because label doesn't exist.
    VecSimIndex_DeleteVector(index, 0);

    // Add one vector.
    float a[dim];
    for (size_t j = 0; j < dim; j++) {
        a[j] = (float)1.7;
    }

    VecSimIndex_AddVector(index, (const void *)a, 1);
    // Try to remove it.
    VecSimIndex_DeleteVector(index, 1);
    // The capacity should change to be aligned with the vector size.

    HNSWIndex<float, float> *hnswIndex = reinterpret_cast<HNSWIndex<float, float> *>(index);
    size_t new_capacity = hnswIndex->getIndexCapacity();
    ASSERT_EQ(new_capacity, n - n % bs - bs);

    // Size equals 0.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove it again.
    // The capacity should remain unchanged, as we are trying to delete a label that doesn't exist.
    VecSimIndex_DeleteVector(index, 1);
    ASSERT_EQ(hnswIndex->getIndexCapacity(), new_capacity);
    // Nor the size.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_vector_search_test) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = 200,
                                                 .M = 16,
                                                 .efConstruction = 200}};
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
    auto verify_res = [&](size_t id, double score, size_t index) {
        size_t diff_id = ((int)(id - 50) > 0) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_vector_search_by_id_test) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 11;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = 200,
                                                 .M = 16,
                                                 .efConstruction = 200}};
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
    auto verify_res = [&](size_t id, double score, size_t index) { ASSERT_EQ(id, (index + 45)); };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_indexing_same_vector) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 10;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = 200,
                                                 .M = 16,
                                                 .efConstruction = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i / 10);
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float query[] = {4.9, 4.95, 5.05, 5.1};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_reindexing_same_vector) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 10;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = 200,
                                                 .M = 16,
                                                 .efConstruction = 200}};
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
    auto verify_res = [&](size_t id, double score, size_t index) {
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

TEST_F(HNSWTest, hnsw_reindexing_same_vector_different_id) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 10;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = 200,
                                                 .M = 16,
                                                 .efConstruction = 200}};
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
    auto verify_res = [&](size_t id, double score, size_t index) {
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
    auto verify_res_different_id = [&](int id, double score, size_t index) {
        ASSERT_TRUE(id >= 60 && id < 70 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res_different_id);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, sanity_reinsert_1280) {
    size_t n = 5;
    size_t d = 1280;
    size_t k = 5;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = d,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .M = 16,
                                                 .efConstruction = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    auto *vectors = (float *)malloc(n * d * sizeof(float));

    // Generate random vectors in every iteration and inert them under different ids.
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
        auto verify_res = [&](size_t id, double score, size_t index) {
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

TEST_F(HNSWTest, test_hnsw_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args
    VecSimParams params{
        .algo = VecSimAlgo_HNSWLIB,
        .hnswParams = HNSWParams{
            .type = VecSimType_FLOAT32, .dim = d, .metric = VecSimMetric_L2, .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_HNSWLIB);
    ASSERT_EQ(info.hnswInfo.dim, d);
    // Default args.
    ASSERT_FALSE(info.hnswInfo.isMulti);
    ASSERT_EQ(info.hnswInfo.blockSize, DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(info.hnswInfo.M, HNSW_DEFAULT_M);
    ASSERT_EQ(info.hnswInfo.efConstruction, HNSW_DEFAULT_EF_C);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);
    ASSERT_DOUBLE_EQ(info.hnswInfo.epsilon, HNSW_DEFAULT_EPSILON);
    VecSimIndex_Free(index);

    d = 1280;
    size_t bs = 42;
    params = {.algo = VecSimAlgo_HNSWLIB,
              .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                       .dim = d,
                                       .metric = VecSimMetric_L2,
                                       .multi = false,
                                       .initialCapacity = n,
                                       .blockSize = bs,
                                       .M = 200,
                                       .efConstruction = 1000,
                                       .efRuntime = 500,
                                       .epsilon = 0.005}};
    index = VecSimIndex_New(&params);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_HNSWLIB);
    ASSERT_EQ(info.hnswInfo.dim, d);
    // User args.
    ASSERT_FALSE(info.hnswInfo.isMulti);
    ASSERT_EQ(info.hnswInfo.blockSize, bs);
    ASSERT_EQ(info.hnswInfo.efConstruction, 1000);
    ASSERT_EQ(info.hnswInfo.M, 200);
    ASSERT_EQ(info.hnswInfo.efRuntime, 500);
    ASSERT_EQ(info.hnswInfo.epsilon, 0.005);
    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, test_basic_hnsw_info_iterator) {
    size_t n = 100;
    size_t d = 128;

    VecSimMetric metrics[3] = {VecSimMetric_Cosine, VecSimMetric_IP, VecSimMetric_L2};
    for (size_t i = 0; i < 3; i++) {
        // Build with default args.
        VecSimParams params{
            .algo = VecSimAlgo_HNSWLIB,
            .hnswParams = HNSWParams{
                .type = VecSimType_FLOAT32, .dim = d, .metric = metrics[i], .initialCapacity = n}};
        VecSimIndex *index = VecSimIndex_New(&params);
        VecSimIndexInfo info = VecSimIndex_Info(index);
        VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
        compareHNSWIndexInfoToIterator(info, infoIter);
        VecSimInfoIterator_Free(infoIter);
        VecSimIndex_Free(index);
    }
}

TEST_F(HNSWTest, test_dynamic_hnsw_info_iterator) {
    size_t n = 100;
    size_t d = 128;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = d,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .M = 100,
                                                 .efConstruction = 250,
                                                 .efRuntime = 400,
                                                 .epsilon = 0.004}};
    float v[d];
    for (size_t i = 0; i < d; i++) {
        v[i] = (float)i;
    }
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(100, info.hnswInfo.M);
    ASSERT_EQ(250, info.hnswInfo.efConstruction);
    ASSERT_EQ(400, info.hnswInfo.efRuntime);
    ASSERT_EQ(0.004, info.hnswInfo.epsilon);
    ASSERT_EQ(0, info.hnswInfo.indexSize);
    ASSERT_EQ(-1, info.hnswInfo.max_level);
    ASSERT_EQ(-1, info.hnswInfo.entrypoint);
    compareHNSWIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Add vector.
    VecSimIndex_AddVector(index, v, 1);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(1, info.hnswInfo.indexSize);
    ASSERT_EQ(1, info.hnswInfo.entrypoint);
    ASSERT_GE(1, info.hnswInfo.max_level);
    compareHNSWIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Delete vector.
    VecSimIndex_DeleteVector(index, 1);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(0, info.bfInfo.indexSize);
    compareHNSWIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Perform (or simulate) Search in all modes.
    VecSimIndex_AddVector(index, v, 0);
    auto res = VecSimIndex_TopKQuery(index, v, 1, nullptr, BY_SCORE);
    VecSimQueryResult_Free(res);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(STANDARD_KNN, info.hnswInfo.last_mode);
    compareHNSWIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    res = VecSimIndex_RangeQuery(index, v, 1, nullptr, BY_SCORE);
    VecSimQueryResult_Free(res);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(RANGE_QUERY, info.hnswInfo.last_mode);
    compareHNSWIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 1, 1, true));
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(HYBRID_ADHOC_BF, info.hnswInfo.last_mode);
    compareHNSWIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Set the index size artificially so that BATCHES mode will be selected by the heuristics.
    reinterpret_cast<HNSWIndex<float, float> *>(index)->cur_element_count = 1e6;
    for (size_t i = 0; i < 1e6; i++) {
        reinterpret_cast<HNSWIndex_Single<float, float> *>(index)->label_lookup_[i] = i;
    }
    ASSERT_FALSE(VecSimIndex_PreferAdHocSearch(index, 10, 1, true));
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(HYBRID_BATCHES, info.hnswInfo.last_mode);
    compareHNSWIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Simulate the case where another call to the heuristics is done after realizing that
    // the subset size is smaller, and change the policy as a result.
    ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 1, 10, false));
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(HYBRID_BATCHES_TO_ADHOC_BF, info.hnswInfo.last_mode);
    compareHNSWIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, test_query_runtime_params_default_build_args) {
    size_t n = 100;
    size_t d = 4;
    size_t k = 11;

    // Build with default args.
    VecSimParams params{
        .algo = VecSimAlgo_HNSWLIB,
        .hnswParams = HNSWParams{
            .type = VecSimType_FLOAT32, .dim = d, .metric = VecSimMetric_L2, .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[d];
        for (size_t j = 0; j < d; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    auto verify_res = [&](size_t id, double score, size_t index) {
        size_t diff_id = ((int)(id - 50) > 0) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    float query[] = {50, 50, 50, 50};
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    // Check that default args did not change.
    ASSERT_EQ(info.hnswInfo.M, HNSW_DEFAULT_M);
    ASSERT_EQ(info.hnswInfo.efConstruction, HNSW_DEFAULT_EF_C);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);

    // Run same query again, set efRuntime to 300.
    VecSimQueryParams queryParams{.hnswRuntimeParams = HNSWRuntimeParams{.efRuntime = 300}};
    runTopKSearchTest(index, query, k, verify_res, &queryParams);

    info = VecSimIndex_Info(index);
    // Check that default args did not change.
    ASSERT_EQ(info.hnswInfo.M, HNSW_DEFAULT_M);
    ASSERT_EQ(info.hnswInfo.efConstruction, HNSW_DEFAULT_EF_C);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);

    // Create batch iterator without query param - verify that ef_runtime didn't change.
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);
    // Run one batch for sanity.
    runBatchIteratorSearchTest(batchIterator, k, verify_res);
    // After releasing the batch iterator, ef_runtime should return to the default one.
    VecSimBatchIterator_Free(batchIterator);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, test_query_runtime_params_user_build_args) {
    size_t n = 100;
    size_t d = 4;
    size_t M = 100;
    size_t efConstruction = 300;
    size_t efRuntime = 500;
    // Build with user args.
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = d,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .M = M,
                                                 .efConstruction = efConstruction,
                                                 .efRuntime = efRuntime}};

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

    auto verify_res = [&](size_t id, double score, size_t index) {
        size_t diff_id = ((int)(id - 50) > 0) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    float query[] = {50, 50, 50, 50};
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    // Check that user args did not change.
    ASSERT_EQ(info.hnswInfo.M, M);
    ASSERT_EQ(info.hnswInfo.efConstruction, efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    // Run same query again, set efRuntime to 300.
    VecSimQueryParams queryParams{.hnswRuntimeParams = HNSWRuntimeParams{.efRuntime = 300}};
    runTopKSearchTest(index, query, k, verify_res, &queryParams);

    info = VecSimIndex_Info(index);
    // Check that user args did not change.
    ASSERT_EQ(info.hnswInfo.M, M);
    ASSERT_EQ(info.hnswInfo.efConstruction, efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    // Create batch iterator with query param - verify that ef_runtime is set.
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, &queryParams);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.hnswInfo.efRuntime, 300);
    // Run one batch for sanity.
    runBatchIteratorSearchTest(batchIterator, k, verify_res);
    // After releasing the batch iterator, ef_runtime should return to the default one.
    VecSimBatchIterator_Free(batchIterator);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_search_empty_index) {
    size_t n = 100;
    size_t k = 11;
    size_t d = 4;
    VecSimParams params{
        .algo = VecSimAlgo_HNSWLIB,
        .hnswParams = HNSWParams{
            .type = VecSimType_FLOAT32, .dim = d, .metric = VecSimMetric_L2, .initialCapacity = 0}};
    VecSimIndex *index = VecSimIndex_New(&params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float query[] = {50, 50, 50, 50};

    // We do not expect any results.
    VecSimQueryResult_List res =
        VecSimIndex_TopKQuery(index, (const void *)query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Iterator *it = VecSimQueryResult_List_GetIterator(res);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);

    res = VecSimIndex_RangeQuery(index, (const void *)query, 1.0f, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
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

    // Again - we do not expect any results.
    res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    it = VecSimQueryResult_List_GetIterator(res);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);

    res = VecSimIndex_RangeQuery(index, (const void *)query, 1.0f, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_inf_score) {
    size_t n = 4;
    size_t k = 4;
    size_t dim = 2;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
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

    auto verify_res = [&](size_t id, double score, size_t index) {
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
TEST_F(HNSWTest, hnsw_bad_params) {
    size_t n = 1000000;
    size_t dim = 2;
    size_t bad_M[] = {
        1,         // Will fail because 1/log(M).
        100000000, // Will fail on this->allocator->allocate(max_elements_ * size_data_per_element_)
        SIZE_MAX,  // Will fail on M * 2 overflow.
        SIZE_MAX / 2, // Will fail on M * 2 overflow.
        SIZE_MAX / 4  // Will fail on size_links_level0_ calculation:
                      // sizeof(linklistsizeint) + M * 2 * sizeof(idType) + sizeof(void *)
    };
    unsigned long len = sizeof(bad_M) / sizeof(size_t);

    for (unsigned long i = 0; i < len; i++) {
        VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                            .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                     .dim = dim,
                                                     .metric = VecSimMetric_L2,
                                                     .initialCapacity = n}};

        params.hnswParams.M = bad_M[i];
        VecSimIndex *index = VecSimIndex_New(&params);
        ASSERT_TRUE(index == NULL);
    }
}

TEST_F(HNSWTest, hnsw_delete_entry_point) {
    size_t n = 10000;
    size_t dim = 4;
    size_t M = 2;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .M = M,
                                                 .efConstruction = 0,
                                                 .efRuntime = 0}};

    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_TRUE(index != NULL);

    int64_t vec[dim];
    for (size_t i = 0; i < dim; i++)
        vec[i] = i;
    for (size_t j = 0; j < n; j++)
        VecSimIndex_AddVector(index, vec, j);

    VecSimIndexInfo info = VecSimIndex_Info(index);

    while (info.hnswInfo.indexSize > 0) {
        ASSERT_NO_THROW(VecSimIndex_DeleteVector(index, info.hnswInfo.entrypoint));
        info = VecSimIndex_Info(index);
    }
    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_override) {
    size_t n = 100;
    size_t dim = 4;
    size_t M = 8;
    size_t ef = 300;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .M = M,
                                                 .efConstruction = 20,
                                                 .efRuntime = ef}};

    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_TRUE(index != nullptr);

    // Insert n == 100 vectors.
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Insert again 300 vectors, the first 100 will be overwritten (deleted first).
    n = 300;
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }

    float query[dim];
    for (size_t j = 0; j < dim; j++) {
        query[j] = (float)n;
    }
    // This is testing a bug fix - before we had the seconder sorting by id in CompareByFirst,
    // the graph got disconnected due to the deletion of some node followed by a bad repairing of
    // one of its neighbours. Here, we ensure that we get all the nodes in the graph as results.
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(id == n - 1 - index);
    };
    runTopKSearchTest(index, query, 300, verify_res);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_batch_iterator_basic) {
    size_t dim = 4;
    size_t M = 8;
    size_t ef = 20;
    size_t n = 1000;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .M = M,
                                                 .efConstruction = ef,
                                                 .efRuntime = ef}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // For every i, add the vector (i,i,i,i) under the label i.
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
    float query[dim];
    for (size_t j = 0; j < dim; j++) {
        query[j] = (float)n;
    }
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
    size_t iteration_num = 0;

    // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet
    // in every iteration. The results order should be sorted by their score (distance from the
    // query vector), which means sorted from the largest id to the lowest.
    size_t n_res = 5;
    while (VecSimBatchIterator_HasNext(batchIterator)) {
        std::vector<size_t> expected_ids(n_res);
        for (size_t i = 0; i < n_res; i++) {
            expected_ids[i] = (n - iteration_num * n_res - i - 1);
        }
        auto verify_res = [&](size_t id, double score, size_t index) {
            ASSERT_TRUE(expected_ids[index] == id);
        };
        runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
        iteration_num++;
    }
    ASSERT_EQ(iteration_num, n / n_res);
    VecSimBatchIterator_Free(batchIterator);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_batch_iterator_reset) {
    size_t dim = 4;
    size_t n = 1000;
    size_t M = 8;
    size_t ef = 20;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .M = M,
                                                 .efConstruction = ef,
                                                 .efRuntime = ef}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
    float query[dim];
    for (size_t j = 0; j < dim; j++) {
        query[j] = (float)n;
    }
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    // Get the 100 vectors whose ids are the maximal among those that hasn't been returned yet, in
    // every iteration. Run this flow for 3 times, and reset the iterator.
    size_t n_res = 100;
    size_t re_runs = 3;

    for (size_t take = 0; take < re_runs; take++) {
        size_t iteration_num = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            std::vector<size_t> expected_ids(n_res);
            for (size_t i = 0; i < n_res; i++) {
                expected_ids[i] = (n - iteration_num * n_res - i - 1);
            }
            auto verify_res = [&](size_t id, double score, size_t index) {
                ASSERT_TRUE(expected_ids[index] == id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_SCORE);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num, n / n_res);
        VecSimBatchIterator_Reset(batchIterator);
    }
    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_batch_iterator_batch_size_1) {
    size_t dim = 4;
    size_t n = 1000;
    size_t M = 8;
    size_t ef = 2;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .M = M,
                                                 .efConstruction = ef,
                                                 .efRuntime = ef}};
    VecSimIndex *index = VecSimIndex_New(&params);

    float query[] = {(float)n, (float)n, (float)n, (float)n};

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        // Set labels to be different than the internal ids.
        VecSimIndex_AddVector(index, (const void *)f, n - i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
    size_t iteration_num = 0;
    size_t n_res = 1, expected_n_res = 1;
    while (VecSimBatchIterator_HasNext(batchIterator)) {
        iteration_num++;
        // Expect to get results in the reverse order of labels - which is the order of the distance
        // from the query vector. Get one result in every iteration.
        auto verify_res = [&](size_t id, double score, size_t index) {
            ASSERT_TRUE(id == iteration_num);
        };
        runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_SCORE, expected_n_res);
    }

    ASSERT_EQ(iteration_num, n);
    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_batch_iterator_advanced) {
    size_t dim = 4;
    size_t n = 1000;
    size_t M = 8;
    size_t ef = 1000;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .M = M,
                                                 .efConstruction = ef,
                                                 .efRuntime = ef}};
    VecSimIndex *index = VecSimIndex_New(&params);

    float query[] = {(float)n, (float)n, (float)n, (float)n};
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    // Try to get results even though there are no vectors in the index.
    VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, 10, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);
    ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));

    // Insert one vector and query again. The internal id will be 0.
    VecSimIndex_AddVector(index, query, n);
    VecSimBatchIterator_Reset(batchIterator);
    res = VecSimBatchIterator_Next(batchIterator, 10, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 1);
    VecSimQueryResult_Free(res);
    ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));

    for (size_t i = 1; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Reset the iterator after it was depleted.
    VecSimBatchIterator_Reset(batchIterator);

    // Try to get 0 results.
    res = VecSimBatchIterator_Next(batchIterator, 0, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);

    // n_res does not divide into ef or vice versa - expect leftovers between the graph scans.
    size_t n_res = 7;
    size_t iteration_num = 0;

    while (VecSimBatchIterator_HasNext(batchIterator)) {
        iteration_num++;
        std::vector<size_t> expected_ids;
        // We ask to get the results sorted by ID in a specific batch (in ascending order), but
        // in every iteration the ids should be lower than the previous one, according to the
        // distance from the query.
        for (size_t i = 1; i <= n_res; i++) {
            expected_ids.push_back(n - iteration_num * n_res + i);
        }
        auto verify_res = [&](size_t id, double score, size_t index) {
            ASSERT_TRUE(expected_ids[index] == id);
        };
        if (iteration_num <= n / n_res) {
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_ID);
        } else {
            // In the last iteration there are n%iteration_num (=6) results left to return.
            expected_ids.erase(expected_ids.begin()); // remove the first id
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_ID, n % n_res);
        }
    }
    ASSERT_EQ(iteration_num, n / n_res + 1);
    // Try to get more results even though there are no.
    res = VecSimBatchIterator_Next(batchIterator, 1, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);

    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, hnsw_resolve_params) {
    size_t dim = 4;
    size_t M = 8;
    size_t ef = 2;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = 0,
                                                 .M = M,
                                                 .efConstruction = ef,
                                                 .efRuntime = ef}};
    VecSimIndex *index = VecSimIndex_New(&params);

    VecSimQueryParams qparams, zero;
    bzero(&zero, sizeof(VecSimQueryParams));

    auto *rparams = array_new<VecSimRawParam>(3);

    // Test with empty runtime params.
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSim_OK);
    ASSERT_EQ(memcmp(&qparams, &zero, sizeof(VecSimQueryParams)), 0);

    array_append(rparams, (VecSimRawParam){
                              .name = "ef_runtime", .nameLen = 10, .value = "100", .valLen = 3});
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSim_OK);
    ASSERT_EQ(qparams.hnswRuntimeParams.efRuntime, 100);

    rparams[0] = (VecSimRawParam){.name = "wrong_name", .nameLen = 10, .value = "100", .valLen = 3};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSimParamResolverErr_UnknownParam);

    // Testing for legal prefix but only partial parameter name.
    rparams[0] = (VecSimRawParam){.name = "ef_run", .nameLen = 6, .value = "100", .valLen = 3};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSimParamResolverErr_UnknownParam);

    rparams[0] =
        (VecSimRawParam){.name = "ef_runtime", .nameLen = 10, .value = "wrong_val", .valLen = 9};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSimParamResolverErr_BadValue);

    rparams[0] = (VecSimRawParam){.name = "ef_runtime", .nameLen = 10, .value = "-30", .valLen = 3};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSimParamResolverErr_BadValue);

    rparams[0] =
        (VecSimRawParam){.name = "ef_runtime", .nameLen = 10, .value = "1.618", .valLen = 5};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSimParamResolverErr_BadValue);

    rparams[0] = (VecSimRawParam){.name = "ef_runtime", .nameLen = 10, .value = "100", .valLen = 3};
    array_append(rparams, (VecSimRawParam){
                              .name = "ef_runtime", .nameLen = 10, .value = "100", .valLen = 3});
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSimParamResolverErr_AlreadySet);

    /** Testing with hybrid query params - cases which are only relevant for HNSW index. **/
    // Cannot set ef_runtime param with "hybrid_policy" which is "ADHOC_BF"
    rparams[1] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "ADHOC_BF",
                                  .valLen = strlen("ADHOC_BF")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_InvalidPolicy_AdHoc_With_EfRuntime);

    rparams[1] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "BATCHES",
                                  .valLen = strlen("BATCHES")};
    array_append(rparams, (VecSimRawParam){.name = "batch_size",
                                           .nameLen = strlen("batch_size"),
                                           .value = "50",
                                           .valLen = strlen("50")});
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 50);
    ASSERT_EQ(qparams.hnswRuntimeParams.efRuntime, 100);

    VecSimIndex_Free(index);
    array_free(rparams);
}

TEST_F(HNSWTest, hnsw_serialization_v1) {
    size_t dim = 4;
    size_t n = 1000;
    size_t M = 8;
    size_t ef = 10;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .blockSize = 1,
                                                 .M = M,
                                                 .efConstruction = ef,
                                                 .efRuntime = ef}};
    VecSimIndex *index = VecSimIndex_New(&params);

    auto serializer = HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float> *>(index));

    auto file_name = std::string(getenv("ROOT")) + "/tests/unit/data/1k-d4-L2-M8-ef_c10.hnsw_v1";
    // Save and load an empty index.
    serializer.saveIndex(file_name);
    serializer.loadIndex(file_name);
    auto res = serializer.checkIntegrity();
    ASSERT_TRUE(res.valid_state);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    // Get index info and copy it, so it will be available after the index is deleted.
    VecSimIndexInfo info = VecSimIndex_Info(index);

    // Persist index with the serializer, and delete it.
    serializer.saveIndex(file_name);
    VecSimIndex_Free(index);

    // Create new index, set it into the serializer and extract the data to it.
    auto new_index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(new_index), 0);

    serializer.reset(reinterpret_cast<HNSWIndex<float, float> *>(new_index));
    serializer.loadIndex(file_name);

    // Validate that the new loaded index has the same meta-data as the original.
    VecSimIndexInfo new_info = VecSimIndex_Info(new_index);
    ASSERT_EQ(info.algo, new_info.algo);
    ASSERT_EQ(info.hnswInfo.M, new_info.hnswInfo.M);
    ASSERT_EQ(info.hnswInfo.efConstruction, new_info.hnswInfo.efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, new_info.hnswInfo.efRuntime);
    ASSERT_EQ(info.hnswInfo.indexSize, new_info.hnswInfo.indexSize);
    ASSERT_EQ(info.hnswInfo.max_level, new_info.hnswInfo.max_level);
    ASSERT_EQ(info.hnswInfo.entrypoint, new_info.hnswInfo.entrypoint);
    ASSERT_EQ(info.hnswInfo.metric, new_info.hnswInfo.metric);
    ASSERT_EQ(info.hnswInfo.type, new_info.hnswInfo.type);
    ASSERT_EQ(info.hnswInfo.dim, new_info.hnswInfo.dim);

    res = serializer.checkIntegrity();
    ASSERT_TRUE(res.valid_state);

    // Add 1000 random vectors, override the existing ones to trigger deletions.
    std::vector<float> data((n + 1) * dim);
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < (n + 1) * dim; ++i) {
        data[i] = (float)distrib(rng);
    }
    for (size_t i = 0; i < n + 1; ++i) {
        VecSimIndex_AddVector(new_index, data.data() + dim * i, i);
    }

    // Delete arbitrary vector (trigger removal of a block).
    VecSimIndex_DeleteVector(new_index, (size_t)(distrib(rng) * (n + 1)));

    HNSWIndex<float, float> *hnswNewIndex = reinterpret_cast<HNSWIndex<float, float> *>(new_index);
    ASSERT_EQ(hnswNewIndex->getIndexCapacity(), n);

    // Persist index, delete it from memory and restore.
    serializer.saveIndex(file_name);
    VecSimIndex_Free(new_index);

    params.hnswParams.initialCapacity = n / 2; // to ensure that we resize in load time.
    auto restored_index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(restored_index), 0);

    serializer.reset(reinterpret_cast<HNSWIndex<float, float> *>(restored_index));
    serializer.loadIndex(file_name);
    ASSERT_EQ(VecSimIndex_IndexSize(restored_index), n);
    res = serializer.checkIntegrity();
    ASSERT_TRUE(res.valid_state);

    // Clean-up.
    remove(file_name.c_str());
    VecSimIndex_Free(restored_index);
    serializer.reset();
}

TEST_F(HNSWTest, hnsw_get_distance) {
    size_t n = 4;
    size_t dim = 2;
    size_t numIndex = 3;
    VecSimIndex *index[numIndex];
    std::vector<double> distances;

    float v1[] = {M_PI, M_PI};
    float v2[] = {M_E, M_E};
    float v3[] = {M_PI, M_E};
    float v4[] = {M_SQRT2, -M_SQRT2};

    VecSimParams params{
        .algo = VecSimAlgo_HNSWLIB,
        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32, .dim = dim, .initialCapacity = n}};

    for (size_t i = 0; i < numIndex; i++) {
        params.bfParams.metric = (VecSimMetric)i;
        index[i] = VecSimIndex_New(&params);
        VecSimIndex_AddVector(index[i], v1, 1);
        VecSimIndex_AddVector(index[i], v2, 2);
        VecSimIndex_AddVector(index[i], v3, 3);
        VecSimIndex_AddVector(index[i], v4, 4);
        ASSERT_EQ(VecSimIndex_IndexSize(index[i]), 4);
    }

    void *query = v1;
    void *norm = v2;                                 // {e, e}
    VecSim_Normalize(norm, dim, VecSimType_FLOAT32); // now {1/sqrt(2), 1/sqrt(2)}
    ASSERT_FLOAT_EQ(((float *)norm)[0], 1.0f / sqrt(2.0f));
    ASSERT_FLOAT_EQ(((float *)norm)[1], 1.0f / sqrt(2.0f));
    double dist;

    // VecSimMetric_L2
    distances = {0, 0.3583844006061554, 0.1791922003030777, 23.739208221435547};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_L2], i + 1, query);
        ASSERT_DOUBLE_EQ(dist, distances[i]);
    }

    // VecSimMetric_IP
    distances = {-18.73921012878418, -16.0794677734375, -17.409339904785156, 1};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_IP], i + 1, query);
        ASSERT_DOUBLE_EQ(dist, distances[i]);
    }

    // VecSimMetric_Cosine
    distances = {5.9604644775390625e-08, 5.9604644775390625e-08, 0.0025991201400756836, 1};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_Cosine], i + 1, norm);
        ASSERT_DOUBLE_EQ(dist, distances[i]);
    }

    // Bad values
    dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_Cosine], 0, norm);
    ASSERT_TRUE(std::isnan(dist));
    dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_L2], 46, query);
    ASSERT_TRUE(std::isnan(dist));

    // Clean-up.
    for (size_t i = 0; i < numIndex; i++) {
        VecSimIndex_Free(index[i]);
    }
}

TEST_F(HNSWTest, preferAdHocOptimization) {
    // Save the expected result for every combination that represent a different leaf in the tree.
    // map: [k, index_size, dim, M, r] -> res
    std::map<std::vector<float>, bool> combinations;
    combinations[{5, 1000, 5, 5, 0.5}] = true;
    combinations[{5, 6000, 5, 5, 0.1}] = true;
    combinations[{5, 6000, 5, 5, 0.2}] = false;
    combinations[{5, 6000, 60, 5, 0.5}] = false;
    combinations[{5, 6000, 60, 15, 0.5}] = true;
    combinations[{15, 6000, 50, 5, 0.5}] = true;
    combinations[{5, 700000, 60, 5, 0.05}] = true;
    combinations[{5, 800000, 60, 5, 0.05}] = false;
    combinations[{10, 800000, 60, 5, 0.01}] = true;
    combinations[{10, 800000, 60, 5, 0.05}] = false;
    combinations[{10, 800000, 60, 5, 0.1}] = false;
    combinations[{10, 60000, 100, 5, 0.1}] = true;
    combinations[{10, 80000, 100, 5, 0.1}] = false;
    combinations[{10, 60000, 100, 60, 0.1}] = true;
    combinations[{10, 60000, 100, 5, 0.3}] = false;
    combinations[{20, 60000, 100, 5, 0.1}] = true;
    combinations[{20, 60000, 100, 5, 0.2}] = false;
    combinations[{20, 60000, 100, 20, 0.1}] = true;
    combinations[{20, 350000, 100, 20, 0.1}] = true;
    combinations[{20, 350000, 100, 20, 0.2}] = false;

    for (auto &comb : combinations) {
        auto k = (size_t)comb.first[0];
        auto index_size = (size_t)comb.first[1];
        auto dim = (size_t)comb.first[2];
        auto M = (size_t)comb.first[3];
        auto r = comb.first[4];

        // Create index and check for the expected output of "prefer ad-hoc" heuristics.
        VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                            .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                     .dim = dim,
                                                     .metric = VecSimMetric_L2,
                                                     .initialCapacity = index_size,
                                                     .M = M,
                                                     .efConstruction = 1,
                                                     .efRuntime = 1}};
        VecSimIndex *index = VecSimIndex_New(&params);

        // Set the index size artificially to be the required one.
        reinterpret_cast<HNSWIndex<float, float> *>(index)->cur_element_count = index_size;
        for (size_t i = 0; i < index_size; i++) {
            reinterpret_cast<HNSWIndex_Single<float, float> *>(index)->label_lookup_[i] = i;
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), index_size);
        bool res = VecSimIndex_PreferAdHocSearch(index, (size_t)(r * (float)index_size), k, true);
        ASSERT_EQ(res, comb.second);
        VecSimIndex_Free(index);
    }
    // Corner cases - empty index.
    VecSimParams params{
        .algo = VecSimAlgo_HNSWLIB,
        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32, .dim = 4, .metric = VecSimMetric_L2}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 0, 50, true));

    // Corner cases - subset size is greater than index size.
    try {
        VecSimIndex_PreferAdHocSearch(index, 1, 50, true);
        FAIL() << "Expected std::runtime error";
    } catch (std::runtime_error const &err) {
        EXPECT_EQ(err.what(),
                  std::string("internal error: subset size cannot be larger than index size"));
    }
    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, testCosine) {
    size_t dim = 4;
    size_t n = 100;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_Cosine,
                                                 .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 1; i <= n; i++) {
        float f[dim];
        f[0] = (float)i / n;
        for (size_t j = 1; j < dim; j++) {
            f[j] = 1.0f;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    float query[dim];
    for (size_t i = 0; i < dim; i++) {
        query[i] = 1.0f;
    }
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, (n - index));
        double first_coordinate = (double)id / n;
        // By cosine definition: 1 - ((A \dot B) / (norm(A)*norm(B))), where A is the query vector
        // and B is the current result vector.
        double expected_score =
            1.0f -
            ((first_coordinate + (double)dim - 1.0f) /
             (sqrtf((double)dim) * sqrtf((double)(dim - 1) + first_coordinate * first_coordinate)));
        // Verify that abs difference between the actual and expected score is at most 1/10^6.
        ASSERT_NEAR(score, expected_score, 1e-5);
    };
    runTopKSearchTest(index, query, 10, verify_res);

    // Test with batch iterator.
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
    size_t iteration_num = 0;

    // Get the 10 vectors whose ids are the maximal among those that hasn't been returned yet,
    // in every iteration. The order should be from the largest to the lowest id.
    size_t n_res = 10;
    while (VecSimBatchIterator_HasNext(batchIterator)) {
        std::vector<size_t> expected_ids(n_res);
        auto verify_res_batch = [&](size_t id, double score, size_t index) {
            ASSERT_EQ(id, (n - n_res * iteration_num - index));
            double first_coordinate = (double)id / n;
            // By cosine definition: 1 - ((A \dot B) / (norm(A)*norm(B))), where A is the query
            // vector and B is the current result vector.
            double expected_score =
                1.0f - ((first_coordinate + (double)dim - 1.0f) /
                        (sqrtf((double)dim) *
                         sqrtf((double)(dim - 1) + first_coordinate * first_coordinate)));
            // Verify that abs difference between the actual and expected score is at most 1/10^6.
            ASSERT_NEAR(score, expected_score, 1e-5);
        };
        runBatchIteratorSearchTest(batchIterator, n_res, verify_res_batch);
        iteration_num++;
    }
    ASSERT_EQ(iteration_num, n / n_res);
    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, testSizeEstimation) {
    size_t dim = 128;
    size_t n = 1000;
    size_t bs = DEFAULT_BLOCK_SIZE;
    size_t M = 32;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .blockSize = bs,
                                                 .M = M}};

    size_t estimation = VecSimIndex_EstimateInitialSize(&params);
    VecSimIndex *index = VecSimIndex_New(&params);

    size_t actual = index->getAllocator()->getAllocationSize();
    // labels_lookup hash table has additional memory, since STL implementation chooses "an
    // appropriate prime number" higher than n as the number of allocated buckets (for n=1000, 1031
    // buckets are created)
    estimation += (reinterpret_cast<HNSWIndex_Single<float, float> *>(index)

                       ->label_lookup_.bucket_count() -
                   n) *
                  sizeof(size_t);

    ASSERT_EQ(estimation, actual);

    float vec[dim];
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)i;
        }
        VecSimIndex_AddVector(index, vec, i);
    }

    // Estimate the memory delta of adding a full new block.
    estimation = VecSimIndex_EstimateElementSize(&params) * (bs % n + bs);

    actual = 0;
    for (size_t i = 0; i < bs; i++) {
        for (size_t j = 0; j < dim; j++) {
            vec[j] = (float)i;
        }
        actual += VecSimIndex_AddVector(index, vec, n + i);
    }
    ASSERT_GE(estimation * 1.01, actual);
    ASSERT_LE(estimation * 0.99, actual);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, testInitialSizeEstimation_No_InitialCapacity) {
    size_t dim = 128;
    size_t n = 0;
    size_t bs = DEFAULT_BLOCK_SIZE;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_Cosine,
                                                 .initialCapacity = n,
                                                 .blockSize = bs}};

    VecSimIndex *index = VecSimIndex_New(&params);
    size_t estimation = VecSimIndex_EstimateInitialSize(&params);

    size_t actual = index->getAllocator()->getAllocationSize();

    // labels_lookup and element_levels containers are not allocated at all in some platforms,
    // when initial capacity is zero, while in other platforms labels_lookup is allocated with a
    // single bucket. This, we get the following range in which we expect the initial memory to be
    // in.
    ASSERT_GE(actual, estimation);
    ASSERT_LE(actual, estimation + sizeof(size_t) + 2 * sizeof(size_t));

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, testTimeoutReturn) {
    size_t dim = 4;
    float vec[] = {1.0f, 1.0f, 1.0f, 1.0f};
    VecSimQueryResult_List rl;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = 1,
                                                 .blockSize = 5}};
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndex_AddVector(index, vec, 0);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out

    // Checks return code on timeout.
    rl = VecSimIndex_TopKQuery(index, vec, 1, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    // Check timeout again - range query.
    VecSimIndex_AddVector(index, vec, 1);
    ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.max_level, 0);
    // Here, the entry point is inserted to the results set before we test for timeout.
    // hence, expect a single result to be returned (instead of 2 that would have return without
    // timeout).
    rl = VecSimIndex_RangeQuery(index, vec, 1, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 1);
    VecSimQueryResult_Free(rl);

    // Fail on searching bottom layer entry point.
    // We need to have at least 1 vector in layer higher than 0 to fail there.
    size_t next = 0;
    while (VecSimIndex_Info(index).hnswInfo.max_level == 0) {
        VecSimIndex_AddVector(index, vec, next++);
    }
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out.

    rl = VecSimIndex_TopKQuery(index, vec, 2, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    // Timeout on searching bottom layer entry point - range query.
    rl = VecSimIndex_RangeQuery(index, vec, 1, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSimIndex_Free(index);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // Cleanup.
}

TEST_F(HNSWTest, testTimeoutReturn_batch_iterator) {
    size_t dim = 4;
    size_t n = 2;
    float vec[] = {1.0f, 1.0f, 1.0f, 1.0f};
    VecSimQueryResult_List rl;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        VecSimIndex_AddVector(index, vec, 46 - i);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Fail on second batch (after some calculation already completed in the first one).
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, vec, nullptr);

    rl = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_OK);
    ASSERT_NE(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out.
    rl = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSimBatchIterator_Free(batchIterator);

    // Fail on first batch (while calculating).
    auto timeoutcb = [](void *ctx) {
        static size_t flag = 1;
        if (flag) {
            flag = 0;
            return 0;
        } else {
            return 1;
        }
    };
    VecSim_SetTimeoutCallbackFunction(timeoutcb); // Fails on second call.
    batchIterator = VecSimBatchIterator_New(index, vec, nullptr);

    rl = VecSimBatchIterator_Next(batchIterator, 2, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSimBatchIterator_Free(batchIterator);

    // Fail on searching bottom layer entry point.
    // We need to have at least 1 vector in layer higher than 0 to fail there.
    size_t next = 0;
    while (VecSimIndex_Info(index).hnswInfo.max_level == 0) {
        VecSimIndex_AddVector(index, vec, next++);
    }
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out.
    batchIterator = VecSimBatchIterator_New(index, vec, nullptr);

    rl = VecSimBatchIterator_Next(batchIterator, 2, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSimBatchIterator_Free(batchIterator);

    VecSimIndex_Free(index);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // Cleanup.
}

TEST_F(HNSWTest, rangeQuery) {
    size_t n = 5000;
    size_t dim = 4;

    VecSimParams params{
        .algo = VecSimAlgo_HNSWLIB,
        .hnswParams = HNSWParams{
            .type = VecSimType_FLOAT32, .dim = dim, .metric = VecSimMetric_L2, .blockSize = n / 2}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, (int)i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    size_t pivot_id = n / 2; // the id to return vectors around it.
    float query[] = {(float)pivot_id, (float)pivot_id, (float)pivot_id, (float)pivot_id};

    auto verify_res_by_score = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(std::abs(int(id - pivot_id)), (index + 1) / 2);
        ASSERT_EQ(score, dim * pow((index + 1) / 2, 2));
    };
    uint expected_num_results = 11;
    // To get 11 results in the range [pivot_id - 5, pivot_id + 5], set the radius as the L2 score
    // in the boundaries.
    double radius = dim * pow(expected_num_results / 2, 2);
    runRangeQueryTest(index, query, radius, verify_res_by_score, expected_num_results, BY_SCORE);

    // Rerun with a given query params. This high epsilon value will cause the range search main
    // loop to break since we insert a candidate whose distance is within the dynamic range
    // boundaries at the beginning of the search, but when this candidate is popped out from the
    // queue, it's no longer within the dynamic range boundaries.
    auto query_params = VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.epsilon = 1.0}};
    runRangeQueryTest(index, query, radius, verify_res_by_score, expected_num_results, BY_SCORE,
                      &query_params);

    // Get results by id.
    auto verify_res_by_id = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, pivot_id - expected_num_results / 2 + index);
        ASSERT_EQ(score, dim * pow(std::abs(int(id - pivot_id)), 2));
    };
    runRangeQueryTest(index, query, radius, verify_res_by_id, expected_num_results);

    VecSimIndex_Free(index);
}

TEST_F(HNSWTest, rangeQueryCosine) {
    size_t n = 800;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_Cosine,
                                                 .blockSize = n / 2}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        f[0] = float(i + 1) / n;
        for (size_t j = 1; j < dim; j++) {
            f[j] = 1.0f;
        }
        // Use as label := n - (internal id)
        VecSimIndex_AddVector(index, (const void *)f, n - i);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    float query[dim];
    for (size_t i = 0; i < dim; i++) {
        query[i] = 1.0f;
    }
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, index + 1);
        double first_coordinate = float(n - index) / n;
        // By cosine definition: 1 - ((A \dot B) / (norm(A)*norm(B))), where A is the query vector
        // and B is the current result vector.
        double expected_score =
            1.0f -
            ((first_coordinate + (double)dim - 1.0f) /
             (sqrtf((double)dim) * sqrtf((double)(dim - 1) + first_coordinate * first_coordinate)));
        // Verify that abs difference between the actual and expected score is at most 1/10^5.
        ASSERT_NEAR(score, expected_score, 1e-5);
    };
    uint expected_num_results = 31;
    // Calculate the score of the 31st distant vector from the query vector (whose id should be 30)
    // to get the radius.
    double edge_first_coordinate = (double)(n - expected_num_results + 1) / n;
    double radius =
        1.0f - ((edge_first_coordinate + (double)dim - 1.0) /
                (sqrt((double)dim) *
                 sqrt((double)(dim - 1) + edge_first_coordinate * edge_first_coordinate)));
    runRangeQueryTest(index, query, radius, verify_res, expected_num_results, BY_SCORE);

    // Return results BY_ID should give the same results.
    runRangeQueryTest(index, query, radius, verify_res, expected_num_results, BY_ID);

    VecSimIndex_Free(index);
}
