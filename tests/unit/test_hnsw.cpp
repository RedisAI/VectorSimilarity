/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "test_utils.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/query_result_struct.h"
#include <climits>
#include <unistd.h>
#include <random>
#include <thread>
#include <atomic>

template <typename index_type_t>
class HNSWTest : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    VecSimIndex *CreateNewIndex(HNSWParams &params, bool is_multi = false) {
        return test_utils::CreateNewIndex(params, index_type_t::get_index_type(), is_multi);
    }
    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex<data_t, dist_t> *>(index);
    }

    HNSWIndex_Single<data_t, dist_t> *CastToHNSW_Single(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex_Single<data_t, dist_t> *>(index);
    }
};

// DataTypeSet, TEST_DATA_T and TEST_DIST_T are defined in test_utils.h

TYPED_TEST_SUITE(HNSWTest, DataTypeSet);

TYPED_TEST(HNSWTest, hnsw_vector_add_test) {
    size_t dim = 4;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 200,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, hnsw_blob_sanity_test) {
    size_t dim = 4;
    size_t bs = 1;
#define ASSERT_HNSW_BLOB_EQ(id, blob)                                                              \
    do {                                                                                           \
        void *v = hnsw_index->getDataByInternalId(id);                                             \
        ASSERT_FALSE(memcmp(v, blob, sizeof(blob)));                                               \
    } while (0)

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T a[dim], b[dim], c[dim], d[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (TEST_DATA_T)0;
        b[i] = (TEST_DATA_T)1;
        c[i] = (TEST_DATA_T)2;
        d[i] = (TEST_DATA_T)3;
    }
    HNSWIndex<TEST_DATA_T, TEST_DIST_T> *hnsw_index = this->CastToHNSW(index);

    VecSimIndex_AddVector(index, a, 42);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    ASSERT_HNSW_BLOB_EQ(0, a);
    ASSERT_EQ(hnsw_index->getExternalLabel(0), 42);

    VecSimIndex_AddVector(index, b, 46);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 2);
    ASSERT_HNSW_BLOB_EQ(1, b);
    ASSERT_EQ(hnsw_index->getExternalLabel(1), 46);

    // After inserting c with label 46, we first delete id 1 from the index.
    // we expect id 0 to not change
    VecSimIndex_AddVector(index, c, 46);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 2);
    ASSERT_HNSW_BLOB_EQ(0, a);
    ASSERT_HNSW_BLOB_EQ(1, c);
    ASSERT_EQ(hnsw_index->getExternalLabel(0), 42);
    ASSERT_EQ(hnsw_index->getExternalLabel(1), 46);

    // After inserting d with label 42, we first delete id 0 and move the last id (1) to be 0.
    // Then we add the new vector d under the internal id 1.
    VecSimIndex_AddVector(index, d, 42);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 2);
    ASSERT_HNSW_BLOB_EQ(0, c);
    ASSERT_HNSW_BLOB_EQ(1, d);
    ASSERT_EQ(hnsw_index->getExternalLabel(0), 46);
    ASSERT_EQ(hnsw_index->getExternalLabel(1), 42);

    VecSimIndex_Free(index);
}

/**** resizing cases ****/

// Add up to capacity.
TYPED_TEST(HNSWTest, resizeNAlignIndex) {
    size_t dim = 4;
    size_t n = 10;
    size_t bs = 3;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);

    // Add up to n.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    // The size and the capacity should be equal.
    HNSWIndex<TEST_DATA_T, TEST_DIST_T> *hnswIndex = this->CastToHNSW(index);
    ASSERT_EQ(hnswIndex->indexCapacity(), VecSimIndex_IndexSize(index));
    // The capacity shouldn't be changed.
    ASSERT_EQ(hnswIndex->indexCapacity(), n);

    // Add another vector to exceed the initial capacity.
    GenerateAndAddVector<TEST_DATA_T>(index, dim, n);

    // The capacity should be now aligned with the block size.
    // bs = 3, size = 11 -> capacity = 12
    // New capacity = initial capacity + blockSize - initial capacity % blockSize.
    ASSERT_EQ(hnswIndex->indexCapacity(), n + bs - n % bs);
    VecSimIndex_Free(index);
}

// Case 1: initial capacity is larger than block size, and it is not aligned.
TYPED_TEST(HNSWTest, resizeNAlignIndex_largeInitialCapacity) {
    size_t dim = 4;
    size_t n = 10;
    size_t bs = 3;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // add up to blocksize + 1 = 3 + 1 = 4
    for (size_t i = 0; i < bs + 1; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }

    // The capacity shouldn't change, should remain n.
    HNSWIndex<TEST_DATA_T, TEST_DIST_T> *hnswIndex = this->CastToHNSW(index);
    ASSERT_EQ(hnswIndex->indexCapacity(), n);

    // Delete last vector, to get size % block_size == 0. size = 3
    VecSimIndex_DeleteVector(index, bs);

    // Index size = bs = 3.
    ASSERT_EQ(VecSimIndex_IndexSize(index), bs);

    // New capacity = initial capacity - block_size - number_of_vectors_to_align =
    // 10  - 3 - 10 % 3 (1) = 6
    size_t curr_capacity = hnswIndex->indexCapacity();
    ASSERT_EQ(curr_capacity, n - bs - n % bs);

    // Delete all the vectors to decrease capacity by another bs.
    size_t i = 0;
    while (VecSimIndex_IndexSize(index) > 0) {
        VecSimIndex_DeleteVector(index, i);
        ++i;
    }
    ASSERT_EQ(hnswIndex->indexCapacity(), bs);
    // Add and delete a vector to achieve:
    // size % block_size == 0 && size + bs <= capacity(3).
    // the capacity should be resized to zero
    GenerateAndAddVector<TEST_DATA_T>(index, dim, 0);
    VecSimIndex_DeleteVector(index, 0);
    ASSERT_EQ(hnswIndex->indexCapacity(), 0);

    // Do it again. This time after adding a vector the capacity is increased by bs.
    // Upon deletion it will be resized to zero again.
    GenerateAndAddVector<TEST_DATA_T>(index, dim, 0);
    ASSERT_EQ(hnswIndex->indexCapacity(), bs);
    VecSimIndex_DeleteVector(index, 0);
    ASSERT_EQ(hnswIndex->indexCapacity(), 0);

    VecSimIndex_Free(index);
}

// Case 2: initial capacity is smaller than block_size.
TYPED_TEST(HNSWTest, resizeNAlignIndex_largerBlockSize) {
    size_t dim = 4;
    size_t n = 4;
    size_t bs = 6;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Add up to initial capacity.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }

    HNSWIndex<TEST_DATA_T, TEST_DIST_T> *hnswIndex = this->CastToHNSW(index);
    // The capacity shouldn't change.
    ASSERT_EQ(hnswIndex->indexCapacity(), n);

    // Size equals capacity.
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Add another vector - > the capacity is increased to a multiplication of block_size.
    GenerateAndAddVector<TEST_DATA_T>(index, dim, n);
    ASSERT_EQ(hnswIndex->indexCapacity(), bs);

    // Size increased by 1.
    ASSERT_EQ(VecSimIndex_IndexSize(index), n + 1);

    // Delete random vector.
    VecSimIndex_DeleteVector(index, 1);

    // The capacity should remain the same.
    ASSERT_EQ(hnswIndex->indexCapacity(), bs);

    VecSimIndex_Free(index);
}

// Test empty index edge cases.
TYPED_TEST(HNSWTest, emptyIndex) {
    size_t dim = 4;
    size_t n = 20;
    size_t bs = 6;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove from an empty index - should fail because label doesn't exist.
    VecSimIndex_DeleteVector(index, 0);

    // Add one vector.
    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1, 1.7);

    // Try to remove it.
    VecSimIndex_DeleteVector(index, 1);
    // The capacity should change to be aligned with the block size.

    HNSWIndex<TEST_DATA_T, TEST_DIST_T> *hnswIndex = this->CastToHNSW(index);
    size_t new_capacity = hnswIndex->indexCapacity();
    ASSERT_EQ(new_capacity, n - n % bs - bs);

    // Size equals 0.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove it again.
    // The capacity should remain unchanged, as we are trying to delete a label that doesn't exist.
    VecSimIndex_DeleteVector(index, 1);
    ASSERT_EQ(hnswIndex->indexCapacity(), new_capacity);
    // Nor the size.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, hnsw_vector_search_test) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 200,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[] = {50, 50, 50, 50};
    auto verify_res = [&](size_t id, double score, size_t index) {
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    runTopKSearchTest(index, query, k, verify_res);
    runTopKSearchTest(index, query, 0, verify_res); // For sanity, search for nothing
    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, hnsw_vector_search_by_id_test) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 11;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 200,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[] = {50, 50, 50, 50};
    auto verify_res = [&](size_t id, double score, size_t index) { ASSERT_EQ(id, (index + 45)); };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, hnsw_indexing_same_vector) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 10;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 200,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    TEST_DATA_T query[] = {4.9, 4.95, 5.05, 5.1};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, hnsw_reindexing_same_vector) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 10;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 200,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    TEST_DATA_T query[] = {4.9, 4.95, 5.05, 5.1};
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
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run the same query again
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, hnsw_reindexing_same_vector_different_id) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 10;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 200,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    TEST_DATA_T query[] = {4.9, 4.95, 5.05, 5.1};
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
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i + 10, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run the same query again
    auto verify_res_different_id = [&](int id, double score, size_t index) {
        ASSERT_TRUE(id >= 60 && id < 70 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res_different_id);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, sanity_reinsert_1280) {
    size_t n = 5;
    size_t d = 1280;
    size_t k = 5;

    HNSWParams params = {
        .dim = d, .metric = VecSimMetric_L2, .initialCapacity = n, .M = 16, .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    auto *vectors = new TEST_DATA_T[n * d];

    // Generate random vectors in every iteration and inert them under different ids.
    for (size_t iter = 1; iter <= 3; iter++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                (vectors + i * d)[j] = (TEST_DATA_T)rand() / (TEST_DATA_T)(RAND_MAX) / 100;
            }
        }
        auto expected_ids = std::set<size_t>();
        for (size_t i = 0; i < n; i++) {
            VecSimIndex_AddVector(index, (vectors + i * d), i * iter);
            expected_ids.insert(i * iter);
        }
        auto verify_res = [&](size_t id, double score, size_t index) {
            ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
            expected_ids.erase(id);
        };

        // Send arbitrary vector (the first) and search for top k. This should return all the
        // vectors that were inserted in this iteration - verify their ids.
        runTopKSearchTest(index, vectors, k, verify_res);

        // Remove vectors form current iteration.
        for (size_t i = 0; i < n; i++) {
            VecSimIndex_DeleteVector(index, i * iter);
        }
    }
    delete[] vectors;
    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, test_hnsw_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args
    HNSWParams params = {.dim = d, .metric = VecSimMetric_L2, .initialCapacity = n};

    VecSimIndex *index = this->CreateNewIndex(params);

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
    ASSERT_EQ(info.hnswInfo.type, params.type);
    VecSimIndex_Free(index);

    d = 1280;
    size_t bs = 42;
    params.dim = d;
    params.blockSize = bs, params.M = 200, params.efConstruction = 1000, params.efRuntime = 500,
    params.epsilon = 0.005;

    index = this->CreateNewIndex(params);
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
    ASSERT_EQ(info.hnswInfo.type, params.type);
    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, test_basic_hnsw_info_iterator) {
    size_t n = 100;
    size_t d = 128;

    VecSimMetric metrics[3] = {VecSimMetric_Cosine, VecSimMetric_IP, VecSimMetric_L2};
    for (size_t i = 0; i < 3; i++) {
        // Build with default args.
        // Build with default args
        HNSWParams params = {.dim = d, .metric = metrics[i], .initialCapacity = n};

        VecSimIndex *index = this->CreateNewIndex(params);

        VecSimIndexInfo info = VecSimIndex_Info(index);
        VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
        compareHNSWIndexInfoToIterator(info, infoIter);
        VecSimInfoIterator_Free(infoIter);
        VecSimIndex_Free(index);
    }
}

TYPED_TEST(HNSWTest, test_dynamic_hnsw_info_iterator) {
    size_t n = 100;
    size_t d = 128;

    HNSWParams params = {.dim = d,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 100,
                         .efConstruction = 250,
                         .efRuntime = 400,
                         .epsilon = 0.004};

    VecSimIndex *index = this->CreateNewIndex(params);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(100, info.hnswInfo.M);
    ASSERT_EQ(250, info.hnswInfo.efConstruction);
    ASSERT_EQ(400, info.hnswInfo.efRuntime);
    ASSERT_EQ(0.004, info.hnswInfo.epsilon);
    ASSERT_EQ(0, info.hnswInfo.indexSize);
    ASSERT_EQ(-1, info.hnswInfo.max_level);
    ASSERT_EQ(-1, info.hnswInfo.entrypoint);
    ASSERT_EQ(params.type, info.hnswInfo.type);
    compareHNSWIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    TEST_DATA_T v[d];
    for (size_t i = 0; i < d; i++) {
        v[i] = (TEST_DATA_T)i;
    }
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
    auto actual_element_count = this->CastToHNSW(index)->cur_element_count;
    this->CastToHNSW(index)->cur_element_count = 1e6;
    auto &label_lookup = this->CastToHNSW_Single(index)->label_lookup_;
    for (size_t i = 0; i < 1e6; i++) {
        label_lookup[i] = i;
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

    this->CastToHNSW(index)->cur_element_count = actual_element_count;
    VecSimIndex_Free(index);
}
TYPED_TEST(HNSWTest, test_query_runtime_params_default_build_args) {
    size_t n = 100;
    size_t d = 4;
    size_t k = 11;

    // Build with default args.

    HNSWParams params = {.dim = d, .metric = VecSimMetric_L2, .initialCapacity = n};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, d, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    auto verify_res = [&](size_t id, double score, size_t index) {
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    TEST_DATA_T query[] = {50, 50, 50, 50};
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    // Check that default args did not change.
    ASSERT_EQ(info.hnswInfo.M, HNSW_DEFAULT_M);
    ASSERT_EQ(info.hnswInfo.efConstruction, HNSW_DEFAULT_EF_C);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);

    // Run same query again, set efRuntime to 300.
    HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = 300};
    VecSimQueryParams queryParams = CreateQueryParams(hnswRuntimeParams);
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

TYPED_TEST(HNSWTest, test_query_runtime_params_user_build_args) {
    size_t n = 100;
    size_t d = 4;
    size_t M = 100;
    size_t efConstruction = 300;
    size_t efRuntime = 500;

    // Build with user args.

    HNSWParams params = {.dim = d,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = M,
                         .efConstruction = efConstruction,
                         .efRuntime = efRuntime};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, d, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    auto verify_res = [&](size_t id, double score, size_t index) {
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    TEST_DATA_T query[] = {50, 50, 50, 50};

    size_t k = 11;
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    // Check that user args did not change.
    ASSERT_EQ(info.hnswInfo.M, M);
    ASSERT_EQ(info.hnswInfo.efConstruction, efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    // Run same query again, set efRuntime to 300.
    HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = 300};
    VecSimQueryParams queryParams = CreateQueryParams(hnswRuntimeParams);
    runTopKSearchTest(index, query, k, verify_res, &queryParams);

    info = VecSimIndex_Info(index);
    // Check that user args did not change.
    ASSERT_EQ(info.hnswInfo.M, M);
    ASSERT_EQ(info.hnswInfo.efConstruction, efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    // Create batch iterator with query param.
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, &queryParams);
    info = VecSimIndex_Info(index);
    // Check that user args did not change.
    ASSERT_EQ(info.hnswInfo.M, M);
    ASSERT_EQ(info.hnswInfo.efConstruction, efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);
    // Run one batch for sanity.
    runBatchIteratorSearchTest(batchIterator, k, verify_res);
    // Check that user args did not change after releasing the batch iterator.
    VecSimBatchIterator_Free(batchIterator);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.hnswInfo.M, M);
    ASSERT_EQ(info.hnswInfo.efConstruction, efConstruction);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, hnsw_search_empty_index) {
    size_t n = 100;
    size_t k = 11;
    size_t d = 4;

    HNSWParams params = {.dim = d, .metric = VecSimMetric_L2, .initialCapacity = 0};

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T query[] = {50, 50, 50, 50};

    // We do not expect any results.
    VecSimQueryResult_List res = VecSimIndex_TopKQuery(index, query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Iterator *it = VecSimQueryResult_List_GetIterator(res);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);

    res = VecSimIndex_RangeQuery(index, query, 1.0, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);

    // Add some vectors and remove them all from index, so it will be empty again.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, d, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Again - we do not expect any results.
    res = VecSimIndex_TopKQuery(index, query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    it = VecSimQueryResult_List_GetIterator(res);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);

    res = VecSimIndex_RangeQuery(index, query, 1.0, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Free(res);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, hnsw_test_inf_score) {
    size_t n = 4;
    size_t k = 4;
    size_t dim = 2;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n};

    VecSimIndex *index = this->CreateNewIndex(params);

    TEST_DATA_T inf_val = GetInfVal(params.type);
    ASSERT_FALSE(std::isinf(inf_val));

    TEST_DATA_T query[] = {M_PI, M_PI};
    TEST_DATA_T v1[] = {M_PI, M_PI};
    TEST_DATA_T v2[] = {inf_val, inf_val};
    TEST_DATA_T v3[] = {M_E, M_E};
    TEST_DATA_T v4[] = {-inf_val, -inf_val};

    VecSimIndex_AddVector(index, v1, 1);
    VecSimIndex_AddVector(index, v2, 2);
    VecSimIndex_AddVector(index, v3, 3);
    VecSimIndex_AddVector(index, v4, 4);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 4);

    auto verify_res = [&](size_t id, double score, size_t index) {
        if (index == 0) {
            ASSERT_EQ(1, id);
        } else if (index == 1) {
            ASSERT_EQ(3, id);
        } else {
            ASSERT_TRUE(id == 2 || id == 4);
            ASSERT_TRUE(std::isinf(score));
        }
    };
    runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

// Tests VecSimIndex_New failure on bad M parameter. Should return null.
TYPED_TEST(HNSWTest, hnsw_bad_params) {
    size_t n = 10000000;
    size_t dim = 10000000;
    size_t bad_M[] = {
        1,          // Will fail because 1/log(M).
        100000000,  // Will fail on M * 2 overflow.
        UINT16_MAX, // Will fail on M * 2 overflow.
        UINT16_MAX /
            2, // Will fail on this->allocator->callocate(max_elements_ * size_data_per_element_)
    };
    size_t len = sizeof(bad_M) / sizeof(size_t);

    for (size_t i = 0; i < len; i++) {

        HNSWParams params = {.dim = dim,
                             .metric = VecSimMetric_L2,
                             .initialCapacity = n,
                             .M = bad_M[i],
                             .efConstruction = 250,
                             .efRuntime = 400,
                             .epsilon = 0.004};

        VecSimIndex *index = this->CreateNewIndex(params);

        ASSERT_TRUE(index == NULL) << "Failed on M=" << bad_M[i];
    }
}

TYPED_TEST(HNSWTest, hnsw_delete_entry_point) {
    size_t n = 10000;
    size_t dim = 4;
    size_t M = 2;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = M,
                         .efConstruction = 0,
                         .efRuntime = 0};

    VecSimIndex *index = this->CreateNewIndex(params);

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

TYPED_TEST(HNSWTest, hnsw_override) {
    size_t n = 100;
    size_t dim = 4;
    size_t M = 8;
    size_t ef = 300;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = M,
                         .efConstruction = 20,
                         .efRuntime = ef};

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_TRUE(index != nullptr);

    // Insert n == 100 vectors.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Try to override when overriding is not allowed.
    TEST_DATA_T vec[dim];
    GenerateVector<TEST_DATA_T>(vec, dim, n);
    ASSERT_EQ(this->CastToHNSW_Single(index)->addVector(vec, 0, false), -1);

    // Insert again 300 vectors, the first 100 will be overwritten (deleted first).
    n = 300;
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, n);
    // This is testing a bug fix - before we had the seconder sorting by id in CompareByFirst,
    // the graph got disconnected due to the deletion of some node followed by a bad repairing of
    // one of its neighbours. Here, we ensure that we get all the nodes in the graph as results.
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(id == n - 1 - index);
    };
    runTopKSearchTest(index, query, 300, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, hnsw_batch_iterator_basic) {
    size_t dim = 4;
    size_t M = 8;
    size_t ef = 20;
    size_t n = 1000;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = M,
                         .efConstruction = ef,
                         .efRuntime = ef};

    VecSimIndex *index = this->CreateNewIndex(params);

    // For every i, add the vector (i,i,i,i) under the label i.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, n);

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

TYPED_TEST(HNSWTest, hnsw_batch_iterator_reset) {
    size_t dim = 4;
    size_t n = 1000;
    size_t M = 8;
    size_t ef = 20;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = M,
                         .efConstruction = ef,
                         .efRuntime = ef};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, n);

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

TYPED_TEST(HNSWTest, hnsw_batch_iterator_batch_size_1) {
    size_t dim = 4;
    size_t n = 1000;
    size_t M = 8;
    size_t ef = 2;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = M,
                         .efConstruction = ef,
                         .efRuntime = ef};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        // Set labels to be different than the internal ids.
        GenerateAndAddVector<TEST_DATA_T>(index, dim, n - i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, n);

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

TYPED_TEST(HNSWTest, hnsw_batch_iterator_advanced) {
    size_t dim = 4;
    size_t n = 1000;
    size_t M = 8;
    size_t ef = 1000;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = M,
                         .efConstruction = ef,
                         .efRuntime = ef};

    VecSimIndex *index = this->CreateNewIndex(params);

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, n);
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
    VecSimBatchIterator_Free(batchIterator);

    // Insert vectors to the index and re-create the batch iterator.
    for (size_t i = 1; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    batchIterator = VecSimBatchIterator_New(index, query, nullptr);

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

TYPED_TEST(HNSWTest, hnsw_resolve_ef_runtime_params) {
    size_t dim = 4;
    size_t M = 8;
    size_t ef = 2;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 0,
                         .M = M,
                         .efConstruction = ef,
                         .efRuntime = ef};

    VecSimIndex *index = this->CreateNewIndex(params);

    VecSimQueryParams qparams, zero;
    bzero(&zero, sizeof(VecSimQueryParams));

    auto *rparams = array_new<VecSimRawParam>(3);

    // Test with empty runtime params.
    for (VecsimQueryType query_type : test_utils::query_types) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, query_type),
            VecSim_OK);
    }
    ASSERT_EQ(memcmp(&qparams, &zero, sizeof(VecSimQueryParams)), 0);

    array_append(rparams, (VecSimRawParam){
                              .name = "ef_runtime", .nameLen = 10, .value = "100", .valLen = 3});
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_KNN),
        VecSim_OK);
    ASSERT_EQ(qparams.hnswRuntimeParams.efRuntime, 100);

    rparams[0] = (VecSimRawParam){.name = "wrong_name", .nameLen = 10, .value = "100", .valLen = 3};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_NONE),
        VecSimParamResolverErr_UnknownParam);

    // Testing for legal prefix but only partial parameter name.
    rparams[0] = (VecSimRawParam){.name = "ef_run", .nameLen = 6, .value = "100", .valLen = 3};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_NONE),
        VecSimParamResolverErr_UnknownParam);

    rparams[0] =
        (VecSimRawParam){.name = "ef_runtime", .nameLen = 10, .value = "wrong_val", .valLen = 9};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_BadValue);

    rparams[0] = (VecSimRawParam){.name = "ef_runtime", .nameLen = 10, .value = "100", .valLen = 3};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_RANGE),
        VecSimParamResolverErr_UnknownParam);

    rparams[0] = (VecSimRawParam){.name = "ef_runtime", .nameLen = 10, .value = "-30", .valLen = 3};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_BadValue);

    rparams[0] =
        (VecSimRawParam){.name = "ef_runtime", .nameLen = 10, .value = "1.618", .valLen = 5};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_BadValue);

    rparams[0] = (VecSimRawParam){.name = "ef_runtime", .nameLen = 10, .value = "100", .valLen = 3};
    array_append(rparams, (VecSimRawParam){
                              .name = "ef_runtime", .nameLen = 10, .value = "100", .valLen = 3});
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_AlreadySet);

    /** Testing with hybrid query params - cases which are only relevant for HNSW index. **/
    // Cannot set ef_runtime param with "hybrid_policy" which is "ADHOC_BF"
    rparams[1] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "ADHOC_BF",
                                  .valLen = strlen("ADHOC_BF")};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSimParamResolverErr_InvalidPolicy_AdHoc_With_EfRuntime);

    rparams[1] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "BATCHES",
                                  .valLen = strlen("BATCHES")};
    array_append(rparams, (VecSimRawParam){.name = "batch_size",
                                           .nameLen = strlen("batch_size"),
                                           .value = "50",
                                           .valLen = strlen("50")});
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_HYBRID),
        VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 50);
    ASSERT_EQ(qparams.hnswRuntimeParams.efRuntime, 100);

    VecSimIndex_Free(index);
    array_free(rparams);
}

TYPED_TEST(HNSWTest, hnsw_resolve_epsilon_runtime_params) {
    size_t dim = 4;
    size_t M = 8;
    size_t ef = 2;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 0,
                         .M = M,
                         .efConstruction = ef,
                         .efRuntime = ef};

    VecSimIndex *index = this->CreateNewIndex(params);

    VecSimQueryParams qparams, zero;
    bzero(&zero, sizeof(VecSimQueryParams));

    auto *rparams = array_new<VecSimRawParam>(2);

    array_append(rparams, (VecSimRawParam){.name = "epsilon",
                                           .nameLen = strlen("epsilon"),
                                           .value = "0.001",
                                           .valLen = strlen("0.001")});

    for (VecsimQueryType query_type : {QUERY_TYPE_NONE, QUERY_TYPE_KNN, QUERY_TYPE_HYBRID}) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, query_type),
            VecSimParamResolverErr_InvalidPolicy_NRange);
    }

    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_RANGE),
        VecSim_OK);
    ASSERT_EQ(qparams.hnswRuntimeParams.epsilon, 0.001);

    rparams[0] = (VecSimRawParam){.name = "wrong_name",
                                  .nameLen = strlen("wrong_name"),
                                  .value = "0.001",
                                  .valLen = strlen("0.001")};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_RANGE),
        VecSimParamResolverErr_UnknownParam);

    // Testing for legal prefix but only partial parameter name.
    rparams[0] = (VecSimRawParam){
        .name = "epsi", .nameLen = strlen("epsi"), .value = "0.001", .valLen = strlen("0.001")};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_NONE),
        VecSimParamResolverErr_UnknownParam);

    rparams[0] = (VecSimRawParam){
        .name = "epsilon", .nameLen = strlen("epsilon"), .value = "wrong_val", .valLen = 9};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_RANGE),
        VecSimParamResolverErr_BadValue);

    rparams[0] = (VecSimRawParam){
        .name = "epsilon", .nameLen = strlen("epsilon"), .value = "-30", .valLen = 3};
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_RANGE),
        VecSimParamResolverErr_BadValue);

    rparams[0] = (VecSimRawParam){.name = "epsilon",
                                  .nameLen = strlen("epsilon"),
                                  .value = "0.001",
                                  .valLen = strlen("0.001")};
    array_append(rparams, (VecSimRawParam){.name = "epsilon",
                                           .nameLen = strlen("epsilon"),
                                           .value = "0.001",
                                           .valLen = strlen("0.001")});
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, QUERY_TYPE_RANGE),
        VecSimParamResolverErr_AlreadySet);

    VecSimIndex_Free(index);
    array_free(rparams);
}

TYPED_TEST(HNSWTest, hnsw_get_distance) {
    size_t n = 4;
    size_t dim = 2;
    size_t numIndex = 3;
    VecSimIndex *index[numIndex];
    std::vector<double> distances;

    TEST_DATA_T v1[] = {M_PI, M_PI};
    TEST_DATA_T v2[] = {M_E, M_E};
    TEST_DATA_T v3[] = {M_PI, M_E};
    TEST_DATA_T v4[] = {M_SQRT2, -M_SQRT2};

    HNSWParams params = {.dim = dim, .initialCapacity = n};

    for (size_t i = 0; i < numIndex; i++) {
        params.metric = (VecSimMetric)i;
        index[i] = this->CreateNewIndex(params);
        VecSimIndex_AddVector(index[i], v1, 1);
        VecSimIndex_AddVector(index[i], v2, 2);
        VecSimIndex_AddVector(index[i], v3, 3);
        VecSimIndex_AddVector(index[i], v4, 4);
        ASSERT_EQ(VecSimIndex_IndexSize(index[i]), 4);
    }

    TEST_DATA_T *query = v1;
    TEST_DATA_T *norm = v2;                   // {e, e}
    VecSim_Normalize(norm, dim, params.type); // now {1/sqrt(2), 1/sqrt(2)}
    ASSERT_FLOAT_EQ(norm[0], 1.0 / sqrt(2.0));
    ASSERT_FLOAT_EQ(norm[1], 1.0 / sqrt(2.0));
    double dist;

    // distances array values were calculated locally for fp32 vectors
    // using VecSim library.

    // VecSimMetric_L2
    distances = {0, 0.3583844006061554, 0.1791922003030777, 23.739208221435547};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_L2], i + 1, query);
        ASSERT_NEAR(dist, distances[i], 1e-5);
    }

    // VecSimMetric_IP
    distances = {-18.73921012878418, -16.0794677734375, -17.409339904785156, 1};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_IP], i + 1, query);
        ASSERT_NEAR(dist, distances[i], 1e-5);
    }

    // VecSimMetric_Cosine
    distances = {5.9604644775390625e-08, 5.9604644775390625e-08, 0.0025991201400756836, 1};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_Cosine], i + 1, norm);
        ASSERT_NEAR(dist, distances[i], 1e-5);
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

TYPED_TEST(HNSWTest, preferAdHocOptimization) {
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
        HNSWParams params = {.dim = dim,
                             .metric = VecSimMetric_L2,
                             .initialCapacity = index_size,
                             .M = M,
                             .efConstruction = 1,
                             .efRuntime = 1};

        VecSimIndex *index = this->CreateNewIndex(params);

        // Set the index size artificially to be the required one.
        this->CastToHNSW(index)->cur_element_count = index_size;
        for (size_t i = 0; i < index_size; i++) {
            this->CastToHNSW_Single(index)->label_lookup_[i] = i;
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), index_size);
        bool res = VecSimIndex_PreferAdHocSearch(index, (size_t)(r * (float)index_size), k, true);
        ASSERT_EQ(res, comb.second);
        VecSimIndex_Free(index);
    }

    // Corner cases - empty index.

    HNSWParams params = {.dim = 4, .metric = VecSimMetric_L2};

    VecSimIndex *index = this->CreateNewIndex(params);

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

TYPED_TEST(HNSWTest, testCosine) {
    size_t dim = 4;
    size_t n = 100;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_Cosine, .initialCapacity = n};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 1; i <= n; i++) {
        TEST_DATA_T f[dim];
        f[0] = (TEST_DATA_T)i / n;
        for (size_t j = 1; j < dim; j++) {
            f[j] = 1.0;
        }
        VecSimIndex_AddVector(index, f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, 1.0);
    VecSim_Normalize(query, dim, params.type);

    auto verify_res = [&](size_t id, double score, size_t result_rank) {
        ASSERT_EQ(id, (n - result_rank));
        TEST_DATA_T expected_score = index->getDistanceFrom(id, query);
        ASSERT_DOUBLE_EQ(score, expected_score);
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
        auto verify_res_batch = [&](size_t id, double score, size_t result_rank) {
            ASSERT_EQ(id, (n - n_res * iteration_num - result_rank));
            double expected_score = index->getDistanceFrom(id, query);
            ASSERT_DOUBLE_EQ(score, expected_score);
        };
        runBatchIteratorSearchTest(batchIterator, n_res, verify_res_batch);
        iteration_num++;
    }
    ASSERT_EQ(iteration_num, n / n_res);
    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, testSizeEstimation) {
    size_t dim = 128;
    size_t n = 1000;
    size_t bs = DEFAULT_BLOCK_SIZE;
    size_t M = 32;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n, .blockSize = bs, .M = M};

    VecSimIndex *index = this->CreateNewIndex(params);
    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();
    // labels_lookup hash table has additional memory, since STL implementation chooses "an
    // appropriate prime number" higher than n as the number of allocated buckets (for n=1000, 1031
    // buckets are created)
    estimation +=
        (this->CastToHNSW_Single(index)->label_lookup_.bucket_count() - n) * sizeof(size_t);

    ASSERT_EQ(estimation, actual);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }

    // Estimate the memory delta of adding a full new block.
    estimation = EstimateElementSize(params) * (bs % n + bs);

    actual = 0;
    for (size_t i = 0; i < bs; i++) {
        actual += GenerateAndAddVector<TEST_DATA_T>(index, dim, n + i, i);
    }
    ASSERT_GE(estimation * 1.01, actual);
    ASSERT_LE(estimation * 0.99, actual);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, testInitialSizeEstimation_No_InitialCapacity) {
    size_t dim = 128;
    size_t n = 0;
    size_t bs = DEFAULT_BLOCK_SIZE;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_Cosine, .initialCapacity = n, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);
    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();

    // labels_lookup and element_levels containers are not allocated at all in some platforms,
    // when initial capacity is zero, while in other platforms labels_lookup is allocated with a
    // single bucket. This, we get the following range in which we expect the initial memory to be
    // in.
    ASSERT_GE(actual, estimation);
    ASSERT_LE(actual, estimation + sizeof(size_t) + 2 * sizeof(size_t));

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, testTimeoutReturn) {
    size_t dim = 4;
    VecSimQueryResult_List rl;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .initialCapacity = 1, .blockSize = 5};

    VecSimIndex *index = this->CreateNewIndex(params);

    GenerateAndAddVector<TEST_DATA_T>(index, dim, 0, 1.0);

    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, 1.0);
    // Checks return code on timeout.
    rl = VecSimIndex_TopKQuery(index, query, 1, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    // Check timeout again - range query.
    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1, 1.0);
    ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.max_level, 0);
    // Here, the entry point is inserted to the results set before we test for timeout.
    // hence, expect a single result to be returned (instead of 2 that would have return without
    // timeout).
    rl = VecSimIndex_RangeQuery(index, query, 1, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 1);
    VecSimQueryResult_Free(rl);

    // Fail on searching bottom layer entry point.
    // We need to have at least 1 vector in layer higher than 0 to fail there.
    size_t next = 0;
    while (VecSimIndex_Info(index).hnswInfo.max_level == 0) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, next, 1.0);
        ++next;
    }
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out.

    rl = VecSimIndex_TopKQuery(index, query, 2, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    // Timeout on searching bottom layer entry point - range query.
    rl = VecSimIndex_RangeQuery(index, query, 1, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSimIndex_Free(index);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // Cleanup.
}

TYPED_TEST(HNSWTest, testTimeoutReturn_batch_iterator) {
    size_t dim = 4;
    size_t n = 2;
    VecSimQueryResult_List rl;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, 46 - i, 1.0);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Fail on second batch (after some calculation already completed in the first one).
    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, 1.0);
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

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
    batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    rl = VecSimBatchIterator_Next(batchIterator, 2, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSimBatchIterator_Free(batchIterator);

    // Fail on searching bottom layer entry point.
    // We need to have at least 1 vector in layer higher than 0 to fail there.
    size_t next = 0;
    while (VecSimIndex_Info(index).hnswInfo.max_level == 0) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, next++, 1.0);
    }
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out.
    batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    rl = VecSimBatchIterator_Next(batchIterator, 2, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSimBatchIterator_Free(batchIterator);

    VecSimIndex_Free(index);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // Cleanup.
}

TYPED_TEST(HNSWTest, rangeQuery) {
    size_t n = 5000;
    size_t dim = 4;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n / 2};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    size_t pivot_id = n / 2; // the id to return vectors around it.
    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, pivot_id);

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
    HNSWRuntimeParams hnswRuntimeParams = {.epsilon = 1.0};
    auto query_params = CreateQueryParams(hnswRuntimeParams);
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

TYPED_TEST(HNSWTest, rangeQueryCosine) {
    size_t n = 800;
    size_t dim = 4;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_Cosine, .initialCapacity = n / 2};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T f[dim];
        f[0] = TEST_DATA_T(i + 1) / n;
        for (size_t j = 1; j < dim; j++) {
            f[j] = 1.0;
        }
        // Use as label := n - (internal id)
        VecSimIndex_AddVector(index, f, n - i);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    TEST_DATA_T query[dim];
    for (size_t i = 0; i < dim; i++) {
        query[i] = 1.0;
    }

    VecSim_Normalize(query, dim, params.type);
    auto verify_res = [&](size_t id, double score, size_t result_rank) {
        ASSERT_EQ(id, result_rank + 1);
        double expected_score = index->getDistanceFrom(id, query);
        ASSERT_EQ(score, expected_score);
    };
    uint expected_num_results = 31;
    // Calculate the score of the 31st distant vector from the query vector (whose id should be 30)
    // to get the radius.
    double radius = index->getDistanceFrom(31, query);
    runRangeQueryTest(index, query, radius, verify_res, expected_num_results, BY_SCORE);

    // Return results BY_ID should give the same results.
    runRangeQueryTest(index, query, radius, verify_res, expected_num_results, BY_ID);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, HNSWSerialization_v2) {

    size_t dim = 4;
    size_t n = 1000;
    size_t n_labels[] = {n, 100};
    size_t M = 8;
    size_t ef = 10;
    double epsilon = 0.004;
    size_t blockSize = 1;
    bool is_multi[] = {false, true};
    std::string multiToString[] = {"single", "multi_100labels_"};

    HNSWParams params{.type = TypeParam::get_index_type(),
                      .dim = dim,
                      .metric = VecSimMetric_L2,
                      .blockSize = blockSize,
                      .M = M,
                      .efConstruction = ef,
                      .efRuntime = ef,
                      .epsilon = epsilon};

    // Test for multi and single

    for (size_t i = 0; i < 2; ++i) {
        // Set index type.
        params.multi = is_multi[i];

        // Generate and add vectors to an index.
        VecSimIndex *index = this->CreateNewIndex(params, is_multi[i]);
        HNSWIndex<TEST_DATA_T, TEST_DIST_T> *hnsw_index = this->CastToHNSW(index);

        std::vector<TEST_DATA_T> data(n * dim);
        std::mt19937 rng;
        rng.seed(47);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < n * dim; ++i) {
            data[i] = (TEST_DATA_T)distrib(rng);
        }
        for (size_t j = 0; j < n; ++j) {
            VecSimIndex_AddVector(index, data.data() + dim * j, j % n_labels[i]);
        }

        auto file_name = std::string(getenv("ROOT")) + "/tests/unit/data/1k-d4-L2-M8-ef_c10_" +
                         VecSimType_ToString(TypeParam::get_index_type()) + "_" + multiToString[i] +
                         ".hnsw_v2";

        // Save the index with the default version (V2).
        hnsw_index->saveIndex(file_name);

        // Fetch info after saving, as memory size change during saving.
        VecSimIndexInfo info = VecSimIndex_Info(index);
        ASSERT_EQ(info.algo, VecSimAlgo_HNSWLIB);
        ASSERT_EQ(info.hnswInfo.M, M);
        ASSERT_EQ(info.hnswInfo.efConstruction, ef);
        ASSERT_EQ(info.hnswInfo.efRuntime, ef);
        ASSERT_EQ(info.hnswInfo.indexSize, n);
        ASSERT_EQ(info.hnswInfo.metric, VecSimMetric_L2);
        ASSERT_EQ(info.hnswInfo.type, TypeParam::get_index_type());
        ASSERT_EQ(info.hnswInfo.dim, dim);
        ASSERT_EQ(info.hnswInfo.indexLabelCount, n_labels[i]);

        VecSimIndex_Free(index);

        // Load the index from the file.
        VecSimIndex *serialized_index = HNSWFactory::NewIndex(file_name);
        auto *serialized_hnsw_index = this->CastToHNSW(serialized_index);

        // Verify that the index was loaded as expected.
        ASSERT_TRUE(serialized_hnsw_index->checkIntegrity().valid_state);

        VecSimIndexInfo info2 = VecSimIndex_Info(serialized_index);
        ASSERT_EQ(info2.algo, VecSimAlgo_HNSWLIB);
        ASSERT_EQ(info2.hnswInfo.M, M);
        ASSERT_EQ(info2.hnswInfo.isMulti, is_multi[i]);
        ASSERT_EQ(info2.hnswInfo.blockSize, blockSize);
        ASSERT_EQ(info2.hnswInfo.efConstruction, ef);
        ASSERT_EQ(info2.hnswInfo.efRuntime, ef);
        ASSERT_EQ(info2.hnswInfo.indexSize, n);
        ASSERT_EQ(info2.hnswInfo.metric, VecSimMetric_L2);
        ASSERT_EQ(info2.hnswInfo.type, TypeParam::get_index_type());
        ASSERT_EQ(info2.hnswInfo.dim, dim);
        ASSERT_EQ(info2.hnswInfo.indexLabelCount, n_labels[i]);
        ASSERT_EQ(info2.hnswInfo.epsilon, epsilon);

        // Check the functionality of the loaded index.

        // Add and delete vector
        GenerateAndAddVector<TEST_DATA_T>(serialized_index, dim, n);

        VecSimIndex_DeleteVector(serialized_index, 1);

        size_t n_per_label = n / n_labels[i];
        ASSERT_TRUE(serialized_hnsw_index->checkIntegrity().valid_state);
        ASSERT_EQ(VecSimIndex_IndexSize(serialized_index), n + 1 - n_per_label);

        // Clean up.
        remove(file_name.c_str());
        VecSimIndex_Free(serialized_index);
    }
}

TYPED_TEST(HNSWTest, LoadHNSWSerialized_v1) {

    size_t dim = 4;
    size_t n = 1000;
    size_t n_labels[] = {n, 100};
    size_t M_serialized = 8;
    size_t M_param = 5;
    size_t ef_serialized = 10;
    size_t ef_param = 12;
    bool is_multi[] = {false, true};
    std::string multiToString[] = {"single", "multi_100labels_"};

    HNSWParams params{.type = TypeParam::get_index_type(),
                      .dim = dim,
                      .metric = VecSimMetric_L2,
                      .M = M_param,
                      .efConstruction = ef_param,
                      .efRuntime = ef_param};
    for (size_t i = 0; i < 2; ++i) {
        // Set index type.
        params.multi = is_multi[i];

        auto file_name = std::string(getenv("ROOT")) + "/tests/unit/data/1k-d4-L2-M8-ef_c10_" +
                         VecSimType_ToString(TypeParam::get_index_type()) + "_" + multiToString[i] +
                         ".hnsw_v1";

        // Try to load with an invalid type
        params.type = VecSimType_INT32;
        ASSERT_EXCEPTION_MESSAGE(HNSWFactory::NewIndex(file_name, &params), std::runtime_error,
                                 "Cannot load index: bad index data type");
        // Restore value.
        params.type = TypeParam::get_index_type();

        // Create new index from file
        VecSimIndex *serialized_index = HNSWFactory::NewIndex(file_name, &params);
        auto *serialized_hnsw_index = this->CastToHNSW(serialized_index);

        // Verify that the index was loaded as expected.
        ASSERT_TRUE(serialized_hnsw_index->checkIntegrity().valid_state);

        VecSimIndexInfo info2 = VecSimIndex_Info(serialized_index);
        ASSERT_EQ(info2.algo, VecSimAlgo_HNSWLIB);
        // Check that M is taken from file and not from @params.
        ASSERT_EQ(info2.hnswInfo.M, M_serialized);
        ASSERT_NE(info2.hnswInfo.M, M_param);

        ASSERT_EQ(info2.hnswInfo.isMulti, is_multi[i]);

        // Check it was initalized with the default blockSize value.
        ASSERT_EQ(info2.hnswInfo.blockSize, DEFAULT_BLOCK_SIZE);

        // Check that ef is taken from file and not from @params.
        ASSERT_EQ(info2.hnswInfo.efConstruction, ef_serialized);
        ASSERT_EQ(info2.hnswInfo.efRuntime, ef_serialized);
        ASSERT_NE(info2.hnswInfo.efRuntime, ef_param);
        ASSERT_NE(info2.hnswInfo.efConstruction, ef_param);

        ASSERT_EQ(info2.hnswInfo.indexSize, n);
        ASSERT_EQ(info2.hnswInfo.metric, VecSimMetric_L2);
        ASSERT_EQ(info2.hnswInfo.type, TypeParam::get_index_type());
        ASSERT_EQ(info2.hnswInfo.dim, dim);
        ASSERT_EQ(info2.hnswInfo.indexLabelCount, n_labels[i]);
        // Check it was initalized with the default epsilon value.
        ASSERT_EQ(info2.hnswInfo.epsilon, HNSW_DEFAULT_EPSILON);

        // Check the functionality of the loaded index.

        // Add and delete vector
        GenerateAndAddVector<TEST_DATA_T>(serialized_index, dim, n);

        VecSimIndex_DeleteVector(serialized_index, 1);

        size_t n_per_label = n / n_labels[i];
        ASSERT_TRUE(serialized_hnsw_index->checkIntegrity().valid_state);
        ASSERT_EQ(VecSimIndex_IndexSize(serialized_index), n + 1 - n_per_label);
        VecSimIndex_Free(serialized_index);
    }
}

TYPED_TEST(HNSWTest, markDelete) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;
    VecSimBatchIterator *batchIterator;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n};

    VecSimIndex *index = this->CreateNewIndex(params);
    // Try marking and a non-existing label
    ASSERT_EQ(this->CastToHNSW(index)->markDelete(0), std::vector<idType>());

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, n / 2);

    // Search for k results around the middle. expect to find them.
    auto verify_res = [&](size_t id, double score, size_t index) {
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    runTopKSearchTest(index, query, k, verify_res);
    runRangeQueryTest(index, query, dim * k * k / 4 - 1, verify_res, k, BY_SCORE);
    batchIterator = VecSimBatchIterator_New(index, query, nullptr);
    runBatchIteratorSearchTest(batchIterator, k, verify_res);
    VecSimBatchIterator_Free(batchIterator);

    unsigned char ep_reminder = index->info().hnswInfo.entrypoint % 2;
    // Mark as deleted half of the vectors, including the entrypoint.
    for (labelType label = 0; label < n; label++)
        if (label % 2 == ep_reminder)
            ASSERT_EQ(this->CastToHNSW(index)->markDelete(label), std::vector<idType>(1, label));

    ASSERT_EQ(this->CastToHNSW(index)->getNumMarkedDeleted(), n / 2);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Search for k results around the middle. expect to find only even results.
    auto verify_res_half = [&](size_t id, double score, size_t index) {
        ASSERT_NE(id % 2, ep_reminder);
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        size_t expected_id = index % 2 ? index + 1 : index;
        ASSERT_EQ(diff_id, expected_id);
        ASSERT_EQ(score, (dim * expected_id * expected_id));
    };
    runTopKSearchTest(index, query, k, verify_res_half);
    runRangeQueryTest(index, query, dim * k * k - 1, verify_res_half, k, BY_SCORE);
    batchIterator = VecSimBatchIterator_New(index, query, nullptr);
    runBatchIteratorSearchTest(batchIterator, k, verify_res_half);
    VecSimBatchIterator_Free(batchIterator);

    // Add a new vector, make sure it has no link to a deleted vector
    GenerateAndAddVector<TEST_DATA_T>(index, dim, n, n);
    for (size_t level = 0; level <= this->CastToHNSW(index)->element_levels_[n]; level++) {
        idType *neighbors = this->CastToHNSW(index)->get_linklist_at_level(n, level);
        linkListSize size = this->CastToHNSW(index)->getListCount(neighbors);
        for (size_t idx = 0; idx < size; idx++) {
            ASSERT_TRUE(neighbors[idx] % 2 != ep_reminder)
                << "Got a link to " << neighbors[idx] << " on level " << level;
        }
    }

    // Re-add the previously marked vectors (under new internal ids).
    for (labelType label = 0; label < n; label++) {
        if (label % 2 == ep_reminder) {
            GenerateAndAddVector<TEST_DATA_T>(index, dim, label, label);
        }
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), n + n / 2 + 1);
    ASSERT_EQ(this->CastToHNSW(index)->getNumMarkedDeleted(), n / 2);

    // Search for k results around the middle again. expect to find the same results we
    // found in the first search.
    runTopKSearchTest(index, query, k, verify_res);
    runRangeQueryTest(index, query, dim * k * k / 4 - 1, verify_res, k, BY_SCORE);
    batchIterator = VecSimBatchIterator_New(index, query, nullptr);
    runBatchIteratorSearchTest(batchIterator, k, verify_res);
    VecSimBatchIterator_Free(batchIterator);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, allMarkedDeletedLevel) {
    size_t dim = 4;
    size_t M = 2;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .M = M};

    VecSimIndex *index = this->CreateNewIndex(params);

    size_t num_multi_layered = 0;
    labelType max_id = 0;

    // Add vectors to the index until we have 10 multi-layered vectors.
    do {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, max_id, max_id);
        if (this->CastToHNSW(index)->element_levels_[max_id] > 0) {
            num_multi_layered++;
        }
        max_id++;
    } while (num_multi_layered < 10);

    // Mark all vectors with multi-layers as deleted.
    for (labelType label = 0; label < max_id; label++) {
        if (this->CastToHNSW(index)->element_levels_[label] > 0) {
            this->CastToHNSW(index)->markDelete(label);
        }
    }

    size_t max_level = index->info().hnswInfo.max_level;

    // Re-add a new vector until its level is equal to the max level of the index.
    do {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, max_id, max_id);
    } while (this->CastToHNSW(index)->element_levels_[max_id] < max_level);

    // If we passed the previous loop, it means that we successfully added a vector without invalid
    // memory access.

    // For completeness, we also check index integrity.
    ASSERT_TRUE(this->CastToHNSW(index)->checkIntegrity().valid_state);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, parallelSearchKnn) {
    size_t n = 1000;
    size_t k = 11;
    size_t dim = 4;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 16,
                         .efConstruction = 200};
    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    std::atomic_int successful_searches(0);
    // Run parallel searches where every searching thread expects to get different labels as results
    // (determined by the thread id), which are labels in the range [50+myID-5, 50+myID+5].
    auto parallel_search = [&](int myID) {
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            // We expect to get the results with increasing order of the distance between the res
            // label and the query val (query_val, query_val-1, query_val+1, query_val-2,
            // query_val+2, ...) The score is the L2 distance between the vectors that correspond
            // the ids.
            size_t diff_id = (id > query_val) ? (id - query_val) : (query_val - id);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, (dim * (diff_id * diff_id)));
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    size_t memory_before = index->info().hnswInfo.memory;
    size_t n_threads = 16;
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_search, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(successful_searches, n_threads);
    // Make sure that we properly update the allocator atomically during the searches. The expected
    // Memory delta should only be the visited nodes handler added to the pool. Note that the
    // initial pool size is 1, so we subtract 1 from the current pool size to get the delta.
    size_t expected_memory = memory_before + (index->info().hnswInfo.visitedNodesPoolSize - 1) *
                                                 (sizeof(VisitedNodesHandler) + sizeof(tag_t) * n +
                                                  2 * sizeof(size_t) + sizeof(void *));
    ASSERT_EQ(expected_memory, index->info().hnswInfo.memory);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTest, parallelSearchCombined) {
    size_t n = 1000;
    size_t k = 11;
    size_t dim = 4;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 16,
                         .efConstruction = 200};
    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    std::atomic_int successful_searches(0);

    /* Run parallel searches of three kinds: KNN, range, and batched search. */

    // In knn, we expect to get different labels as results (determined by the thread id), which are
    // labels in the range [50+myID-5, 50+myID+5].
    auto parallel_knn_search = [&](int myID) {
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            // We expect to get the results with increasing order of the distance between the res
            // label and the query val (query_val, query_val-1, query_val+1, query_val-2,
            // query_val+2, ...) The score is the L2 distance between the vectors that correspond
            // the ids.
            size_t diff_id = std::abs(id - query_val);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, (dim * (diff_id * diff_id)));
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    auto parallel_range_search = [&](int myID) {
        TEST_DATA_T pivot_id = 100 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, pivot_id);
        auto verify_res_by_score = [&](size_t id, double score, size_t res_index) {
            size_t diff_id = std::abs(id - pivot_id);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, dim * (diff_id * diff_id));
        };
        uint expected_num_results = 11;
        // To get 11 results in the range [pivot_id-5, pivot_id+5], set the radius as the L2 score
        // in the boundaries.
        double radius = (double)dim * pow((double)expected_num_results / 2, 2);
        runRangeQueryTest(index, query, radius, verify_res_by_score, expected_num_results,
                          BY_SCORE);
        successful_searches++;
    };

    auto parallel_batched_search = [&](int myID) {
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
        size_t iteration_num = 0;

        // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet
        // in every iteration. The results order should be sorted by their score (distance from the
        // query vector), which means sorted from the largest id to the lowest.
        // Run different number of iterations for every thread id.
        size_t total_iterations = myID;
        size_t n_res = 5;
        while (VecSimBatchIterator_HasNext(batchIterator) && iteration_num < total_iterations) {
            std::vector<size_t> expected_ids(n_res);
            for (size_t i = 0; i < n_res; i++) {
                expected_ids[i] = (n - iteration_num * n_res - i - 1);
            }
            auto verify_res = [&](size_t id, double score, size_t res_index) {
                ASSERT_TRUE(expected_ids[res_index] == id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num, total_iterations);
        VecSimBatchIterator_Free(batchIterator);
        successful_searches++;
    };

    size_t n_threads = 15;
    std::thread thread_objs[n_threads];
    size_t memory_before = index->info().hnswInfo.memory;
    for (size_t i = 0; i < n_threads; i++) {
        if (i % 3 == 0) {
            thread_objs[i] = std::thread(parallel_knn_search, i);
        } else if (i % 3 == 1) {
            thread_objs[i] = std::thread(parallel_range_search, i);
        } else {
            thread_objs[i] = std::thread(parallel_batched_search, i);
        }
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(successful_searches, n_threads);
    // Make sure that we properly update the allocator atomically during the searches.
    // Memory delta should only be the visited nodes handler added to the pool. Note that the
    // initial pool size is 1, so we subtract 1 from the current pool size to get the delta.
    size_t expected_memory = memory_before + (index->info().hnswInfo.visitedNodesPoolSize - 1) *
                                                 (sizeof(VisitedNodesHandler) + sizeof(tag_t) * n +
                                                  2 * sizeof(size_t) + sizeof(void *));
    ASSERT_EQ(expected_memory, index->info().hnswInfo.memory);
    VecSimIndex_Free(index);
}
