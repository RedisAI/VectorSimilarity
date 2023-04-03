#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/raft_ivf/ivf_flat.cuh"
#include "VecSim/algorithms/raft_ivf/ivf_pq.cuh"
#include "test_utils.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/query_result_struct.h"
#include <climits>
#include <unistd.h>
#include <random>
#include <thread>
#include <atomic>

template <typename index_type_t>
class RaftIvfTest : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    VecSimIndex *CreateNewIndex(RaftIVFFlatParams &params, bool is_multi = false) {
        return test_utils::CreateNewIndex(params, index_type_t::get_index_type(), is_multi);
    }
    VecSimIndex *CreateNewIndex(RaftIVFPQParams &params, bool is_multi = false) {
        return test_utils::CreateNewIndex(params, index_type_t::get_index_type(), is_multi);
    }
    RaftIVFFlatIndex *CastToFlat(VecSimIndex *index) {
        return dynamic_cast<RaftIVFFlatIndex *>(index);
    }

    RaftIVFPQIndex *CastToPQ(VecSimIndex *index) {
        return dynamic_cast<RaftIVFPQIndex *>(index);
    }
};

// TEST_DATA_T and TEST_DIST_T are defined in test_utils.h

RaftIVFPQParams createDefaultPQParams(size_t dim)
{
    RaftIVFPQParams params = {.dim = dim,
                              .metric = VecSimMetric_L2,
                              .nLists = 1,
                              .pqBits = 8,
                              .pqDim = 0,
                              .codebookKind = RaftIVFPQ_PerSubspace,
                              .kmeans_nIters = 20,
                              .kmeans_trainsetFraction = 0.5,
                              .nProbes = 20,
                              .lutType = CUDAType_R_32F,
                              .internalDistanceType = CUDAType_R_32F,
                              .preferredShmemCarveout = 1.0};
    return params;
}

RaftIVFFlatParams createDefaultFlatParams(size_t dim)
{
    RaftIVFFlatParams params = {.dim = dim,
                                .metric = VecSimMetric_L2,
                                .nLists = 1,
                                .kmeans_nIters = 20,
                                .kmeans_trainsetFraction = 0.5,
                                .nProbes = 20};
    return params;
}

using DataTypeSetFloat =
    ::testing::Types<IndexType<VecSimType_FLOAT32, float>>;

TYPED_TEST_SUITE(RaftIvfTest, DataTypeSetFloat);

TYPED_TEST(RaftIvfTest, RaftIVFFlat_vector_add_test) {
    size_t dim = 4;

    RaftIVFFlatParams params = createDefaultFlatParams(dim);

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    VecSimIndex_Free(index);
}

TYPED_TEST(RaftIvfTest, RaftIVFPQ_vector_add_test) {
    size_t dim = 4;

    RaftIVFPQParams params = createDefaultPQParams(dim);

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    VecSimIndex_Free(index);
}

TYPED_TEST(RaftIvfTest, RaftIVFFlat_blob_sanity_test) {
    size_t dim = 4;

    RaftIVFFlatParams params = createDefaultFlatParams(dim);

    VecSimIndex *index = this->CreateNewIndex(params);
    VecSimQueryParams queryParams = {.batchSize = 1};

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T a[dim], b[dim], c[dim], d[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (TEST_DATA_T)0;
        b[i] = (TEST_DATA_T)1;
        c[i] = (TEST_DATA_T)2;
        d[i] = (TEST_DATA_T)3;
    }

    VecSimQueryResult_List resultQuery;
    VecSimQueryResult_Iterator *it = nullptr;
    // Add first vector. Check it was inserted by searching it.
    VecSimIndex_AddVector(index, a, 0);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    resultQuery = index->topKQuery(a, 1, &queryParams);
    it = VecSimQueryResult_List_GetIterator(resultQuery);
    VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 0);
    ASSERT_EQ(currentResult->score, 0);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_Free(resultQuery);
    VecSimQueryResult_IteratorFree(it);

    // Add second and third vector. Check the topK distances
    VecSimIndex_AddVector(index, b, 1);
    VecSimIndex_AddVector(index, c, 2);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 3);
    resultQuery = index->topKQuery(c, 2, &queryParams);
    //VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(resultQuery.results[0].id, 2);
    ASSERT_EQ(resultQuery.results[0].score, 0);
    ASSERT_EQ(resultQuery.results[1].id, 1);
    ASSERT_EQ(resultQuery.results[1].score, dim);
    VecSimQueryResult_Free(resultQuery);
    VecSimIndex_Free(index);
}

/**** resizing cases ****/
/*
// Add up to capacity.
TYPED_TEST(RaftIvfTest, resizeNAlignIndex) {
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
TYPED_TEST(RaftIvfTest, resizeNAlignIndex_largeInitialCapacity) {
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
TYPED_TEST(RaftIvfTest, resizeNAlignIndex_largerBlockSize) {
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
TYPED_TEST(RaftIvfTest, emptyIndex) {
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

TYPED_TEST(RaftIvfTest, hnsw_vector_search_test) {
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

TYPED_TEST(RaftIvfTest, hnsw_vector_search_by_id_test) {
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

TYPED_TEST(RaftIvfTest, hnsw_indexing_same_vector) {
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

TYPED_TEST(RaftIvfTest, hnsw_reindexing_same_vector) {
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

TYPED_TEST(RaftIvfTest, hnsw_reindexing_same_vector_different_id) {
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

TYPED_TEST(RaftIvfTest, sanity_reinsert_1280) {
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

TYPED_TEST(RaftIvfTest, test_hnsw_info) {
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

TYPED_TEST(RaftIvfTest, HNSWSerialization_v2) {

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

TYPED_TEST(RaftIvfTest, LoadHNSWSerialized_v1) {

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
*/