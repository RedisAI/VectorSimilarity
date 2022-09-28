#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "test_utils.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include <cmath>

template <VecSimType type, typename DataType, typename DistType = DataType>
struct IndexType {
    static VecSimType get_index_type() { return type; }
    typedef DataType data_t;
    typedef DistType dist_t;
};

template <typename index_type_t>
class BruteForceTest : public ::testing::Test {
    using data_t = typename index_type_t::data_t;

protected:
    void GenerateVector(data_t *output, size_t dim, data_t value = 1.0) {
        for (size_t i = 0; i < dim; i++) {
            output[i] = (data_t)value;
        }
    }
    void GenerateNAddVector(VecSimIndex *index, size_t dim, size_t id, data_t value = 1.0) {
        data_t v[dim];
        this->GenerateVector(v, dim, value); // i / 10 is in integer (take the "floor" value).
        VecSimIndex_AddVector(index, v, id);
    }
};

#define TEST_DATA_T typename TypeParam::data_t
#define TEST_DIST_T typename TypeParam::dist_t
using DataTypeSet =
    ::testing::Types<IndexType<VecSimType_FLOAT32, float>, IndexType<VecSimType_FLOAT64, double>>;

TYPED_TEST_CASE(BruteForceTest, DataTypeSet);

TYPED_TEST(BruteForceTest, brute_force_vector_add_test) {

    size_t dim = 4;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_IP,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    this->GenerateNAddVector(index, dim, 1);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceTest, brute_force_vector_update_test) {
    size_t dim = 4;
    size_t n = 1;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_IP,
                                             .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *bf_index =
        reinterpret_cast<BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *>(index);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    this->GenerateNAddVector(index, dim, 1);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

    // Prepare new vector data and call addVEctor with the same id, different data.
    this->GenerateNAddVector(index, dim, 1, 2.0);

    // Index size shouldn't change.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

    // id2label size should remain the same, although we seemingly tried to exceed
    // initial capacity.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), n);

    // Check update.
    TEST_DATA_T *vector_data = bf_index->getDataByInternalId(0);
    for (size_t i = 0; i < dim; ++i) {
        ASSERT_EQ(*vector_data, 2.0);
        ++vector_data;
    }

    // Delete the last vector.
    VecSimIndex_DeleteVector(index, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // VectorBlocks vector is empty.
    ASSERT_EQ(bf_index->vectorBlocks.size(), 0);
    // id2label size shouldn't change.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), n);

    BruteForceIndex_Single<TEST_DATA_T, TEST_DIST_T> *bf_single_index =
        reinterpret_cast<BruteForceIndex_Single<TEST_DATA_T, TEST_DIST_T> *>(index);

    // Label2id of the last vector doesn't exist.
    ASSERT_EQ(bf_single_index->labelToIdLookup.find(1), bf_single_index->labelToIdLookup.end());

    VecSimIndex_Free(index);
}

/**** resizing cases ****/

TYPED_TEST(BruteForceTest, resize_and_align_index) {
    size_t dim = 4;
    size_t n = 15;
    size_t blockSize = 10;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = n,
                                             .blockSize = blockSize}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *bf_index =
        reinterpret_cast<BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *>(index);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i, i);
    }
    ASSERT_EQ(bf_index->idToLabelMapping.size(), n);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // remove invalid id
    VecSimIndex_DeleteVector(index, 3459);

    // This should do nothing
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(bf_index->idToLabelMapping.size(), n);

    // Add another vector, since index size equals to the capacity, this should cause resizing
    // (to fit a multiplication of block_size).
    this->GenerateNAddVector(index, dim, n + 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n + 1);
    // Check new capacity size, should be blockSize * 2.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), 2 * blockSize);

    // Now size = n + 1 = 16, capacity = 2* bs = 20. Test capacity overflow again
    // to check that it stays aligned with blocksize.

    size_t add_vectors_count = 8;
    for (size_t i = 0; i < add_vectors_count; i++) {
        this->GenerateNAddVector(index, dim, n + 2 + i, i);
    }

    // Size should be n + 1 + 8 = 24.
    ASSERT_EQ(VecSimIndex_IndexSize(index), n + 1 + add_vectors_count);

    // Check new capacity size, should be blockSize * 3.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), 3 * blockSize);

    VecSimIndex_Free(index);
}

// Case 1: initial capacity is larger than block size, and it is not aligned.
TYPED_TEST(BruteForceTest, resize_and_align_index_largeInitialCapacity) {
    size_t dim = 4;
    size_t n = 10; // Determines the initial size of idToLabelMapping.
    size_t bs = 3;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = n,
                                             .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *bf_index =
        reinterpret_cast<BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *>(index);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // add up to blocksize + 1 = 3 + 1 = 4
    for (size_t i = 0; i < bs + 1; i++) {
        this->GenerateNAddVector(index, dim, i, i);
    }

    size_t idToLabelMapping_size = bf_index->idToLabelMapping.size();
    // The idToLabelMapping size shouldn't change, should remain n.
    ASSERT_EQ(idToLabelMapping_size, n);
    ASSERT_EQ(VecSimIndex_IndexSize(index), bs + 1);

    // Delete last vector, to get size % block_size == 0. size = 3
    VecSimIndex_DeleteVector(index, bs);

    // Index size = bs = 3.
    ASSERT_EQ(VecSimIndex_IndexSize(index), bs);

    // New idToLabelMapping size = idToLabelMapping_size - block_size - number_of_vectors_to_align =
    // 10  - 3 - 10 % 3 (1) = 6
    idToLabelMapping_size = bf_index->idToLabelMapping.size();
    ASSERT_EQ(idToLabelMapping_size, n - bs - n % bs);

    // Delete all the vectors to decrease idToLabelMapping size by another bs.
    size_t i = 0;
    while (VecSimIndex_IndexSize(index) > 0) {
        VecSimIndex_DeleteVector(index, i);
        ++i;
    }
    ASSERT_EQ(bf_index->idToLabelMapping.size(), bs);
    // Add and delete a vector to achieve:
    // size % block_size == 0 && size + bs <= idToLabelMapping_size(3).
    // idToLabelMapping_size should be resized to zero.
    this->GenerateNAddVector(index, dim, 0);
    VecSimIndex_DeleteVector(index, 0);
    ASSERT_EQ(bf_index->idToLabelMapping.size(), 0);

    // Do it again. This time after adding a vector idToLabelMapping_size is increased by bs.
    // Upon deletion it will be resized to zero again.
    this->GenerateNAddVector(index, dim, 0);
    ASSERT_EQ(bf_index->idToLabelMapping.size(), bs);
    VecSimIndex_DeleteVector(index, 0);
    ASSERT_EQ(bf_index->idToLabelMapping.size(), 0);

    VecSimIndex_Free(index);
}

// Test empty index edge cases.
TYPED_TEST(BruteForceTest, brute_force_empty_index) {
    size_t dim = 4;
    size_t n = 20;
    size_t bs = 6;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = n,
                                             .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *bf_index =
        reinterpret_cast<BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *>(index);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove from an empty index - should fail because label doesn't exist.
    VecSimIndex_DeleteVector(index, 0);

    // Add one vector.
    this->GenerateNAddVector(index, dim, 1, 1.7);

    // Try to remove it.
    VecSimIndex_DeleteVector(index, 1);
    // The idToLabelMapping_size should change to be aligned with the vector size.
    size_t idToLabelMapping_size = bf_index->idToLabelMapping.size();

    ASSERT_EQ(idToLabelMapping_size, n - n % bs - bs);

    // Size equals 0.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove it again.
    // The idToLabelMapping_size should remain unchanged, as we are trying to delete a label that
    // doesn't exist.
    VecSimIndex_DeleteVector(index, 1);
    ASSERT_EQ(bf_index->idToLabelMapping.size(), idToLabelMapping_size);
    // Nor the size.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceTest, brute_force_vector_search_test_ip) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_IP,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[] = {50, 50, 50, 50};
    std::set<size_t> expected_ids;
    for (size_t i = n - 1; i > n - 1 - k; i--) {
        expected_ids.insert(i);
    }
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
        expected_ids.erase(id);
    };
    runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceTest, brute_force_vector_search_test_l2) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    auto verify_res = [&](size_t id, double score, size_t index) {
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    TEST_DATA_T query[] = {50, 50, 50, 50};
    runTopKSearchTest(index, query, k, verify_res);
    runTopKSearchTest(index, query, 0, verify_res); // For sanity, search for nothing

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceTest, brute_force_vector_search_by_id_test) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[] = {50, 50, 50, 50};
    auto verify_res = [&](size_t id, double score, size_t index) { ASSERT_EQ(id, (index + 45)); };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceTest, brute_force_indexing_same_vector) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i,
                                 i / 10); // i / 10 is in integer (take the "floor" value).
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

TYPED_TEST(BruteForceTest, brute_force_reindexing_same_vector) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;
    size_t initial_capacity = 200;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = initial_capacity}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *bf_index =
        reinterpret_cast<BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *>(index);

    for (size_t i = 0; i < n; i++) {
        // i / 10 is in integer (take the "floor" value).
        this->GenerateNAddVector(index, dim, i, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    TEST_DATA_T query[] = {4.9, 4.95, 5.05, 5.1};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    // Delete all vectors.
    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // The vector block should be removed.
    ASSERT_EQ(bf_index->getVectorBlocks().size(), 0);

    // id2label size should remain the same.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), initial_capacity);

    // Reinsert the same vectors under the same ids.
    for (size_t i = 0; i < n; i++) {
        // i / 10 is in integer (take the "floor value).
        this->GenerateNAddVector(index, dim, i, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run the same query again.
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceTest, brute_force_reindexing_same_vector_different_id) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i,
                                 i / 10); // i / 10 is in integer (take the "floor" value).
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

    // Reinsert the same vectors under different ids than before.
    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i + 10,
                                 i / 10); // i / 10 is in integer (take the "floor" value).
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run the same query again.
    auto verify_res_different_id = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(id >= 60 && id < 70 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res_different_id);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceTest, test_delete_swap_block) {
    size_t initial_capacity = 5; // idToLabelMapping initial size.
    size_t k = 5;
    size_t dim = 2;

    // This test creates 2 vector blocks with size of 3
    // Insert 6 vectors with ascending ids; The vector blocks will look like
    // 0 [0, 1, 2]
    // 1 [3, 4, 5]
    // Delete the id 1 will delete it from the first vector block 0 [0 ,1, 2] and will move vector
    // data of id 5 to vector block 0 at index 1. id2label[1] should hold the label of the vector
    // that was in id 5.
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = initial_capacity,
                                             .blockSize = 3}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *bf_index =
        reinterpret_cast<BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *>(index);

    // idToLabelMapping initial size equals n.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), initial_capacity);

    size_t n = 6;
    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i, i);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    // id2label is increased and aligned with bs.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), n);

    labelType id1_prev_label = bf_index->getVectorLabel(1);
    labelType id5_prev_label = bf_index->getVectorLabel(5);

    // Here the shift should happen.
    VecSimIndex_DeleteVector(index, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n - 1);
    // id2label size should remain unchanged..
    ASSERT_EQ(bf_index->idToLabelMapping.size(), n);

    // id1 gets what was previously id5's label.
    ASSERT_EQ(bf_index->getVectorLabel(1), id5_prev_label);

    BruteForceIndex_Single<TEST_DATA_T, TEST_DIST_T> *bf_single_index =
        reinterpret_cast<BruteForceIndex_Single<TEST_DATA_T, TEST_DIST_T> *>(index);

    // label2id value at label5 should be 1
    auto last_vector_new_id = bf_single_index->labelToIdLookup[id5_prev_label];
    ASSERT_EQ(last_vector_new_id, 1);

    // The label of what initially was in id1 should be removed.
    auto deleted_label_id_pair = bf_single_index->labelToIdLookup.find(id1_prev_label);
    ASSERT_EQ(deleted_label_id_pair, bf_single_index->labelToIdLookup.end());

    // The vector in index1 should hold id5 data.
    TEST_DATA_T *vector_data = bf_index->getDataByInternalId(1);
    for (size_t i = 0; i < dim; ++i) {
        ASSERT_EQ(*vector_data, 5);
        ++vector_data;
    }

    TEST_DATA_T query[] = {0.0, 0.0};
    auto verify_res = [&](size_t id, double score, size_t index) {
        if (index == 0) {
            ASSERT_EQ(id, index);
        } else {
            ASSERT_EQ(id, index + 1);
        }
    };
    runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceTest, sanity_reinsert_1280) {
    size_t n = 5;
    size_t d = 1280;
    size_t k = 5;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = d,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = n}};

    VecSimIndex *index = VecSimIndex_New(&params);

    auto *vectors = (TEST_DATA_T *)malloc(n * d * sizeof(TEST_DATA_T));

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
    free(vectors);
    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceTest, test_bf_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args.
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = d,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.bfInfo.dim, d);
    ASSERT_FALSE(info.bfInfo.isMulti);
    // Default args.
    ASSERT_EQ(info.bfInfo.blockSize, DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(info.bfInfo.indexSize, 0);
    VecSimIndex_Free(index);

    d = 1280;
    params = VecSimParams{.algo = VecSimAlgo_BF,
                          .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                               .dim = d,
                                               .metric = VecSimMetric_L2,
                                               .initialCapacity = n,
                                               .blockSize = 1}};
    index = VecSimIndex_New(&params);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.bfInfo.dim, d);
    ASSERT_FALSE(info.bfInfo.isMulti);
    // User args.
    ASSERT_EQ(info.bfInfo.blockSize, 1);
    ASSERT_EQ(info.bfInfo.indexSize, 0);
    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceTest, test_basic_bf_info_iterator) {
    size_t n = 100;
    size_t d = 128;
    VecSimMetric metrics[3] = {VecSimMetric_Cosine, VecSimMetric_IP, VecSimMetric_L2};

    for (size_t i = 0; i < 3; i++) {
        // Build with default args.
        VecSimParams params{.algo = VecSimAlgo_BF,
                            .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                                 .dim = d,
                                                 .metric = metrics[i],
                                                 .initialCapacity = n}};
        VecSimIndex *index = VecSimIndex_New(&params);
        VecSimIndexInfo info = VecSimIndex_Info(index);
        VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
        compareFlatIndexInfoToIterator(info, infoIter);
        VecSimInfoIterator_Free(infoIter);
        VecSimIndex_Free(index);
    }
}

TYPED_TEST(BruteForceTest, test_dynamic_bf_info_iterator) {
    size_t d = 128;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = d,
                                             .metric = VecSimMetric_L2,
                                             .blockSize = 1}};
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(1, info.bfInfo.blockSize);
    ASSERT_EQ(0, info.bfInfo.indexSize);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    TEST_DATA_T v[d];
    for (size_t i = 0; i < d; i++) {
        v[i] = (TEST_DATA_T)i;
    }
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

    // Perform (or simulate) Search in all modes.
    VecSimIndex_AddVector(index, v, 0);
    auto res = VecSimIndex_TopKQuery(index, v, 1, nullptr, BY_SCORE);
    VecSimQueryResult_Free(res);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(STANDARD_KNN, info.bfInfo.last_mode);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    res = VecSimIndex_RangeQuery(index, v, 1, nullptr, BY_SCORE);
    VecSimQueryResult_Free(res);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(RANGE_QUERY, info.bfInfo.last_mode);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 1, 1, true));
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(HYBRID_ADHOC_BF, info.bfInfo.last_mode);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Set the index size artificially so that BATCHES mode will be selected by the heuristics.
    reinterpret_cast<BruteForceIndex<TEST_DATA_T, TEST_DIST_T> *>(index)->count = 1e4;
    ASSERT_FALSE(VecSimIndex_PreferAdHocSearch(index, 7e3, 1, true));
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(HYBRID_BATCHES, info.bfInfo.last_mode);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Simulate the case where another call to the heuristics is done after realizing that
    // the subset size is smaller, and change the policy as a result.
    ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 1, 1, false));
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(HYBRID_BATCHES_TO_ADHOC_BF, info.bfInfo.last_mode);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    VecSimIndex_Free(index);
}
