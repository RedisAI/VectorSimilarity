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
/* 
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

    TEST_DATA_T a[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (TEST_DATA_T)1;
    }
    VecSimIndex_AddVector(index, (const void *)a, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

    // Prepare new vector data.
    for (size_t i = 0; i < dim; i++) {
        a[i] = (TEST_DATA_T)2;
    }
    // Call addVEctor with the same id, different data.
    VecSimIndex_AddVector(index, (const void *)a, 1);

    // Index size shouldn't change.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

    // id2label size should remain the same, although we seemingly tried to exceed
    // initial capacity.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), n);

    // Check update.
    TEST_DATA_T *vector_data = bf_index->getDataByInternalId(0);
    for (size_t i = 0; i < dim; ++i) {
        ASSERT_EQ(*vector_data, 2);
        ++vector_data;
    }

    for (size_t i = 0; i < dim; i++) {
        a[i] = (TEST_DATA_T)3;
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
}*/

/**** resizing cases ****/
/*
TEST_F(BruteForceTest, resize_and_align_index) {
    size_t dim = 4;
    size_t n = 15;
    size_t blockSize = 10;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = n,
                                             .blockSize = blockSize}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex<float, float> *bf_index =
        reinterpret_cast<BruteForceIndex<float, float> *>(index);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[dim];
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            a[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)a, i);
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
    VecSimIndex_AddVector(index, (const void *)a, n + 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n + 1);
    // Check new capacity size, should be blockSize * 2.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), 2 * blockSize);

    // Now size = n + 1 = 16, capacity = 2* bs = 20. Test capacity overflow again
    // to check that it stays aligned with blocksize.

    size_t add_vectors_count = 8;
    for (size_t i = 0; i < add_vectors_count; i++) {
        for (size_t j = 0; j < dim; j++) {
            a[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)a, n + 2 + i);
    }

    // Size should be n + 1 + 8 = 24.
    ASSERT_EQ(VecSimIndex_IndexSize(index), n + 1 + add_vectors_count);

    // Check new capacity size, should be blockSize * 3.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), 3 * blockSize);

    VecSimIndex_Free(index);
}*/
/*
// Case 1: initial capacity is larger than block size, and it is not aligned.
TEST_F(BruteForceTest, resize_and_align_index_largeInitialCapacity) {
    size_t dim = 4;
    size_t n = 10; // Determines the initial size of idToLabelMapping.
    size_t bs = 3;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = n,
                                             .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex<float, float> *bf_index =
        reinterpret_cast<BruteForceIndex<float, float> *>(index);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[dim];

    // add up to blocksize + 1 = 3 + 1 = 4
    for (size_t i = 0; i < bs + 1; i++) {
        for (size_t j = 0; j < dim; j++) {
            a[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)a, i);
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
    VecSimIndex_AddVector(index, (const void *)a, 0);
    VecSimIndex_DeleteVector(index, 0);
    ASSERT_EQ(bf_index->idToLabelMapping.size(), 0);

    // Do it again. This time after adding a vector idToLabelMapping_size is increased by bs.
    // Upon deletion it will be resized to zero again.
    VecSimIndex_AddVector(index, (const void *)a, 0);
    ASSERT_EQ(bf_index->idToLabelMapping.size(), bs);
    VecSimIndex_DeleteVector(index, 0);
    ASSERT_EQ(bf_index->idToLabelMapping.size(), 0);

    VecSimIndex_Free(index);
}*/
/*
// Test empty index edge cases.
TEST_F(BruteForceTest, brute_force_empty_index) {
    size_t dim = 4;
    size_t n = 20;
    size_t bs = 6;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .initialCapacity = n,
                                             .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex<float, float> *bf_index =
        reinterpret_cast<BruteForceIndex<float, float> *>(index);

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
 */

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
        this->GenerateNAddVector(index, dim, i, i / 10);// i / 10 is in integer (take the "floor" value).
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
        this->GenerateNAddVector(index, dim, i, i / 10);// i / 10 is in integer (take the "floor" value).
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
        this->GenerateNAddVector(index, dim, i, i / 10);// i / 10 is in integer (take the "floor" value).
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run the same query again.
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}
