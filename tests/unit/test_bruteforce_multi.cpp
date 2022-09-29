#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "test_utils.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/brute_force/brute_force_multi.h"
#include <cmath>

template <VecSimType type, typename DataType, typename DistType = DataType>
struct IndexType {
    static VecSimType get_index_type() { return type; }
    typedef DataType data_t;
    typedef DistType dist_t;
};

template <typename index_type_t>
class BruteForceMultiTest : public ::testing::Test {
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    void GenerateVector(data_t *output, size_t dim, data_t value = 1.0) {
        for (size_t i = 0; i < dim; i++) {
            output[i] = (data_t)value;
        }
    }
    // Returns the memory addition after adding the vector to the index.
    int GenerateNAddVector(VecSimIndex *index, size_t dim, size_t id, data_t value = 1.0) {
        data_t v[dim];
        this->GenerateVector(v, dim, value); // i / 10 is in integer (take the "floor" value).
        return VecSimIndex_AddVector(index, v, id);
    }

    BruteForceIndex_Multi<data_t, dist_t> *CastToBF_Multi(VecSimIndex *index) {
        return reinterpret_cast<BruteForceIndex_Multi<data_t, dist_t> *>(index);
    }
};

#define TEST_DATA_T typename TypeParam::data_t
#define TEST_DIST_T typename TypeParam::dist_t
using DataTypeSet =
    ::testing::Types<IndexType<VecSimType_FLOAT32, float>, IndexType<VecSimType_FLOAT64, double>>;

TYPED_TEST_CASE(BruteForceMultiTest, DataTypeSet);

TYPED_TEST(BruteForceMultiTest, vector_add_multiple_test) {
    size_t dim = 4;
    size_t rep = 5;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_IP,
                                             .multi = true,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Adding multiple vectors under the same label
    for (size_t j = 0; j < rep; j++) {
        TEST_DATA_T a[dim];
        for (size_t i = 0; i < dim; i++) {
            a[i] = (TEST_DATA_T)i * j + i;
        }
        VecSimIndex_AddVector(index, a, 46);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), rep);
    ASSERT_EQ(this->CastToBF_Multi(index)->indexLabelCount(), 1);

    // Deleting the label. All the vectors should be deleted.
    VecSimIndex_DeleteVector(index, 46);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);
    ASSERT_EQ(this->CastToBF_Multi(index)->indexLabelCount(), 0);

    VecSimIndex_Free(index);
}

/**** resizing cases ****/

TYPED_TEST(BruteForceMultiTest, resize_and_align_index) {
    size_t dim = 4;
    size_t n = 15;
    size_t blockSize = 10;
    size_t n_labels = 3;
    VecSimIndexInfo info;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = n,
                                             .blockSize = blockSize}};
    VecSimIndex *index = VecSimIndex_New(&params);
    auto bf_index = this->CastToBF_Multi(index);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i % n_labels, i);
    }
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.bfInfo.indexSize, n);
    ASSERT_EQ(info.bfInfo.indexLabelCount, n_labels);
    ASSERT_EQ(bf_index->idToLabelMapping.size(), n);
    ASSERT_EQ(bf_index->getVectorBlocks().size(), n / blockSize + 1);

    // remove invalid id
    VecSimIndex_DeleteVector(index, 3459);

    // This should do nothing
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.bfInfo.indexSize, n);
    ASSERT_EQ(info.bfInfo.indexLabelCount, n_labels);
    ASSERT_EQ(bf_index->idToLabelMapping.size(), n);
    ASSERT_EQ(bf_index->getVectorBlocks().size(), n / blockSize + 1);

    // Add another vector, since index size equals to the capacity, this should cause resizing
    // (to fit a multiplication of block_size).
    this->GenerateNAddVector(index, dim, 0);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.bfInfo.indexSize, n + 1);
    // Label count doesn't increase because label 0 already exists
    ASSERT_EQ(info.bfInfo.indexLabelCount, n_labels);
    // Check new capacity size, should be blockSize * 2.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), 2 * blockSize);

    // Now size = n + 1 = 16, capacity = 2* bs = 20. Test capacity overflow again
    // to check that it stays aligned with blocksize.

    size_t add_vectors_count = 8;
    for (size_t i = 0; i < add_vectors_count; i++) {
        this->GenerateNAddVector(index, dim, i % n_labels, i);
    }

    // Size should be n + 1 + 8 = 24.
    size_t new_n = n + 1 + add_vectors_count;
    info = VecSimIndex_Info(index);

    ASSERT_EQ(info.bfInfo.indexSize, new_n);
    // Label count doesn't increase because label 0 already exists
    ASSERT_EQ(info.bfInfo.indexLabelCount, n_labels);
    size_t total_vectors = 0;
    for (auto label_ids : bf_index->labelToIdsLookup) {
        total_vectors += label_ids.second.size();
    }
    ASSERT_EQ(total_vectors, new_n);

    // Check new capacity size, should be blockSize * 3.
    ASSERT_EQ(bf_index->idToLabelMapping.size(), 3 * blockSize);

    VecSimIndex_Free(index);
}

// Test empty index edge cases.
TYPED_TEST(BruteForceMultiTest, empty_index) {
    size_t dim = 4;
    size_t n = 20;
    size_t bs = 6;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = n,
                                             .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove from an empty index - should fail because label doesn't exist.
    VecSimIndex_DeleteVector(index, 0);

    // Add one vector.
    this->GenerateNAddVector(index, dim, 1, 1.7);

    // Try to remove it.
    VecSimIndex_DeleteVector(index, 1);
    // The idToLabelMapping_size should change to be aligned with the vector size.
    size_t idToLabelMapping_size = this->CastToBF_Multi(index)->idToLabelMapping.size();

    ASSERT_EQ(idToLabelMapping_size, n - n % bs - bs);

    // Size equals 0.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove it again.
    // The idToLabelMapping_size should remain unchanged, as we are trying to delete a label that
    // doesn't exist.
    VecSimIndex_DeleteVector(index, 1);
    ASSERT_EQ(this->CastToBF_Multi(index)->idToLabelMapping.size(), idToLabelMapping_size);
    // Nor the size.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, search_more_than_there_is) {
    size_t dim = 4;
    size_t n = 5;
    size_t perLabel = 3;
    size_t n_labels = ceil((float)n / perLabel);
    size_t k = 3;
    // This test add 5 vectors under 2 labels, and then query for 3 results.
    // We want to make sure we get only 2 results back (because the results should have unique
    // labels), although the index contains 5 vectors.

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i / perLabel, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(VecSimIndex_Info(index).bfInfo.indexLabelCount, n_labels);

    TEST_DATA_T query[] = {0, 0, 0, 0};
    VecSimQueryResult_List res = VecSimIndex_TopKQuery(index, query, k, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), n_labels);
    auto it = VecSimQueryResult_List_GetIterator(res);
    for (size_t i = 0; i < n_labels; i++) {
        auto el = VecSimQueryResult_IteratorNext(it);
        labelType element_label = VecSimQueryResult_GetId(el);
        ASSERT_EQ(element_label, i);
        auto ids = this->CastToBF_Multi(index)->labelToIdsLookup.at(element_label);
        for (size_t j = 0; j < ids.size(); j++) {
            // Verifying that each vector is labeled correctly.
            // ID is calculated according to insertion order.
            ASSERT_EQ(ids[j], i * perLabel + j);
        }
    }
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}
TYPED_TEST(BruteForceMultiTest, indexing_same_vector) {
    size_t n = 100;
    size_t k = 10;
    size_t perLabel = 10;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i / perLabel, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[] = {0, 0, 0, 0};
    auto verify_res = [&](size_t id, double score, size_t index) { ASSERT_EQ(id, index); };
    runTopKSearchTest(index, query, k, verify_res);
    auto res = VecSimIndex_TopKQuery(index, query, k, nullptr, BY_SCORE);
    auto it = VecSimQueryResult_List_GetIterator(res);
    for (size_t i = 0; i < k; i++) {
        auto el = VecSimQueryResult_IteratorNext(it);
        labelType element_label = VecSimQueryResult_GetId(el);
        ASSERT_EQ(element_label, i);
        auto ids = this->CastToBF_Multi(index)->labelToIdsLookup.at(element_label);
        for (size_t j = 0; j < ids.size(); j++) {
            // Verifying that each vector is labeled correctly.
            // ID is calculated according to insertion order.
            ASSERT_EQ(ids[j], i * perLabel + j);
        }
    }
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, find_better_score) {
    size_t n = 100;
    size_t k = 10;
    size_t n_labels = 10;
    size_t dim = 4;
    size_t initial_capacity = 200;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = initial_capacity}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // Building the index. Each label gets 10 vectors with decreasing (by insertion order) element
    // value, so when we search, each vector is better than the previous one. Furthermore, each
    // label gets at least one better vector than the previous label and one with a score equals to
    // the best of the previous label, so the multimap holds at least two labels with the same
    // score.
    std::map<size_t, double> scores;
    for (size_t i = 0; i < n; i++) {
        // For example, with n_labels == 10 and n == 100,
        // label 0 will get vector elements 18 -> 9 (aka (9 -> 0) + 9),
        // label 1 will get vector elements 17 -> 8 (aka (9 -> 0) + 8),
        // label 2 will get vector elements 16 -> 7 (aka (9 -> 0) + 7),
        // . . . . .
        // label 9 will get vector elements 9 -> 0 (aka (9 -> 0) + 0),
        // and so on, so each label has some common vectors with all the previous labels.
        size_t el = ((n - i - 1) % n_labels) + ((n - i - 1) / n_labels);

        this->GenerateNAddVector(index, dim, i / n_labels, el);
        // This should be the best score for each label.
        if (i % n_labels == n_labels - 1) {
            // `el * el * dim` is the L2-squared value with the 0 vector.
            scores.emplace(i / n_labels, el * el * dim);
        }
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(VecSimIndex_Info(index).bfInfo.indexLabelCount, n_labels);

    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, k - index - 1);
        ASSERT_DOUBLE_EQ(score, scores[id]);
    };

    TEST_DATA_T query[] = {0, 0, 0, 0};
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, find_better_score_after_pop) {
    size_t n = 12;
    size_t n_labels = 3;
    size_t dim = 4;
    size_t initial_capacity = 200;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = initial_capacity}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // Building the index. Each is better than the previous one.
    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i % n_labels, n - i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(VecSimIndex_Info(index).bfInfo.indexLabelCount, n_labels);

    TEST_DATA_T query[] = {0, 0, 0, 0};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, n_labels - index - 1);
    };
    // Having k = n_labels - 1, the heap will continuously pop the worst label before finding the
    // next vector, which is of the same label and is the best vector yet.
    runTopKSearchTest(index, query, n_labels - 1, verify_res);

    // Having k = n_labels, the heap will never get to pop the worst label, but we will update the
    // scores each time.
    runTopKSearchTest(index, query, n_labels, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, reindexing_same_vector_different_id) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;
    size_t perLabel = 3;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        // i / 10 is in integer (take the "floor" value)
        this->GenerateNAddVector(index, dim, i, TEST_DATA_T(i / 10));
    }
    // Add more vectors under the same labels. their scores should be worst.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T v[dim];
        this->GenerateVector(v, dim, TEST_DATA_T(i / 10) + n);
        for (size_t j = 0; j < perLabel - 1; j++) {
            VecSimIndex_AddVector(index, v, i);
        }
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n * perLabel);

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
        // i / 10 is in integer (take the "floor" value)
        this->GenerateNAddVector(index, dim, i + 10, TEST_DATA_T(i / 10));
    }
    // Add more vectors under the same labels. their scores should be worst.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T v[dim];
        this->GenerateVector(v, dim, TEST_DATA_T(i / 10) + n);
        for (size_t j = 0; j < perLabel - 1; j++) {
            VecSimIndex_AddVector(index, v, i + 10);
        }
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n * perLabel);

    // Run the same query again.
    auto verify_res_different_id = [&](int id, double score, size_t index) {
        ASSERT_TRUE(id >= 60 && id < 70 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res_different_id);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, test_delete_swap_block) {
    size_t initial_capacity = 5; // idToLabelMapping initial size.
    size_t n = 6;
    size_t dim = 2;
    size_t n_labels = 3;
    size_t bs = 3;

    // This test creates 2 vector blocks with size of 3
    // Insert 6 vectors with ascending ids; The vector blocks will look like
    // block 0 [0, 1, 2]
    // block 1 [3, 4, 5]
    // Delete the id 1 will delete it from the first vector block 0 [0 ,1, 2] and will move vector
    // data of id 5 to vector block 0 at index 1. id2label[1] should hold the label of the vector
    // that was in id 5.
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = initial_capacity,
                                             .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex_Multi<TEST_DATA_T, TEST_DIST_T> *bfm_index = this->CastToBF_Multi(index);

    // idToLabelMapping initial size equals n.
    ASSERT_EQ(bfm_index->idToLabelMapping.size(), initial_capacity);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i % n_labels, i);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(bfm_index->indexLabelCount(), n_labels);
    for (auto label_ids : bfm_index->labelToIdsLookup) {
        ASSERT_EQ(label_ids.second.size(), n / n_labels);
    }
    // id2label is increased and aligned with bs.
    ASSERT_EQ(bfm_index->idToLabelMapping.size(), n);

    labelType id1_prev_label = bfm_index->getVectorLabel(1);
    labelType id5_prev_label = bfm_index->getVectorLabel(5);

    // Here the shift should happen.
    //
    //         initial ids     labels
    // block 0 [0, 1, 2] ~~~~~ [0, 1, 2]
    // block 1 [3, 4, 5] ~~~~~ [0, 1, 2]
    //
    // We labeled each vector as its relative index in the block, so we delete ids 1 and 4 now.
    // We should get the followed result:
    //
    //         initial ids     labels
    // block 0 [0, 5, 2] ~~~~~ [0, 2, 2]
    // block 1 [3]       ~~~~~ [0]

    VecSimIndex_DeleteVector(index, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n - (n / n_labels));
    ASSERT_EQ(bfm_index->indexLabelCount(), n_labels - 1);
    ASSERT_EQ(bfm_index->getVectorLabel(0), 0);
    ASSERT_EQ(bfm_index->getVectorLabel(1), 2);
    ASSERT_EQ(bfm_index->getVectorLabel(2), 2);
    ASSERT_EQ(bfm_index->getVectorLabel(3), 0);
    // id2label size should remain unchanged..
    ASSERT_EQ(bfm_index->idToLabelMapping.size(), n);

    // id1 gets what was previously id5's label.
    ASSERT_EQ(bfm_index->getVectorLabel(1), id5_prev_label);

    // The label of what initially was in id1 should be removed.
    auto deleted_label_id_pair = bfm_index->labelToIdsLookup.find(id1_prev_label);
    ASSERT_EQ(deleted_label_id_pair, bfm_index->labelToIdsLookup.end());

    // The vector in index1 should hold id5 data.
    VectorBlock *block = bfm_index->getVectorVectorBlock(1);
    TEST_DATA_T *vector_data = (TEST_DATA_T *)block->getVector(1);
    for (size_t i = 0; i < dim; ++i) {
        ASSERT_EQ(*vector_data, 5);
        ++vector_data;
    }

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, test_bf_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args.
    VecSimParams params = {.algo = VecSimAlgo_BF,
                           .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                                .dim = d,
                                                .metric = VecSimMetric_L2,
                                                .multi = true,
                                                .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.bfInfo.dim, d);
    ASSERT_TRUE(info.bfInfo.isMulti);
    // Default args.
    ASSERT_EQ(info.bfInfo.blockSize, DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(info.bfInfo.indexSize, 0);
    VecSimIndex_Free(index);

    d = 1280;
    params = VecSimParams{.algo = VecSimAlgo_BF,
                          .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                               .dim = d,
                                               .metric = VecSimMetric_L2,
                                               .multi = true,
                                               .initialCapacity = n,
                                               .blockSize = 1}};
    index = VecSimIndex_New(&params);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.bfInfo.dim, d);
    ASSERT_TRUE(info.bfInfo.isMulti);
    // User args.
    ASSERT_EQ(info.bfInfo.blockSize, 1);
    ASSERT_EQ(info.bfInfo.indexSize, 0);
    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, test_basic_bf_info_iterator) {
    size_t n = 100;
    size_t d = 128;
    VecSimMetric metrics[3] = {VecSimMetric_Cosine, VecSimMetric_IP, VecSimMetric_L2};

    for (size_t i = 0; i < 3; i++) {
        // Build with default args.
        VecSimParams params{.algo = VecSimAlgo_BF,
                            .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                                 .dim = d,
                                                 .metric = metrics[i],
                                                 .multi = true,
                                                 .initialCapacity = n}};
        VecSimIndex *index = VecSimIndex_New(&params);
        VecSimIndexInfo info = VecSimIndex_Info(index);
        VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
        compareFlatIndexInfoToIterator(info, infoIter);
        VecSimInfoIterator_Free(infoIter);
        VecSimIndex_Free(index);
    }
}

TYPED_TEST(BruteForceMultiTest, test_dynamic_bf_info_iterator) {
    size_t d = 128;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = d,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
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

    // Add vectors.
    VecSimIndex_AddVector(index, v, 0);
    VecSimIndex_AddVector(index, v, 0);
    VecSimIndex_AddVector(index, v, 1);
    VecSimIndex_AddVector(index, v, 1);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(4, info.bfInfo.indexSize);
    ASSERT_EQ(2, info.bfInfo.indexLabelCount);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Delete vectors.
    VecSimIndex_DeleteVector(index, 0);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(2, info.bfInfo.indexSize);
    ASSERT_EQ(1, info.bfInfo.indexLabelCount);
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
    size_t perLabel = 3;
    for (size_t i = 0; i < 1e4; i++) {
        VecSimIndex_AddVector(index, v, i / perLabel);
    }
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

TYPED_TEST(BruteForceMultiTest, vector_search_test_l2) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;
    size_t perLabel = 4;

    for (size_t blocksize : {12, DEFAULT_BLOCK_SIZE}) {

        VecSimParams params{.algo = VecSimAlgo_BF,
                            .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = 200,
                                                 .blockSize = blocksize}};
        VecSimIndex *index = VecSimIndex_New(&params);

        VecSimIndexInfo info = VecSimIndex_Info(index);
        ASSERT_EQ(info.algo, VecSimAlgo_BF);
        ASSERT_EQ(info.bfInfo.blockSize, blocksize);

        for (size_t i = 0; i < n; i++) {
            this->GenerateNAddVector(index, dim, i, i);
        }
        // Add more vectors under the same labels. their scores should be worst.
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < perLabel - 1; j++) {
                this->GenerateNAddVector(index, dim, i, i + n);
            }
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), n * perLabel);

        auto verify_res = [&](size_t id, double score, size_t index) {
            size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
            ASSERT_EQ(diff_id, (index + 1) / 2);
            ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
        };
        TEST_DATA_T query[] = {50, 50, 50, 50};
        runTopKSearchTest(index, query, k, verify_res);

        VecSimIndex_Free(index);
    }
}

TYPED_TEST(BruteForceMultiTest, search_empty_index) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T query[] = {50, 50, 50, 50};

    // We do not expect any results.
    VecSimQueryResult_List res = VecSimIndex_TopKQuery(index, query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    VecSimQueryResult_Iterator *it = VecSimQueryResult_List_GetIterator(res);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);

    // TODO: uncomment when support for BFM range is enabled
    // res = VecSimIndex_RangeQuery(index, (const void *)query, 1.0f, NULL, BY_SCORE);
    // ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    // VecSimQueryResult_Free(res);

    // Add some vectors and remove them all from index, so it will be empty again.
    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i, i);
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

    // TODO: uncomment when support for BFM range is enabled
    // res = VecSimIndex_RangeQuery(index, (const void *)query, 1.0f, NULL, BY_SCORE);
    // ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    // VecSimQueryResult_Free(res);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, remove_vector_after_replacing_block) {
    size_t dim = 4;
    size_t bs = 2;
    size_t n = 6;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = 200,
                                             .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Setting up vectors
    TEST_DATA_T f[n][dim];
    for (size_t i = 0; i < n; i++) {
        this->GenerateVector(f[i], dim, i);
    }
    // Add 1 vector with label 1
    VecSimIndex_AddVector(index, f[0], 1);

    // Add 3 vectors with label 3
    VecSimIndex_AddVector(index, f[1], 3);
    VecSimIndex_AddVector(index, f[2], 3);
    VecSimIndex_AddVector(index, f[3], 3);

    // Add 2 vectors with label 2
    VecSimIndex_AddVector(index, f[4], 2);
    VecSimIndex_AddVector(index, f[5], 2);

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Delete label 3. the following drawing present the expected changes
    // [[1, 3], [3, 3], [2, 2]] -> [[1, 2], [3, 3], [2]] -> [[1, 2], [2, 3]] -> [[1, 2], [2]]
    // [[0, 1], [2, 3], [4, 5]] -> [[0, 5], [2, 3], [4]] -> [[0, 5], [4, 3]] -> [[0, 5], [4]]
    VecSimIndex_DeleteVector(index, 3);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 3);
    ASSERT_EQ(VecSimIndex_Info(index).bfInfo.indexLabelCount, 2);
    auto bf_index = this->CastToBF_Multi(index);
    ASSERT_EQ(bf_index->getVectorLabel(0), 1);
    ASSERT_EQ(bf_index->getVectorLabel(1), 2);
    ASSERT_EQ(bf_index->getVectorLabel(2), 2);
    // checking the blob swaps.
    ASSERT_EQ(bf_index->getDataByInternalId(0)[0], 0);
    ASSERT_EQ(bf_index->getDataByInternalId(1)[0], 5);
    ASSERT_EQ(bf_index->getDataByInternalId(2)[0], 4);

    VecSimIndex_DeleteVector(index, 1);
    VecSimIndex_DeleteVector(index, 2);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, batch_iterator) {
    size_t dim = 4;
    size_t perLabel = 5;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = 200,
                                             .blockSize = 7}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // run the test twice - for index of size 100, every iteration will run select-based search,
    // as the number of results is 5, which is more than 0.1% of the index size. for index of size
    // 10000, we will run the heap-based search until we return 5000 results, and then switch to
    // select-based search.
    for (size_t m : {100, 10000}) {
        size_t n = m * perLabel;
        for (size_t i = 0; i < n; i++) {
            this->GenerateNAddVector(index, dim, i / perLabel, i);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), n);
        ASSERT_EQ(VecSimIndex_Info(index).bfInfo.indexLabelCount, m);

        // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
        TEST_DATA_T query[dim];
        this->GenerateVector(query, dim, n);
        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
        size_t iteration_num = 0;

        // Get the 10 vectors whose ids are the maximal among those that hasn't been returned yet,
        // in every iteration. The order should be from the largest to the lowest id.
        size_t n_res = 5;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            std::vector<size_t> expected_ids(n_res);
            for (size_t i = 0; i < n_res; i++) {
                expected_ids[i] = (m - iteration_num * n_res - i - 1);
            }
            auto verify_res = [&](size_t id, double score, size_t index) {
                ASSERT_EQ(expected_ids[index], id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num, m / n_res);
        VecSimBatchIterator_Free(batchIterator);
        ASSERT_EQ(VecSimIndex_IndexSize(index), n);
        ASSERT_EQ(VecSimIndex_Info(index).bfInfo.indexLabelCount, m);
        // Cleanup before next round.
        for (size_t i = 0; i < m; i++) {
            VecSimIndex_DeleteVector(index, i);
        }
    }
    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, brute_force_batch_iterator_non_unique_scores) {
    size_t dim = 4;
    size_t perLabel = 5;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = 200,
                                             .blockSize = 5}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // Run the test twice - for index of size 100, every iteration will run select-based search,
    // as the number of results is 5, which is more than 0.1% of the index size. for index of size
    // 10000, we will run the heap-based search until we return 5000 results, and then switch to
    // select-based search.
    for (size_t m : {100, 10000}) {
        size_t n = m * perLabel;
        for (size_t i = 0; i < n; i++) {
            this->GenerateNAddVector(index, dim, i / perLabel, i / (10 * perLabel));
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), n);

        // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
        TEST_DATA_T query[dim];
        this->GenerateVector(query, dim, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
        size_t iteration_num = 0;

        // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet,
        // in every iteration. there are n/10 groups of 10 different vectors with the same score.
        size_t n_res = 5;
        bool even_iteration = false;
        std::set<size_t> expected_ids;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            // Insert the maximal 10 ids in every odd iteration.
            if (!even_iteration) {
                for (size_t i = 1; i <= 2 * n_res; i++) {
                    expected_ids.insert(m - iteration_num * n_res - i);
                }
            }
            auto verify_res = [&](size_t id, double score, size_t index) {
                ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
                expected_ids.erase(id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            // Make sure that the expected ids set is empty after two iterations.
            if (even_iteration) {
                ASSERT_TRUE(expected_ids.empty());
            }
            iteration_num++;
            even_iteration = !even_iteration;
        }
        ASSERT_EQ(iteration_num, m / n_res);
        VecSimBatchIterator_Free(batchIterator);
        ASSERT_EQ(VecSimIndex_IndexSize(index), n);
        ASSERT_EQ(VecSimIndex_Info(index).bfInfo.indexLabelCount, m);
        // Cleanup before next round.
        for (size_t i = 0; i < m; i++) {
            VecSimIndex_DeleteVector(index, i);
        }
    }
    VecSimIndex_Free(index);
}
// TODO:FIX!!!!
TYPED_TEST(BruteForceMultiTest, batch_iterator_validate_scores) {
    size_t dim = 4;
    size_t perLabel = 10;
    size_t n_labels = 100;

    size_t init_n = n_labels * (perLabel - 1);

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = init_n,
                                             .blockSize = 5}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // Inserting some big vectors to the index
    for (size_t i = 0; i < init_n; i++) {
        this->GenerateNAddVector(index, dim, i % n_labels, i + n_labels + 46);
    }
    // Lastly, inserting small vector for each label
    for (size_t label = 0; label < n_labels; label++) {
        this->GenerateNAddVector(index, dim, label, label);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n_labels * perLabel);

    // Query for (0,0,0,...,0) vector.
    TEST_DATA_T query[dim];
    this->GenerateVector(query, dim, 0);
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
    size_t iteration_num = 0;

    size_t n_res = 5;
    // ids should be in ascending order
    // scores should match to the score of the last vector for each label.
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, index + iteration_num * n_res);
        ASSERT_DOUBLE_EQ(score, id * id * dim);
    };
    while (VecSimBatchIterator_HasNext(batchIterator)) {
        runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
        iteration_num++;
    }

    ASSERT_EQ(iteration_num, n_labels / n_res);
    VecSimBatchIterator_Free(batchIterator);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n_labels * perLabel);
    ASSERT_EQ(VecSimIndex_Info(index).bfInfo.indexLabelCount, n_labels);

    VecSimIndex_Free(index);
}

/* TYPED_TEST(BruteForceMultiTest, brute_get_distance) {
    size_t n_labels = 2;
    size_t dim = 2;
    size_t numIndex = 3;
    VecSimIndex *index[numIndex];
    std::vector<double> distances;

    float v1_0[] = {M_PI, M_PI};
    float v2_0[] = {M_E, M_E};
    float v3_1[] = {M_PI, M_E};
    float v4_1[] = {M_SQRT2, -M_SQRT2};

    VecSimParams params{
        .algo = VecSimAlgo_BF,
        .bfParams =
            BFParams{.type = VecSimType_FLOAT32, .dim = dim, .multi = true, .initialCapacity = 4}};

    for (size_t i = 0; i < numIndex; i++) {
        params.bfParams.metric = (VecSimMetric)i;
        index[i] = VecSimIndex_New(&params);
        VecSimIndex_AddVector(index[i], v1_0, 0);
        VecSimIndex_AddVector(index[i], v2_0, 0);
        VecSimIndex_AddVector(index[i], v3_1, 1);
        VecSimIndex_AddVector(index[i], v4_1, 1);
        ASSERT_EQ(VecSimIndex_IndexSize(index[i]), 4);
    }

    void *query = v1_0;
    void *norm = v2_0;                               // {e, e}
    VecSim_Normalize(norm, dim, VecSimType_FLOAT32); // now {1/sqrt(2), 1/sqrt(2)}
    ASSERT_FLOAT_EQ(((float *)norm)[0], 1.0f / sqrt(2.0f));
    ASSERT_FLOAT_EQ(((float *)norm)[1], 1.0f / sqrt(2.0f));
    double dist;

    // VecSimMetric_L2
    // distances are [[0.000, 0.358], [0.179, 23.739]]
    // minimum of each label are:
    distances = {0, 0.1791922003030777};
    for (size_t i = 0; i < n_labels; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_L2], i, query);
        ASSERT_DOUBLE_EQ(dist, distances[i]);
    }

    // VecSimMetric_IP
    // distances are [[-18.739, -16.079], [-17.409, 1.000]]
    // minimum of each label are:
    distances = {-18.73921012878418, -17.409339904785156};
    for (size_t i = 0; i < n_labels; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_IP], i, query);
        ASSERT_DOUBLE_EQ(dist, distances[i]);
    }

    // VecSimMetric_Cosine
    // distances are [[5.960e-08, 5.960e-08], [0.0026, 1.000]]
    // minimum of each label are:
    distances = {5.9604644775390625e-08, 0.0025991201400756836};
    for (size_t i = 0; i < n_labels; i++) {
        dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_Cosine], i, norm);
        ASSERT_DOUBLE_EQ(dist, distances[i]);
    }

    // Bad values
    dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_Cosine], -1, norm);
    ASSERT_TRUE(std::isnan(dist));
    dist = VecSimIndex_GetDistanceFrom(index[VecSimMetric_L2], 46, query);
    ASSERT_TRUE(std::isnan(dist));

    // Clean-up.
    for (size_t i = 0; i < numIndex; i++) {
        VecSimIndex_Free(index[i]);
    }
} */
/*
TYPED_TEST(BruteForceMultiTest, testCosine) {
    size_t dim = 128;
    size_t n = 100;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_Cosine,
                                             .multi = true,
                                             .initialCapacity = n}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 1; i <= n; i++) {
        TEST_DATA_T f[dim];
        f[0] = (TEST_DATA_T)i / n;
        for (size_t j = 1; j < dim; j++) {
            f[j] = 1.0;
        }
        VecSimIndex_AddVector(index, f, i);
    }
    // Add more worst vector for each label
    for (TEST_DATA_T i = 1; i <= n; i++) {
        TEST_DATA_T f[dim];
        f[0] = (TEST_DATA_T)i + n;
        for (size_t j = 1; j < dim; j++) {
            f[j] = 1.0;
        }
        VecSimIndex_AddVector(index, f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 2 * n);
    TEST_DATA_T query[dim];
    this->GenerateVector(query, dim);

    auto verify_res = [&](size_t id, double score, size_t res_rank) {
        ASSERT_EQ(id, (n - res_rank));

        double expected_score = index->getDistanceFrom(id, query); */
/*         double first_coordinate = (double)id / n;
        // By cosine definition: 1 - ((A \dot B) / (norm(A)*norm(B))), where A is the query vector
        // and B is the current result vector.
        double expected_score =
            1.0 -
            ((first_coordinate + (double)dim - 1.0) /
             (sqrt((double)dim) * sqrt((double)(dim - 1) + first_coordinate * first_coordinate)));
        // Verify that abs difference between the actual and expected score is at most 1/10^6. */
/*       ASSERT_NEAR(score, expected_score, 1e-5);
   };
   runTopKSearchTest(index, query, 10, verify_res);

   // Test with batch iterator.
   VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
   size_t iteration_num = 0;

   // get the 10 vectors whose ids are the maximal among those that hasn't been returned yet,
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
               1.0 - ((first_coordinate + (double)dim - 1.0) /
                       (sqrt((double)dim) *
                        sqrt((double)(dim - 1) + first_coordinate * first_coordinate)));
           // Verify that abs difference between the actual and expected score is at most 1/10^6.
           ASSERT_NEAR(score, expected_score, 1e-5);
       };
       runBatchIteratorSearchTest(batchIterator, n_res, verify_res_batch);
       iteration_num++;
   }
   ASSERT_EQ(iteration_num, n / n_res);
   VecSimBatchIterator_Free(batchIterator);
   VecSimIndex_Free(index);
} */

TYPED_TEST(BruteForceMultiTest, testSizeEstimation) {
    size_t dim = 128;
    size_t n = 0;
    size_t bs = DEFAULT_BLOCK_SIZE;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_Cosine,
                                             .multi = true,
                                             .initialCapacity = n,
                                             .blockSize = bs}};

    size_t estimation = VecSimIndex_EstimateInitialSize(&params);
    VecSimIndex *index = VecSimIndex_New(&params);

    size_t actual = index->getAllocator()->getAllocationSize();
    ASSERT_EQ(estimation, actual);

    estimation = VecSimIndex_EstimateElementSize(&params) * bs;
    actual = this->GenerateNAddVector(index, dim, 0);

    ASSERT_GE(estimation * 1.01, actual);
    ASSERT_LE(estimation * 0.99, actual);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, testInitialSizeEstimationWithInitialCapacity) {
    size_t dim = 128;
    size_t n = 100;
    size_t bs = DEFAULT_BLOCK_SIZE;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_Cosine,
                                             .multi = true,
                                             .initialCapacity = n,
                                             .blockSize = bs}};

    size_t estimation = VecSimIndex_EstimateInitialSize(&params);
    VecSimIndex *index = VecSimIndex_New(&params);

    size_t actual = index->getAllocator()->getAllocationSize();
    ASSERT_EQ(estimation, actual);

    VecSimIndex_Free(index);
}

TYPED_TEST(BruteForceMultiTest, testTimeoutReturn) {
    size_t dim = 4;
    float vec[] = {1.0f, 1.0f, 1.0f, 1.0f};
    VecSimQueryResult_List rl;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = 1,
                                             .blockSize = 5}};
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndex_AddVector(index, vec, 0);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out

    // Checks return code on timeout - knn
    rl = VecSimIndex_TopKQuery(index, vec, 1, NULL, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    // Check timeout again - range query
    // TODO: uncomment when support for BFM range is enabled
    // rl = VecSimIndex_RangeQuery(index, vec, 1, NULL, BY_ID);
    // ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    // ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    // VecSimQueryResult_Free(rl);

    VecSimIndex_Free(index);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // cleanup
}

TYPED_TEST(BruteForceMultiTest, testTimeoutReturn_batch_iterator) {
    size_t dim = 4;
    size_t n = 10;
    VecSimQueryResult_List rl;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = TypeParam::get_index_type(),
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = n,
                                             .blockSize = 5}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[dim];
    this->GenerateVector(query, dim, n);

    // Fail on second batch (after calculation already completed)
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    rl = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_OK);
    ASSERT_NE(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out
    rl = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSimBatchIterator_Free(batchIterator);

    // Fail on first batch (while calculating)
    // Timeout callback function already set to always time out
    batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    rl = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
    ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    VecSimQueryResult_Free(rl);

    VecSimBatchIterator_Free(batchIterator);

    VecSimIndex_Free(index);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // cleanup
}

// TEST_F(BruteForceMultiTest, rangeQuery) {
//     size_t n = 2000;
//     size_t dim = 4;

//     VecSimParams params{
//         .algo = VecSimAlgo_BF,
//         .bfParams = BFParams{
//             .type = VecSimType_FLOAT32, .dim = dim, .metric = VecSimMetric_L2, .blockSize = n /
//             2}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, (int)i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     size_t pivot_id = n / 2; // The id to return vectors around it.
//     float query[] = {(float)pivot_id, (float)pivot_id, (float)pivot_id, (float)pivot_id};

//     // Validate invalid params are caught with runtime exception.
//     try {
//         VecSimIndex_RangeQuery(index, (const void *)query, -1, nullptr, BY_SCORE);
//         FAIL();
//     } catch (std::runtime_error const &err) {
//         EXPECT_EQ(err.what(), std::string("radius must be non-negative"));
//     }
//     try {
//         VecSimIndex_RangeQuery(index, (const void *)query, 1, nullptr,
//         VecSimQueryResult_Order(2)); FAIL();
//     } catch (std::runtime_error const &err) {
//         EXPECT_EQ(err.what(), std::string("Possible order values are only 'BY_ID' or
//         'BY_SCORE'"));
//     }

//     auto verify_res_by_score = [&](size_t id, float score, size_t index) {
//         ASSERT_EQ(std::abs(int(id - pivot_id)), (index + 1) / 2);
//         ASSERT_EQ(score, dim * powf((index + 1) / 2, 2));
//     };
//     uint expected_num_results = 11;
//     // To get 11 results in the range [pivot_id - 5, pivot_id + 5], set the radius as the L2
//     score
//     // in the boundaries.
//     float radius = dim * powf(expected_num_results / 2, 2);
//     runRangeQueryTest(index, query, radius, verify_res_by_score, expected_num_results, BY_SCORE);

//     // Get results by id.
//     auto verify_res_by_id = [&](size_t id, float score, size_t index) {
//         ASSERT_EQ(id, pivot_id - expected_num_results / 2 + index);
//         ASSERT_EQ(score, dim * pow(std::abs(int(id - pivot_id)), 2));
//     };
//     runRangeQueryTest(index, query, radius, verify_res_by_id, expected_num_results);

//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, rangeQueryCosine) {
//     size_t n = 100;
//     size_t dim = 4;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_Cosine,
//                                              .blockSize = n / 2}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         f[0] = float(i + 1) / n;
//         for (size_t j = 1; j < dim; j++) {
//             f[j] = 1.0f;
//         }
//         // Use as label := n - (internal id)
//         VecSimIndex_AddVector(index, (const void *)f, n - i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);
//     float query[dim];
//     for (size_t i = 0; i < dim; i++) {
//         query[i] = 1.0f;
//     }
//     auto verify_res = [&](size_t id, float score, size_t index) {
//         ASSERT_EQ(id, index + 1);
//         float first_coordinate = float(n - index) / n;
//         // By cosine definition: 1 - ((A \dot B) / (norm(A)*norm(B))), where A is the query
//         vector
//         // and B is the current result vector.
//         float expected_score =
//             1.0f -
//             ((first_coordinate + (float)dim - 1.0f) /
//              (sqrtf((float)dim) * sqrtf((float)(dim - 1) + first_coordinate *
//              first_coordinate)));
//         // Verify that abs difference between the actual and expected score is at most 1/10^5.
//         ASSERT_NEAR(score, expected_score, 1e-5);
//     };
//     uint expected_num_results = 31;
//     // Calculate the score of the 31st distant vector from the query vector (whose id should be
//     30)
//     // to get the radius.
//     float edge_first_coordinate = (float)(n - expected_num_results + 1) / n;
//     float radius =
//         1.0f - ((edge_first_coordinate + (float)dim - 1.0f) /
//                 (sqrtf((float)dim) *
//                  sqrtf((float)(dim - 1) + edge_first_coordinate * edge_first_coordinate)));
//     runRangeQueryTest(index, query, radius, verify_res, expected_num_results, BY_SCORE);
//     // Return results BY_ID should give the same results.
//     runRangeQueryTest(index, query, radius, verify_res, expected_num_results, BY_ID);

//     VecSimIndex_Free(index);
// }
