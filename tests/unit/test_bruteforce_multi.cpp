#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "test_utils.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/brute_force/brute_force_multi.h"
#include <cmath>

class BruteForceMultiTest : public ::testing::Test {
protected:
    BruteForceMultiTest() {}

    ~BruteForceMultiTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(BruteForceMultiTest, brute_force_vector_add_multiple_test) {
    size_t dim = 4;
    int rep = 5;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_IP,
                                             .multi = true,
                                             .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Adding multiple vectors under the same label
    for (int j = 0; j < rep; j++) {
        float a[dim];
        for (size_t i = 0; i < dim; i++) {
            a[i] = (float)i * j + i;
        }
        VecSimIndex_AddVector(index, (const void *)a, 46);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), rep);
    ASSERT_EQ(reinterpret_cast<BruteForceIndex_Multi *>(index)->indexLabelCount(), 1);

    // Deleting the label. All the vectors should be deleted.
    VecSimIndex_DeleteVector(index, 46);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);
    ASSERT_EQ(reinterpret_cast<BruteForceIndex_Multi *>(index)->indexLabelCount(), 0);

    VecSimIndex_Free(index);
}

/**** resizing cases ****/

TEST_F(BruteForceMultiTest, resizeNAlignIndex) {
    size_t dim = 4;
    size_t n = 15;
    size_t blockSize = 10;
    size_t n_labels = 3;
    VecSimIndexInfo info;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = n,
                                             .blockSize = blockSize}};
    VecSimIndex *index = VecSimIndex_New(&params);
    auto bf_index = reinterpret_cast<BruteForceIndex_Multi *>(index);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[dim];
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            a[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)a, i % n_labels);
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
    VecSimIndex_AddVector(index, (const void *)a, 0);
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
        for (size_t j = 0; j < dim; j++) {
            a[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)a, i % n_labels);
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
TEST_F(BruteForceMultiTest, brute_force_empty_index) {
    size_t dim = 4;
    size_t n = 20;
    size_t bs = 6;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
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
    float a[dim];
    for (size_t j = 0; j < dim; j++) {
        a[j] = (float)1.7;
    }

    VecSimIndex_AddVector(index, (const void *)a, 1);
    // Try to remove it.
    VecSimIndex_DeleteVector(index, 1);
    // The idToLabelMapping_size should change to be aligned with the vector size.
    size_t idToLabelMapping_size =
        reinterpret_cast<BruteForceIndex_Multi *>(index)->idToLabelMapping.size();

    ASSERT_EQ(idToLabelMapping_size, n - n % bs - bs);

    // Size equals 0.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove it again.
    // The idToLabelMapping_size should remain unchanged, as we are trying to delete a label that
    // doesn't exist.
    VecSimIndex_DeleteVector(index, 1);
    ASSERT_EQ(reinterpret_cast<BruteForceIndex_Multi *>(index)->idToLabelMapping.size(),
              idToLabelMapping_size);
    // Nor the size.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

// TEST_F(BruteForceMultiTest, brute_force_vector_search_test_ip) {
//     size_t dim = 4;
//     size_t n = 100;
//     size_t k = 11;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_IP,
//                                              .multi = true,
//                                              .initialCapacity = 200}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, (size_t)i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     float query[] = {50, 50, 50, 50};
//     std::set<size_t> expected_ids;
//     for (size_t i = n - 1; i > n - 1 - k; i--) {
//         expected_ids.insert(i);
//     }
//     auto verify_res = [&](size_t id, float score, size_t index) {
//         ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
//         expected_ids.erase(id);
//     };
//     runTopKSearchTest(index, query, k, verify_res);
//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_vector_search_test_l2) {
//     size_t n = 100;
//     size_t k = 11;
//     size_t dim = 4;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = 200}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, (int)i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     auto verify_res = [&](size_t id, float score, size_t index) {
//         size_t diff_id = ((int)(id - 50) > 0) ? (id - 50) : (50 - id);
//         ASSERT_EQ(diff_id, (index + 1) / 2);
//         ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
//     };
//     float query[] = {50, 50, 50, 50};
//     runTopKSearchTest(index, query, k, verify_res);

//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_vector_search_by_id_test) {
//     size_t n = 100;
//     size_t k = 11;
//     size_t dim = 4;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = 200}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, (int)i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     float query[] = {50, 50, 50, 50};
//     auto verify_res = [&](size_t id, float score, size_t index) { ASSERT_EQ(id, (index + 45)); };
//     runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_indexing_same_vector) {
//     size_t n = 100;
//     size_t k = 10;
//     size_t dim = 4;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = 200}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value).
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
//     float query[] = {4.9, 4.95, 5.05, 5.1};
//     auto verify_res = [&](size_t id, float score, size_t index) {
//         ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
//     };
//     runTopKSearchTest(index, query, k, verify_res);

//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_reindexing_same_vector) {
//     size_t n = 100;
//     size_t k = 10;
//     size_t dim = 4;
//     size_t initial_capacity = 200;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = initial_capacity}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value)
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
//     float query[] = {4.9, 4.95, 5.05, 5.1};
//     auto verify_res = [&](size_t id, float score, size_t index) {
//         ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
//     };
//     runTopKSearchTest(index, query, k, verify_res);

//     // Delete all vectors.
//     for (size_t i = 0; i < n; i++) {
//         VecSimIndex_DeleteVector(index, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

//     // The vector block should be removed.
//     ASSERT_EQ(reinterpret_cast<BruteForceIndex_Multi *>(index)->getVectorBlocks().size(), 0);

//     // id2label size should remain the same.
//     ASSERT_EQ(reinterpret_cast<BruteForceIndex_Multi *>(index)->idToLabelMapping.size(),
//               initial_capacity);

//     // Reinsert the same vectors under the same ids.
//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value)
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     // Run the same query again.
//     runTopKSearchTest(index, query, k, verify_res);

//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_reindexing_same_vector_different_id) {
//     size_t n = 100;
//     size_t k = 10;
//     size_t dim = 4;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = 200}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value)
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
//     float query[] = {4.9, 4.95, 5.05, 5.1};
//     auto verify_res = [&](size_t id, float score, size_t index) {
//         ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
//     };
//     runTopKSearchTest(index, query, k, verify_res);

//     for (size_t i = 0; i < n; i++) {
//         VecSimIndex_DeleteVector(index, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

//     // Reinsert the same vectors under different ids than before.
//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value)
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i + 10);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     // Run the same query again.
//     auto verify_res_different_id = [&](int id, float score, size_t index) {
//         ASSERT_TRUE(id >= 60 && id < 70 && score <= 1);
//     };
//     runTopKSearchTest(index, query, k, verify_res_different_id);

//     VecSimIndex_Free(index);
// }

TEST_F(BruteForceMultiTest, test_delete_swap_block) {
    size_t initial_capacity = 5; // idToLabelMapping initial size.
    size_t n = 6;
    size_t k = 5;
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
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = initial_capacity,
                                             .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    BruteForceIndex_Multi *bfm_index = reinterpret_cast<BruteForceIndex_Multi *>(index);

    // idToLabelMapping initial size equals n.
    ASSERT_EQ(bfm_index->idToLabelMapping.size(), initial_capacity);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i % n_labels);
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
    float *vector_data = block->getVector(1);
    for (size_t i = 0; i < dim; ++i) {
        ASSERT_EQ(*vector_data, 5);
        ++vector_data;
    }

    // float query[] = {0.0, 0.0};
    // auto verify_res = [&](size_t id, float score, size_t index) {
    //     if (index == 0) {
    //         ASSERT_EQ(id, index);
    //     } else {
    //         ASSERT_EQ(id, index + 1);
    //     }
    // };
    // runTopKSearchTest(index, query, k, verify_res);
    ASSERT_TRUE(k);
    VecSimIndex_Free(index);
}

// TEST_F(BruteForceMultiTest, sanity_reinsert_1280) {
//     size_t n = 5;
//     size_t d = 1280;
//     size_t k = 5;

//     VecSimParams params{
//         .algo = VecSimAlgo_BF,
//         .bfParams = BFParams{
//             .type = VecSimType_FLOAT32, .dim = d, .metric = VecSimMetric_L2, .multi = true,
//                                              .initialCapacity = n}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     auto *vectors = (float *)malloc(n * d * sizeof(float));

//     // Generate random vectors in every iteration and inert them under different ids.
//     for (size_t iter = 1; iter <= 3; iter++) {
//         for (size_t i = 0; i < n; i++) {
//             for (size_t j = 0; j < d; j++) {
//                 (vectors + i * d)[j] = (float)rand() / (float)(RAND_MAX) / 100;
//             }
//         }
//         auto expected_ids = std::set<size_t>();
//         for (size_t i = 0; i < n; i++) {
//             VecSimIndex_AddVector(index, (const void *)(vectors + i * d), i * iter);
//             expected_ids.insert(i * iter);
//         }
//         auto verify_res = [&](size_t id, float score, size_t index) {
//             ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
//             expected_ids.erase(id);
//         };

//         // Send arbitrary vector (the first) and search for top k. This should return all the
//         // vectors that were inserted in this iteration - verify their ids.
//         runTopKSearchTest(index, (const void *)vectors, k, verify_res);

//         // Remove vectors form current iteration.
//         for (size_t i = 0; i < n; i++) {
//             VecSimIndex_DeleteVector(index, i * iter);
//         }
//     }
//     free(vectors);
//     VecSimIndex_Free(index);
// }

TEST_F(BruteForceMultiTest, test_bf_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args.
    VecSimParams params = {.algo = VecSimAlgo_BF,
                           .bfParams = BFParams{.type = VecSimType_FLOAT32,
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
                          .bfParams = BFParams{.type = VecSimType_FLOAT32,
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

TEST_F(BruteForceMultiTest, test_basic_bf_info_iterator) {
    size_t n = 100;
    size_t d = 128;
    VecSimMetric metrics[3] = {VecSimMetric_Cosine, VecSimMetric_IP, VecSimMetric_L2};

    for (size_t i = 0; i < 3; i++) {
        // Build with default args.
        VecSimParams params{.algo = VecSimAlgo_BF,
                            .bfParams = BFParams{.type = VecSimType_FLOAT32,
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

TEST_F(BruteForceMultiTest, test_dynamic_bf_info_iterator) {
    size_t d = 128;
    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = d,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .blockSize = 1}};
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

    // // Set the index size artificially so that BATCHES mode will be selected by the heuristics.
    // reinterpret_cast<BruteForceIndex_Multi *>(index)->count = 1e4;
    // ASSERT_FALSE(VecSimIndex_PreferAdHocSearch(index, 7e3, 1, true));
    // info = VecSimIndex_Info(index);
    // infoIter = VecSimIndex_InfoIterator(index);
    // ASSERT_EQ(HYBRID_BATCHES, info.bfInfo.last_mode);
    // compareFlatIndexInfoToIterator(info, infoIter);
    // VecSimInfoIterator_Free(infoIter);

    // // Simulate the case where another call to the heuristics is done after realizing that
    // // the subset size is smaller, and change the policy as a result.
    // ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 1, 1, false));
    // info = VecSimIndex_Info(index);
    // infoIter = VecSimIndex_InfoIterator(index);
    // ASSERT_EQ(HYBRID_BATCHES_TO_ADHOC_BF, info.bfInfo.last_mode);
    // compareFlatIndexInfoToIterator(info, infoIter);
    // VecSimInfoIterator_Free(infoIter);

    VecSimIndex_Free(index);
}

// TEST_F(BruteForceMultiTest, brute_force_vector_search_test_ip_blocksize_1) {
//     size_t dim = 4;
//     size_t n = 100;
//     size_t k = 11;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_IP,
//                                              .multi = true,
//                                              .initialCapacity = 200,
//                                              .blockSize = 1}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     VecSimIndexInfo info = VecSimIndex_Info(index);
//     ASSERT_EQ(info.algo, VecSimAlgo_BF);
//     ASSERT_EQ(info.bfInfo.blockSize, 1);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     float query[] = {50, 50, 50, 50};
//     std::set<size_t> expected_ids;
//     for (size_t i = n - 1; i > n - 1 - k; i--) {
//         expected_ids.insert(i);
//     }
//     auto verify_res = [&](size_t id, float score, size_t index) {
//         ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
//         expected_ids.erase(id);
//     };
//     runTopKSearchTest(index, query, k, verify_res);
//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_vector_search_test_l2_blocksize_1) {
//     size_t dim = 4;
//     size_t n = 100;
//     size_t k = 11;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = 200,
//                                              .blockSize = 1}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     VecSimIndexInfo info = VecSimIndex_Info(index);
//     ASSERT_EQ(info.algo, VecSimAlgo_BF);
//     ASSERT_EQ(info.bfInfo.blockSize, 1);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     auto verify_res = [&](size_t id, float score, size_t index) {
//         size_t diff_id = ((int)(id - 50) > 0) ? (id - 50) : (50 - id);
//         ASSERT_EQ(diff_id, (index + 1) / 2);
//         ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
//     };
//     float query[] = {50, 50, 50, 50};
//     runTopKSearchTest(index, query, k, verify_res);

//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_search_empty_index) {
//     size_t dim = 4;
//     size_t n = 100;
//     size_t k = 11;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = 200}};
//     VecSimIndex *index = VecSimIndex_New(&params);
//     ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

//     float query[] = {50, 50, 50, 50};

//     // We do not expect any results.
//     VecSimQueryResult_List res =
//         VecSimIndex_TopKQuery(index, (const void *)query, k, NULL, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     VecSimQueryResult_Iterator *it = VecSimQueryResult_List_GetIterator(res);
//     ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
//     VecSimQueryResult_IteratorFree(it);
//     VecSimQueryResult_Free(res);

//     res = VecSimIndex_RangeQuery(index, (const void *)query, 1.0f, NULL, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     VecSimQueryResult_Free(res);

//     // Add some vectors and remove them all from index, so it will be empty again.
//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);
//     for (size_t i = 0; i < n; i++) {
//         VecSimIndex_DeleteVector(index, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

//     // Again - we do not expect any results.
//     res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     it = VecSimQueryResult_List_GetIterator(res);
//     ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
//     VecSimQueryResult_IteratorFree(it);
//     VecSimQueryResult_Free(res);

//     res = VecSimIndex_RangeQuery(index, (const void *)query, 1.0f, NULL, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     VecSimQueryResult_Free(res);

//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_test_inf_score) {
//     size_t n = 4;
//     size_t k = 4;
//     size_t dim = 2;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = n}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     // The 32 bits of "efgh" and "efgg", and the 32 bits of "abcd" and "abbd" will
//     // yield "inf" result when we calculate distance between the vectors.
//     VecSimIndex_AddVector(index, "abcdefgh", 1);
//     VecSimIndex_AddVector(index, "abcdefgg", 2);
//     VecSimIndex_AddVector(index, "aacdefgh", 3);
//     VecSimIndex_AddVector(index, "abbdefgh", 4);
//     ASSERT_EQ(VecSimIndex_IndexSize(index), 4);

//     auto verify_res = [&](size_t id, float score, size_t index) {
//         if (index == 0) {
//             ASSERT_EQ(1, id);
//         } else if (index == 1) {
//             ASSERT_EQ(3, id);
//         } else {
//             ASSERT_TRUE(id == 2 || id == 4);
//         }
//     };
//     runTopKSearchTest(index, "abcdefgh", k, verify_res);
//     VecSimIndex_Free(index);
// }

TEST_F(BruteForceMultiTest, brute_force_remove_vector_after_replacing_block) {
    size_t dim = 4;
    size_t bs = 2;
    size_t n = 6;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_L2,
                                             .multi = true,
                                             .initialCapacity = 200,
                                             .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Setting up vectors
    float f[n][dim];
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            f[i][j] = i;
        }
    }
    // Add 1 vector with label 1
    VecSimIndex_AddVector(index, (const void *)f[0], 1);

    // Add 3 vectors with label 3
    VecSimIndex_AddVector(index, (const void *)f[1], 3);
    VecSimIndex_AddVector(index, (const void *)f[2], 3);
    VecSimIndex_AddVector(index, (const void *)f[3], 3);

    // Add 2 vectors with label 2
    VecSimIndex_AddVector(index, (const void *)f[4], 2);
    VecSimIndex_AddVector(index, (const void *)f[5], 2);

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Delete label 3. the following drawing present the expected changes
    // [[1, 3], [3, 3], [2, 2]] -> [[1, 2], [3, 3], [2]] -> [[1, 2], [2, 3]] -> [[1, 2], [2]]
    // [[0, 1], [2, 3], [4, 5]] -> [[0, 5], [2, 3], [4]] -> [[0, 5], [4, 3]] -> [[0, 5], [4]]
    VecSimIndex_DeleteVector(index, 3);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 3);
    ASSERT_EQ(VecSimIndex_Info(index).bfInfo.indexLabelCount, 2);
    auto bf_index = reinterpret_cast<BruteForceIndex_Multi *>(index);
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

// TEST_F(BruteForceMultiTest, brute_force_batch_iterator) {
//     size_t dim = 4;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = 200,
//                                              .blockSize = 5}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     // run the test twice - for index of size 100, every iteration will run select-based search,
//     // as the number of results is 5, which is more than 0.1% of the index size. for index of
//     size
//     // 10000, we will run the heap-based search until we return 5000 results, and then switch to
//     // select-based search.
//     for (size_t n : {100, 10000}) {
//         for (size_t i = 0; i < n; i++) {
//             float f[dim];
//             for (size_t j = 0; j < dim; j++) {
//                 f[j] = (float)i;
//             }
//             VecSimIndex_AddVector(index, (const void *)f, i);
//         }
//         ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//         // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
//         float query[dim];
//         for (size_t j = 0; j < dim; j++) {
//             query[j] = (float)n;
//         }
//         VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
//         size_t iteration_num = 0;

//         // Get the 10 vectors whose ids are the maximal among those that hasn't been returned
//         yet,
//         // in every iteration. The order should be from the largest to the lowest id.
//         size_t n_res = 5;
//         while (VecSimBatchIterator_HasNext(batchIterator)) {
//             std::vector<size_t> expected_ids(n_res);
//             for (size_t i = 0; i < n_res; i++) {
//                 expected_ids[i] = (n - iteration_num * n_res - i - 1);
//             }
//             auto verify_res = [&](size_t id, float score, size_t index) {
//                 ASSERT_TRUE(expected_ids[index] == id);
//             };
//             runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
//             iteration_num++;
//         }
//         ASSERT_EQ(iteration_num, n / n_res);
//         VecSimBatchIterator_Free(batchIterator);
//     }
//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_batch_iterator_non_unique_scores) {
//     size_t dim = 4;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = 200,
//                                              .blockSize = 5}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     // Run the test twice - for index of size 100, every iteration will run select-based search,
//     // as the number of results is 5, which is more than 0.1% of the index size. for index of
//     size
//     // 10000, we will run the heap-based search until we return 5000 results, and then switch to
//     // select-based search.
//     for (size_t n : {100, 10000}) {
//         for (size_t i = 0; i < n; i++) {
//             float f[dim];
//             for (size_t j = 0; j < dim; j++) {
//                 f[j] = (float)(i / 10);
//             }
//             VecSimIndex_AddVector(index, (const void *)f, i);
//         }
//         ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//         // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
//         float query[dim];
//         for (size_t j = 0; j < dim; j++) {
//             query[j] = (float)n;
//         }
//         VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
//         size_t iteration_num = 0;

//         // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet,
//         in
//         // every iteration. there are n/10 groups of 10 different vectors with the same score.
//         size_t n_res = 5;
//         bool even_iteration = false;
//         std::set<size_t> expected_ids;
//         while (VecSimBatchIterator_HasNext(batchIterator)) {
//             // Insert the maximal 10 ids in every odd iteration.
//             if (!even_iteration) {
//                 for (size_t i = 1; i <= 2 * n_res; i++) {
//                     expected_ids.insert(n - iteration_num * n_res - i);
//                 }
//             }
//             auto verify_res = [&](size_t id, float score, size_t index) {
//                 ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
//                 expected_ids.erase(id);
//             };
//             runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
//             // Make sure that the expected ids set is empty after two iterations.
//             if (even_iteration) {
//                 ASSERT_TRUE(expected_ids.empty());
//             }
//             iteration_num++;
//             even_iteration = !even_iteration;
//         }
//         ASSERT_EQ(iteration_num, n / n_res);
//         VecSimBatchIterator_Free(batchIterator);
//     }
//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_batch_iterator_reset) {
//     size_t dim = 4;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = 100000,
//                                              .blockSize = 100000}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     size_t n = 10000;
//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
//     float query[dim];
//     for (size_t j = 0; j < dim; j++) {
//         query[j] = (float)n;
//     }
//     VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

//     // Get the 100 vectors whose ids are the maximal among those that hasn't been returned yet,
//     in
//     // every iteration. run this flow for 5 times, each time for 10 iteration, and reset the
//     // iterator.
//     size_t n_res = 100;
//     size_t total_iteration = 5;
//     size_t re_runs = 3;

//     for (size_t take = 0; take < re_runs; take++) {
//         size_t iteration_num = 0;
//         while (VecSimBatchIterator_HasNext(batchIterator)) {
//             std::set<size_t> expected_ids;
//             for (size_t i = 1; i <= n_res; i++) {
//                 expected_ids.insert(n - iteration_num * n_res - i);
//             }
//             auto verify_res = [&](size_t id, float score, size_t index) {
//                 ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
//                 expected_ids.erase(id);
//             };
//             runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
//             iteration_num++;
//             if (iteration_num == total_iteration) {
//                 break;
//             }
//         }
//         VecSimBatchIterator_Reset(batchIterator);
//     }
//     VecSimBatchIterator_Free(batchIterator);
//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, brute_force_batch_iterator_corner_cases) {
//     size_t dim = 4;
//     size_t n = 1000;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = n}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     // Query for (n,n,...,n) vector (recall that n is the largest id in te index)
//     float query[dim];
//     for (size_t j = 0; j < dim; j++) {
//         query[j] = (float)n;
//     }

//     // Create batch iterator for empty index.
//     VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
//     // Try to get more results even though there are no.
//     VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, 1, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     VecSimQueryResult_Free(res);
//     // Retry to get results.
//     res = VecSimBatchIterator_Next(batchIterator, 1, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     VecSimQueryResult_Free(res);
//     VecSimBatchIterator_Free(batchIterator);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     batchIterator = VecSimBatchIterator_New(index, query, nullptr);

//     // Ask for zero results.
//     res = VecSimBatchIterator_Next(batchIterator, 0, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     VecSimQueryResult_Free(res);

//     // Get all in first iteration, expect to use select search.
//     size_t n_res = n;
//     auto verify_res = [&](size_t id, float score, size_t index) {
//         ASSERT_TRUE(id == n - 1 - index);
//     };
//     runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
//     ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));

//     // Try to get more results even though there are no.
//     res = VecSimBatchIterator_Next(batchIterator, n_res, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     VecSimQueryResult_Free(res);

//     // Reset, and run in batches, but the final batch is partial.
//     VecSimBatchIterator_Reset(batchIterator);
//     res = VecSimBatchIterator_Next(batchIterator, n_res / 2, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), n / 2);
//     VecSimQueryResult_Free(res);
//     res = VecSimBatchIterator_Next(batchIterator, n_res / 2 + 1, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), n / 2);
//     VecSimQueryResult_Free(res);
//     ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));

//     VecSimBatchIterator_Free(batchIterator);
//     VecSimIndex_Free(index);
// }

TEST_F(BruteForceMultiTest, brute_get_distance) {
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
}

// TODO: check what is the preferred optimization when we have multi values
// TEST_F(BruteForceMultiTest, preferAdHocOptimization)

// TEST_F(BruteForceMultiTest, batchIteratorSwapIndices) {
//     size_t dim = 4;
//     size_t n = 10000;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = n}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     float close_vec[] = {1.0, 1.0, 1.0, 1.0};
//     float further_vec[] = {2.0, 2.0, 2.0, 2.0};
//     VecSimIndex_AddVector(index, (const void *)further_vec, 0);
//     VecSimIndex_AddVector(index, (const void *)close_vec, 1);
//     VecSimIndex_AddVector(index, (const void *)further_vec, 2);
//     VecSimIndex_AddVector(index, (const void *)close_vec, 3);
//     VecSimIndex_AddVector(index, (const void *)close_vec, 4);
//     VecSimIndex_AddVector(index, (const void *)close_vec, 5);
//     for (size_t i = 6; i < n; i++) {
//         float f[dim];
//         f[0] = f[1] = f[2] = f[3] = (float)i;
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     // Query for (1,1,1,1) vector.
//     float query[dim];
//     query[0] = query[1] = query[2] = query[3] = 1.0;
//     VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

//     // Get first batch - expect to get ids 1,3,4,5.
//     VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, 4, BY_ID);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 4);
//     VecSimQueryResult_Iterator *iterator = VecSimQueryResult_List_GetIterator(res);
//     int res_ind = 0;
//     size_t expected_res[] = {1, 3, 4, 5};
//     while (VecSimQueryResult_IteratorHasNext(iterator)) {
//         VecSimQueryResult *item = VecSimQueryResult_IteratorNext(iterator);
//         int id = (int)VecSimQueryResult_GetId(item);
//         ASSERT_EQ(expected_res[res_ind++], id);
//     }
//     VecSimQueryResult_IteratorFree(iterator);
//     VecSimQueryResult_Free(res);

//     // Get another batch - expect to get ids 0,2,6,7. Make sure that ids 0,2 swapped properly.
//     res = VecSimBatchIterator_Next(batchIterator, 4, BY_ID);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 4);
//     iterator = VecSimQueryResult_List_GetIterator(res);
//     res_ind = 0;
//     size_t expected_res_2[] = {0, 2, 6, 7};
//     while (VecSimQueryResult_IteratorHasNext(iterator)) {
//         VecSimQueryResult *item = VecSimQueryResult_IteratorNext(iterator);
//         int id = (int)VecSimQueryResult_GetId(item);
//         ASSERT_EQ(expected_res_2[res_ind++], id);
//     }
//     VecSimQueryResult_IteratorFree(iterator);
//     VecSimQueryResult_Free(res);

//     VecSimBatchIterator_Free(batchIterator);
//     VecSimIndex_Free(index);
// }

// TEST_F(BruteForceMultiTest, testCosine) {
//     size_t dim = 128;
//     size_t n = 100;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_Cosine,
//                                              .multi = true,
//                                              .initialCapacity = n}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 1; i <= n; i++) {
//         float f[dim];
//         f[0] = (float)i / n;
//         for (size_t j = 1; j < dim; j++) {
//             f[j] = 1.0f;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);
//     float query[dim];
//     for (size_t i = 0; i < dim; i++) {
//         query[i] = 1.0f;
//     }
//     auto verify_res = [&](size_t id, float score, size_t index) {
//         ASSERT_EQ(id, (n - index));
//         float first_coordinate = (float)id / n;
//         // By cosine definition: 1 - ((A \dot B) / (norm(A)*norm(B))), where A is the query
//         vector
//         // and B is the current result vector.
//         float expected_score =
//             1.0f -
//             ((first_coordinate + (float)dim - 1.0f) /
//              (sqrtf((float)dim) * sqrtf((float)(dim - 1) + first_coordinate *
//              first_coordinate)));
//         // Verify that abs difference between the actual and expected score is at most 1/10^6.
//         ASSERT_NEAR(score, expected_score, 1e-5);
//     };
//     runTopKSearchTest(index, query, 10, verify_res);

//     // Test with batch iterator.
//     VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
//     size_t iteration_num = 0;

//     // get the 10 vectors whose ids are the maximal among those that hasn't been returned yet,
//     // in every iteration. The order should be from the largest to the lowest id.
//     size_t n_res = 10;
//     while (VecSimBatchIterator_HasNext(batchIterator)) {
//         std::vector<size_t> expected_ids(n_res);
//         auto verify_res_batch = [&](size_t id, float score, size_t index) {
//             ASSERT_EQ(id, (n - n_res * iteration_num - index));
//             float first_coordinate = (float)id / n;
//             // By cosine definition: 1 - ((A \dot B) / (norm(A)*norm(B))), where A is the query
//             // vector and B is the current result vector.
//             float expected_score =
//                 1.0f - ((first_coordinate + (float)dim - 1.0f) /
//                         (sqrtf((float)dim) *
//                          sqrtf((float)(dim - 1) + first_coordinate * first_coordinate)));
//             // Verify that abs difference between the actual and expected score is at most
//             1/10^6. ASSERT_NEAR(score, expected_score, 1e-5);
//         };
//         runBatchIteratorSearchTest(batchIterator, n_res, verify_res_batch);
//         iteration_num++;
//     }
//     ASSERT_EQ(iteration_num, n / n_res);
//     VecSimBatchIterator_Free(batchIterator);
//     VecSimIndex_Free(index);
// }

TEST_F(BruteForceMultiTest, testSizeEstimation) {
    size_t dim = 128;
    size_t n = 0;
    size_t bs = DEFAULT_BLOCK_SIZE;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                             .dim = dim,
                                             .metric = VecSimMetric_Cosine,
                                             .multi = true,
                                             .initialCapacity = n,
                                             .blockSize = bs}};
    float vec[dim];
    for (size_t i = 0; i < dim; i++) {
        vec[i] = 1.0f;
    }

    size_t estimation = VecSimIndex_EstimateInitialSize(&params);
    VecSimIndex *index = VecSimIndex_New(&params);

    size_t actual = index->getAllocator()->getAllocationSize();
    ASSERT_EQ(estimation, actual);

    estimation = VecSimIndex_EstimateElementSize(&params) * bs;
    actual = VecSimIndex_AddVector(index, vec, 0);
    ASSERT_GE(estimation * 1.01, actual);
    ASSERT_LE(estimation * 0.99, actual);

    VecSimIndex_Free(index);
}

TEST_F(BruteForceMultiTest, testInitialSizeEstimationWithInitialCapacity) {
    size_t dim = 128;
    size_t n = 100;
    size_t bs = DEFAULT_BLOCK_SIZE;

    VecSimParams params{.algo = VecSimAlgo_BF,
                        .bfParams = BFParams{.type = VecSimType_FLOAT32,
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

// TEST_F(BruteForceMultiTest, testTimeoutReturn) {
//     size_t dim = 4;
//     float vec[] = {1.0f, 1.0f, 1.0f, 1.0f};
//     VecSimQueryResult_List rl;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = 1,
//                                              .blockSize = 5}};
//     VecSimIndex *index = VecSimIndex_New(&params);
//     VecSimIndex_AddVector(index, vec, 0);
//     VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out

//     // Checks return code on timeout - knn
//     rl = VecSimIndex_TopKQuery(index, vec, 1, NULL, BY_ID);
//     ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
//     ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
//     VecSimQueryResult_Free(rl);

//     // Check timeout again - range query
//     rl = VecSimIndex_RangeQuery(index, vec, 1, NULL, BY_ID);
//     ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
//     ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
//     VecSimQueryResult_Free(rl);

//     VecSimIndex_Free(index);
//     VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // cleanup
// }

// TEST_F(BruteForceMultiTest, testTimeoutReturn_batch_iterator) {
//     size_t dim = 4;
//     size_t n = 10;
//     VecSimQueryResult_List rl;

//     VecSimParams params{.algo = VecSimAlgo_BF,
//                         .bfParams = BFParams{.type = VecSimType_FLOAT32,
//                                              .dim = dim,
//                                              .metric = VecSimMetric_L2,
//                                              .multi = true,
//                                              .initialCapacity = n,
//                                              .blockSize = 5}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     float query[dim];
//     for (size_t j = 0; j < dim; j++) {
//         query[j] = (float)n;
//     }

//     // Fail on second batch (after calculation already completed)
//     VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

//     rl = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
//     ASSERT_EQ(rl.code, VecSim_QueryResult_OK);
//     ASSERT_NE(VecSimQueryResult_Len(rl), 0);
//     VecSimQueryResult_Free(rl);

//     VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out
//     rl = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
//     ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
//     ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
//     VecSimQueryResult_Free(rl);

//     VecSimBatchIterator_Free(batchIterator);

//     // Fail on first batch (while calculating)
//     // Timeout callback function already set to always time out
//     batchIterator = VecSimBatchIterator_New(index, query, nullptr);

//     rl = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
//     ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
//     ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
//     VecSimQueryResult_Free(rl);

//     VecSimBatchIterator_Free(batchIterator);

//     VecSimIndex_Free(index);
//     VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // cleanup
// }

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
