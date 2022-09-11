#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "test_utils.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"
#include <cmath>
#include <map>

class HNSWMultiTest : public ::testing::Test {
protected:
    HNSWMultiTest() {}

    ~HNSWMultiTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(HNSWMultiTest, vector_add_multiple_test) {
    size_t dim = 4;
    int rep = 5;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
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
    ASSERT_EQ((reinterpret_cast<HNSWIndex_Multi<float, float> *>(index))->indexLabelCount(), 1);

    // Deleting the label. All the vectors should be deleted.
    VecSimIndex_DeleteVector(index, 46);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);
    ASSERT_EQ((reinterpret_cast<HNSWIndex_Multi<float, float> *>(index))->indexLabelCount(), 0);

    VecSimIndex_Free(index);
}

// Test empty index edge cases.
TEST_F(HNSWMultiTest, empty_index) {
    size_t dim = 4;
    size_t n = 20;
    size_t bs = 6;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
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

    // Size equals 0.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove it again.
    VecSimIndex_DeleteVector(index, 1);

    // Nor the size.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, vector_search_test) {
    size_t dim = 4;
    size_t n = 1000;
    size_t n_labels = 100;
    size_t k = 11;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, (size_t)i % n_labels);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, n_labels);

    float query[] = {50, 50, 50, 50};
    std::set<size_t> expected_ids;
    for (size_t i = n - 1; i > n - 1 - k; i--) {
        expected_ids.insert(i);
    }
    auto verify_res = [&](size_t id, float score, size_t index) {
        size_t diff_id = ((int)(id - 50) > 0) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, search_more_then_there_is) {
    size_t dim = 4;
    size_t n = 5;
    size_t perLabel = 3;
    size_t n_labels = ceil((float)n / perLabel);
    size_t k = 3;
    // This test add 5 vectors under 2 labels, and then query for 3 results.
    // We want to make sure we get only 2 results back (because the results should have unique
    // labels), although the index contains 5 vectors.

    VecSimParams params{
        .algo = VecSimAlgo_HNSWLIB,
        .hnswParams = HNSWParams{
            .type = VecSimType_FLOAT32, .dim = dim, .metric = VecSimMetric_L2, .multi = true}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t x = 0; x < dim; x++) {
            f[x] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, (size_t)(i / perLabel));
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, n_labels);

    float query[] = {0, 0, 0, 0};
    VecSimQueryResult_List res = VecSimIndex_TopKQuery(index, query, k, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), n_labels);
    auto it = VecSimQueryResult_List_GetIterator(res);
    for (size_t i = 0; i < n_labels; i++) {
        auto el = VecSimQueryResult_IteratorNext(it);
        labelType element_label = VecSimQueryResult_GetId(el);
        ASSERT_EQ(element_label, i);
        auto ids = reinterpret_cast<HNSWIndex_Multi<float, float> *>(index)->label_lookup_.at(
            element_label);
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

TEST_F(HNSWMultiTest, indexing_same_vector) {
    size_t n = 100;
    size_t k = 10;
    size_t perLabel = 10;
    size_t dim = 4;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i);
        }
        VecSimIndex_AddVector(index, (const void *)f, (i / perLabel));
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[] = {0, 0, 0, 0};
    auto verify_res = [&](size_t id, float score, size_t index) { ASSERT_EQ(id, index); };
    runTopKSearchTest(index, query, k, verify_res);
    auto res = VecSimIndex_TopKQuery(index, query, k, nullptr, BY_SCORE);
    auto it = VecSimQueryResult_List_GetIterator(res);
    for (size_t i = 0; i < k; i++) {
        auto el = VecSimQueryResult_IteratorNext(it);
        labelType element_label = VecSimQueryResult_GetId(el);
        ASSERT_EQ(element_label, i);
        auto ids = reinterpret_cast<HNSWIndex_Multi<float, float> *>(index)->label_lookup_.at(
            element_label);
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

TEST_F(HNSWMultiTest, find_better_score) {
    size_t n = 100;
    size_t k = 10;
    size_t n_labels = 10;
    size_t dim = 4;
    size_t initial_capacity = 200;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = initial_capacity}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // Building the index. Each label gets 10 vectors with decreasing (by insertion order) element
    // value, so when we search, each vector is better then the previous one. Furthermore, each
    // label gets at least one better vector than the previous label and one with a score equals to
    // the best of the previous label, so the multimap holds at least two labels with the same
    // score.
    std::map<size_t, float> scores;
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        // For example, with n_labels == 10 and n == 100,
        // label 0 will get vector elements 18 -> 9 (aka (9 -> 0) + 9),
        // label 1 will get vector elements 17 -> 8 (aka (9 -> 0) + 8),
        // label 2 will get vector elements 16 -> 7 (aka (9 -> 0) + 7),
        // . . . . .
        // label 9 will get vector elements 9 -> 0 (aka (9 -> 0) + 0),
        // and so on, so each label has some common vectors with all the previous labels.
        size_t el = ((n - i - 1) % n_labels) + ((n - i - 1) / n_labels);
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)el;
        }
        VecSimIndex_AddVector(index, (const void *)f, i / n_labels);
        // This should be the best score for each label.
        if (i % n_labels == n_labels - 1) {
            // `el * el * dim` is the L2-squared value with the 0 vector.
            scores.emplace(i / n_labels, el * el * dim);
        }
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, n_labels);

    auto verify_res = [&](size_t id, float score, size_t index) {
        ASSERT_EQ(id, k - index - 1);
        ASSERT_FLOAT_EQ(score, scores[id]);
    };

    float query[] = {0, 0, 0, 0};
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, find_better_score_after_pop) {
    size_t n = 12;
    size_t n_labels = 3;
    size_t dim = 4;
    size_t initial_capacity = 200;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = initial_capacity}};
    VecSimIndex *index = VecSimIndex_New(&params);

    // Building the index. Each is better than the previous one.
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        size_t el = n - i;
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)el;
        }
        VecSimIndex_AddVector(index, (const void *)f, i % n_labels);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, n_labels);

    float query[] = {0, 0, 0, 0};
    auto verify_res = [&](size_t id, float score, size_t index) {
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

TEST_F(HNSWMultiTest, reindexing_same_vector_different_id) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;
    size_t perLabel = 3;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value)
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    // Add more vectors under the same labels. their scores should be worst.
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i / 10) + n;
        }
        for (size_t j = 0; j < perLabel - 1; j++) {
            VecSimIndex_AddVector(index, (const void *)f, i);
        }
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n * perLabel);

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

    // Reinsert the same vectors under different ids than before.
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i / 10); // i / 10 is in integer (take the "floor" value)
        }
        VecSimIndex_AddVector(index, (const void *)f, i + 10);
    }
    // Add more vectors under the same labels. their scores should be worst.
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)(i / 10) + n;
        }
        for (size_t j = 0; j < perLabel - 1; j++) {
            VecSimIndex_AddVector(index, (const void *)f, i + 10);
        }
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n * perLabel);

    // Run the same query again.
    auto verify_res_different_id = [&](int id, float score, size_t index) {
        ASSERT_TRUE(id >= 60 && id < 70 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res_different_id);

    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, test_hnsw_info) {
    // Build with default args.
    size_t n = 100;
    size_t d = 128;

    // Build with default args
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = d,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = n,
                                                 .M = 16,
                                                 .efConstruction = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_HNSWLIB);
    ASSERT_EQ(info.hnswInfo.dim, d);
    ASSERT_TRUE(info.hnswInfo.isMulti);
    // Default args.
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
                                       .multi = true,
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
    ASSERT_TRUE(info.hnswInfo.isMulti);
    // User args.
    ASSERT_EQ(info.hnswInfo.blockSize, bs);
    ASSERT_EQ(info.hnswInfo.efConstruction, 1000);
    ASSERT_EQ(info.hnswInfo.M, 200);
    ASSERT_EQ(info.hnswInfo.efRuntime, 500);
    ASSERT_EQ(info.hnswInfo.epsilon, 0.005);
    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, test_basic_hnsw_info_iterator) {
    size_t n = 100;
    size_t d = 128;
    VecSimMetric metrics[3] = {VecSimMetric_Cosine, VecSimMetric_IP, VecSimMetric_L2};

    for (size_t i = 0; i < 3; i++) {
        // Build with default args.
        VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                            .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                     .dim = d,
                                                     .metric = metrics[i],
                                                     .multi = true,
                                                     .initialCapacity = n}};
        VecSimIndex *index = VecSimIndex_New(&params);
        VecSimIndexInfo info = VecSimIndex_Info(index);
        VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
        compareHNSWIndexInfoToIterator(info, infoIter);
        VecSimInfoIterator_Free(infoIter);
        VecSimIndex_Free(index);
    }
}

TEST_F(HNSWMultiTest, test_dynamic_hnsw_info_iterator) {
    size_t n = 100;
    size_t d = 128;
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = d,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
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

    // Add vectors.
    VecSimIndex_AddVector(index, v, 0);
    VecSimIndex_AddVector(index, v, 0);
    VecSimIndex_AddVector(index, v, 1);
    VecSimIndex_AddVector(index, v, 1);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(4, info.hnswInfo.indexSize);
    ASSERT_EQ(2, info.hnswInfo.indexLabelCount);
    ASSERT_GE(1, info.hnswInfo.max_level);
    ASSERT_EQ(0, info.hnswInfo.entrypoint);
    compareHNSWIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Delete vectors.
    VecSimIndex_DeleteVector(index, 0);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(2, info.hnswInfo.indexSize);
    ASSERT_EQ(1, info.hnswInfo.indexLabelCount);
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
    vecsim_stl::vector<idType> vec(index->getAllocator());
    for (size_t i = 0; i < 1e5; i++) {
        reinterpret_cast<HNSWIndex_Multi<float, float> *>(index)->label_lookup_.emplace(i, vec);
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

// TODO: test different label count.
TEST_F(HNSWMultiTest, preferAdHocOptimization) {
    // Save the expected result for every combination that represent a different leaf in the tree.
    // map: [k, index_size, dim, M, r] -> res
    std::map<std::vector<float>, bool> combinations;
    combinations[{5, 1000, 1000, 5, 5, 0.5}] = true;
    combinations[{5, 6000, 6000, 5, 5, 0.1}] = true;
    combinations[{5, 6000, 6000, 5, 5, 0.2}] = false;
    combinations[{5, 6000, 6000, 60, 5, 0.5}] = false;
    combinations[{5, 6000, 6000, 60, 15, 0.5}] = true;
    combinations[{15, 6000, 6000, 50, 5, 0.5}] = true;
    combinations[{5, 700000, 700000, 60, 5, 0.05}] = true;
    combinations[{5, 800000, 800000, 60, 5, 0.05}] = false;
    combinations[{10, 800000, 800000, 60, 5, 0.01}] = true;
    combinations[{10, 800000, 800000, 60, 5, 0.05}] = false;
    combinations[{10, 800000, 800000, 60, 5, 0.1}] = false;
    combinations[{10, 60000, 60000, 100, 5, 0.1}] = true;
    combinations[{10, 80000, 80000, 100, 5, 0.1}] = false;
    combinations[{10, 60000, 60000, 100, 60, 0.1}] = true;
    combinations[{10, 60000, 60000, 100, 5, 0.3}] = false;
    combinations[{20, 60000, 60000, 100, 5, 0.1}] = true;
    combinations[{20, 60000, 60000, 100, 5, 0.2}] = false;
    combinations[{20, 60000, 60000, 100, 20, 0.1}] = true;
    combinations[{20, 350000, 350000, 100, 20, 0.1}] = true;
    combinations[{20, 350000, 350000, 100, 20, 0.2}] = false;

    for (auto &comb : combinations) {
        auto k = (size_t)comb.first[0];
        auto index_size = (size_t)comb.first[1];
        auto label_count = (size_t)comb.first[2];
        auto dim = (size_t)comb.first[3];
        auto M = (size_t)comb.first[4];
        auto r = comb.first[5];

        // Create index and check for the expected output of "prefer ad-hoc" heuristics.
        VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                            .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                     .dim = dim,
                                                     .metric = VecSimMetric_L2,
                                                     .multi = true,
                                                     .initialCapacity = index_size,
                                                     .M = M,
                                                     .efConstruction = 1,
                                                     .efRuntime = 1}};
        VecSimIndex *index = VecSimIndex_New(&params);

        // Set the index size artificially to be the required one.
        reinterpret_cast<HNSWIndex<float, float> *>(index)->cur_element_count = index_size;
        vecsim_stl::vector<idType> vec(index->getAllocator());
        for (size_t i = 0; i < label_count; i++) {
            reinterpret_cast<HNSWIndex_Multi<float, float> *>(index)->label_lookup_.emplace(i, vec);
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

TEST_F(HNSWMultiTest, vector_search_test_l2_blocksize_1) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;
    size_t perLabel = 4;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = 200,
                                                 .blockSize = 1}};
    VecSimIndex *index = VecSimIndex_New(&params);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.algo, VecSimAlgo_HNSWLIB);
    ASSERT_EQ(info.hnswInfo.blockSize, 1);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    // Add more vectors under the same labels. their scores should be worst.
    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i + n;
        }
        for (size_t j = 0; j < perLabel - 1; j++) {
            VecSimIndex_AddVector(index, (const void *)f, i);
        }
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n * perLabel);

    auto verify_res = [&](size_t id, float score, size_t index) {
        size_t diff_id = ((int)(id - 50) > 0) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    float query[] = {50, 50, 50, 50};
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, search_empty_index) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = 200}};
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

    // TODO: uncomment when support for HNSW Multi range is enabled
    // res = VecSimIndex_RangeQuery(index, (const void *)query, 1.0f, NULL, BY_SCORE);
    // ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    // VecSimQueryResult_Free(res);

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

    // Again - we do not expect any results.
    res = VecSimIndex_TopKQuery(index, (const void *)query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    it = VecSimQueryResult_List_GetIterator(res);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);

    // TODO: uncomment when support for HNSW Multi range is enabled
    // res = VecSimIndex_RangeQuery(index, (const void *)query, 1.0f, NULL, BY_SCORE);
    // ASSERT_EQ(VecSimQueryResult_Len(res), 0);
    // VecSimQueryResult_Free(res);

    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, remove_vector_after_replacing_block) {
    size_t dim = 4;
    size_t bs = 2;
    size_t n = 6;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
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
    ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, 2);
    auto hnsw_index = reinterpret_cast<HNSWIndex_Multi<float, float> *>(index);
    ASSERT_EQ(hnsw_index->getExternalLabel(0), 1);
    ASSERT_EQ(hnsw_index->getExternalLabel(1), 2);
    ASSERT_EQ(hnsw_index->getExternalLabel(2), 2);
    // checking the blob swaps.
    ASSERT_EQ(*(float *)(hnsw_index->getDataByInternalId(0)), 0);
    ASSERT_EQ(*(float *)(hnsw_index->getDataByInternalId(1)), 5);
    ASSERT_EQ(*(float *)(hnsw_index->getDataByInternalId(2)), 4);

    VecSimIndex_DeleteVector(index, 1);
    VecSimIndex_DeleteVector(index, 2);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, hnsw_get_distance) {
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
        .algo = VecSimAlgo_HNSWLIB,
        .hnswParams = HNSWParams{
            .type = VecSimType_FLOAT32, .dim = dim, .multi = true, .initialCapacity = 4}};

    for (size_t i = 0; i < numIndex; i++) {
        params.hnswParams.metric = (VecSimMetric)i;
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

TEST_F(HNSWMultiTest, testSizeEstimation) {
    size_t dim = 128;
    size_t n = 1000;
    size_t bs = DEFAULT_BLOCK_SIZE;
    size_t M = 32;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = n,
                                                 .blockSize = bs,
                                                 .M = M}};

    size_t estimation = VecSimIndex_EstimateInitialSize(&params);
    VecSimIndex *index = VecSimIndex_New(&params);

    size_t actual = index->getAllocator()->getAllocationSize();
    // labels_lookup hash table has additional memory, since STL implementation chooses "an
    // appropriate prime number" higher than n as the number of allocated buckets (for n=1000, 1031
    // buckets are created)
    estimation +=
        (reinterpret_cast<HNSWIndex_Multi<float, float> *>(index)->label_lookup_.bucket_count() -
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

TEST_F(HNSWMultiTest, testInitialSizeEstimation_No_InitialCapacity) {
    size_t dim = 128;
    size_t n = 0;
    size_t bs = DEFAULT_BLOCK_SIZE;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_Cosine,
                                                 .multi = true,
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

TEST_F(HNSWMultiTest, testTimeoutReturn) {
    size_t dim = 4;
    float vec[] = {1.0f, 1.0f, 1.0f, 1.0f};
    VecSimQueryResult_List rl;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
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
    // TODO: uncomment when support for HNSW Multi range is enabled
    // rl = VecSimIndex_RangeQuery(index, vec, 1, NULL, BY_ID);
    // ASSERT_EQ(rl.code, VecSim_QueryResult_TimedOut);
    // ASSERT_EQ(VecSimQueryResult_Len(rl), 0);
    // VecSimQueryResult_Free(rl);

    VecSimIndex_Free(index);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // cleanup
}

// TEST_F(HNSWMultiTest, testTimeoutReturn_batch_iterator) {
//     size_t dim = 4;
//     size_t n = 10;
//     VecSimQueryResult_List rl;

//     VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
//                         .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
//                                                  .dim = dim,
//                                                  .metric = VecSimMetric_L2,
//                                                  .multi = true,
//                                                  .initialCapacity = n,
//                                                  .blockSize = 5}};
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

/**** resizing cases ****/

// Add up to capacity.
TEST_F(HNSWMultiTest, resize_and_align_index) {
    size_t dim = 4;
    size_t n = 10;
    size_t bs = 3;
    size_t n_labels = 3;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
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
        VecSimIndex_AddVector(index, (const void *)a, i % n_labels);
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
TEST_F(HNSWMultiTest, resize_and_align_index_largeInitialCapacity) {
    size_t dim = 4;
    size_t n = 10;
    size_t bs = 3;
    size_t n_labels = 3;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = n,
                                                 .blockSize = bs}};
    VecSimIndex *index = VecSimIndex_New(&params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float a[dim];

    // add up to blocksize + 1 = 3 + 1 = 4
    for (size_t i = 0; i < bs; i++) {
        for (size_t j = 0; j < dim; j++) {
            a[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)a, i % n_labels);
    }
    VecSimIndex_AddVector(index, (const void *)a, n_labels);

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
TEST_F(HNSWMultiTest, resize_and_align_index_largerBlockSize) {
    size_t dim = 4;
    size_t n = 4;
    size_t bs = 6;
    size_t n_labels = 3;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
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
        VecSimIndex_AddVector(index, (const void *)a, i % n_labels);
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
TEST_F(HNSWMultiTest, emptyIndex) {
    size_t dim = 4;
    size_t n = 20;
    size_t bs = 6;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
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

TEST_F(HNSWMultiTest, hnsw_vector_search_by_id_test) {
    size_t n = 100;
    size_t dim = 4;
    size_t k = 11;
    size_t per_label = 5;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
                                                 .initialCapacity = 200,
                                                 .M = 16,
                                                 .efConstruction = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        float f[dim];
        for (size_t j = 0; j < dim; j++) {
            f[j] = (float)i;
        }
        VecSimIndex_AddVector(index, (const void *)f, i / per_label);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, n / per_label);

    float query[] = {50, 50, 50, 50};
    auto verify_res = [&](size_t id, float score, size_t index) { ASSERT_EQ(id, (index + 5)); };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, sanity_reinsert_1280) {
    size_t n_labels = 5;
    size_t per_label = 3;
    size_t d = 1280;
    size_t k = 5;

    size_t n = n_labels * per_label;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = d,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
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
            VecSimIndex_AddVector(index, (const void *)(vectors + i * d), (i % n_labels) * iter);
            expected_ids.insert((i % n_labels) * iter);
        }
        auto verify_res = [&](size_t id, float score, size_t index) {
            ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
            expected_ids.erase(id);
        };

        // Send arbitrary vector (the first) and search for top k. This should return all the
        // vectors that were inserted in this iteration - verify their ids.
        runTopKSearchTest(index, (const void *)vectors, k, verify_res);

        // Remove vectors form current iteration.
        for (size_t i = 0; i < n_labels; i++) {
            VecSimIndex_DeleteVector(index, i * iter);
        }
    }
    free(vectors);
    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, test_query_runtime_params_user_build_args) {
    size_t n = 100;
    size_t n_labels = 25;
    size_t per_label = 4;
    size_t d = 4;
    size_t M = 100;
    size_t efConstruction = 300;
    size_t efRuntime = 500;
    // Build with user args.
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = d,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
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
        VecSimIndex_AddVector(index, (const void *)f, i / per_label);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, n_labels);

    float query_element = (n_labels / 2) * per_label;
    auto verify_res = [&](size_t id, float score, size_t index) {
        size_t diff_id = (id < n_labels / 2) ? (n_labels / 2) - id : id - (n_labels / 2);
        float exp_elem = (id < n_labels / 2) ? id * per_label + per_label - 1 : id * per_label;
        float exp_score = d * (exp_elem - query_element) * (exp_elem - query_element);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, exp_score);
    };
    float query[] = {query_element, query_element, query_element, query_element};
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
    // TODO: uncomment when support for multi batch iterator is enabled
    // VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, &queryParams);
    // info = VecSimIndex_Info(index);
    // ASSERT_EQ(info.hnswInfo.efRuntime, 300);
    // // Run one batch for sanity.
    // runBatchIteratorSearchTest(batchIterator, k, verify_res);
    // // After releasing the batch iterator, ef_runtime should return to the default one.
    // VecSimBatchIterator_Free(batchIterator);
    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.hnswInfo.efRuntime, efRuntime);

    VecSimIndex_Free(index);
}

TEST_F(HNSWMultiTest, hnsw_delete_entry_point) {
    size_t n = 10000;
    size_t per_label = 5;
    size_t dim = 4;
    size_t M = 2;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_L2,
                                                 .multi = true,
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
        VecSimIndex_AddVector(index, vec, j / per_label);

    VecSimIndexInfo info = VecSimIndex_Info(index);

    while (info.hnswInfo.indexSize > 0) {
        ASSERT_NO_THROW(VecSimIndex_DeleteVector(index, info.hnswInfo.entrypoint));
        info = VecSimIndex_Info(index);
    }
    VecSimIndex_Free(index);
}

// TEST_F(HNSWMultiTest, hnsw_batch_iterator_basic) {
//     size_t dim = 4;
//     size_t M = 8;
//     size_t ef = 20;
//     size_t n_labels = 1000;
//     size_t perLabel = 5;

//     size_t n = n_labels * perLabel;

//     VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
//                         .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
//                                                  .dim = dim,
//                                                  .metric = VecSimMetric_L2,
//                                                  .multi = true,
//                                                  .initialCapacity = n,
//                                                  .M = M,
//                                                  .efConstruction = ef,
//                                                  .efRuntime = ef}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     // For every i, add the vector (i,i,i,i) under the label i.
//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i / perLabel);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);
//     ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, n_labels);

//     // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
//     float query[dim];
//     for (size_t j = 0; j < dim; j++) {
//         query[j] = (float)n;
//     }
//     VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
//     size_t iteration_num = 0;

//     // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet
//     // in every iteration. The results order should be sorted by their score (distance from the
//     // query vector), which means sorted from the largest id to the lowest.
//     size_t n_res = 5;
//     while (VecSimBatchIterator_HasNext(batchIterator)) {
//         std::vector<size_t> expected_ids(n_res);
//         for (size_t i = 0; i < n_res; i++) {
//             expected_ids[i] = (n - iteration_num * n_res - i - 1);
//         }
//         auto verify_res = [&](size_t id, float score, size_t index) {
//             ASSERT_EQ(expected_ids[index], id);
//         };
//         runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
//         iteration_num++;
//     }
//     ASSERT_EQ(iteration_num, n / n_res);
//     VecSimBatchIterator_Free(batchIterator);

//     VecSimIndex_Free(index);
// }

// TEST_F(HNSWMultiTest, hnsw_batch_iterator_reset) {
//     size_t dim = 4;
//     size_t M = 8;
//     size_t ef = 20;
//     size_t n_labels = 1000;
//     size_t perLabel = 5;

//     size_t n = n_labels * perLabel;

//     VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
//                         .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
//                                                  .dim = dim,
//                                                  .metric = VecSimMetric_L2,
//                                                  .multi = true,
//                                                  .initialCapacity = n,
//                                                  .M = M,
//                                                  .efConstruction = ef,
//                                                  .efRuntime = ef}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i % n_labels);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);
//     ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, n_labels);

//     // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
//     float query[dim];
//     for (size_t j = 0; j < dim; j++) {
//         query[j] = (float)n;
//     }
//     VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

//     // Get the 100 vectors whose ids are the maximal among those that hasn't been returned yet,
//     in
//     // every iteration. Run this flow for 3 times, and reset the iterator.
//     size_t n_res = 100;
//     size_t re_runs = 3;

//     for (size_t take = 0; take < re_runs; take++) {
//         size_t iteration_num = 0;
//         while (VecSimBatchIterator_HasNext(batchIterator)) {
//             std::vector<size_t> expected_ids(n_res);
//             for (size_t i = 0; i < n_res; i++) {
//                 expected_ids[i] = (n - iteration_num * n_res - i - 1);
//             }
//             auto verify_res = [&](size_t id, float score, size_t index) {
//                 ASSERT_EQ(expected_ids[index], id);
//             };
//             runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_SCORE);
//             iteration_num++;
//         }
//         ASSERT_EQ(iteration_num, n / n_res);
//         VecSimBatchIterator_Reset(batchIterator);
//     }
//     VecSimBatchIterator_Free(batchIterator);
//     VecSimIndex_Free(index);
// }

// TEST_F(HNSWMultiTest, hnsw_batch_iterator_batch_size_1) {
//     size_t dim = 4;
//     size_t M = 8;
//     size_t ef = 2;
//     size_t n_labels = 1000;
//     size_t perLabel = 5;

//     size_t n = n_labels * perLabel;

//     VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
//                         .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
//                                                  .dim = dim,
//                                                  .metric = VecSimMetric_L2,
//                                                  .multi = true,
//                                                  .initialCapacity = n,
//                                                  .M = M,
//                                                  .efConstruction = ef,
//                                                  .efRuntime = ef}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     float query[] = {(float)n, (float)n, (float)n, (float)n};

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         // Set labels to be different than the internal ids.
//         VecSimIndex_AddVector(index, (const void *)f, (n - i - 1)/perLabel);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);
//     ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, n_labels);

//     VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
//     size_t iteration_num = 0;
//     size_t n_res = 1, expected_n_res = 1;
//     while (VecSimBatchIterator_HasNext(batchIterator)) {
//         iteration_num++;
//         // Expect to get results in the reverse order of labels - which is the order of the
//         distance
//         // from the query vector. Get one result in every iteration.
//         auto verify_res = [&](size_t id, float score, size_t index) {
//             ASSERT_EQ(id, iteration_num);
//         };
//         runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_SCORE, expected_n_res);
//     }

//     ASSERT_EQ(iteration_num, n);
//     VecSimBatchIterator_Free(batchIterator);
//     VecSimIndex_Free(index);
// }

// TEST_F(HNSWMultiTest, hnsw_batch_iterator_advanced) {
//     size_t dim = 4;
//     size_t M = 8;
//     size_t ef = 1000;
//     size_t n_labels = 1000;
//     size_t perLabel = 5;

//     size_t n = n_labels * perLabel;

//     VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
//                         .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
//                                                  .dim = dim,
//                                                  .metric = VecSimMetric_L2,
//                                                  .multi = true,
//                                                  .initialCapacity = n,
//                                                  .M = M,
//                                                  .efConstruction = ef,
//                                                  .efRuntime = ef}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     float query[] = {(float)n, (float)n, (float)n, (float)n};
//     VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

//     // Try to get results even though there are no vectors in the index.
//     VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, 10, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     VecSimQueryResult_Free(res);
//     ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));

//     // Insert one vector and query again. The internal id will be 0.
//     VecSimIndex_AddVector(index, query, n_labels - 1);
//     VecSimBatchIterator_Reset(batchIterator);
//     res = VecSimBatchIterator_Next(batchIterator, 10, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 1);
//     VecSimQueryResult_Free(res);
//     ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));

//     for (size_t i = 1; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i%n_labels);
//     }
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);
//     ASSERT_EQ(VecSimIndex_Info(index).hnswInfo.indexLabelCount, n_labels);

//     // Reset the iterator after it was depleted.
//     VecSimBatchIterator_Reset(batchIterator);

//     // Try to get 0 results.
//     res = VecSimBatchIterator_Next(batchIterator, 0, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     VecSimQueryResult_Free(res);

//     // n_res does not divide into ef or vice versa - expect leftovers between the graph scans.
//     size_t n_res = 7;
//     size_t iteration_num = 0;

//     while (VecSimBatchIterator_HasNext(batchIterator)) {
//         iteration_num++;
//         std::vector<size_t> expected_ids;
//         // We ask to get the results sorted by ID in a specific batch (in ascending order), but
//         // in every iteration the ids should be lower than the previous one, according to the
//         // distance from the query.
//         for (size_t i = 1; i <= n_res; i++) {
//             expected_ids.push_back(n - iteration_num * n_res + i);
//         }
//         auto verify_res = [&](size_t id, float score, size_t index) {
//             ASSERT_EQ(expected_ids[index], id);
//         };
//         if (iteration_num <= n / n_res) {
//             runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_ID);
//         } else {
//             // In the last iteration there are n%iteration_num (=6) results left to return.
//             expected_ids.erase(expected_ids.begin()); // remove the first id
//             runBatchIteratorSearchTest(batchIterator, n_res, verify_res, BY_ID, n % n_res);
//         }
//     }
//     ASSERT_EQ(iteration_num, n / n_res + 1);
//     // Try to get more results even though there are no.
//     res = VecSimBatchIterator_Next(batchIterator, 1, BY_SCORE);
//     ASSERT_EQ(VecSimQueryResult_Len(res), 0);
//     VecSimQueryResult_Free(res);

//     VecSimBatchIterator_Free(batchIterator);
//     VecSimIndex_Free(index);
// }

// TEST_F(HNSWMultiTest, hnsw_multi_serialization_v1) {
//     size_t dim = 4;
//     size_t n = 1000;
//     size_t M = 8;
//     size_t ef = 10;

//     VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
//                         .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
//                                                  .dim = dim,
//                                                  .metric = VecSimMetric_L2,
//                                                  .initialCapacity = n,
//                                                  .blockSize = 1,
//                                                  .M = M,
//                                                  .efConstruction = ef,
//                                                  .efRuntime = ef}};
//     VecSimIndex *index = VecSimIndex_New(&params);

//     auto serializer = HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float> *>(index));

//     auto file_name = std::string(getenv("ROOT")) + "/tests/unit/data/1k-d4-L2-M8-ef_c10.hnsw_v1";
//     // Save and load an empty index.
//     serializer.saveIndex(file_name);
//     serializer.loadIndex(file_name);
//     auto res = serializer.checkIntegrity();
//     ASSERT_TRUE(res.valid_state);

//     for (size_t i = 0; i < n; i++) {
//         float f[dim];
//         for (size_t j = 0; j < dim; j++) {
//             f[j] = (float)i;
//         }
//         VecSimIndex_AddVector(index, (const void *)f, i);
//     }
//     // Get index info and copy it, so it will be available after the index is deleted.
//     VecSimIndexInfo info = VecSimIndex_Info(index);

//     // Persist index with the serializer, and delete it.
//     serializer.saveIndex(file_name);
//     VecSimIndex_Free(index);

//     // Create new index, set it into the serializer and extract the data to it.
//     auto new_index = VecSimIndex_New(&params);
//     ASSERT_EQ(VecSimIndex_IndexSize(new_index), 0);

//     serializer.reset(reinterpret_cast<HNSWIndex<float, float> *>(new_index));
//     serializer.loadIndex(file_name);

//     // Validate that the new loaded index has the same meta-data as the original.
//     VecSimIndexInfo new_info = VecSimIndex_Info(new_index);
//     ASSERT_EQ(info.algo, new_info.algo);
//     ASSERT_EQ(info.hnswInfo.M, new_info.hnswInfo.M);
//     ASSERT_EQ(info.hnswInfo.efConstruction, new_info.hnswInfo.efConstruction);
//     ASSERT_EQ(info.hnswInfo.efRuntime, new_info.hnswInfo.efRuntime);
//     ASSERT_EQ(info.hnswInfo.indexSize, new_info.hnswInfo.indexSize);
//     ASSERT_EQ(info.hnswInfo.max_level, new_info.hnswInfo.max_level);
//     ASSERT_EQ(info.hnswInfo.entrypoint, new_info.hnswInfo.entrypoint);
//     ASSERT_EQ(info.hnswInfo.metric, new_info.hnswInfo.metric);
//     ASSERT_EQ(info.hnswInfo.type, new_info.hnswInfo.type);
//     ASSERT_EQ(info.hnswInfo.dim, new_info.hnswInfo.dim);

//     res = serializer.checkIntegrity();
//     ASSERT_TRUE(res.valid_state);

//     // Add 1000 random vectors, override the existing ones to trigger deletions.
//     std::vector<float> data((n + 1) * dim);
//     std::mt19937 rng;
//     rng.seed(47);
//     std::uniform_real_distribution<> distrib;
//     for (size_t i = 0; i < (n + 1) * dim; ++i) {
//         data[i] = (float)distrib(rng);
//     }
//     for (size_t i = 0; i < n + 1; ++i) {
//         VecSimIndex_AddVector(new_index, data.data() + dim * i, i);
//     }

//     // Delete arbitrary vector (trigger removal of a block).
//     VecSimIndex_DeleteVector(new_index, (size_t)(distrib(rng) * (n + 1)));

//     HNSWIndex<float, float> *hnswNewIndex = reinterpret_cast<HNSWIndex<float, float>
//     *>(new_index); ASSERT_EQ(hnswNewIndex->getIndexCapacity(), n);

//     // Persist index, delete it from memory and restore.
//     serializer.saveIndex(file_name);
//     VecSimIndex_Free(new_index);

//     params.hnswParams.initialCapacity = n / 2; // to ensure that we resize in load time.
//     auto restored_index = VecSimIndex_New(&params);
//     ASSERT_EQ(VecSimIndex_IndexSize(restored_index), 0);

//     serializer.reset(reinterpret_cast<HNSWIndex<float, float> *>(restored_index));
//     serializer.loadIndex(file_name);
//     ASSERT_EQ(VecSimIndex_IndexSize(restored_index), n);
//     res = serializer.checkIntegrity();
//     ASSERT_TRUE(res.valid_state);

//     // Clean-up.
//     remove(file_name.c_str());
//     VecSimIndex_Free(restored_index);
//     serializer.reset();
// }

TEST_F(HNSWMultiTest, testCosine) {
    size_t dim = 4;
    size_t n = 100;

    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = dim,
                                                 .metric = VecSimMetric_Cosine,
                                                 .multi = true,
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
    // Add more worst vector for each label
    for (size_t i = 1; i <= n; i++) {
        float f[dim];
        f[0] = (float)i + n;
        for (size_t j = 1; j < dim; j++) {
            f[j] = 1.0f;
        }
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 2 * n);
    float query[dim];
    for (size_t i = 0; i < dim; i++) {
        query[i] = 1.0f;
    }
    auto verify_res = [&](size_t id, float score, size_t index) {
        ASSERT_EQ(id, (n - index));
        float first_coordinate = (float)id / n;
        // By cosine definition: 1 - ((A \dot B) / (norm(A)*norm(B))), where A is the query vector
        // and B is the current result vector.
        float expected_score =
            1.0f -
            ((first_coordinate + (float)dim - 1.0f) /
             (sqrtf((float)dim) * sqrtf((float)(dim - 1) + first_coordinate * first_coordinate)));
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
        auto verify_res_batch = [&](size_t id, float score, size_t index) {
            ASSERT_EQ(id, (n - n_res * iteration_num - index));
            float first_coordinate = (float)id / n;
            // By cosine definition: 1 - ((A \dot B) / (norm(A)*norm(B))), where A is the query
            // vector and B is the current result vector.
            float expected_score =
                1.0f - ((first_coordinate + (float)dim - 1.0f) /
                        (sqrtf((float)dim) *
                         sqrtf((float)(dim - 1) + first_coordinate * first_coordinate)));
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

// TEST_F(HNSWMultiTest, rangeQuery) {
//     size_t n = 5000;
//     size_t dim = 4;

//     VecSimParams params{
//         .algo = VecSimAlgo_HNSWLIB,
//         .hnswParams = HNSWParams{
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

//     size_t pivot_id = n / 2; // the id to return vectors around it.
//     float query[] = {(float)pivot_id, (float)pivot_id, (float)pivot_id, (float)pivot_id};

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

//     // Rerun with a given query params. This high epsilon value will cause the range search main
//     // loop to break since we insert a candidate whose distance is within the dynamic range
//     // boundaries at the beginning of the search, but when this candidate is popped out from the
//     // queue, it's no longer within the dynamic range boundaries.
//     auto query_params = VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.epsilon
//     = 1.0}}; runRangeQueryTest(index, query, radius, verify_res_by_score, expected_num_results,
//     BY_SCORE,
//                       &query_params);

//     // Get results by id.
//     auto verify_res_by_id = [&](size_t id, float score, size_t index) {
//         ASSERT_EQ(id, pivot_id - expected_num_results / 2 + index);
//         ASSERT_EQ(score, dim * pow(std::abs(int(id - pivot_id)), 2));
//     };
//     runRangeQueryTest(index, query, radius, verify_res_by_id, expected_num_results);

//     VecSimIndex_Free(index);
// }

// TEST_F(HNSWMultiTest, rangeQueryCosine) {
//     size_t n = 800;
//     size_t dim = 4;

//     VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
//                         .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
//                                                  .dim = dim,
//                                                  .metric = VecSimMetric_Cosine,
//                                                  .blockSize = n / 2}};
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
