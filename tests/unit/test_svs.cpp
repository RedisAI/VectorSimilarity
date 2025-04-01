#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "unit_test_utils.h"
#include "VecSim/algorithms/svs/svs.h"
#include "cpu_features_macros.h"
#include <cmath>

template <typename index_type_t>
class SVSTest : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;

protected:
    VecSimIndex *CreateNewIndex(SVSParams &params) {
        params.quantBits = index_type_t::get_quant_bits();
        params.type = index_type_t::get_index_type();
        VecSimParams index_params = CreateParams(params);
        return VecSimIndex_New(&index_params);
    }

    SVSIndexBase *CastToSVS(VecSimIndex *index) {
        auto indexBase = dynamic_cast<SVSIndexBase *>(index);
        assert(indexBase != nullptr);
        return indexBase;
    }
};

// TEST_DATA_T and TEST_DIST_T are defined in test_utils.h

template <VecSimType type, typename DataType, VecSimSvsQuantBits quantBits>
struct SVSIndexType {
    static constexpr VecSimType get_index_type() { return type; }
    static constexpr VecSimSvsQuantBits get_quant_bits() { return quantBits; }
    typedef DataType data_t;
};

// clang-format off
using SVSDataTypeSet = ::testing::Types<SVSIndexType<VecSimType_FLOAT32, float, VecSimSvsQuant_NONE>
#if HAVE_SVS_LVQ
                                       ,SVSIndexType<VecSimType_FLOAT32, float, VecSimSvsQuant_8>
#endif
                                        >;
// clang-format on

TYPED_TEST_SUITE(SVSTest, SVSDataTypeSet);

TYPED_TEST(SVSTest, svs_vector_add_test) {

    size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_IP,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    EXPECT_EQ(VecSimIndex_IndexSize(index), 0);

    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1);

    EXPECT_EQ(VecSimIndex_IndexSize(index), 1);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, svs_vector_update_test) {
    size_t dim = 4;
    size_t n = 1;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_IP,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    auto *svs_index = this->CastToSVS(index);

    EXPECT_EQ(VecSimIndex_IndexSize(index), 0);

    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1);

    EXPECT_EQ(VecSimIndex_IndexSize(index), 1);

    // Prepare new vector data and call addVector with the same id, different data.
    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1, 2.0);

    // Index size shouldn't change.
    EXPECT_EQ(VecSimIndex_IndexSize(index), 1);

    // Delete the last vector.
    VecSimIndex_DeleteVector(index, 1);
    EXPECT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, svs_vector_search_by_id_test) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[] = {50, 50, 50, 50};
    auto verify_res = [&](size_t id, double score, size_t index) { EXPECT_EQ(id, (index + 45)); };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, svs_bulk_vectors_add_delete_test) {
    size_t n = 1000;
    size_t k = 11;
    const size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    auto svs_index = this->CastToSVS(index); // CAST_TO_SVS(index, svs::distance::DistanceL2);

    std::vector<TEST_DATA_T[dim]> v(n);
    for (size_t i = 0; i < n; i++) {
        GenerateVector<TEST_DATA_T>(v[i], dim, i);
    }

    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    svs_index->addVectors(v.data(), ids.data(), n);

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[] = {50, 50, 50, 50};
    auto verify_res = [&](size_t id, double score, size_t index) { EXPECT_EQ(id, (index + 45)); };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    // Delete almost all vectors
    const size_t keep_num = 1;
    ASSERT_EQ(svs_index->deleteVectors(ids.data(), n - keep_num), n - keep_num);
    ASSERT_EQ(VecSimIndex_IndexSize(index), keep_num);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, svs_get_distance) {
    size_t n = 4;
    size_t dim = 2;
    size_t numIndex = 3;
    VecSimIndex *index[numIndex];
    std::vector<double> distances;

    TEST_DATA_T v1[] = {M_PI, M_PI};
    TEST_DATA_T v2[] = {M_E, M_E};
    TEST_DATA_T v3[] = {M_PI, M_E};
    TEST_DATA_T v4[] = {M_SQRT2, -M_SQRT2};

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

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

    ASSERT_TYPE_EQ(norm[0], TEST_DATA_T(1.0 / sqrt(2.0)));
    ASSERT_TYPE_EQ(norm[1], TEST_DATA_T(1.0 / sqrt(2.0)));
    double dist;

    auto qbits = TypeParam::get_quant_bits();
    double relative_err = qbits ? 1e-2 : 1.e-5;

    // VecSimMetric_L2
    distances = {0, 0.3583844006061554, 0.1791922003030777, 23.739208221435547};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom_Unsafe(index[VecSimMetric_L2], i + 1, query);
        EXPECT_NEAR(dist, distances[i], std::abs(distances[i] * relative_err));
    }

    // VecSimMetric_IP
    distances = {-18.73921012878418, -16.0794677734375, -17.409339904785156, 1};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom_Unsafe(index[VecSimMetric_IP], i + 1, query);
        EXPECT_NEAR(dist, distances[i], std::abs(distances[i] * relative_err));
    }

    // VecSimMetric_Cosine
    distances = {5.9604644775390625e-08, 5.9604644775390625e-08, 0.0025991201400756836, 1};
    for (size_t i = 0; i < n; i++) {
        dist = VecSimIndex_GetDistanceFrom_Unsafe(index[VecSimMetric_Cosine], i + 1, norm);
        EXPECT_NEAR(dist, distances[i], std::abs(distances[i] * relative_err));
    }

    // Bad values
    dist = VecSimIndex_GetDistanceFrom_Unsafe(index[VecSimMetric_Cosine], 0, norm);
    EXPECT_TRUE(std::isnan(dist));
    dist = VecSimIndex_GetDistanceFrom_Unsafe(index[VecSimMetric_L2], 46, query);
    EXPECT_TRUE(std::isnan(dist));

    // Clean-up.
    for (size_t i = 0; i < numIndex; i++) {
        VecSimIndex_Free(index[i]);
    }
}

TYPED_TEST(SVSTest, svs_indexing_same_vector) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i,
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

TYPED_TEST(SVSTest, svs_reindexing_same_vector) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    // SVSIndex<TEST_DATA_T, TEST_DIST_T> *bf_index = this->CastToBF(index);

    for (size_t i = 0; i < n; i++) {
        // i / 10 is in integer (take the "floor" value).
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i / 10);
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

    // Reinsert the same vectors under the same ids.
    for (size_t i = 0; i < n; i++) {
        // i / 10 is in integer (take the "floor value).
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run the same query again.
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, svs_reindexing_same_vector_different_id) {
    size_t n = 100;
    size_t k = 10;
    size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i,
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
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i + 10,
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

TYPED_TEST(SVSTest, svs_batch_iterator) {
    size_t dim = 4;

    // run the test twice - for index of size 100, every iteration will run select-based search,
    // as the number of results is 5, which is more than 0.1% of the index size. for index of size
    // 10000, we will run the heap-based search until we return 5000 results, and then switch to
    // select-based search.
    for (size_t n : {100, 1000}) {
        SVSParams params = {
            .dim = dim,
            .metric = VecSimMetric_L2,
            /* SVS-Vamana specifics */
            .alpha = 1.2,
            .graph_max_degree = 64,
            .construction_window_size = 20,
            .max_candidate_pool_size = 1024,
            .prune_to = 60,
            .use_search_history = VecSimOption_ENABLE,
        };

        VecSimIndex *index = this->CreateNewIndex(params);
        for (size_t i = 0; i < n; i++) {
            GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), n);

        // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
        size_t iteration_num = 0;

        // Get the 10 vectors whose ids are the maximal among those that hasn't been returned yet,
        // in every iteration. The order should be from the largest to the lowest id.
        size_t n_res = 20;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            std::vector<size_t> expected_ids(n_res);
            for (size_t i = 0; i < n_res; i++) {
                expected_ids[i] = (n - iteration_num * n_res - i - 1);
            }
            auto verify_res = [&](size_t id, double score, size_t index) {
                ASSERT_EQ(id, expected_ids[index]);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num, n / n_res);
        VecSimBatchIterator_Free(batchIterator);

        VecSimIndex_Free(index);
    }
}

TYPED_TEST(SVSTest, svs_batch_iterator_non_unique_scores) {
    size_t dim = 4;

    // Run the test twice - for index of size 100, every iteration will run select-based search,
    // as the number of results is 5, which is more than 0.1% of the index size. for index of size
    // 10000, we will run the heap-based search until we return 5000 results, and then switch to
    // select-based search.
    for (size_t n : {100, 10000}) {
        SVSParams params = {
            .dim = dim,
            .metric = VecSimMetric_L2,
            /* SVS-Vamana specifics */
            .alpha = 1.2,
            .graph_max_degree = 64,
            .construction_window_size = 20,
            .max_candidate_pool_size = 1024,
            .prune_to = 60,
            .use_search_history = VecSimOption_ENABLE,
        };

        VecSimIndex *index = this->CreateNewIndex(params);

        for (size_t i = 0; i < n; i++) {
            GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i / 10);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), n);

        // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
        size_t iteration_num = 0;

        // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet, in
        // every iteration. there are n/10 groups of 10 different vectors with the same score.
        size_t n_res = 5;
        bool even_iteration = false;
        std::set<size_t> expected_ids;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            // Insert the maximal 10 ids in every odd iteration.
            if (!even_iteration) {
                for (size_t i = 1; i <= 2 * n_res; i++) {
                    expected_ids.insert(n - iteration_num * n_res - i);
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
        ASSERT_EQ(iteration_num, n / n_res);
        VecSimBatchIterator_Free(batchIterator);

        VecSimIndex_Free(index);
    }
}

TYPED_TEST(SVSTest, svs_batch_iterator_reset) {
    size_t dim = 4;
    size_t n = 10000;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, n);
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    // Get the 100 vectors whose ids are the maximal among those that hasn't been returned yet, in
    // every iteration. run this flow for 5 times, each time for 10 iteration, and reset the
    // iterator.
    size_t n_res = 100;
    size_t total_iteration = 5;
    size_t re_runs = 3;

    for (size_t take = 0; take < re_runs; take++) {
        size_t iteration_num = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            std::set<size_t> expected_ids;
            for (size_t i = 1; i <= n_res * 2; i++) {
                expected_ids.insert(n - iteration_num * n_res - i);
            }
            auto verify_res = [&](size_t id, double score, size_t index) {
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

TYPED_TEST(SVSTest, svs_batch_iterator_corner_cases) {
    size_t dim = 4;
    size_t n = 1000;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, n);

    // Create batch iterator for empty index.
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
    // Try to get more results even though there are no.
    VecSimQueryReply *res = VecSimBatchIterator_Next(batchIterator, 1, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    VecSimQueryReply_Free(res);
    // Retry to get results.
    VecSimBatchIterator_Reset(batchIterator);
    res = VecSimBatchIterator_Next(batchIterator, 1, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    VecSimQueryReply_Free(res);

    // Check if depleted
    ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));
    VecSimBatchIterator_Free(batchIterator);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    // Ask for zero results.
    res = VecSimBatchIterator_Next(batchIterator, 0, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    VecSimQueryReply_Free(res);

    // Get all in first iteration, expect to use select search.
    size_t n_res = n;
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(id == n - 1 - index);
    };
    runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
    ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));

    // Try to get more results even though there are no.
    res = VecSimBatchIterator_Next(batchIterator, n_res, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    VecSimQueryReply_Free(res);

    // Reset, and run in batches, but the final batch is partial.
    VecSimBatchIterator_Reset(batchIterator);
    res = VecSimBatchIterator_Next(batchIterator, n_res / 2, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), n / 2);
    VecSimQueryReply_Free(res);
    res = VecSimBatchIterator_Next(batchIterator, n_res / 2 + 1, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), n / 2);
    VecSimQueryReply_Free(res);
    ASSERT_FALSE(VecSimBatchIterator_HasNext(batchIterator));

    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

// Add up to capacity.
TYPED_TEST(SVSTest, resizeIndex) {
    size_t dim = 4;
    size_t n = 10;
    size_t bs = 4;

    SVSParams params = {.dim = dim, .metric = VecSimMetric_L2, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);

    // Add up to n.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }

    // Initial capacity is rounded up to the block size.
    size_t extra_cap = n % bs == 0 ? 0 : bs - n % bs;
    if constexpr (TypeParam::get_quant_bits() > 0) {
        // LVQDataset does not provide a capacity method
        extra_cap = 0;
    }
    // The size (+extra) and the capacity should be equal.
    ASSERT_EQ(index->indexCapacity(), VecSimIndex_IndexSize(index) + extra_cap);
    // The capacity shouldn't be changed.
    ASSERT_EQ(index->indexCapacity(), n + extra_cap);

    VecSimIndex_Free(index);
}

// Test empty index edge cases.
TYPED_TEST(SVSTest, svs_empty_index) {
    size_t dim = 4;
    size_t n = 20;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove from an empty index - should fail because label doesn't exist.
    VecSimIndex_DeleteVector(index, 0);

    // Add one vector.
    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1, 1.7);

    // Size equals 1.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

    // Try to remove it.
    VecSimIndex_DeleteVector(index, 1);

    // Size equals 0.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // The expected capacity should be 0 for empty index.
    ASSERT_EQ(index->indexCapacity(), 0);

    // Try to remove it again.
    VecSimIndex_DeleteVector(index, 1);
    // Nor the size.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, test_delete_vector) {
    size_t k = 5;
    size_t dim = 2;
    size_t block_size = 3;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = block_size,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    // Delete from empty index
    ASSERT_EQ(VecSimIndex_DeleteVector(index, 111), 0);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    size_t n = 6;
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Here the shift should happen.
    VecSimIndex_DeleteVector(index, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n - 1);

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

TYPED_TEST(SVSTest, sanity_reinsert_1280) {
    size_t n = 5;
    size_t d = 1280;
    size_t k = 5;

    SVSParams params = {
        .dim = d,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

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

TYPED_TEST(SVSTest, test_svs_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args.

    SVSParams params = {
        .dim = d,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVS);
    ASSERT_EQ(info.commonInfo.basicInfo.dim, d);
    ASSERT_FALSE(info.commonInfo.basicInfo.isMulti);
    // Default args.
    ASSERT_EQ(info.commonInfo.indexSize, 0);
    VecSimIndex_Free(index);

    d = 1280;
    params.dim = d;
    params.blockSize = 1;

    index = this->CreateNewIndex(params);

    info = VecSimIndex_Info(index);
    ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVS);
    ASSERT_EQ(info.commonInfo.basicInfo.dim, d);
    ASSERT_FALSE(info.commonInfo.basicInfo.isMulti);
    ASSERT_FALSE(info.commonInfo.basicInfo.isTiered);

    // User args.
    ASSERT_EQ(info.commonInfo.basicInfo.blockSize, 1);
    ASSERT_EQ(info.commonInfo.indexSize, 0);

    // Validate that Static info returns the right restricted info as well.
    VecSimIndexBasicInfo s_info = VecSimIndex_BasicInfo(index);
    ASSERT_EQ(info.commonInfo.basicInfo.algo, s_info.algo);
    ASSERT_EQ(info.commonInfo.basicInfo.dim, s_info.dim);
    ASSERT_EQ(info.commonInfo.basicInfo.blockSize, s_info.blockSize);
    ASSERT_EQ(info.commonInfo.basicInfo.type, s_info.type);
    ASSERT_EQ(info.commonInfo.basicInfo.isMulti, s_info.isMulti);
    ASSERT_EQ(info.commonInfo.basicInfo.type, s_info.type);
    ASSERT_EQ(info.commonInfo.basicInfo.isTiered, s_info.isTiered);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, test_basic_svs_info_iterator) {
    size_t n = 100;
    size_t d = 128;
    VecSimMetric metrics[3] = {VecSimMetric_Cosine, VecSimMetric_IP, VecSimMetric_L2};

    for (size_t i = 0; i < 3; i++) {

        // Build with default args.
        SVSParams params = {
            .dim = d,
            .metric = metrics[i],
            /* SVS-Vamana specifics */
            .alpha = 1.2,
            .graph_max_degree = 64,
            .construction_window_size = 20,
            .max_candidate_pool_size = 1024,
            .prune_to = 60,
            .use_search_history = VecSimOption_ENABLE,
        };

        VecSimIndex *index = this->CreateNewIndex(params);

        VecSimIndexInfo info = VecSimIndex_Info(index);
        VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
        compareFlatIndexInfoToIterator(info, infoIter);
        VecSimInfoIterator_Free(infoIter);
        VecSimIndex_Free(index);
    }
}

TYPED_TEST(SVSTest, test_dynamic_svs_info_iterator) {
    size_t d = 128;

    SVSParams params = {
        .dim = d,
        .metric = VecSimMetric_L2,
        .blockSize = 1,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(1, info.commonInfo.basicInfo.blockSize);
    ASSERT_EQ(0, info.commonInfo.indexSize);
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
    ASSERT_EQ(1, info.commonInfo.indexSize);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Delete vector.
    VecSimIndex_DeleteVector(index, 0);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(0, info.commonInfo.indexSize);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Perform (or simulate) Search in all modes.
    VecSimIndex_AddVector(index, v, 0);
    auto res = VecSimIndex_TopKQuery(index, v, 1, nullptr, BY_SCORE);
    VecSimQueryReply_Free(res);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(STANDARD_KNN, info.commonInfo.lastMode);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    res = VecSimIndex_RangeQuery(index, v, 1, nullptr, BY_SCORE);
    VecSimQueryReply_Free(res);
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(RANGE_QUERY, info.commonInfo.lastMode);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 1, 1, true));
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(HYBRID_ADHOC_BF, info.commonInfo.lastMode);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    // Simulate the case where another call to the heuristics is done after realizing that
    // the subset size is smaller, and change the policy as a result.
    ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 1, 1, false));
    info = VecSimIndex_Info(index);
    infoIter = VecSimIndex_InfoIterator(index);
    ASSERT_EQ(HYBRID_BATCHES_TO_ADHOC_BF, info.commonInfo.lastMode);
    compareFlatIndexInfoToIterator(info, infoIter);
    VecSimInfoIterator_Free(infoIter);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, svs_vector_search_test_ip) {
    size_t dim = 4;
    size_t n = 10;
    size_t k = 5;

    for (size_t blocksize : {1, 12, DEFAULT_BLOCK_SIZE}) {

        SVSParams params = {
            .dim = dim,
            .metric = VecSimMetric_IP,
            .blockSize = blocksize,
            /* SVS-Vamana specifics */
            .alpha = 0.9,
            .graph_max_degree = 64,
            .construction_window_size = 20,
            .max_candidate_pool_size = 1024,
            .prune_to = 60,
            .use_search_history = VecSimOption_ENABLE,
        };

        VecSimIndex *index = this->CreateNewIndex(params);

        VecSimIndexInfo info = VecSimIndex_Info(index);
        ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVS);
        ASSERT_EQ(info.commonInfo.basicInfo.blockSize, blocksize);

        for (size_t i = 0; i < n; i++) {
            GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
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
}

TYPED_TEST(SVSTest, svs_vector_search_test_l2) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    for (size_t blocksize : {1, 12, DEFAULT_BLOCK_SIZE}) {

        SVSParams params = {
            .dim = dim,
            .metric = VecSimMetric_L2,
            .blockSize = blocksize,
            /* SVS-Vamana specifics */
            .alpha = 1.2,
            .graph_max_degree = 64,
            .construction_window_size = 20,
            .max_candidate_pool_size = 1024,
            .prune_to = 60,
            .use_search_history = VecSimOption_ENABLE,
        };

        VecSimIndex *index = this->CreateNewIndex(params);

        VecSimIndexInfo info = VecSimIndex_Info(index);
        ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVS);
        ASSERT_EQ(info.commonInfo.basicInfo.blockSize, blocksize);

        for (size_t i = 0; i < n; i++) {
            GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
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
}

TYPED_TEST(SVSTest, svs_search_empty_index) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = 1,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T query[] = {50, 50, 50, 50};

    // We do not expect any results.
    VecSimQueryReply *res = VecSimIndex_TopKQuery(index, query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    VecSimQueryReply_Iterator *it = VecSimQueryReply_GetIterator(res);
    ASSERT_EQ(VecSimQueryReply_IteratorNext(it), nullptr);
    VecSimQueryReply_IteratorFree(it);
    VecSimQueryReply_Free(res);

    res = VecSimIndex_RangeQuery(index, query, 1.0, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    VecSimQueryReply_Free(res);

    // Add some vectors and remove them all from index, so it will be empty again.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Again - we do not expect any results.
    res = VecSimIndex_TopKQuery(index, query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    it = VecSimQueryReply_GetIterator(res);
    ASSERT_EQ(VecSimQueryReply_IteratorNext(it), nullptr);
    VecSimQueryReply_IteratorFree(it);
    VecSimQueryReply_Free(res);

    res = VecSimIndex_RangeQuery(index, query, 1.0, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    VecSimQueryReply_Free(res);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, svs_test_inf_score) {
    size_t n = 4;
    size_t k = 4;
    size_t dim = 2;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = 1,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

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

TYPED_TEST(SVSTest, preferAdHocOptimization) {
    // Save the expected ratio which is the threshold between ad-hoc and batches mode
    // for every combination of index size and dim.
    // std::map<std::pair<size_t, size_t>, float> threshold;
    // threshold[{1000, 4}] = threshold[{1000, 80}] = threshold[{1000, 350}] = threshold[{1000,
    // 780}] =
    //     1.0;
    // threshold[{6000, 4}] = 0.2;
    // threshold[{6000, 80}] = 0.4;
    // threshold[{6000, 350}] = 0.6;
    // threshold[{6000, 780}] = 0.8;
    // threshold[{600000, 4}] = threshold[{600000, 80}] = 0.2;
    // threshold[{600000, 350}] = 0.6;
    // threshold[{600000, 780}] = 0.8;

    for (size_t index_size : {1000, 6000, 600000}) {
        for (size_t dim : {4, 80, 350, 780}) {
            // Create index and check for the expected output of "prefer ad-hoc".

            SVSParams params = {
                .dim = dim,
                .metric = VecSimMetric_IP,
                .blockSize = 5,
                /* SVS-Vamana specifics */
                .alpha = 1.2,
                .graph_max_degree = 64,
                .construction_window_size = 20,
                .max_candidate_pool_size = 1024,
                .prune_to = 60,
                .use_search_history = VecSimOption_ENABLE,
            };

            VecSimIndex *index = this->CreateNewIndex(params);

            // Set the index size artificially to be the required one.
            // (this->CastToBF(index))->count = index_size;
            // ASSERT_EQ(VecSimIndex_IndexSize(index), index_size);
            for (float r : {0.1f, 0.3f, 0.5f, 0.7f, 0.9f}) {
                bool res = VecSimIndex_PreferAdHocSearch(index, (size_t)(r * index_size), 50, true);
                // If r is below the threshold for this specific configuration of (index_size, dim),
                // expect that result will be ad-hoc (i.e., true), and otherwise, batches (i.e.,
                // false)
                // bool expected_res = r < threshold[{index_size, dim}];
                bool expected_res = true;
                ASSERT_EQ(res, expected_res);
            }
            VecSimIndex_Free(index);
        }
    }
    // Corner cases - empty index.

    SVSParams params = {
        .dim = 4,
        .metric = VecSimMetric_IP,
        .blockSize = 5,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 0, 50, true));

    // Corner cases - subset size is greater than index size.
    ASSERT_EQ(VecSimIndex_PreferAdHocSearch(index, 42, 50, true),
              VecSimIndex_PreferAdHocSearch(index, 0, 50, true));

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, batchIteratorSwapIndices) {
    size_t dim = 4;
    size_t n = 10000;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    TEST_DATA_T close_vec[] = {1.0, 1.0, 1.0, 1.0};
    TEST_DATA_T further_vec[] = {2.0, 2.0, 2.0, 2.0};
    VecSimIndex_AddVector(index, further_vec, 0);
    VecSimIndex_AddVector(index, close_vec, 1);
    VecSimIndex_AddVector(index, further_vec, 2);
    VecSimIndex_AddVector(index, close_vec, 3);
    VecSimIndex_AddVector(index, close_vec, 4);
    VecSimIndex_AddVector(index, close_vec, 5);
    for (size_t i = 6; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Query for (1,1,1,1) vector.
    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, 1.0);
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    // Get first batch - expect to get ids 1,3,4,5.
    VecSimQueryReply *res = VecSimBatchIterator_Next(batchIterator, 4, BY_ID);
    ASSERT_EQ(VecSimQueryReply_Len(res), 4);
    VecSimQueryReply_Iterator *iterator = VecSimQueryReply_GetIterator(res);
    int res_ind = 0;
    size_t expected_res[] = {1, 3, 4, 5};
    while (VecSimQueryReply_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryReply_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        ASSERT_EQ(expected_res[res_ind++], id);
    }
    VecSimQueryReply_IteratorFree(iterator);
    VecSimQueryReply_Free(res);

    // Get another batch - expect to get ids 0,2,6,7. Make sure that ids 0,2 swapped properly.
    res = VecSimBatchIterator_Next(batchIterator, 4, BY_ID);
    ASSERT_EQ(VecSimQueryReply_Len(res), 4);
    iterator = VecSimQueryReply_GetIterator(res);
    res_ind = 0;
    size_t expected_res_2[] = {0, 2, 6, 7};
    while (VecSimQueryReply_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryReply_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        ASSERT_EQ(expected_res_2[res_ind++], id);
    }
    VecSimQueryReply_IteratorFree(iterator);
    VecSimQueryReply_Free(res);

    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, svs_vector_search_test_cosine) {
    size_t dim = 128;
    size_t n = 100;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_Cosine,
        /* SVS-Vamana specifics */
        .alpha = 0.9,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

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

    // topK search will normalize the query so we keep the original data to
    // avoid normalizing twice.
    TEST_DATA_T normalized_query[dim];
    memcpy(normalized_query, query, dim * sizeof(TEST_DATA_T));
    VecSim_Normalize(normalized_query, dim, params.type);

    auto verify_res = [&](size_t id, double score, size_t result_rank) {
        ASSERT_EQ(id, (n - result_rank));
        TEST_DATA_T expected_score = index->getDistanceFrom_Unsafe(id, normalized_query);
        // Verify that abs difference between the actual and expected score is at most 1/10^5.
        ASSERT_NEAR((TEST_DATA_T)score, expected_score, 1e-5f);
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
        auto verify_res_batch = [&](size_t id, double score, size_t result_rank) {
            ASSERT_EQ(id, (n - n_res * iteration_num - result_rank));
            TEST_DATA_T expected_score = index->getDistanceFrom_Unsafe(id, normalized_query);
            // Verify that abs difference between the actual and expected score is at most 1/10^5.
            ASSERT_NEAR((TEST_DATA_T)score, expected_score, 1e-5f);
        };
        runBatchIteratorSearchTest(batchIterator, n_res, verify_res_batch);
        iteration_num++;
    }
    ASSERT_EQ(iteration_num, n / n_res);
    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, testSizeEstimation) {
    size_t dim = 128;
#if HAVE_SVS_LVQ
    // SVS block sizes always rounded to a power of 2
    // This why, in case of quantization, actual block size can be differ than requested
    // In addition, block size to be passed to graph and dataset counted in bytes,
    // converted then to a number of elements.
    // IMHO, would be better to always interpret block size to a number of elements
    // rather than conversion to-from number of bytes
    if (TypeParam::get_quant_bits() > 0) {
        // Extra data in LVQ vector
        const auto lvq_vector_extra = sizeof(svs::quantization::lvq::ScalarBundle);
        dim -= (lvq_vector_extra * 8) / TypeParam::get_quant_bits();
    }
#endif
    size_t n = 0;
    size_t bs = DEFAULT_BLOCK_SIZE;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_Cosine,
        .blockSize = bs,
        /* SVS-Vamana specifics */
        .alpha = 0.9,
        .graph_max_degree = 63, // x^2-1 to round the graph block size
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();
    ASSERT_EQ(estimation, actual);

    estimation = EstimateElementSize(params) * bs;

    GenerateAndAddVector<TEST_DATA_T>(index, dim, 0);
    actual = index->getAllocationSize() - actual; // get the delta
    ASSERT_GT(actual, 0);
    ASSERT_GE(estimation * 1.01, actual);
    ASSERT_LE(estimation * 0.99, actual);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, testInitialSizeEstimation) {
    size_t dim = 128;
    size_t n = 100;
    size_t bs = DEFAULT_BLOCK_SIZE;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_Cosine,
        .blockSize = bs,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();
    ASSERT_EQ(estimation, actual);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, testTimeoutReturn_topK) {
    size_t dim = 4;
    VecSimQueryReply *rep;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = 5,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out

    TEST_DATA_T vec[dim];
    GenerateVector<TEST_DATA_T>(vec, dim);

    VecSimIndex_AddVector(index, vec, 0);
    // Checks return code on timeout - knn
    rep = VecSimIndex_TopKQuery(index, vec, 1, NULL, BY_ID);
    ASSERT_EQ(VecSimQueryReply_GetCode(rep), VecSim_QueryReply_TimedOut);
    ASSERT_EQ(VecSimQueryReply_Len(rep), 0);
    VecSimQueryReply_Free(rep);

    VecSimIndex_Free(index);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // cleanup
}

TYPED_TEST(SVSTest, testTimeoutReturn_range) {
    size_t dim = 4;
    VecSimQueryReply *rep;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = 5,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out

    TEST_DATA_T vec[dim];
    GenerateVector<TEST_DATA_T>(vec, dim);

    VecSimIndex_AddVector(index, vec, 0);

    // Checks return code on timeout - range query
    rep = VecSimIndex_RangeQuery(index, vec, 1, NULL, BY_ID);
    ASSERT_EQ(VecSimQueryReply_GetCode(rep), VecSim_QueryReply_TimedOut);
    ASSERT_EQ(VecSimQueryReply_Len(rep), 0);
    VecSimQueryReply_Free(rep);

    VecSimIndex_Free(index);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // cleanup
}

TYPED_TEST(SVSTest, testTimeoutReturn_batch_iterator) {
    size_t dim = 4;
    size_t n = 10;
    VecSimQueryReply *rep;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = 5,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, n);

    // Fail on second batch (after calculation already completed)
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    rep = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
    ASSERT_EQ(VecSimQueryReply_GetCode(rep), VecSim_QueryReply_OK);
    ASSERT_NE(VecSimQueryReply_Len(rep), 0);
    VecSimQueryReply_Free(rep);

    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 1; }); // Always times out
    rep = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
    ASSERT_EQ(VecSimQueryReply_GetCode(rep), VecSim_QueryReply_TimedOut);
    ASSERT_EQ(VecSimQueryReply_Len(rep), 0);
    VecSimQueryReply_Free(rep);

    VecSimBatchIterator_Free(batchIterator);

    // Fail on first batch (while calculating)
    // Timeout callback function already set to always time out
    batchIterator = VecSimBatchIterator_New(index, query, nullptr);

    rep = VecSimBatchIterator_Next(batchIterator, 1, BY_ID);
    ASSERT_EQ(VecSimQueryReply_GetCode(rep), VecSim_QueryReply_TimedOut);
    ASSERT_EQ(VecSimQueryReply_Len(rep), 0);
    VecSimQueryReply_Free(rep);

    VecSimBatchIterator_Free(batchIterator);

    VecSimIndex_Free(index);
    VecSim_SetTimeoutCallbackFunction([](void *ctx) { return 0; }); // cleanup
}

TYPED_TEST(SVSTest, rangeQuery) {
    size_t n = 2000;
    size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = n / 2,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    size_t pivot_id = n / 2; // The id to return vectors around it.
    TEST_DATA_T query[] = {(TEST_DATA_T)pivot_id, (TEST_DATA_T)pivot_id, (TEST_DATA_T)pivot_id,
                           (TEST_DATA_T)pivot_id};

    // Validate invalid params are caught with runtime exception.
    try {
        VecSimIndex_RangeQuery(index, query, -1, nullptr, BY_SCORE);
        FAIL();
    } catch (const std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("radius must be non-negative"));
    }
    try {
        VecSimIndex_RangeQuery(index, query, 1, nullptr, VecSimQueryReply_Order(2));
        FAIL();
    } catch (const std::runtime_error &err) {
        EXPECT_EQ(err.what(), std::string("Possible order values are only 'BY_ID' or 'BY_SCORE'"));
    }

    auto verify_res_by_score = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(std::abs(int(id - pivot_id)), (index + 1) / 2);
        ASSERT_EQ(score, dim * pow((index + 1) / 2, 2));
    };
    uint expected_num_results = 11;
    // To get 11 results in the range [pivot_id - 5, pivot_id + 5], set the radius as the L2 score
    // in the boundaries.
    double radius = dim * pow(expected_num_results / 2, 2);
    runRangeQueryTest(index, query, radius, verify_res_by_score, expected_num_results, BY_SCORE);

    // Rerun with a given query params.
    SVSRuntimeParams svsRuntimeParams = {.epsilon = 1.0};
    auto query_params = CreateQueryParams(svsRuntimeParams);
    query_params.batchSize = 100;
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

TYPED_TEST(SVSTest, rangeQueryCosine) {
    size_t n = 100;
    size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_Cosine,
        .blockSize = n / 2,
        /* SVS-Vamana specifics */
        .alpha = 0.9,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T f[dim];
        f[0] = TEST_DATA_T(i + 1) / n;
        for (size_t j = 1; j < dim; j++) {
            f[j] = 1.0;
        }
        // Use as label := n - (internal id)
        VecSimIndex_AddVector(index, (const void *)f, n - i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    TEST_DATA_T query[dim];
    for (size_t i = 0; i < dim; i++) {
        query[i] = 1.0;
    }
    auto verify_res = [&](size_t id, double score, size_t result_rank) {
        ASSERT_EQ(id, result_rank + 1);
        TEST_DATA_T expected_score = index->getDistanceFrom_Unsafe(id, query);
        // Verify that abs difference between the actual and expected score is at most 1/10^5.
        ASSERT_NEAR((TEST_DATA_T)score, expected_score, 1e-5f);
    };

    uint expected_num_results = 31;
    // Calculate the score of the 31st distant vector from the query vector (whose id should be 30)
    // to get the radius.
    VecSim_Normalize(query, dim, params.type);
    double radius = index->getDistanceFrom_Unsafe(31, query);
    runRangeQueryTest(index, query, radius, verify_res, expected_num_results, BY_SCORE);
    // Return results BY_ID should give the same results.
    runRangeQueryTest(index, query, radius, verify_res, expected_num_results, BY_ID);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, FitMemoryTest) {
    size_t dim = 4;
    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        .blockSize = DEFAULT_BLOCK_SIZE,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);

    size_t initial_memory = index->getAllocationSize();
    index->fitMemory();
    ASSERT_GE(index->getAllocationSize(), initial_memory);

    // Add vector
    GenerateAndAddVector<TEST_DATA_T>(index, dim, 0);
    initial_memory = index->getAllocationSize();
    index->fitMemory();
    size_t final_size = index->getAllocationSize();
    // Due to the initial capacity, the memory for the vector was already allocated
    ASSERT_EQ(final_size, initial_memory);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, resolve_ws_search_runtime_params) {
    SVSParams params = {.dim = 4, .metric = VecSimMetric_L2};

    VecSimIndex *index = this->CreateNewIndex(params);

    VecSimQueryParams qparams, zero;
    bzero(&zero, sizeof(VecSimQueryParams));

    std::vector<VecSimRawParam> rparams;

    auto mkRawParams = [](const std::string &name, const std::string &val) {
        return VecSimRawParam{name.c_str(), name.length(), val.c_str(), val.length()};
    };

    // Test with empty runtime params.
    for (VecsimQueryType query_type : test_utils::query_types) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, query_type),
            VecSim_OK);
    }
    ASSERT_EQ(memcmp(&qparams, &zero, sizeof(VecSimQueryParams)), 0);

    std::string param_name = "ws_search";
    std::string param_val = "100";
    rparams.push_back(mkRawParams(param_name, param_val));

    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSim_OK);
    ASSERT_EQ(qparams.svsRuntimeParams.windowSize, 100);

    param_name = "wrong_name";
    param_val = "100";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_NONE),
        VecSimParamResolverErr_UnknownParam);

    // Testing for legal prefix but only partial parameter name.
    param_name = "ws_sea";
    param_val = "100";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_NONE),
        VecSimParamResolverErr_UnknownParam);

    param_name = "ws_search";
    param_val = "wrong_val";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_BadValue);

    param_name = "ws_search";
    param_val = "-30";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_BadValue);

    param_name = "ws_search";
    param_val = "1.618";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_BadValue);

    param_name = "ws_search";
    param_val = "100";
    rparams[0] = mkRawParams(param_name, param_val);
    rparams.push_back(mkRawParams(param_name, param_val));
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_AlreadySet);

    rparams[1] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "BATCHES",
                                  .valLen = strlen("BATCHES")};
    rparams.push_back((VecSimRawParam){.name = "batch_size",
                                       .nameLen = strlen("batch_size"),
                                       .value = "50",
                                       .valLen = strlen("50")});
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 50);
    ASSERT_EQ(qparams.svsRuntimeParams.windowSize, 100);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, resolve_use_search_history_runtime_params) {
    SVSParams params = {.dim = 4, .metric = VecSimMetric_L2};

    VecSimIndex *index = this->CreateNewIndex(params);

    VecSimQueryParams qparams, zero;
    bzero(&zero, sizeof(VecSimQueryParams));

    std::vector<VecSimRawParam> rparams;

    auto mkRawParams = [](const std::string &name, const std::string &val) {
        return VecSimRawParam{name.c_str(), name.length(), val.c_str(), val.length()};
    };

    // Test with empty runtime params.
    for (VecsimQueryType query_type : test_utils::query_types) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, query_type),
            VecSim_OK);
    }
    ASSERT_EQ(memcmp(&qparams, &zero, sizeof(VecSimQueryParams)), 0);

    std::string param_name = "use_search_history";
    std::string param_val = "on";
    rparams.push_back(mkRawParams(param_name, param_val));
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSim_OK);
    ASSERT_EQ(qparams.svsRuntimeParams.searchHistory, VecSimOption_ENABLE);

    param_name = "use_search_history";
    param_val = "off";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSim_OK);
    ASSERT_EQ(qparams.svsRuntimeParams.searchHistory, VecSimOption_DISABLE);

    param_name = "use_search_history";
    param_val = "auto";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSim_OK);
    ASSERT_EQ(qparams.svsRuntimeParams.searchHistory, VecSimOption_AUTO);

    param_name = "wrong_name";
    param_val = "on";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_NONE),
        VecSimParamResolverErr_UnknownParam);

    // Testing for legal prefix but only partial parameter name.
    param_name = "use_search";
    param_val = "on";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_NONE),
        VecSimParamResolverErr_UnknownParam);

    param_name = "use_search_history";
    param_val = "wrong_val";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_BadValue);

    param_name = "use_search_history";
    param_val = "1";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_BadValue);

    param_name = "use_search_history";
    param_val = "disable";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_BadValue);

    param_name = "use_search_history";
    param_val = "on";
    rparams[0] = mkRawParams(param_name, param_val);
    rparams.push_back(mkRawParams(param_name, param_val));
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_KNN),
        VecSimParamResolverErr_AlreadySet);

    rparams[1] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "BATCHES",
                                  .valLen = strlen("BATCHES")};
    rparams.push_back((VecSimRawParam){.name = "batch_size",
                                       .nameLen = strlen("batch_size"),
                                       .value = "50",
                                       .valLen = strlen("50")});
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_HYBRID),
              VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 50);
    ASSERT_EQ(qparams.svsRuntimeParams.searchHistory, VecSimOption_ENABLE);

    VecSimIndex_Free(index);
}

TYPED_TEST(SVSTest, resolve_epsilon_runtime_params) {
    SVSParams params = {.dim = 4, .metric = VecSimMetric_L2};

    VecSimIndex *index = this->CreateNewIndex(params);

    VecSimQueryParams qparams, zero;
    bzero(&zero, sizeof(VecSimQueryParams));

    std::vector<VecSimRawParam> rparams;

    auto mkRawParams = [](const std::string &name, const std::string &val) {
        return VecSimRawParam{name.c_str(), name.length(), val.c_str(), val.length()};
    };

    // Test with empty runtime params.
    for (VecsimQueryType query_type : test_utils::query_types) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, query_type),
            VecSim_OK);
    }
    ASSERT_EQ(memcmp(&qparams, &zero, sizeof(VecSimQueryParams)), 0);

    std::string param_name = "epsilon";
    std::string param_val = "0.001";
    rparams.push_back(mkRawParams(param_name, param_val));
    for (VecsimQueryType query_type : {QUERY_TYPE_NONE, QUERY_TYPE_KNN, QUERY_TYPE_HYBRID}) {
        ASSERT_EQ(
            VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, query_type),
            VecSimParamResolverErr_InvalidPolicy_NRange);
    }

    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_RANGE),
              VecSim_OK);
    ASSERT_FLOAT_EQ(qparams.svsRuntimeParams.epsilon, 0.001);

    param_name = "wrong_name";
    param_val = "0.001";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_RANGE),
              VecSimParamResolverErr_UnknownParam);

    // Testing for legal prefix but only partial parameter name.
    param_name = "epsi";
    param_val = "0.001";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(
        VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams, QUERY_TYPE_NONE),
        VecSimParamResolverErr_UnknownParam);

    param_name = "epsilon";
    param_val = "wrong_val";
    rparams[0] = mkRawParams(param_name, param_val);
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_RANGE),
              VecSimParamResolverErr_BadValue);

    rparams[0] = (VecSimRawParam){
        .name = "epsilon", .nameLen = strlen("epsilon"), .value = "-30", .valLen = 3};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_RANGE),
              VecSimParamResolverErr_BadValue);

    param_name = "epsilon";
    param_val = "0.001";
    rparams[0] = mkRawParams(param_name, param_val);
    rparams.push_back(mkRawParams(param_name, param_val));
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams.data(), rparams.size(), &qparams,
                                        QUERY_TYPE_RANGE),
              VecSimParamResolverErr_AlreadySet);

    VecSimIndex_Free(index);
}
