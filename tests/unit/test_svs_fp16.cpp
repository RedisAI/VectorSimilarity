/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "unit_test_utils.h"
#include "tests_utils.h"
#include "VecSim/spaces/L2/L2.h"
#include <array>
#include <cmath>
#include <random>
#include <vector>

using float16 = vecsim_types::float16;
/**
 *
 *
 * DIDNT TAKE test_svs_parameter_combinations_and_defaults and
 * test_svs_parameter_consistency_across_metrics
 * as it mostly covered in test_svs_info
 *
 * also test_inf wasn't covered as i coudnlt get an inf score with fp16
 *
 * in batchIteratorSwapIndices decreased n to 1000
 *
 * not taking joinSearchParams (type is not relevant)
 *
 * FitMemoryTest does nothing
 *
 * resolve_ws_search_runtime_params, resolve_bc_search_runtime_params,
 * resolve_use_search_history_runtime_params, resolve_epsilon_runtime_params, nothing to do with the
 * type
 *
 * testTimeoutReturn_batch_iterator, testTimeoutReturn_range, testTimeoutReturn_topK removed
 *
 * testInitialSizeEstimation covered by testSizeEstimation
 * test_override_all also in float32 (test_svs.cpp)
 *
 * TODO: revert setLogCallbackFunction
 */

#if HAVE_SVS
#include <sstream>
#include "spdlog/sinks/ostream_sink.h"
#include "VecSim/algorithms/svs/svs.h"

// There are possible cases when SVS Index cannot be created with the requested quantization mode
// due to platform and/or hardware limitations or combination of requested 'compression' modes.
// This assert handle those cases and skip a test if the mode is not supported.
// Elsewhere, test will fail if the index creation failed with no reason explained above.
#define ASSERT_INDEX(index)                                                                        \
    if (index == nullptr) {                                                                        \
        if (std::get<1>(svs_details::isSVSQuantBitsSupported(TypeParam::get_quant_bits()))) {      \
            GTEST_FAIL() << "Failed to create SVS index";                                          \
        } else {                                                                                   \
            GTEST_SKIP() << "SVS LVQ is not supported.";                                           \
        }                                                                                          \
    }

// Log callback function to print non-debug log messages
static void svsTestLogCallBackNoDebug(void *ctx, const char *level, const char *message) {
    if (level == nullptr || message == nullptr) {
        return; // Skip null messages
    }
    if (std::string_view{level} == VecSimCommonStrings::LOG_DEBUG_STRING) {
        return; // Skip debug messages
    }
    // Print other log levels
    std::cout << level << ": " << message << std::endl;
}
template <typename index_type_t>
class FP16SVSTest : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;

protected:
    void SetTypeParams(SVSParams &params) {
        params.quantBits = params.quantBits == VecSimSvsQuant_NONE ? index_type_t::get_quant_bits()
                                                                   : params.quantBits;
        params.type = index_type_t::get_index_type();
        params.multi = params.multi == 0 ? false : params.multi;
    }

    virtual VecSimIndex *CreateNewIndex(SVSParams &params) {
        VecSimIndexInterface::setLogCallbackFunction(svsTestLogCallBackNoDebug);
        SetTypeParams(params);
        VecSimParams index_params = CreateParams(params);
        return VecSimIndex_New(&index_params);
    }

    SVSIndexBase *CastToSVS(VecSimIndex *index) {
        auto indexBase = dynamic_cast<SVSIndexBase *>(index);
        assert(indexBase != nullptr);
        return indexBase;
    }

    // Check if the test is running in fallback mode to scalar quantization.
    bool isFallbackToSQ() const {
        // Get the fallback quantization mode and compare it to the scalar quantization mode.
        return VecSimSvsQuant_Scalar ==
               std::get<0>(svs_details::isSVSQuantBitsSupported(index_type_t::get_quant_bits()));
    }

    void GenerateVector(float16 *out_vec, size_t dim, float initial_value = 0.25f,
                        float step = 0.0f) {
        for (size_t i = 0; i < dim; i++) {
            out_vec[i] = vecsim_types::FP32_to_FP16(initial_value + step * i);
        }
    }

    int GenerateAndAddVector(VecSimIndex *index, size_t dim, size_t id, float initial_value = 0.25f,
                             float step = 0.0f) {
        float16 v[dim];
        this->GenerateVector(v, dim, initial_value, step);
        return VecSimIndex_AddVector(index, v, id);
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


using SVSDataTypeSet = ::testing::Types<SVSIndexType<VecSimType_FLOAT16, float16, VecSimSvsQuant_NONE>
                                       ,SVSIndexType<VecSimType_FLOAT16, float16, VecSimSvsQuant_8>
                                       ,SVSIndexType<VecSimType_FLOAT16, float16, VecSimSvsQuant_8x8_LeanVec>
                                        >;
// clang-format on

TYPED_TEST_SUITE(FP16SVSTest, SVSDataTypeSet);

TYPED_TEST(FP16SVSTest, svs_vector_add_test) {

    constexpr size_t dim = 4;
    float initial_value = 0.5f;
    float step = 1.0f;
    size_t label = 0;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    EXPECT_EQ(VecSimIndex_IndexSize(index), 0);

    float16 vector[dim];
    this->GenerateVector(vector, dim, initial_value, step);
    VecSimIndex_AddVector(index, vector, label);

    EXPECT_EQ(VecSimIndex_IndexSize(index), 1);
    ASSERT_EQ(index->getDistanceFrom_Unsafe(label, vector), 0);
    if (!(TypeParam::get_quant_bits() == VecSimSvsQuant_8 ||
          TypeParam::get_quant_bits() == VecSimSvsQuant_8x8_LeanVec)) {
        // Get vector is only implemented for non-compressed indices.
        auto vector_data =
            (dynamic_cast<VecSimIndexAbstract<svs_details::vecsim_dt<svs::Float16>, float> *>(
                 index))
                ->getStoredVectorDataByLabel(label);
        const char *data_ptr = vector_data.at(0).data();
        for (size_t i = 0; i < dim; i++) {
            ASSERT_EQ(vecsim_types::FP16_to_FP32(((float16 *)data_ptr)[i]),
                      initial_value + step * float(i));
        }
    }

    size_t k = 1;
    auto verify_res = [&](size_t id, double score, size_t index) { EXPECT_EQ(score, 0.0); };
    runTopKSearchTest(index, vector, k, verify_res, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, svs_vector_update_test) {
    size_t dim = 4;
    size_t n = 1;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_IP,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    auto *svs_index = this->CastToSVS(index);

    EXPECT_EQ(VecSimIndex_IndexSize(index), 0);

    this->GenerateAndAddVector(index, dim, 1);

    EXPECT_EQ(VecSimIndex_IndexSize(index), 1);

    // Prepare new vector data and call addVector with the same id, different data.
    this->GenerateAndAddVector(index, dim, 1, 2.0);

    // Index size shouldn't change.
    EXPECT_EQ(VecSimIndex_IndexSize(index), 1);

    // Delete the last vector.
    VecSimIndex_DeleteVector(index, 1);
    EXPECT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, svs_vector_search_test) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    constexpr size_t n = 100;
    constexpr size_t k = 11;
    constexpr size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float16 query[dim];
    this->GenerateVector(query, dim, 50);
    auto verify_res_by_id = [&](size_t id, double score, size_t index) {
        EXPECT_EQ(id, (index + 45));
        ASSERT_EQ(score, 4 * (50 - id) * (50 - id)); // L2 distance
    };
    runTopKSearchTest(index, query, k, verify_res_by_id, nullptr, BY_ID);

    auto verify_res_by_score = [&](size_t id, double score, size_t index) {
        size_t abs_index_offset = (index + 1) / 2;
        int direction = id < 50 ? -1 : 1;
        EXPECT_EQ(id, 50 + direction * abs_index_offset);
        ASSERT_EQ(score, 4 * (50 - id) * (50 - id)); // L2 distance
    };
    runTopKSearchTest(index, query, k, verify_res_by_score, nullptr, BY_SCORE);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, svs_bulk_vectors_add_delete_test) {
    constexpr size_t n = 256;
    size_t k = 11;
    const size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        .construction_window_size = 20,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    auto svs_index = this->CastToSVS(index); // CAST_TO_SVS(index, svs::distance::DistanceL2);

    std::vector<std::array<float16, dim>> v(n);
    for (size_t i = 0; i < n; i++) {
        this->GenerateVector(v[i].data(), dim, i);
    }

    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    svs_index->addVectors(v.data(), ids.data(), n);

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float16 query[dim];
    this->GenerateVector(query, dim, 50);
    auto verify_res = [&](size_t id, double score, size_t index) { EXPECT_EQ(id, (index + 45)); };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    // Delete almost all vectors
    const size_t keep_num = 10;
    ASSERT_EQ(svs_index->deleteVectors(ids.data(), n - keep_num), n - keep_num);
    ASSERT_EQ(VecSimIndex_IndexSize(index), keep_num);

    auto verify_res_after_delete = [&](size_t id, double score, size_t index) {
        EXPECT_EQ(id, n - keep_num + index);
    };
    // TODO : THIS IS CURRENTLY THROWING A C++ EXCEPTION ORIGINATED FROM robin_hash.h: 917
    // =====================================================================
    runTopKSearchTest(index, query, keep_num, verify_res_after_delete, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, svs_indexing_same_vector) {
    const size_t n = 100;
    const size_t k = 10;
    const size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    auto svs_index = this->CastToSVS(index);
    ASSERT_NE(svs_index, nullptr);

    std::vector<std::array<float16, dim>> v(n);
    for (size_t i = 0; i < n; i++) {
        this->GenerateVector(v[i].data(), dim,
                             i / 10); // i / 10 is in integer (take the "floor" value).
    }

    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    svs_index->addVectors(v.data(), ids.data(), n);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float16 query[dim] = {vecsim_types::FP32_to_FP16(4.9), vecsim_types::FP32_to_FP16(4.95),
                          vecsim_types::FP32_to_FP16(5.05), vecsim_types::FP32_to_FP16(5.1)};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, svs_reindexing_same_vector) {
    const size_t n = 100;
    const size_t k = 10;
    const size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    auto svs_index = this->CastToSVS(index);
    ASSERT_NE(svs_index, nullptr);

    std::vector<std::array<TEST_DATA_T, dim>> v(n);
    for (size_t i = 0; i < n; i++) {
        // i / 10 is in integer (take the "floor" value).
        this->GenerateVector(v[i].data(), dim, i / 10);
    }

    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    svs_index->addVectors(v.data(), ids.data(), n);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float16 query[dim] = {vecsim_types::FP32_to_FP16(4.9), vecsim_types::FP32_to_FP16(4.95),
                          vecsim_types::FP32_to_FP16(5.05), vecsim_types::FP32_to_FP16(5.1)};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    // Delete almost all vectors - keeping SVS index implementation alive.
    for (size_t i = 0; i < n - 1; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

    // Reinsert the same vectors under the same ids.
    for (size_t i = 0; i < n; i++) {
        // i / 10 is in integer (take the "floor value).
        this->GenerateAndAddVector(index, dim, i, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run the same query again.
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, svs_reindexing_same_vector_different_id) {
    const size_t n = 100;
    const size_t k = 10;
    const size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    auto svs_index = this->CastToSVS(index);
    ASSERT_NE(svs_index, nullptr);

    std::vector<std::array<float16, dim>> v(n);
    for (size_t i = 0; i < n; i++) {
        this->GenerateVector(v[i].data(), dim,
                             i / 10); // i / 10 is in integer (take the "floor" value).
    }

    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    svs_index->addVectors(v.data(), ids.data(), n);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float16 query[dim] = {vecsim_types::FP32_to_FP16(4.9), vecsim_types::FP32_to_FP16(4.95),
                          vecsim_types::FP32_to_FP16(5.05), vecsim_types::FP32_to_FP16(5.1)};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_TRUE(id >= 50 && id < 60 && score <= 1);
    };
    runTopKSearchTest(index, query, k, verify_res);

    for (size_t i = 0; i < n - 1; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

    // Reinsert the same vectors under different ids than before.
    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i + 10,
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

TYPED_TEST(FP16SVSTest, svs_batch_iterator) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t dim = 4;

    // run the test twice - for index of size 100, every iteration will run select-based search,
    // as the number of results is 5, which is more than 0.1% of the index size. for index of size
    // 10000, we will run the heap-based search until we return 5000 results, and then switch to
    // select-based search.
    for (size_t n : {100, 1000}) {
        SVSParams params = {
            .dim = dim,
            .metric = VecSimMetric_L2,
        };

        VecSimIndex *index = this->CreateNewIndex(params);
        ASSERT_INDEX(index);

        for (size_t i = 0; i < n; i++) {
            this->GenerateAndAddVector(index, dim, i, i);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), n) << "Running for n = " << n;

        // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
        float16 query[dim];
        this->GenerateVector(query, dim, n);

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
                ASSERT_EQ(id, expected_ids[index])
                    << "Running for n = " << n << " iteration_num: " << iteration_num
                    << " index: " << index << " score: " << score;
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num, n / n_res) << "Running for n = " << n;
        VecSimBatchIterator_Free(batchIterator);

        VecSimIndex_Free(index);
    }
}

TYPED_TEST(FP16SVSTest, svs_batch_iterator_non_unique_scores) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t dim = 4;

    // Run the test twice - for index of size 100, every iteration will run select-based search,
    // as the number of results is 5, which is more than 0.1% of the index size. for index of size
    // 10000, we will run the heap-based search until we return 5000 results, and then switch to
    // select-based search.
    for (size_t n : {100, 1000}) {
        SVSParams params = {
            .dim = dim,
            .metric = VecSimMetric_L2,
        };

        VecSimIndex *index = this->CreateNewIndex(params);
        ASSERT_INDEX(index);

        for (size_t i = 0; i < n; i++) {
            this->GenerateAndAddVector(index, dim, i, i / 10);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), n) << "Running for n = " << n;

        // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
        float16 query[dim];
        this->GenerateVector(query, dim, n);

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
                ASSERT_TRUE(expected_ids.find(id) != expected_ids.end())
                    << "Running for n = " << n << " iteration_num: " << iteration_num
                    << " index: " << index << " score: " << score;
                ;
                expected_ids.erase(id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            // Make sure that the expected ids set is empty after two iterations.
            if (even_iteration) {
                ASSERT_TRUE(expected_ids.empty())
                    << "Running for n = " << n << " iteration_num: " << iteration_num;
            }
            iteration_num++;
            even_iteration = !even_iteration;
        }
        ASSERT_EQ(iteration_num, n / n_res);
        VecSimBatchIterator_Free(batchIterator);

        VecSimIndex_Free(index);
    }
}

TYPED_TEST(FP16SVSTest, svs_batch_iterator_reset) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    constexpr size_t dim = 4;
    size_t n = 10000;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
        .construction_window_size = 20,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i, i / 10);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
    float16 query[dim];
    this->GenerateVector(query, dim, n);
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
                ASSERT_TRUE(expected_ids.find(id) != expected_ids.end())
                    << "take: " << take << " iteration_num: " << iteration_num
                    << " index: " << index << " score: " << score;
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

TYPED_TEST(FP16SVSTest, svs_batch_iterator_corner_cases) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t dim = 4;
    size_t n = 1000;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    // Query for (n,n,...,n) vector (recall that n is the largest id in te index).
    float16 query[dim];
    this->GenerateVector(query, dim, n);

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
        this->GenerateAndAddVector(index, dim, i, i);
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
        ASSERT_TRUE(id == n - 1 - index) << "index: " << index << " score: " << score;
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
TYPED_TEST(FP16SVSTest, resizeIndex) {
    size_t dim = 4;
    size_t n = 10;
    size_t bs = 4;

    SVSParams params = {.dim = dim, .metric = VecSimMetric_L2, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    // Add up to n.
    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i, i);
    }

    // Initial capacity is rounded up to the block size.
    size_t extra_cap = n % bs == 0 ? 0 : bs - n % bs;
    auto quantBits = TypeParam::get_quant_bits();
    // Get the fallback quantization mode
    quantBits = std::get<0>(svs_details::isSVSQuantBitsSupported(quantBits));
    if (quantBits != VecSimSvsQuant_NONE) {
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
TYPED_TEST(FP16SVSTest, svs_empty_index) {
    size_t dim = 4;
    size_t n = 20;
    size_t k = 10;

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
    ASSERT_INDEX(index);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove from an empty index - should fail because label doesn't exist.
    VecSimIndex_DeleteVector(index, 0);

    float16 query[dim];
    this->GenerateVector(query, dim);

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

    // Add one vector.
    this->GenerateAndAddVector(index, dim, 1, 1.7);

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

TYPED_TEST(FP16SVSTest, test_delete_vector) {
    size_t k = 5;
    constexpr size_t dim = 10;
    size_t block_size = 3;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_IP,
        .blockSize = block_size,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        // .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    // Delete from empty index
    ASSERT_EQ(VecSimIndex_DeleteVector(index, 111), 0);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    size_t n = 6;
    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float16 query[dim];
    this->GenerateVector(query, dim, 0.0);
    auto verify_res = [&](size_t id, double score, size_t index) {
        if (index == 0) {
            ASSERT_EQ(id, index);
        } else {
            ASSERT_EQ(id, index + 1);
        }
    };
    runTopKSearchTest(index, query, k, verify_res);

    // Here the shift should happen.
    VecSimIndex_DeleteVector(index, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n - 1);

    // float16 query[dim];
    // this->GenerateVector(query, dim, 0.0);
    // auto verify_res = [&](size_t id, double score, size_t index) {
    //     if (index == 0) {
    //         ASSERT_EQ(id, index);
    //     } else {
    //         ASSERT_EQ(id, index + 1);
    //     }
    // };
    // runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, sanity_reinsert_1280) {
    constexpr size_t n = 5;
    constexpr size_t d = 1280;
    size_t k = 5;

    SVSParams params = {
        .dim = d,
        .metric = VecSimMetric_Cosine,
        /* SVS-Vamana specifics */
        .alpha = 1.2,
        .graph_max_degree = 64,
        // .construction_window_size = 20,
        .max_candidate_pool_size = 1024,
        .prune_to = 60,
        .use_search_history = VecSimOption_ENABLE,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    auto *vectors = new float16[n * d];

    // Generate random vectors in every iteration and inert them under different ids.
    for (size_t iter = 1; iter <= 1; iter++) {
        for (size_t i = 0; i < n; i++) {
            size_t internal_iter = n * (iter - 1) + i;
            test_utils::populate_float16_vec(vectors + i * d, d, internal_iter, 0.0, 1.0f);
        }
        auto expected_ids = std::set<size_t>();
        for (size_t i = 0; i < n; i++) {
            ASSERT_EQ(VecSimIndex_AddVector(index, (vectors + i * d), i * iter + 1), 1);
            expected_ids.insert(i * iter);
        }
        auto verify_res = [&](size_t id, double score, size_t index) {
            ASSERT_TRUE(expected_ids.find(id) != expected_ids.end());
            expected_ids.erase(id);
        };

        // Send arbitrary vector (the first) and search for top k. This should return all the
        // vectors that were inserted in this iteration - verify their ids.
        ASSERT_NO_FATAL_FAILURE(runTopKSearchTest(index, vectors, k, verify_res))
            << "iter: " << iter;

        // Remove vectors form current iteration.
        for (size_t i = 0; i < n; i++) {
            VecSimIndex_DeleteVector(index, i * iter);
        }
    }
    delete[] vectors;
    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, test_svs_info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args.

    for (auto metric : {VecSimMetric_L2, VecSimMetric_IP, VecSimMetric_Cosine}) {
        SCOPED_TRACE("Testing metric: " + std::string(VecSimMetric_ToString(metric)));
        SVSParams params = {
            .dim = d,
            .metric = metric,
        };

        VecSimIndex *index = this->CreateNewIndex(params);
        ASSERT_INDEX(index);

        VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(index);
        ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVS);
        ASSERT_EQ(info.commonInfo.basicInfo.dim, d);
        ASSERT_FALSE(info.commonInfo.basicInfo.isMulti);
        ASSERT_EQ(info.commonInfo.basicInfo.type, VecSimType_FLOAT16);
        // Default args.
        ASSERT_EQ(info.commonInfo.indexSize, 0);
        ASSERT_EQ(info.commonInfo.indexLabelCount, 0);
        EXPECT_EQ(info.commonInfo.memory, index->getAllocationSize());
        validateSVSIndexAttributesInfo(info.svsInfo, params);
        VecSimIndex_Free(index);

        d = 1280;
        params.dim = d;
        params.blockSize = DEFAULT_BLOCK_SIZE - 10;
        params.alpha = 52;
        params.graph_max_degree = SVS_VAMANA_DEFAULT_GRAPH_MAX_DEGREE + 2;
        params.construction_window_size = SVS_VAMANA_DEFAULT_CONSTRUCTION_WINDOW_SIZE + 4;
        params.max_candidate_pool_size = params.construction_window_size * 8;
        params.prune_to = params.graph_max_degree + 56;
        params.use_search_history = VecSimOption_DISABLE;

        index = this->CreateNewIndex(params);
        ASSERT_INDEX(index);

        info = VecSimIndex_DebugInfo(index);
        ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVS);
        ASSERT_EQ(info.commonInfo.basicInfo.dim, d);
        ASSERT_FALSE(info.commonInfo.basicInfo.isMulti);
        ASSERT_FALSE(info.commonInfo.basicInfo.isTiered);

        // User args.
        ASSERT_EQ(info.commonInfo.basicInfo.blockSize, DEFAULT_BLOCK_SIZE - 10);
        ASSERT_EQ(info.commonInfo.indexSize, 0);
        validateSVSIndexAttributesInfo(info.svsInfo, params);

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
}

TYPED_TEST(FP16SVSTest, test_basic_svs_info_iterator) {
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
        ASSERT_INDEX(index);

        VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(index);
        VecSimDebugInfoIterator *infoIter = VecSimIndex_DebugInfoIterator(index);
        compareSVSIndexInfoToIterator(info, infoIter);
        VecSimDebugInfoIterator_Free(infoIter);
        VecSimIndex_Free(index);
    }
}

TYPED_TEST(FP16SVSTest, test_dynamic_svs_info_iterator) {
    if (TypeParam::get_quant_bits() == VecSimSvsQuant_8 ||
        TypeParam::get_quant_bits() == VecSimSvsQuant_8x8_LeanVec) {
        GTEST_SKIP() << "Already included in the test loop.";
    }
    constexpr size_t d = 128;
    for (auto quant_bits : {VecSimSvsQuant_NONE, VecSimSvsQuant_Scalar, VecSimSvsQuant_8,
                            VecSimSvsQuant_4, VecSimSvsQuant_4x4, VecSimSvsQuant_4x8,
                            VecSimSvsQuant_4x8_LeanVec, VecSimSvsQuant_8x8_LeanVec}) {

        SVSParams params = {
            .dim = d,
            .metric = VecSimMetric_Cosine,
            .blockSize = 1,
            /* SVS-Vamana specifics */
            .quantBits = quant_bits,
            .alpha = 1.2,
            .graph_max_degree = 64,
            .construction_window_size = 20,
            .max_candidate_pool_size = 1024,
            .prune_to = 60,
            .use_search_history = VecSimOption_ENABLE,
        };
        VecSimIndex *index = this->CreateNewIndex(params);
        ASSERT_INDEX(index);

        VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(index);
        validateSVSIndexAttributesInfo(info.svsInfo, params);

        VecSimDebugInfoIterator *infoIter = VecSimIndex_DebugInfoIterator(index);
        ASSERT_EQ(1, info.commonInfo.basicInfo.blockSize);
        ASSERT_EQ(0, info.commonInfo.indexSize);
        compareSVSIndexInfoToIterator(info, infoIter);
        VecSimDebugInfoIterator_Free(infoIter);

        float16 v[d];
        this->GenerateVector(v, d);
        // Add vector.
        VecSimIndex_AddVector(index, v, 0);
        info = VecSimIndex_DebugInfo(index);
        infoIter = VecSimIndex_DebugInfoIterator(index);
        ASSERT_EQ(1, info.commonInfo.indexSize);
        compareSVSIndexInfoToIterator(info, infoIter);
        VecSimDebugInfoIterator_Free(infoIter);

        // Delete vector.
        VecSimIndex_DeleteVector(index, 0);
        info = VecSimIndex_DebugInfo(index);
        infoIter = VecSimIndex_DebugInfoIterator(index);
        ASSERT_EQ(0, info.commonInfo.indexSize);
        compareSVSIndexInfoToIterator(info, infoIter);
        VecSimDebugInfoIterator_Free(infoIter);

        // Perform (or simulate) Search in all modes.
        VecSimIndex_AddVector(index, v, 0);
        auto res = VecSimIndex_TopKQuery(index, v, 1, nullptr, BY_SCORE);
        VecSimQueryReply_Free(res);
        info = VecSimIndex_DebugInfo(index);
        infoIter = VecSimIndex_DebugInfoIterator(index);
        ASSERT_EQ(STANDARD_KNN, info.commonInfo.lastMode);
        compareSVSIndexInfoToIterator(info, infoIter);
        VecSimDebugInfoIterator_Free(infoIter);

        res = VecSimIndex_RangeQuery(index, v, 1, nullptr, BY_SCORE);
        VecSimQueryReply_Free(res);
        info = VecSimIndex_DebugInfo(index);
        infoIter = VecSimIndex_DebugInfoIterator(index);
        ASSERT_EQ(RANGE_QUERY, info.commonInfo.lastMode);
        compareSVSIndexInfoToIterator(info, infoIter);
        VecSimDebugInfoIterator_Free(infoIter);

        ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 1, 1, true));
        info = VecSimIndex_DebugInfo(index);
        infoIter = VecSimIndex_DebugInfoIterator(index);
        ASSERT_EQ(HYBRID_ADHOC_BF, info.commonInfo.lastMode);
        compareSVSIndexInfoToIterator(info, infoIter);
        VecSimDebugInfoIterator_Free(infoIter);

        // Simulate the case where another call to the heuristics is done after realizing that
        // the subset size is smaller, and change the policy as a result.
        ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 1, 1, false));
        info = VecSimIndex_DebugInfo(index);
        infoIter = VecSimIndex_DebugInfoIterator(index);
        ASSERT_EQ(HYBRID_BATCHES_TO_ADHOC_BF, info.commonInfo.lastMode);
        compareSVSIndexInfoToIterator(info, infoIter);
        VecSimDebugInfoIterator_Free(infoIter);

        VecSimIndex_Free(index);
    }
}

TYPED_TEST(FP16SVSTest, test_get_distance) {
    const size_t dim = 4;
    static double constexpr expected_dists[2] = {0.25, -1.5}; // L2, IP

    for (VecSimMetric metric : {VecSimMetric_L2, VecSimMetric_IP}) {

        SVSParams params = {
            .dim = dim,
            .metric = metric,
        };

        VecSimIndex *index = this->CreateNewIndex(params);
        ASSERT_INDEX(index);

        float16 vec[dim];
        this->GenerateVector(vec, dim, 0.25, 0.25); // {0.25, 0.5, 0.75, 1}
        VecSimIndex_AddVector(index, vec, 0);
        ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

        float16 query[dim];
        this->GenerateVector(query, dim, 0.5, 0.25); // {0.5, 0.75, 1, 1.25}

        double dist = VecSimIndex_GetDistanceFrom_Unsafe(index, 0, query);

        // manually calculated. Values were chosen as such that don't cause any accuracy loss in
        // conversion from bfloat16 to float.
        ASSERT_EQ(dist, expected_dists[metric]) << "metric: " << metric;

        VecSimIndex_Free(index);
    }
}

TYPED_TEST(FP16SVSTest, svs_vector_search_test_ip) {
    const size_t dim = 4;
    const size_t n = 10;
    const size_t k = 5;

    for (size_t blocksize : {1, 12, DEFAULT_BLOCK_SIZE}) {

        SVSParams params = {.dim = dim,
                            .metric = VecSimMetric_IP,
                            .blockSize = blocksize,
                            /* SVS-Vamana specifics */
                            .alpha = 0.9,
                            .graph_max_degree = 64,
                            .construction_window_size = 20,
                            .max_candidate_pool_size = 1024,
                            .prune_to = 60,
                            .use_search_history = VecSimOption_ENABLE,
                            .leanvec_dim = dim / 4};

        VecSimIndex *index = this->CreateNewIndex(params);
        ASSERT_INDEX(index);

        VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(index);
        ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVS);
        ASSERT_EQ(info.commonInfo.basicInfo.blockSize, blocksize);

        auto svs_index = this->CastToSVS(index);
        ASSERT_NE(svs_index, nullptr);

        std::vector<std::array<float16, dim>> v(n);
        for (size_t i = 0; i < n; i++) {
            this->GenerateVector(v[i].data(), dim, i);
        }

        std::vector<size_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0);

        svs_index->addVectors(v.data(), ids.data(), n);
        ASSERT_EQ(VecSimIndex_IndexSize(index), n);

        float16 query[dim];
        this->GenerateVector(query, dim, 0, 1.0);
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

TYPED_TEST(FP16SVSTest, svs_vector_search_test_l2) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    constexpr size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    for (size_t blocksize : {1, 12, DEFAULT_BLOCK_SIZE}) {

        SVSParams params = {.dim = dim,
                            .metric = VecSimMetric_L2,
                            .blockSize = blocksize,
                            /* SVS-Vamana specifics */
                            .graph_max_degree = 64,
                            .construction_window_size = 20,
                            .max_candidate_pool_size = 1024,
                            .prune_to = 60,
                            .use_search_history = VecSimOption_ENABLE,
                            .leanvec_dim = dim / 4};

        VecSimIndex *index = this->CreateNewIndex(params);
        ASSERT_INDEX(index);

        VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(index);
        ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVS);
        ASSERT_EQ(info.commonInfo.basicInfo.blockSize, blocksize);

        for (size_t i = 0; i < n; i++) {
            this->GenerateAndAddVector(index, dim, i, i);
        }
        ASSERT_EQ(VecSimIndex_IndexSize(index), n);

        auto verify_res = [&](size_t id, double score, size_t index) {
            size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
            ASSERT_EQ(diff_id, (index + 1) / 2)
                << "index: " << index << " id: " << id << " score: " << score;
            ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)))
                << "index: " << index << " id: " << id << " score: " << score;
        };
        float16 query[dim];
        this->GenerateVector(query, dim, 50);
        runTopKSearchTest(index, query, k, verify_res);
        runTopKSearchTest(index, query, 0, verify_res); // For sanity, search for nothing

        VecSimIndex_Free(index);
    }
}

TYPED_TEST(FP16SVSTest, svs_search_empty_index) {
    size_t dim = 4;
    size_t n = 100;
    size_t k = 11;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    float16 query[dim];
    this->GenerateVector(query, dim, 50);

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
        this->GenerateAndAddVector(index, dim, i, i);
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

TYPED_TEST(FP16SVSTest, preferAdHocOptimization) {
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
            ASSERT_INDEX(index);

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
    ASSERT_INDEX(index);

    ASSERT_TRUE(VecSimIndex_PreferAdHocSearch(index, 0, 50, true));

    // Corner cases - subset size is greater than index size.
    ASSERT_EQ(VecSimIndex_PreferAdHocSearch(index, 42, 50, true),
              VecSimIndex_PreferAdHocSearch(index, 0, 50, true));

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, batchIteratorSwapIndices) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    constexpr size_t dim = 4;
    size_t n = 1000;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,

    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    float16 close_vec[dim];
    this->GenerateVector(close_vec, dim, 1.0);
    float16 further_vec[dim];
    this->GenerateVector(further_vec, dim, 2.0);

    VecSimIndex_AddVector(index, further_vec, 0);
    VecSimIndex_AddVector(index, close_vec, 1);
    VecSimIndex_AddVector(index, further_vec, 2);
    VecSimIndex_AddVector(index, close_vec, 3);
    VecSimIndex_AddVector(index, close_vec, 4);
    VecSimIndex_AddVector(index, close_vec, 5);
    for (size_t i = 6; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    // Query for (1,1,1,1) vector.
    TEST_DATA_T query[dim];
    this->GenerateVector(query, dim, 1.0);
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

TYPED_TEST(FP16SVSTest, svs_vector_search_test_cosine) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    const size_t dim = 128;
    const size_t n = 50;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_Cosine,
        .alpha = 0.9,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    // To meet accuracy in LVQ case we have to add bulk of vectors at once.
    std::vector<std::array<float16, dim>> v(n);
    for (size_t i = 1; i <= n; i++) {
        auto &f = v[i - 1];
        // f[0] = vecsim_types::FP32_to_FP16((float)i / n);
        for (size_t j = 0; j < dim; j++) {
            // for (size_t j = 1; j < dim; j++) {
            f[j] = vecsim_types::FP32_to_FP16((float)i / n);
            // f[j] = vecsim_types::FP32_to_FP16(1.0);
        }
        // test_utils::populate_float16_vec(v[i].data(), dim, i, -1.0f, 1.0f);
    }

    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 1);

    auto svs_index = this->CastToSVS(index);
    svs_index->addVectors(v.data(), ids.data(), n);

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    // float16 query[dim];
    // this->GenerateVector(query, dim, 0.1);
    auto *query = v[0].data();

    // topK search will normalize the query so we keep the original data to
    // avoid normalizing twice.
    float16 normalized_query[dim];
    memcpy(normalized_query, query, dim * sizeof(float16));
    VecSim_Normalize(normalized_query, dim, params.type);

    auto verify_res = [&](size_t id, double score, size_t result_rank) {
        double expected_score = index->getDistanceFrom_Unsafe(id, normalized_query);
        // Verify that abs difference between the actual and expected score is at most 1/10^5.
        float16 fp16_score = vecsim_types::FP32_to_FP16(score);
        float16 fp16_expected_score = vecsim_types::FP32_to_FP16(expected_score);
        ASSERT_EQ(id, (result_rank + 1))
            << "result_rank: " << result_rank << " id: " << id << " score: " << score
            << " expected_score: " << expected_score << " fp16_score: " << fp16_score
            << " fp16_expected_score: " << fp16_expected_score;
        ASSERT_EQ(fp16_score, fp16_expected_score)
            << "result_rank: " << result_rank << " id: " << id;
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
            // In case of quantization, the result is not guaranteed to be properly ordered
            if constexpr (TypeParam::get_quant_bits() == 0) {
                ASSERT_EQ(id, (n - n_res * iteration_num - result_rank));
            }
            double expected_score = index->getDistanceFrom_Unsafe(id, normalized_query);
            // Verify that abs difference between the actual and expected score is at most 1/10^5.
            ASSERT_NEAR(score, expected_score, 1e-5f);
        };
        runBatchIteratorSearchTest(batchIterator, n_res, verify_res_batch);
        iteration_num++;
    }
    ASSERT_EQ(iteration_num, n / n_res);
    VecSimBatchIterator_Free(batchIterator);
    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, testSizeEstimation) {
    size_t dim = 64;
    auto quantBits = TypeParam::get_quant_bits();
#if HAVE_SVS_LVQ
    // SVS block sizes always rounded to a power of 2
    // This why, in case of quantization, actual block size can be differ than requested
    // In addition, block size to be passed to graph and dataset counted in bytes,
    // converted then to a number of elements.
    // IMHO, would be better to always interpret block size to a number of elements
    // rather than conversion to-from number of bytes
    if (quantBits != VecSimSvsQuant_NONE && !this->isFallbackToSQ()) {
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
        /* SVS-Vamana specifics */
        .graph_max_degree = 63, // x^2-1 to round the graph block size
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);
    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();
    ASSERT_EQ(estimation, actual);

    estimation = EstimateElementSize(params) * bs;

    this->GenerateAndAddVector(index, dim, 0);
    actual = index->getAllocationSize() - actual; // get the delta
    ASSERT_GT(actual, 0);
    // LVQ element estimation accuracy is low
    double estimation_accuracy = (quantBits != VecSimSvsQuant_NONE) ? 0.1 : 0.01;
    ASSERT_GE(estimation * (1.0 + estimation_accuracy), actual);
    ASSERT_LE(estimation * (1.0 - estimation_accuracy), actual);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, rangeQuery) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t n = 100;
    size_t dim = 4;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
    };

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    float pivot_value = 1.0f;
    float16 pivot_vec[dim];
    this->GenerateVector(pivot_vec, dim, pivot_value);

    float radius = 1.5f;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(pivot_value - radius, pivot_value + radius);

    // insert 20 vectors near a pivot vector.
    size_t n_close = 20;
    for (size_t i = 0; i < n_close; i++) {
        float random_number = dis(gen);
        this->GenerateAndAddVector(index, dim, i, random_number);
    }

    float16 max_vec[dim];
    this->GenerateVector(max_vec, dim, pivot_value + radius);
    double max_dist = FP16_L2Sqr(pivot_vec, max_vec, dim);

    // Add more vectors far from the pivot vector, under the same labels
    for (size_t i = n_close; i < n; i++) {
        float random_number = dis(gen);
        this->GenerateAndAddVector(index, dim, i, 5.0 + random_number);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    auto verify_res_by_score = [&](size_t id, double score, size_t index) {
        ASSERT_LE(id, n_close - 1) << "score: " << score;
        ASSERT_LE(score, max_dist);
    };
    uint expected_num_results = n_close;

    runRangeQueryTest(index, pivot_vec, max_dist, verify_res_by_score, expected_num_results,
                      BY_SCORE);

    // Rerun with a given query params.
    SVSRuntimeParams svsRuntimeParams = {.epsilon = 100.0};
    auto query_params = CreateQueryParams(svsRuntimeParams);
    query_params.batchSize = 100;
    runRangeQueryTest(index, pivot_vec, max_dist, verify_res_by_score, expected_num_results,
                      BY_SCORE, &query_params);

    // Get results by id.
    auto verify_res_by_id = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, index);
        ASSERT_LE(score, max_dist);
    };
    runRangeQueryTest(index, pivot_vec, max_dist, verify_res_by_id, expected_num_results);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, rangeQueryCosine) {
    GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";

    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    const size_t n = 100;
    const size_t dim = 4;

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
    ASSERT_INDEX(index);

    // To meet accurary in LVQ case we have to add bulk of vectors at once.
    std::vector<std::array<float16, dim>> v(n);
    std::vector<size_t> ids(n);

    for (size_t i = 0; i < n; i++) {
        auto &f = v[i - 1];
        f[0] = vecsim_types::FP32_to_FP16(float(i + 1) / n);
        for (size_t j = 1; j < dim; j++) {
            f[j] = vecsim_types::FP32_to_FP16(1.0);
        }
        // Use as label := n - (internal id)
        ids[i] = n - i;
    }

    auto svs_index = this->CastToSVS(index);
    svs_index->addVectors(v.data(), ids.data(), n);

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    float16 query[dim];
    query[0] = vecsim_types::FP32_to_FP16(1.1);
    for (size_t i = 1; i < dim; i++) {
        query[i] = vecsim_types::FP32_to_FP16(1.0);
    }
    auto verify_res = [&](size_t id, double score, size_t result_rank) {
        ASSERT_EQ(id, result_rank + 1);
        double expected_score = index->getDistanceFrom_Unsafe(id, query);
        // Verify that abs difference between the actual and expected score is at most 1/10^5.
        ASSERT_NEAR(score, expected_score, 1e-5f);
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

TYPED_TEST(FP16SVSTest, quant_modes) {
    if (TypeParam::get_quant_bits() == VecSimSvsQuant_8 ||
        TypeParam::get_quant_bits() == VecSimSvsQuant_8x8_LeanVec) {
        GTEST_SKIP() << "Already included in the test loop.";
    }

    // Limit VecSim log level to avoid printing too much information
    VecSimIndexInterface::setLogCallbackFunction(svsTestLogCallBackNoDebug);

    const size_t dim = 4;
    const size_t n = 100;
    const size_t k = 11;

    for (auto quant_bits : {VecSimSvsQuant_NONE, VecSimSvsQuant_Scalar, VecSimSvsQuant_8,
                            VecSimSvsQuant_4, VecSimSvsQuant_4x4, VecSimSvsQuant_4x8,
                            VecSimSvsQuant_4x8_LeanVec, VecSimSvsQuant_8x8_LeanVec}) {

        SCOPED_TRACE("quant_bits = " + std::string(VecSimQuantBits_ToString(quant_bits)));
        SVSParams params = {
            .dim = dim,
            .metric = VecSimMetric_L2,
            /* SVS-Vamana specifics */
            .quantBits = quant_bits,
            .graph_max_degree = 63,
        };

        VecSimIndex *index = this->CreateNewIndex(params);
        ASSERT_INDEX(index);

        // Test initial size estimation
        // EstimateInitialSize is called after CreateNewIndex because params struct is
        // changed in CreateNewIndex.
        size_t estimation = EstimateInitialSize(params);
        size_t actual = index->getAllocationSize();
        EXPECT_EQ(estimation, actual);

        EXPECT_EQ(VecSimIndex_IndexSize(index), 0);
        EXPECT_EQ(index->debugInfo().svsInfo.quantBits,
                  std::get<0>(svs_details::isSVSQuantBitsSupported(quant_bits)));

        std::vector<std::array<float16, dim>> v(n);
        for (size_t i = 0; i < n; i++) {
            this->GenerateVector(v[i].data(), dim, i);
        }

        std::vector<size_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0);

        auto svs_index = dynamic_cast<SVSIndexBase *>(index);
        ASSERT_NE(svs_index, nullptr);
        svs_index->addVectors(v.data(), ids.data(), n);

        ASSERT_EQ(VecSimIndex_IndexSize(index), n);

        estimation = EstimateElementSize(params) * DEFAULT_BLOCK_SIZE;
        actual = index->getAllocationSize() - actual; // get the delta
        ASSERT_GT(actual, 0);
        // LVQ element size estimation accuracy is low
        auto quant_bits_fallback = std::get<0>(svs_details::isSVSQuantBitsSupported(quant_bits));
        double estimation_accuracy = (quant_bits_fallback != VecSimSvsQuant_NONE) ? 0.12 : 0.01;
        ASSERT_GE(estimation * (1.0 + estimation_accuracy), actual);
        ASSERT_LE(estimation * (1.0 - estimation_accuracy), actual);

        float16 query[dim];
        this->GenerateVector(query, dim, 50);
        auto verify_res = [&](size_t id, double score, size_t idx) {
            EXPECT_DOUBLE_EQ(VecSimIndex_GetDistanceFrom_Unsafe(index, id, query), score)
                << "idx: " << idx << " id: " << id;
            EXPECT_EQ(id, (idx + 45)) << "idx: " << idx << " id: " << id << " score: " << score;
        };
        runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

        VecSimIndex_Free(index);
    }
}

TYPED_TEST(FP16SVSTest, test_override_all) {
    const size_t dim = 4;
    const size_t n = 100;
    size_t new_n = 250;
    const size_t k = 11;

    SVSParams params = {
        .dim = dim,
        .metric = VecSimMetric_L2,
    };
    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    auto svs_index = this->CastToSVS(index);
    ASSERT_NE(svs_index, nullptr);

    std::vector<std::array<float16, dim>> v(n);
    for (size_t i = 0; i < n; i++) {
        this->GenerateVector(v[i].data(), dim, i);
    }

    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    svs_index->addVectors(v.data(), ids.data(), n);

    // Override vectors one-by-one
    for (size_t i = 0; i < n; i++) {
        index->addVector(v[i].data(), ids[i]);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float16 query[dim];
    this->GenerateVector(query, dim, 50);
    auto verify_res = [&](size_t id, double score, size_t index) { EXPECT_EQ(id, (index + 45)); };
    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);

    // Add up to new_n vectors.
    for (size_t i = n; i < new_n; i++) {
        ASSERT_EQ(this->GenerateAndAddVector(index, dim, i, i), 1);
    }

    this->GenerateVector(query, dim, new_n);
    auto verify_res_with_new_n = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, new_n - 1 - index) << "id: " << id << " score: " << score;
        float16 a = vecsim_types::FP32_to_FP16(new_n);
        float16 b = vecsim_types::FP32_to_FP16(id);
        float diff = vecsim_types::FP16_to_FP32(a) - vecsim_types::FP16_to_FP32(b);
        float exp_score = 4 * diff * diff;
        ASSERT_EQ(score, exp_score) << "id: " << id << " score: " << score;
    };
    runTopKSearchTest(index, query, new_n, verify_res_with_new_n);
    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSTest, scalar_quantization_query) {
    if (TypeParam::get_quant_bits() == VecSimSvsQuant_8 ||
        TypeParam::get_quant_bits() == VecSimSvsQuant_8x8_LeanVec) {
        GTEST_SKIP() << "Test only VecSimSvsQuant_NONE and VecSimSvsQuant_Scalar.";
    }
    // Limit VecSim log level to avoid printing too much information
    VecSimIndexInterface::setLogCallbackFunction(svsTestLogCallBackNoDebug);

    const size_t dim = 32;
    const size_t bs = 1024;
    const size_t n = 100;
    const size_t k = 10;
    const double quant_precision = 1.0 / (1 << 7); // int8 quantization precision

    std::vector<std::array<float16, dim>> dataset(n);
    for (size_t i = 0; i < n; i++) {
        test_utils::populate_float16_vec(dataset[i].data(), dim, i, -1.0, 1.0);
    }
    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    float16 query[dim];
    test_utils::populate_float16_vec(query, dim, n, -1.0, 1.0);

    VecSimQueryReply *fp_results = nullptr;
    auto verify_res = [&](size_t id, double score, size_t result_rank) {
        const auto &fp_result = fp_results->results[result_rank];
        ASSERT_EQ(id, fp_result.id);
        // Verify that relative difference between the actual and expected score is within 8-bit
        // quantization precision.
        auto expected_diff = std::abs(score * quant_precision);
        ASSERT_NEAR(score, fp_result.score, expected_diff);
    };

    const std::pair<VecSimMetric, double> metrics[] = {
        {VecSimMetric_L2, 30.},
        {VecSimMetric_Cosine, 1.0},
    };

    for (auto [metric, radius] : metrics) {
        SCOPED_TRACE(VecSimMetric_ToString(metric));
        SVSParams params = {
            .dim = dim,
            .metric = metric,
            .blockSize = bs,
            /* SVS-Vamana specifics */
            .graph_max_degree = 63, // x^2-1 to round the graph block size
            .construction_window_size = 20,
            .max_candidate_pool_size = 1024,
            .prune_to = 60,
            .use_search_history = VecSimOption_ENABLE,
        };
        params.quantBits = VecSimSvsQuant_NONE;

        VecSimIndex *index_fp = this->CreateNewIndex(params);
        ASSERT_INDEX(index_fp);

        dynamic_cast<SVSIndexBase *>(index_fp)->addVectors(dataset.data(), ids.data(), n);
        ASSERT_EQ(VecSimIndex_IndexSize(index_fp), n);

        params.quantBits = VecSimSvsQuant_Scalar;
        VecSimIndex *index_sq = this->CreateNewIndex(params);
        ASSERT_INDEX(index_sq);

        auto estimation = EstimateInitialSize(params);
        auto actual = index_sq->getAllocationSize();
        ASSERT_EQ(estimation, actual);

        dynamic_cast<SVSIndexBase *>(index_sq)->addVectors(dataset.data(), ids.data(), n);
        ASSERT_EQ(VecSimIndex_IndexSize(index_sq), n);
        ASSERT_EQ(index_sq->indexCapacity(), n);

        estimation = EstimateElementSize(params) * params.blockSize;
        actual = index_sq->getAllocationSize() - actual; // get the delta
        ASSERT_GT(actual, 0);
        ASSERT_GE(estimation * 1.01, actual);
        ASSERT_LE(estimation * 0.99, actual);

        // test topK search
        fp_results = VecSimIndex_TopKQuery(index_fp, query, k, nullptr, BY_ID);
        runTopKSearchTest(index_sq, query, k, verify_res, nullptr, BY_ID);
        VecSimQueryReply_Free(fp_results);

        // test range search
        fp_results = VecSimIndex_RangeQuery(index_fp, query, radius, nullptr, BY_ID);
        ASSERT_GT(fp_results->results.size(), 0);
        runRangeQueryTest(index_sq, query, radius, verify_res, fp_results->results.size(), BY_ID);
        VecSimQueryReply_Free(fp_results);

        VecSimIndex_Free(index_sq);
        VecSimIndex_Free(index_fp);
    }
}

/**
 * * * * * * * * * * * * * * * * * *
 *     Multi Vector index Tests    *
 * * * * * * * * * * * * * * * * * *
 */

template <typename index_type_t>
class FP16SVSMultiTest : public FP16SVSTest<index_type_t> {
public:
    VecSimIndex *CreateNewIndex(SVSParams &params) override {
        VecSimIndexInterface::setLogCallbackFunction(svsTestLogCallBackNoDebug);
        params.multi = true;
        this->SetTypeParams(params);
        VecSimParams index_params = CreateParams(params);
        return VecSimIndex_New(&index_params);
    }
};

TYPED_TEST_SUITE(FP16SVSMultiTest, SVSDataTypeSet);

TYPED_TEST(FP16SVSMultiTest, vector_add_multiple_test) {
    const size_t dim = 4;
    float initial_value = 0.5f;
    float step = 1.0f;
    const size_t n = 5;
    constexpr size_t labels_count = 1;
    const size_t label = 46;

    SVSParams params = {.dim = dim, .metric = VecSimMetric_IP, .multi = true};
    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    std::vector<float16> v(dim * n);
    for (size_t i = 0; i < n; i++) {
        this->GenerateVector(v.data() + i * dim, dim, i * initial_value, step);
    }
    std::vector<size_t> ids(n, label);

    // Adding same vector multiple times under the same label
    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(VecSimIndex_AddVector(index, v.data() + i * dim, label), 1);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), labels_count);

    // Deleting the label. All the vectors should be deleted.
    ASSERT_EQ(VecSimIndex_DeleteVector(index, label), n);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);
    ASSERT_EQ(index->indexLabelCount(), 0);

    // Adding multiple vectors at once under the same label
    auto svs_index = this->CastToSVS(index);
    svs_index->addVectors(v.data(), ids.data(), n);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), labels_count);

    if (!(TypeParam::get_quant_bits() == VecSimSvsQuant_8 ||
          TypeParam::get_quant_bits() == VecSimSvsQuant_8x8_LeanVec)) {
        // Get vector is only implemented for non-compressed indices.
        auto vectors_data =
            (dynamic_cast<VecSimIndexAbstract<svs_details::vecsim_dt<svs::Float16>, float> *>(
                 index))
                ->getStoredVectorDataByLabel(label);
        for (size_t i = 0; i < n; i++) {
            const char *data_ptr = vectors_data.at(i).data();
            for (size_t j = 0; j < dim; j++) {
                ASSERT_EQ(vecsim_types::FP16_to_FP32(((float16 *)data_ptr)[j]),
                          i * initial_value + step * float(j));
            }
        }
    }

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSMultiTest, vector_search_test) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t dim = 4;
    size_t n = 1000;
    size_t n_labels = 100;
    size_t k = 11;

    SVSParams params = {.dim = dim, .metric = VecSimMetric_L2};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i % n_labels, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    TEST_DATA_T query[dim];
    this->GenerateVector(query, dim, 50);

    auto verify_res = [&](size_t id, double score, size_t index) {
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2)
            << "index: " << index << " id: " << id << " score: " << score;
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)))
            << "index: " << index << " id: " << id << " score: " << score;
    };
    runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSMultiTest, search_more_than_there_is) {
    const size_t dim = 4;
    const size_t n = 5;
    const size_t perLabel = 3;
    const size_t n_labels = ceil((float)n / perLabel);
    const size_t k = 3;
    // This test add 5 vectors under 2 labels, and then query for 3 results.
    // We want to make sure we get only 2 results back (because the results should have unique
    // labels), although the index contains 5 vectors.

    SVSParams params = {.dim = dim, .metric = VecSimMetric_L2};

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    auto svs_index = this->CastToSVS(index);
    ASSERT_NE(svs_index, nullptr);

    std::vector<std::array<float16, dim>> v(n);
    std::vector<size_t> ids(n);

    for (size_t i = 0; i < n; i++) {
        ids[i] = i / perLabel;
        this->GenerateVector(v[i].data(), dim, i);
    }

    svs_index->addVectors(v.data(), ids.data(), n);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    float16 query[dim];
    this->GenerateVector(query, dim, 0);

    VecSimQueryReply *res = VecSimIndex_TopKQuery(index, query, k, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), n_labels);
    auto it = VecSimQueryReply_GetIterator(res);
    for (size_t i = 0; i < n_labels; i++) {
        auto el = VecSimQueryReply_IteratorNext(it);
        // SVS Scalar quantization is not enough precise
        if (!this->isFallbackToSQ()) {
            ASSERT_EQ(VecSimQueryResult_GetScore(el), i * perLabel * i * perLabel * dim);
        }
        labelType element_label = VecSimQueryResult_GetId(el);
        ASSERT_EQ(element_label, i);
    }
    VecSimQueryReply_IteratorFree(it);
    VecSimQueryReply_Free(res);
    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSMultiTest, find_better_score) {
    const size_t n = 100;
    const size_t k = 10;
    const size_t n_labels = 10;
    const size_t dim = 4;

    SVSParams params = {.dim = dim, .metric = VecSimMetric_L2};

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    auto svs_index = this->CastToSVS(index);
    ASSERT_NE(svs_index, nullptr);

    // Building the index. Each label gets 10 vectors with decreasing (by insertion order) element
    // value.
    std::vector<size_t> ids(n);
    std::vector<std::array<float16, dim>> v(n);
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
        ids[i] = i / n_labels;
        this->GenerateVector(v[i].data(), dim, el);
        // This should be the best score for each label.
        if (i % n_labels == n_labels - 1) {
            // `el * el * dim` is the L2-squared value with the 0 vector.
            scores.emplace(i / n_labels, el * el * dim);
        }
    }
    svs_index->addVectors(v.data(), ids.data(), n);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, k - index - 1);
        // SVS scalar quantization is not enough precise
        if (!this->isFallbackToSQ()) {
            ASSERT_FLOAT_EQ(score, scores[id]);
        }
    };

    float16 query[dim];
    this->GenerateVector(query, dim, 0.0);
    runTopKSearchTest(index, query, k, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSMultiTest, find_better_score_after_pop) {
    const size_t n = 12;
    const size_t n_labels = 3;
    const size_t dim = 4;

    SVSParams params = {.dim = dim, .metric = VecSimMetric_L2};

    VecSimIndex *index = this->CreateNewIndex(params);
    ASSERT_INDEX(index);

    auto svs_index = this->CastToSVS(index);
    ASSERT_NE(svs_index, nullptr);

    // Building the index. Each is better than the previous one.
    std::vector<size_t> ids(n);
    std::vector<std::array<float16, dim>> v(n);
    for (size_t i = 0; i < n; i++) {
        size_t el = n - i;
        ids[i] = i % n_labels;
        this->GenerateVector(v[i].data(), dim, el);
    }
    svs_index->addVectors(v.data(), ids.data(), n);
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    float16 query[dim];
    this->GenerateVector(query, dim, 0);
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, n_labels - index - 1);
    };

    runTopKSearchTest(index, query, n_labels, verify_res);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSMultiTest, reindexing_same_vector_different_id) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t n = 100;
    size_t k = 10;
    constexpr size_t dim = 4;
    size_t perLabel = 3;

    SVSParams params = {.dim = dim, .metric = VecSimMetric_L2};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        // i / 10 is in integer (take the "floor" value)
        this->GenerateAndAddVector(index, dim, i, i / 10);
    }
    // Add more vectors under the same labels. their scores should be worst.
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < perLabel - 1; j++) {
            this->GenerateAndAddVector(index, dim, i, float(i / 10) + n);
        }
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n * perLabel);

    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float16 query[dim];
    this->GenerateVector(query, dim, 4.9, 0.05);
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
        this->GenerateAndAddVector(index, dim, i + 10, i / 10);
    }
    // Add more vectors under the same labels. their scores should be worst.
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < perLabel - 1; j++) {
            this->GenerateAndAddVector(index, dim, i + 10, float(i / 10) + n);
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

TYPED_TEST(FP16SVSMultiTest, test_svs_info) {
    size_t d = 128;
    // Build with default args
    SVSParams params = {.dim = d, .metric = VecSimMetric_L2};
    VecSimIndex *index = this->CreateNewIndex(params);

    VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(index);
    ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_SVS);
    ASSERT_EQ(info.commonInfo.basicInfo.dim, d);
    ASSERT_TRUE(info.commonInfo.basicInfo.isMulti);
    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSMultiTest, test_basic_svs_info_iterator) {
    size_t n = 100;
    size_t d = 128;
    VecSimMetric metrics[3] = {VecSimMetric_Cosine, VecSimMetric_IP, VecSimMetric_L2};

    for (size_t i = 0; i < 3; i++) {
        SCOPED_TRACE("Testing metric: " + std::string(VecSimMetric_ToString(metrics[i])));

        // Build with default args.
        SVSParams params = {.dim = d, .metric = metrics[i]};
        VecSimIndex *index = this->CreateNewIndex(params);

        VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(index);
        VecSimDebugInfoIterator *infoIter = VecSimIndex_DebugInfoIterator(index);
        compareSVSIndexInfoToIterator(info, infoIter);
        VecSimDebugInfoIterator_Free(infoIter);
        VecSimIndex_Free(index);
    }
}

TYPED_TEST(FP16SVSMultiTest, rangeQuery) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t n_labels = 100;
    size_t n_close = 20;
    size_t per_label = 5;
    size_t dim = 4;

    size_t n = n_labels * per_label;

    SVSParams params{.dim = dim, .metric = VecSimMetric_L2, .blockSize = n / 2};
    VecSimIndex *index = this->CreateNewIndex(params);

    float pivot_value = 1.0f;
    float16 pivot_vec[dim];
    this->GenerateVector(pivot_vec, dim, pivot_value);

    float radius = 1.5f;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(pivot_value - radius, pivot_value + radius);

    for (size_t i = 0; i < n_close; i++) {
        // First vector of each label among n_close labels is close to the pivot vector.
        float random_number = dis(gen);
        this->GenerateAndAddVector(index, dim, i, random_number);
    }

    float16 max_vec[dim];
    this->GenerateVector(max_vec, dim, pivot_value + radius);
    double max_dist = FP16_L2Sqr(pivot_vec, max_vec, dim);

    for (size_t i = 0; i < n_labels; i++) {
        size_t max_vec_for_label = per_label;
        if (i < n_close) {
            max_vec_for_label -= 1;
        }
        // Add more vectors, some under the same labels, worse than the previous vector (for the
        // given query)
        for (size_t j = 0; j < max_vec_for_label; j++) {
            float random_number = dis(gen);
            this->GenerateAndAddVector(index, dim, i, 5.0 + random_number);
        }
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    auto verify_res_by_score = [&](size_t id, double score, size_t index) {
        ASSERT_LE(id, n_close - 1) << "score: " << score;
        ASSERT_LE(score, max_dist);
    };
    uint expected_num_results = n_close;

    runRangeQueryTest(index, pivot_vec, max_dist, verify_res_by_score, expected_num_results,
                      BY_SCORE);

    // Rerun with a given query params.
    SVSRuntimeParams svsRuntimeParams = {.epsilon = 100.0};
    auto query_params = CreateQueryParams(svsRuntimeParams);
    query_params.batchSize = 100;
    runRangeQueryTest(index, pivot_vec, max_dist, verify_res_by_score, expected_num_results,
                      BY_SCORE, &query_params);

    // Get results by id.
    auto verify_res_by_id = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, index);
        ASSERT_LE(score, max_dist);
    };
    runRangeQueryTest(index, pivot_vec, max_dist, verify_res_by_id, expected_num_results);

    VecSimIndex_Free(index);
}

TYPED_TEST(FP16SVSMultiTest, svs_batch_iterator_basic) {
    // Scalar quantization accuracy is insufficient for this test.
    if (this->isFallbackToSQ()) {
        GTEST_SKIP() << "SVS Scalar quantization accuracy is insufficient for this test.";
    }
    size_t dim = 4;
    size_t n_labels = 1000;
    size_t perLabel = 5;

    size_t n = n_labels * perLabel;

    SVSParams params = {.dim = dim, .metric = VecSimMetric_L2};

    VecSimIndex *index = this->CreateNewIndex(params);

    // For every i, add the vector (i,i,i,i) under the label i.
    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i / perLabel, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    // Query for (n,n,n,n) vector (recall that n-1 is the largest id in te index).
    float16 query[dim];
    this->GenerateVector(query, dim, n);

    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
    size_t iteration_num = 0;

    // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet
    // in every iteration. The results order should be sorted by their score (distance from the
    // query vector), which means sorted from the largest id to the lowest.
    size_t n_res = 5;
    while (VecSimBatchIterator_HasNext(batchIterator)) {
        std::vector<size_t> expected_ids(n_res);
        for (size_t i = 0; i < n_res; i++) {
            expected_ids[i] = (n_labels - iteration_num * n_res - i - 1);
        }
        auto verify_res = [&](size_t id, double score, size_t index) {
            ASSERT_EQ(expected_ids[index], id);
        };
        runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
        iteration_num++;
    }
    ASSERT_EQ(iteration_num, n_labels / n_res);
    VecSimBatchIterator_Free(batchIterator);

    VecSimIndex_Free(index);
}

#else // HAVE_SVS

TEST(FP16SVSTest, svs_not_supported) {
    SVSParams params = {
        .type = VecSimType_FLOAT16,
        .dim = 16,
        .metric = VecSimMetric_IP,
    };
    auto index_params = CreateParams(params);
    auto index = VecSimIndex_New(&index_params);
    ASSERT_EQ(index, nullptr);

    auto size = VecSimIndex_EstimateInitialSize(&index_params);
    ASSERT_EQ(size, -1);

    auto size2 = VecSimIndex_EstimateElementSize(&index_params);
    ASSERT_EQ(size2, -1);
}

#endif
