/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/types/float16.h"
#include "VecSim/types/sq8.h"
#include "VecSim/vec_sim.h"
#include "unit_test_utils.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <type_traits>

template <VecSimType type, typename DataType, bool WithQuantParams>
struct HNSWSQ8IndexType : IndexType<type, DataType, float> {
    static constexpr bool with_quant_params = WithQuantParams;
};

// Typed tests default to L2, where mean-centered FLOAT16 is unsupported. A parameter test below
// covers both that restriction and the supported inner-product case.
using HNSWSQ8DataTypeSet =
    ::testing::Types<HNSWSQ8IndexType<VecSimType_FLOAT32, float, false>,
                     HNSWSQ8IndexType<VecSimType_FLOAT32, float, true>,
                     HNSWSQ8IndexType<VecSimType_FLOAT16, vecsim_types::float16, false>>;

template <typename index_type_t>
class HNSWSQ8Test : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;

protected:
    static constexpr float quantization_mean_value = 1.0f;

    static data_t ToDataType(float value) {
        if constexpr (std::is_same_v<data_t, vecsim_types::float16>) {
            return vecsim_types::FP32_to_FP16(value);
        } else {
            return value;
        }
    }

    void SetUp(HNSWParams &params) {
        params.type = index_type_t::get_index_type();
        params.quantType = VecSimQuant_SQ8;
        if constexpr (index_type_t::with_quant_params) {
            quantization_mean.assign(params.dim, quantization_mean_value);
            params.quantParams = quantization_mean.data();
        }
        VecSimParams vecsim_params = CreateParams(params);
        index = VecSimIndex_New(&vecsim_params);
        ASSERT_NE(index, nullptr);
        dim = params.dim;
    }

    void TearDown() override {
        if (index) {
            VecSimIndex_Free(index);
        }
    }

    HNSWIndex<data_t, float> *CastToHNSW() {
        return dynamic_cast<HNSWIndex<data_t, float> *>(index);
    }

    void GenerateVector(data_t *out_vec, float initial_value = 0.25f, float step = 0.0f) {
        for (size_t i = 0; i < dim; i++) {
            out_vec[i] = ToDataType(initial_value + step * static_cast<float>(i));
        }
    }

    int GenerateAndAddVector(size_t label, float initial_value = 0.25f, float step = 0.0f) {
        std::vector<data_t> vector(dim);
        GenerateVector(vector.data(), initial_value, step);
        return VecSimIndex_AddVector(index, vector.data(), label);
    }

    void create_index_test();
    void search_by_id_test();
    void search_by_score_test();
    void search_empty_index_test();
    void test_override();
    void test_range_query();
    void test_get_distance(VecSimMetric metric, bool multi);
    void test_batch_iterator_basic();

    VecSimIndex *index = nullptr;
    size_t dim = 0;
    std::vector<float> quantization_mean;
};

TYPED_TEST_SUITE(HNSWSQ8Test, HNSWSQ8DataTypeSet);

// Index creation

template <typename index_type_t>
void HNSWSQ8Test<index_type_t>::create_index_test() {
    HNSWParams params = {.dim = 40, .M = 16, .efConstruction = 200};
    SetUp(params);

    constexpr float initial_value = 0.5f;
    constexpr float step = 1.0f;
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0u);
    ASSERT_EQ(GenerateAndAddVector(0, initial_value, step), 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1u);

    auto *hnsw_index = CastToHNSW();
    ASSERT_NE(hnsw_index, nullptr);
    const auto *stored = reinterpret_cast<const uint8_t *>(hnsw_index->getDataByInternalId(0));
    EXPECT_EQ(stored[0], 0);
    EXPECT_EQ(stored[dim - 1], 255);

    // Stored layout: quantized bytes followed by the minimum value and quantization delta.
    float stored_min;
    float stored_delta;
    std::memcpy(&stored_min, stored + dim + sq8::MIN_VAL * sizeof(float), sizeof(float));
    std::memcpy(&stored_delta, stored + dim + sq8::DELTA * sizeof(float), sizeof(float));
    const float expected_min =
        initial_value - (index_type_t::with_quant_params ? quantization_mean_value : 0.0f);
    EXPECT_FLOAT_EQ(stored_min, expected_min);
    EXPECT_FLOAT_EQ(stored_delta, step * static_cast<float>(dim - 1) / 255.0f);

    EXPECT_EQ(index->basicInfo().type, index_type_t::get_index_type());
    EXPECT_EQ(index->basicInfo().algo, VecSimAlgo_HNSWLIB);
}

TYPED_TEST(HNSWSQ8Test, CreateIndex) { this->create_index_test(); }

TYPED_TEST(HNSWSQ8Test, RejectStandaloneCosine) {
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = 4,
                         .metric = VecSimMetric_Cosine,
                         .quantType = VecSimQuant_SQ8};
    if constexpr (TypeParam::with_quant_params) {
        this->quantization_mean.assign(params.dim, this->quantization_mean_value);
        params.quantParams = this->quantization_mean.data();
    }

    VecSimParams vecsim_params = CreateParams(params);
    this->index = VecSimIndex_New(&vecsim_params);
    EXPECT_EQ(this->index, nullptr);

    EXPECT_EQ(VecSimIndex_EstimateInitialSize(&vecsim_params), SIZE_MAX);
}

// Size estimation

TYPED_TEST(HNSWSQ8Test, SizeEstimation) {
    constexpr size_t block_size = 256;
    HNSWParams params = {.dim = 128, .blockSize = block_size, .M = 64};
    this->SetUp(params);

    // SetUp fills the data type and SQ8 settings used by the estimator.
    EXPECT_EQ(EstimateInitialSize(params), this->index->getAllocationSize());

    size_t label = 0;
    while (this->index->indexSize() < 200 || this->index->indexSize() % block_size != 0) {
        ASSERT_EQ(this->GenerateAndAddVector(label, static_cast<float>(label)), 1);
        label++;
    }

    // Measure the allocation caused by growing the index by one block.
    const size_t estimation = EstimateElementSize(params) * block_size;
    const size_t before = this->index->getAllocationSize();
    ASSERT_EQ(this->GenerateAndAddVector(label, static_cast<float>(label)), 1);
    const size_t actual = this->index->getAllocationSize() - before;

    EXPECT_GE(estimation, actual * 0.99);
    EXPECT_LE(estimation, actual * 1.01);
}

// Search behavior

template <typename index_type_t>
void HNSWSQ8Test<index_type_t>::search_by_id_test() {
    HNSWParams params = {
        .dim = 4, .initialCapacity = 200, .M = 16, .efConstruction = 200, .efRuntime = 100};
    SetUp(params);

    for (size_t i = 0; i < 100; i++) {
        ASSERT_EQ(GenerateAndAddVector(i, static_cast<float>(i)), 1);
    }

    data_t query[4];
    GenerateVector(query, 50.0f);
    // BY_ID returns the 11 nearest labels (45 through 55) in label order.
    static constexpr size_t expected[] = {45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55};
    auto verify = [&](size_t id, double score, size_t result_index) {
        EXPECT_EQ(id, expected[result_index]);
        EXPECT_FLOAT_EQ(score, 4.0f * (50.0f - id) * (50.0f - id));
    };
    runTopKSearchTest(index, query, std::size(expected), verify, nullptr, BY_ID);
}

TYPED_TEST(HNSWSQ8Test, SearchByID) { this->search_by_id_test(); }

template <typename index_type_t>
void HNSWSQ8Test<index_type_t>::search_by_score_test() {
    HNSWParams params = {
        .dim = 4, .initialCapacity = 200, .M = 16, .efConstruction = 200, .efRuntime = 100};
    SetUp(params);

    for (size_t i = 0; i < 100; i++) {
        ASSERT_EQ(GenerateAndAddVector(i, static_cast<float>(i)), 1);
    }

    data_t query[4];
    GenerateVector(query, 50.0f);
    // BY_SCORE orders labels by their distance from label 50.
    static constexpr size_t expected[] = {50, 49, 51, 48, 52, 47, 53, 46, 54, 45, 55};
    auto verify = [&](size_t id, double score, size_t result_index) {
        EXPECT_EQ(id, expected[result_index]);
        EXPECT_FLOAT_EQ(score, 4.0f * (50.0f - id) * (50.0f - id));
    };
    runTopKSearchTest(index, query, std::size(expected), verify);
}

TYPED_TEST(HNSWSQ8Test, SearchByScore) { this->search_by_score_test(); }

template <typename index_type_t>
void HNSWSQ8Test<index_type_t>::search_empty_index_test() {
    HNSWParams params = {.dim = 4, .initialCapacity = 0};
    SetUp(params);

    data_t query[4];
    GenerateVector(query, 50.0f);

    VecSimQueryReply *reply = VecSimIndex_TopKQuery(index, query, 11, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(reply), 0u);
    VecSimQueryReply_Free(reply);

    reply = VecSimIndex_RangeQuery(index, query, 1.0, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(reply), 0u);
    VecSimQueryReply_Free(reply);

    for (size_t i = 0; i < 100; i++) {
        GenerateAndAddVector(i, static_cast<float>(i));
    }
    for (size_t i = 0; i < 100; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0u);

    reply = VecSimIndex_TopKQuery(index, query, 11, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(reply), 0u);
    VecSimQueryReply_Free(reply);

    reply = VecSimIndex_RangeQuery(index, query, 1.0, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(reply), 0u);
    VecSimQueryReply_Free(reply);
}

TYPED_TEST(HNSWSQ8Test, SearchEmptyIndex) { this->search_empty_index_test(); }

template <typename index_type_t>
void HNSWSQ8Test<index_type_t>::test_override() {
    constexpr size_t count = 250;
    HNSWParams params = {
        .dim = 4, .initialCapacity = 100, .M = 8, .efConstruction = 20, .efRuntime = count};
    SetUp(params);

    for (size_t i = 0; i < 100; i++) {
        ASSERT_EQ(GenerateAndAddVector(i, static_cast<float>(i)), 1);
        ASSERT_EQ(GenerateAndAddVector(i, static_cast<float>(i)), 0);
    }
    for (size_t i = 100; i < count; i++) {
        ASSERT_EQ(GenerateAndAddVector(i, static_cast<float>(i)), 1);
    }

    data_t query[4];
    GenerateVector(query, static_cast<float>(count));
    // Distance decreases as the label increases, so results are in descending label order.
    auto verify = [&](size_t id, double score, size_t result_index) {
        EXPECT_EQ(id, count - result_index - 1);
        EXPECT_FLOAT_EQ(score, 4.0f * (count - id) * (count - id));
    };
    runTopKSearchTest(index, query, count, verify);
}

TYPED_TEST(HNSWSQ8Test, Override) { this->test_override(); }

template <typename index_type_t>
void HNSWSQ8Test<index_type_t>::test_range_query() {
    constexpr size_t count = 100;
    constexpr size_t close_count = 20;
    HNSWParams params = {.dim = 4, .initialCapacity = count, .efRuntime = count};
    SetUp(params);

    constexpr float pivot = 1.0f;
    constexpr float value_radius = 1.5f;
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(pivot - value_radius, pivot + value_radius);
    for (size_t i = 0; i < close_count; i++) {
        GenerateAndAddVector(i, distribution(generator));
    }
    for (size_t i = close_count; i < count; i++) {
        GenerateAndAddVector(i, 5.0f + distribution(generator));
    }

    data_t query[4];
    GenerateVector(query, pivot);
    constexpr double max_distance = 4.0 * value_radius * value_radius;
    auto verify = [&](size_t id, double score, size_t) {
        EXPECT_LT(id, close_count);
        EXPECT_LE(score, max_distance);
    };
    runRangeQueryTest(index, query, max_distance, verify, close_count, BY_SCORE);
}

TYPED_TEST(HNSWSQ8Test, RangeQuery) { this->test_range_query(); }

template <typename index_type_t>
void HNSWSQ8Test<index_type_t>::test_get_distance(VecSimMetric metric, bool multi) {
    HNSWParams params = {.dim = 4, .metric = metric, .initialCapacity = 1};
    params.multi = multi;
    SetUp(params);

    ASSERT_EQ(GenerateAndAddVector(0, 0.25f, 0.25f), 1);
    if (multi) {
        ASSERT_EQ(GenerateAndAddVector(0, -2.0f, 0.25f), 1);
    }
    data_t query[4];
    GenerateVector(query, 0.5f, 0.25f);

    const double expected = metric == VecSimMetric_L2 ? 0.25 : -1.5;
    EXPECT_NEAR(VecSimIndex_GetDistanceFrom_Unsafe(index, 0, query), expected, 1e-5);
    EXPECT_TRUE(std::isnan(VecSimIndex_GetDistanceFrom_Unsafe(index, 1, query)));
}

TYPED_TEST(HNSWSQ8Test, GetDistanceL2) { this->test_get_distance(VecSimMetric_L2, false); }
TYPED_TEST(HNSWSQ8Test, GetDistanceIP) { this->test_get_distance(VecSimMetric_IP, false); }
TYPED_TEST(HNSWSQ8Test, GetDistanceMultiL2) { this->test_get_distance(VecSimMetric_L2, true); }
TYPED_TEST(HNSWSQ8Test, GetDistanceMultiIP) { this->test_get_distance(VecSimMetric_IP, true); }

// Batch iteration

template <typename index_type_t>
void HNSWSQ8Test<index_type_t>::test_batch_iterator_basic() {
    constexpr size_t count = 250;
    constexpr size_t batch_size = 5;
    HNSWParams params = {
        .dim = 4, .initialCapacity = count, .M = 8, .efConstruction = 20, .efRuntime = count};
    SetUp(params);

    // Store [i, i, i, i] under label i.
    for (size_t i = 0; i < count; i++) {
        ASSERT_EQ(GenerateAndAddVector(i, static_cast<float>(i)), 1);
    }

    data_t query[4];
    GenerateVector(query, static_cast<float>(count));
    VecSimBatchIterator *iterator = VecSimBatchIterator_New(index, query, nullptr);
    ASSERT_NE(iterator, nullptr);

    // Each batch contains the five largest remaining labels, ordered by distance.
    size_t iteration = 0;
    while (VecSimBatchIterator_HasNext(iterator)) {
        auto verify = [&](size_t id, double, size_t result_index) {
            EXPECT_EQ(id, count - iteration * batch_size - result_index - 1);
        };
        runBatchIteratorSearchTest(iterator, batch_size, verify);
        iteration++;
    }
    EXPECT_EQ(iteration, count / batch_size);
    VecSimBatchIterator_Free(iterator);
}

TYPED_TEST(HNSWSQ8Test, BatchIteratorBasic) { this->test_batch_iterator_basic(); }

// SQ8 kernels support only FLOAT32 and FLOAT16 input vectors.
TEST(HNSWSQ8ParamsTest, RejectsUnsupportedDataType) {
    for (auto type : {VecSimType_FLOAT64, VecSimType_BFLOAT16, VecSimType_INT8, VecSimType_UINT8}) {
        HNSWParams hnsw_params = {
            .type = type, .dim = 4, .metric = VecSimMetric_L2, .quantType = VecSimQuant_SQ8};
        VecSimParams params = CreateParams(hnsw_params);

        EXPECT_EQ(VecSimIndex_New(&params), nullptr) << "data type " << type;
        EXPECT_EQ(EstimateInitialSize(hnsw_params), SIZE_MAX) << "data type " << type;
    }
}

// Value 3 has no VecSimMetric enumerator but is within the enum's representable range, so the
// factory must reject it before dispatch.
TEST(HNSWSQ8ParamsTest, RejectsOutOfRangeMetric) {
    HNSWParams hnsw_params = {.type = VecSimType_FLOAT32,
                              .dim = 4,
                              .metric = static_cast<VecSimMetric>(3),
                              .quantType = VecSimQuant_SQ8};
    VecSimParams params = CreateParams(hnsw_params);

    EXPECT_EQ(VecSimIndex_New(&params), nullptr);
    EXPECT_EQ(EstimateInitialSize(hnsw_params), SIZE_MAX);
}

// Mean-centering a FLOAT16 L2 query can lose precision or overflow when it is narrowed back to
// FLOAT16. Inner-product queries are not centered and remain supported.
TEST(HNSWSQ8ParamsTest, RejectsMeanCenteredFP16L2) {
    std::vector<float> mean(4, 1.0f);

    HNSWParams l2 = {.type = VecSimType_FLOAT16,
                     .dim = 4,
                     .metric = VecSimMetric_L2,
                     .quantType = VecSimQuant_SQ8,
                     .quantParams = mean.data()};
    VecSimParams l2_params = CreateParams(l2);
    EXPECT_EQ(VecSimIndex_New(&l2_params), nullptr);
    EXPECT_EQ(EstimateInitialSize(l2), SIZE_MAX);

    HNSWParams ip = {.type = VecSimType_FLOAT16,
                     .dim = 4,
                     .metric = VecSimMetric_IP,
                     .quantType = VecSimQuant_SQ8,
                     .quantParams = mean.data()};
    VecSimParams ip_params = CreateParams(ip);
    VecSimIndex *ip_index = VecSimIndex_New(&ip_params);
    ASSERT_NE(ip_index, nullptr);
    VecSimIndex_Free(ip_index);
    EXPECT_NE(EstimateInitialSize(ip), SIZE_MAX);
}

// V4 cannot encode the quantization settings needed to reload an SQ8 index.
TYPED_TEST(HNSWSQ8Test, RejectsSerialization) {
    HNSWParams params = {.dim = 4, .initialCapacity = 1};
    this->SetUp(params);
    ASSERT_EQ(this->GenerateAndAddVector(0, 0.25f, 0.25f), 1);

    const auto file_name = std::string(getenv("ROOT")) + "/tests/unit/sq8_should_not_be_written";
    const std::string existing_contents = "existing snapshot";
    {
        std::ofstream output(file_name, std::ios::binary);
        output.write(existing_contents.data(), existing_contents.size());
    }

    EXPECT_THROW(this->CastToHNSW()->saveIndex(file_name), std::runtime_error);

    std::ifstream input(file_name, std::ios::binary);
    const std::string saved_contents{std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>()};
    EXPECT_EQ(saved_contents, existing_contents);
    std::remove(file_name.c_str());
}

// Exercise graph construction's stored-to-stored IP kernel with non-degenerate quantization ranges.
TYPED_TEST(HNSWSQ8Test, GraphConstructionIP) {
    constexpr size_t n = 100;
    constexpr size_t dim = 16;
    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_IP, .initialCapacity = n, .M = 16, .efRuntime = n};
    this->SetUp(params);

    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(this->GenerateAndAddVector(i, static_cast<float>(i) * 0.5f, 0.25f), 1);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(this->index), n);

    // For inner-product distance (1 - dot product), these positive vectors rank by descending
    // label.
    std::vector<typename TestFixture::data_t> query(dim);
    this->GenerateVector(query.data(), 1.0f, 0.25f);

    auto verify = [&](size_t id, double, size_t result_index) {
        EXPECT_EQ(id, n - 1 - result_index);
    };
    runTopKSearchTest(this->index, query.data(), 10, verify);
}

// The tiered frontend does not propagate SQ8 settings, so creation and initial-size estimation must
// reject quantization.
TEST(HNSWSQ8TieredTest, RejectsQuantizedTieredIndex) {
    HNSWParams hnsw_params = {.type = VecSimType_FLOAT32,
                              .dim = 4,
                              .metric = VecSimMetric_L2,
                              .quantType = VecSimQuant_SQ8};
    VecSimParams primary_params = CreateParams(hnsw_params);
    // Rejection happens before the factory needs a job queue or thread pool.
    TieredIndexParams tiered_params = {.primaryIndexParams = &primary_params};
    VecSimParams params = CreateParams(tiered_params);

    EXPECT_EQ(VecSimIndex_New(&params), nullptr);
    EXPECT_EQ(EstimateInitialSize(tiered_params), SIZE_MAX);
}
