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
#include <random>
#include <type_traits>

template <VecSimType type, typename DataType, bool WithQuantParams>
struct HNSWSQ8IndexType : IndexType<type, DataType, float> {
    static constexpr bool with_quant_params = WithQuantParams;
};

// FLOAT16 with a mean vector is absent on purpose: the functional tests below all use L2, and
// mean-centred FP16 L2 is rejected at construction (see HNSWFactory::NewIndex). That combination is
// covered explicitly by HNSWSQ8ParamsTest.RejectsMeanCenteredFP16L2 instead.
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
    void test_get_distance(VecSimMetric metric);
    void test_batch_iterator_basic();

    VecSimIndex *index = nullptr;
    size_t dim = 0;
    std::vector<float> quantization_mean;
};

TYPED_TEST_SUITE(HNSWSQ8Test, HNSWSQ8DataTypeSet);

/* ---------------------------- Create index tests ---------------------------- */

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

    // The quantized vector is followed by the minimum value and quantization delta.
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

    // The size estimate must reject what creation rejects, so a caller cannot size its capacity
    // from a configuration it will then fail to build.
    EXPECT_THROW(EstimateInitialSize(params), std::invalid_argument);
}

/* ---------------------------- Size Estimation tests ---------------------------- */

TYPED_TEST(HNSWSQ8Test, SizeEstimation) {
    constexpr size_t block_size = 256;
    HNSWParams params = {.dim = 128, .blockSize = block_size, .M = 64};
    this->SetUp(params);

    // EstimateInitialSize is called after creating the index because index creation normalizes
    // the parameters.
    EXPECT_EQ(EstimateInitialSize(params), this->index->getAllocationSize());

    size_t label = 0;
    while (this->index->indexSize() < 200 || this->index->indexSize() % block_size != 0) {
        ASSERT_EQ(this->GenerateAndAddVector(label, static_cast<float>(label)), 1);
        label++;
    }

    // Estimate the memory delta of adding a vector that requires a full new block.
    const size_t estimation = EstimateElementSize(params) * block_size;
    const size_t before = this->index->getAllocationSize();
    ASSERT_EQ(this->GenerateAndAddVector(label, static_cast<float>(label)), 1);
    const size_t actual = this->index->getAllocationSize() - before;

    // Check that the actual size is within 1% of the estimation.
    EXPECT_GE(estimation, actual * 0.99);
    EXPECT_LE(estimation, actual * 1.01);
}

/* ---------------------------- Functionality tests ---------------------------- */

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
    // Vector values are equal to their labels, so the closest vectors have labels 45 through 55.
    static constexpr size_t expected[] = {45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55};
    auto verify = [&](size_t id, double score, size_t result_index) {
        // Results are sorted by ID.
        EXPECT_EQ(id, expected[result_index]);
        EXPECT_FLOAT_EQ(score, 4.0f * (50.0f - id) * (50.0f - id)); // L2 distance.
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
    // Vector values are equal to their labels, so results are ordered by distance from label 50.
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

    // We do not expect any results.
    VecSimQueryReply *reply = VecSimIndex_TopKQuery(index, query, 11, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(reply), 0u);
    VecSimQueryReply_Free(reply);

    reply = VecSimIndex_RangeQuery(index, query, 1.0, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(reply), 0u);
    VecSimQueryReply_Free(reply);

    // Add some vectors and remove them all from the index, so it will be empty again.
    for (size_t i = 0; i < 100; i++) {
        GenerateAndAddVector(i, static_cast<float>(i));
    }
    for (size_t i = 0; i < 100; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0u);

    // Again, we do not expect any results.
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

    // Insert 100 vectors and then overwrite each one with the same value.
    for (size_t i = 0; i < 100; i++) {
        ASSERT_EQ(GenerateAndAddVector(i, static_cast<float>(i)), 1);
        ASSERT_EQ(GenerateAndAddVector(i, static_cast<float>(i)), 0);
    }
    // Add vectors up to count.
    for (size_t i = 100; i < count; i++) {
        ASSERT_EQ(GenerateAndAddVector(i, static_cast<float>(i)), 1);
    }

    data_t query[4];
    GenerateVector(query, static_cast<float>(count));
    // The largest label is closest to the query, so labels are returned in descending order.
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
    // Insert close_count vectors near the pivot vector.
    for (size_t i = 0; i < close_count; i++) {
        GenerateAndAddVector(i, distribution(generator));
    }
    // Add the remaining vectors far from the pivot vector.
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
void HNSWSQ8Test<index_type_t>::test_get_distance(VecSimMetric metric) {
    HNSWParams params = {.dim = 4, .metric = metric, .initialCapacity = 1};
    SetUp(params);

    ASSERT_EQ(GenerateAndAddVector(0, 0.25f, 0.25f), 1);
    data_t query[4];
    GenerateVector(query, 0.5f, 0.25f);

    // Values were chosen so the expected distances can be calculated exactly. Comparing against a
    // stored SQ8 blob needs a preprocessed query, which only the index itself can produce.
    auto *hnsw_index = CastToHNSW();
    auto processed_query = hnsw_index->preprocessQuery(query);
    const double expected = metric == VecSimMetric_L2 ? 0.25 : -1.5;
    EXPECT_NEAR(
        hnsw_index->calcDistanceForQuery(hnsw_index->getDataByInternalId(0), processed_query.get()),
        expected, 1e-5);

    // The public API documents blob as a raw dim-by-type vector, which is not a usable query blob
    // for a quantized index: the kernels read query metadata appended past it. It must report no
    // answer rather than read past the caller's buffer.
    EXPECT_TRUE(std::isnan(VecSimIndex_GetDistanceFrom_Unsafe(index, 0, query)));
}

TYPED_TEST(HNSWSQ8Test, GetDistanceL2) { this->test_get_distance(VecSimMetric_L2); }
TYPED_TEST(HNSWSQ8Test, GetDistanceIP) { this->test_get_distance(VecSimMetric_IP); }

/* ---------------------------- Batch iterator tests ---------------------------- */

template <typename index_type_t>
void HNSWSQ8Test<index_type_t>::test_batch_iterator_basic() {
    constexpr size_t count = 250;
    constexpr size_t batch_size = 5;
    HNSWParams params = {
        .dim = 4, .initialCapacity = count, .M = 8, .efConstruction = 20, .efRuntime = count};
    SetUp(params);

    // For every i, add the vector (i, i, i, i) under label i.
    for (size_t i = 0; i < count; i++) {
        ASSERT_EQ(GenerateAndAddVector(i, static_cast<float>(i)), 1);
    }

    data_t query[4];
    GenerateVector(query, static_cast<float>(count));
    VecSimBatchIterator *iterator = VecSimBatchIterator_New(index, query, nullptr);
    ASSERT_NE(iterator, nullptr);

    // Get the five largest remaining labels in each iteration. Since vector values equal their
    // labels, this is also their order by distance from the query vector.
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

// SQ8 quantizes to uint8 with FP32 metadata and only has kernels for FP32 and FP16 sources, so
// every other data type must be rejected outright rather than produce an index. EstimateInitialSize
// rejects the same set; EstimateElementSize deliberately does not, because it has no error channel
// (see the note at its definition).
TEST(HNSWSQ8ParamsTest, RejectsUnsupportedDataType) {
    for (auto type : {VecSimType_FLOAT64, VecSimType_BFLOAT16, VecSimType_INT8, VecSimType_UINT8}) {
        HNSWParams hnsw_params = {
            .type = type, .dim = 4, .metric = VecSimMetric_L2, .quantType = VecSimQuant_SQ8};
        VecSimParams params = CreateParams(hnsw_params);

        EXPECT_EQ(VecSimIndex_New(&params), nullptr) << "data type " << type;
        EXPECT_THROW(EstimateInitialSize(hnsw_params), std::invalid_argument)
            << "data type " << type;
    }
}

// An out-of-range metric must be rejected before dispatch. Reaching the unreachable-branch assert
// in the SQ8 dispatcher would abort an assertions-enabled host, where the unquantized path throws
// and VecSimIndex_New catches it. VecSimMetric has three enumerators, so 3 is the smallest value
// outside the valid set that is still inside the enum's value range and therefore safe to form.
TEST(HNSWSQ8ParamsTest, RejectsOutOfRangeMetric) {
    HNSWParams hnsw_params = {.type = VecSimType_FLOAT32,
                              .dim = 4,
                              .metric = static_cast<VecSimMetric>(3),
                              .quantType = VecSimQuant_SQ8};
    VecSimParams params = CreateParams(hnsw_params);

    EXPECT_EQ(VecSimIndex_New(&params), nullptr);
    EXPECT_THROW(EstimateInitialSize(hnsw_params), std::invalid_argument);
}

// Mean-centred FP16 with L2 must be rejected: QuantPreprocessor narrows the centred query back into
// the FP16 query body while storage keeps its centred min/delta in FP32, so identical vector and
// query pairs diverge and a large mean overflows FP16 to infinity. The same combination with IP is
// supported, because that path does not centre the query.
TEST(HNSWSQ8ParamsTest, RejectsMeanCenteredFP16L2) {
    std::vector<float> mean(4, 1.0f);

    HNSWParams l2 = {.type = VecSimType_FLOAT16,
                     .dim = 4,
                     .metric = VecSimMetric_L2,
                     .quantType = VecSimQuant_SQ8,
                     .quantParams = mean.data()};
    VecSimParams l2_params = CreateParams(l2);
    EXPECT_EQ(VecSimIndex_New(&l2_params), nullptr);
    EXPECT_THROW(EstimateInitialSize(l2), std::invalid_argument);

    HNSWParams ip = {.type = VecSimType_FLOAT16,
                     .dim = 4,
                     .metric = VecSimMetric_IP,
                     .quantType = VecSimQuant_SQ8,
                     .quantParams = mean.data()};
    VecSimParams ip_params = CreateParams(ip);
    VecSimIndex *ip_index = VecSimIndex_New(&ip_params);
    ASSERT_NE(ip_index, nullptr);
    VecSimIndex_Free(ip_index);
    EXPECT_NO_THROW(EstimateInitialSize(ip));
}

// Serialization does not record quantType or the mean vector, and the loading path always builds
// unquantized components, so saving a quantized index would produce a file the loader misreads.
// saveIndex must refuse rather than emit one.
TYPED_TEST(HNSWSQ8Test, RejectsSerialization) {
    HNSWParams params = {.dim = 4, .initialCapacity = 1};
    this->SetUp(params);
    ASSERT_EQ(this->GenerateAndAddVector(0, 0.25f, 0.25f), 1);

    const auto file_name = std::string(getenv("ROOT")) + "/tests/unit/sq8_should_not_be_written";
    EXPECT_THROW(this->CastToHNSW()->saveIndex(file_name), std::runtime_error);
    std::remove(file_name.c_str());
}

// Every other functional test uses L2, so without this the symmetric SQ8-to-SQ8 IP kernel that
// graph construction selects for an IP index would never run. Vectors vary per component as well as
// per label, so quantization does not collapse into the degenerate min == max branch.
TYPED_TEST(HNSWSQ8Test, GraphConstructionIP) {
    constexpr size_t n = 100;
    constexpr size_t dim = 16;
    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_IP, .initialCapacity = n, .M = 16, .efRuntime = n};
    this->SetUp(params);

    // Each label i gets a vector whose components ramp from i upward, so no two vectors share a
    // quantization range and every vector has a non-zero delta.
    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(this->GenerateAndAddVector(i, static_cast<float>(i) * 0.5f, 0.25f), 1);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(this->index), n);

    // This is plain inner product, not cosine: the distance is 1 - IP, so the closest vector is the
    // one with the largest projection onto the query rather than the query's own twin. Every vector
    // and the query are positive and magnitude grows with the label, so IP is strictly increasing
    // in the label and results must come back from the highest label downward.
    std::vector<typename TestFixture::data_t> query(dim);
    this->GenerateVector(query.data(), 1.0f, 0.25f);

    auto verify = [&](size_t id, double, size_t result_index) {
        EXPECT_EQ(id, n - 1 - result_index);
    };
    runTopKSearchTest(this->index, query.data(), 10, verify);
}

// SQ8 is not wired into the tiered index yet, so the tiered factory must reject it instead of
// building a quantized primary index against an unquantized frontend. Without the guard this aborts
// on a debug build and silently mismatches the two blob layouts on a release one. Whoever wires
// quantization through the tiered index should replace this expectation rather than delete it.
TEST(HNSWSQ8TieredTest, RejectsQuantizedTieredIndex) {
    HNSWParams hnsw_params = {.type = VecSimType_FLOAT32,
                              .dim = 4,
                              .metric = VecSimMetric_L2,
                              .quantType = VecSimQuant_SQ8};
    VecSimParams primary_params = CreateParams(hnsw_params);
    // No job queue or thread pool is needed: the factory rejects these params before it reaches
    // anything that would use them.
    TieredIndexParams tiered_params = {.primaryIndexParams = &primary_params};
    VecSimParams params = CreateParams(tiered_params);

    EXPECT_EQ(VecSimIndex_New(&params), nullptr);
    // The primary index alone would accept these params, so the tiered estimate needs its own check
    // rather than inheriting one from HNSWFactory.
    EXPECT_THROW(EstimateInitialSize(tiered_params), std::invalid_argument);
}
