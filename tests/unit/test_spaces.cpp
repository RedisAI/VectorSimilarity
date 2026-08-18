/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <utility>
#include <random>
#include <cmath>
#include <limits>
#include <string>
#include <set>
#include <cstdlib>
#include <iostream>

#include "gtest/gtest.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/types/float16.h"
#include "VecSim/types/sq8.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512BF16_VL.h"
#include "VecSim/spaces/functions/AVX512FP16_VL.h"
#include "VecSim/spaces/functions/AVX512F_BW_VL_VNNI.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/AVX2_F16C.h"
#include "VecSim/spaces/functions/AVX2_FMA.h"
#include "VecSim/spaces/functions/AVX2_FMA_F16C.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/SSE4.h"
#include "VecSim/spaces/functions/SSE4_F16C.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/NEON.h"
#include "VecSim/spaces/functions/NEON_DOTPROD.h"
#include "VecSim/spaces/functions/NEON_HP.h"
#include "VecSim/spaces/functions/NEON_BF16.h"
#include "VecSim/spaces/functions/SVE.h"
#include "VecSim/spaces/functions/SVE_BF16.h"
#include "VecSim/spaces/functions/SVE2.h"
#include "VecSim/spaces/uint8_chunking.h"
#include "tests_utils.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;
using sq8 = vecsim_types::sq8;
using namespace spaces;

class SpacesTest : public ::testing::Test {

protected:
    SpacesTest() {}

    ~SpacesTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(SpacesTest, float_l2_no_optimization_func_test) {
    size_t dim = 5;

    float a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = float(i + 1.5);
        b[i] = float(i + 1.5);
    }

    float dist = FP32_L2Sqr((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, 0.0);
}

TEST_F(SpacesTest, double_l2_no_optimization_func_test) {
    size_t dim = 5;

    double a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = double(i + 1.5);
        b[i] = double(i + 1.5);
    }

    double dist = FP64_L2Sqr((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, 0.0);
}

TEST_F(SpacesTest, bf16_l2_no_optimization_func_test) {
    size_t dim = 4;

    bfloat16 a[dim], b[dim];
    float sanity_a[dim], sanity_b[dim];
    for (size_t i = 0; i < dim; i++) {
        // multiplication of 0.25 have no rounding error when converted to bfloat16
        sanity_a[i] = 0.5f + i * 0.25f;
        a[i] = vecsim_types::float_to_bf16(sanity_a[i]);
        sanity_b[i] = (float)i * 0.25f;
        b[i] = vecsim_types::float_to_bf16(sanity_b[i]);
    }

    float dist = BF16_L2Sqr_LittleEndian((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, FP32_L2Sqr((const void *)sanity_a, (const void *)sanity_b, dim));
}

TEST_F(SpacesTest, fp16_l2_no_optimization_func_test) {
    size_t dim = 4;

    float16 a[dim], b[dim];
    float sanity_a[dim], sanity_b[dim];
    for (size_t i = 0; i < dim; i++) {
        // multiplication of 0.25 have no rounding error when converted to bfloat16
        sanity_a[i] = 0.5f + i * 0.25f;
        a[i] = vecsim_types::FP32_to_FP16(sanity_a[i]);
        sanity_b[i] = (float)i * 0.25f;
        b[i] = vecsim_types::FP32_to_FP16(sanity_b[i]);
    }

    float dist = FP16_L2Sqr((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, FP32_L2Sqr((const void *)sanity_a, (const void *)sanity_b, dim));
}

TEST_F(SpacesTest, int8_l2_no_optimization_func_test) {
    size_t dim = 5;

    int8_t a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (i + 1);
        b[i] = (i + 2);
    }

    float dist = INT8_L2Sqr((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, 5.0);
}

TEST_F(SpacesTest, uint8_l2_no_optimization_func_test) {
    size_t dim = 5;

    uint8_t a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (i + 1);
        b[i] = (i + 2);
    }

    float dist = UINT8_L2Sqr((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, 5.0);
}

/* ======================== IP NO OPT ======================== */

TEST_F(SpacesTest, float_ip_no_optimization_func_test) {
    size_t dim = 5;

    float a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = float(i + 1.5);
        b[i] = float(i + 1.5);
    }

    spaces::GetNormalizeFunc<float>()(a, dim);
    spaces::GetNormalizeFunc<float>()(b, dim);

    float dist = FP32_InnerProduct((const void *)a, (const void *)b, dim);
    ASSERT_FLOAT_EQ(dist, 0.0f);
}

TEST_F(SpacesTest, double_ip_no_optimization_func_test) {
    size_t dim = 5;

    double a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = double(i + 1.5);
        b[i] = double(i + 1.5);
    }

    spaces::GetNormalizeFunc<double>()(a, dim);
    spaces::GetNormalizeFunc<double>()(b, dim);

    double dist = FP64_InnerProduct((const void *)a, (const void *)b, dim);
    ASSERT_NEAR(dist, 0.0, 0.00000001);
}

TEST_F(SpacesTest, bf16_normalize_test) {
    size_t dim = 4;

    bfloat16 a[dim];
    float sanity_a[dim];
    for (size_t i = 0; i < dim; i++) {
        // unit vector
        sanity_a[i] = float(4);
        a[i] = vecsim_types::float_to_bf16(sanity_a[i]);
    }

    spaces::GetNormalizeFunc<bfloat16>()(a, dim);
    spaces::GetNormalizeFunc<float>()(sanity_a, dim);
    // Convert assuming little endian system
    for (size_t i = 0; i < dim; i++) {
        ASSERT_EQ(vecsim_types::bfloat16_to_float32(a[i]), sanity_a[i])
            << " bf16 normalization failed for i = " << i;
        ASSERT_EQ(vecsim_types::bfloat16_to_float32(a[i]), 0.5)
            << " bf16 normalization failed for i = " << i;
    }
}

TEST_F(SpacesTest, fp16_normalize_test) {
    size_t dim = 4;

    float16 a[dim];
    float sanity_a[dim];
    for (size_t i = 0; i < dim; i++) {
        // unit vector
        sanity_a[i] = float(4);
        a[i] = vecsim_types::FP32_to_FP16(sanity_a[i]);
    }

    spaces::GetNormalizeFunc<float16>()(a, dim);
    spaces::GetNormalizeFunc<float>()(sanity_a, dim);
    for (size_t i = 0; i < dim; i++) {
        ASSERT_EQ(vecsim_types::FP16_to_FP32(a[i]), sanity_a[i])
            << " fp16 normalization failed for i = " << i;
        ASSERT_EQ(vecsim_types::FP16_to_FP32(a[i]), 0.5)
            << " fp16 normalization failed for i = " << i;
    }
}

TEST_F(SpacesTest, bf16_ip_no_optimization_func_test) {
    size_t dim = 4;

    bfloat16 a[dim], b[dim];
    float sanity_a[dim], sanity_b[dim];
    for (size_t i = 0; i < dim; i++) {
        // multiplication of 0.25 have no rounding error when converted to bfloat16
        sanity_a[i] = 0.5f + i * 0.25f;
        a[i] = vecsim_types::float_to_bf16(sanity_a[i]);
        sanity_b[i] = (float)i * 0.25f;
        b[i] = vecsim_types::float_to_bf16(sanity_b[i]);
    }

    float dist = BF16_InnerProduct_LittleEndian((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, FP32_InnerProduct((const void *)sanity_a, (const void *)sanity_b, dim));
}

TEST_F(SpacesTest, fp16_ip_no_optimization_func_test) {
    size_t dim = 4;

    float16 a[dim], b[dim];
    float sanity_a[dim], sanity_b[dim];
    for (size_t i = 0; i < dim; i++) {
        // multiplication of 0.25 have no rounding error when converted to bfloat16
        sanity_a[i] = 0.5f + i * 0.25f;
        a[i] = vecsim_types::FP32_to_FP16(sanity_a[i]);
        sanity_b[i] = (float)i * 0.25f;
        b[i] = vecsim_types::FP32_to_FP16(sanity_b[i]);
    }

    float dist = FP16_InnerProduct((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, FP32_InnerProduct((const void *)sanity_a, (const void *)sanity_b, dim));
}

TEST_F(SpacesTest, int8_ip_no_optimization_func_test) {
    size_t dim = 4;
    int8_t a[] = {1, 0, 0, 0};
    int8_t b[] = {1, 0, 0, 0};

    float dist = INT8_InnerProduct((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, 0.0);
}

TEST_F(SpacesTest, uint8_ip_no_optimization_func_test) {
    size_t dim = 4;
    uint8_t a[] = {1, 0, 0, 0};
    uint8_t b[] = {1, 0, 0, 0};

    float dist = UINT8_InnerProduct((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, 0.0);
}

/* ======================== Cosine NO OPT ======================== */

TEST_F(SpacesTest, int8_Cosine_no_optimization_func_test) {
    size_t dim = 4;
    // create a vector with extra space for the norm
    int8_t v1[dim + sizeof(float)];
    int8_t v2[dim + sizeof(float)];

    test_utils::populate_int8_vec(v1, dim, 123);
    test_utils::populate_int8_vec(v2, dim, 123);

    // write the norm at the end of the vector
    const float norm_v1 = test_utils::integral_compute_norm(v1, dim);
    const float norm_v2 = test_utils::integral_compute_norm(v2, dim);
    std::memcpy(v1 + dim, &norm_v1, sizeof(norm_v1));
    std::memcpy(v2 + dim, &norm_v2, sizeof(norm_v2));

    float dist = INT8_Cosine((const void *)v1, (const void *)v2, dim);
    ASSERT_NEAR(dist, 0.0, 0.000001);
}

TEST_F(SpacesTest, uint8_Cosine_no_optimization_func_test) {
    size_t dim = 4;
    // create a vector with extra space for the norm
    uint8_t v1[dim + sizeof(float)];
    uint8_t v2[dim + sizeof(float)];

    test_utils::populate_uint8_vec(v1, dim, 123);
    test_utils::populate_uint8_vec(v2, dim, 123);

    // write the norm at the end of the vector
    const float norm_v1 = test_utils::integral_compute_norm(v1, dim);
    const float norm_v2 = test_utils::integral_compute_norm(v2, dim);
    std::memcpy(v1 + dim, &norm_v1, sizeof(norm_v1));
    std::memcpy(v2 + dim, &norm_v2, sizeof(norm_v2));

    float dist = UINT8_Cosine((const void *)v1, (const void *)v2, dim);
    ASSERT_NEAR(dist, 0.0, 0.000001);
}

/* ======================== Tests SQ8 ========================= */

TEST_F(SpacesTest, SQ8_FP32_ip_no_optimization_norm_func_test) {
    size_t dim = 5;

    // Create V1 fp32 query with precomputed sum and sum_squares
    // Query layout: [float values (dim)] [sum] [sum_squares]
    size_t query_size = dim + sq8::query_metadata_count<VecSimMetric_L2>();
    std::vector<float> v1_orig(query_size);
    test_utils::populate_sq8_fp32_query(v1_orig.data(), dim, true, 1234);

    // Create V2 as SQ8 quantized vector with different seed
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, true, 5678);

    float baseline =
        test_utils::SQ8_FP32_NotOptimized_InnerProduct(v2_compressed.data(), v1_orig.data(), dim);

    float dist = SQ8_FP32_InnerProduct((const void *)v2_compressed.data(),
                                       (const void *)v1_orig.data(), dim);

    ASSERT_NEAR(dist, baseline, 0.01) << "SQ8_FP32_InnerProduct failed to match expected distance";
}

TEST_F(SpacesTest, SQ8_FP32_l2sqr_no_optimization_func_test) {
    size_t dim = 5;

    // Create V1 fp32 query with precomputed sum and sum_squares
    // Query layout: [float values (dim)] [sum] [sum_squares]
    size_t query_size = dim + sq8::query_metadata_count<VecSimMetric_L2>();
    std::vector<float> v1_orig(query_size);
    test_utils::populate_sq8_fp32_query(v1_orig.data(), dim, false, 1234);

    // Create V2 as SQ8 quantized vector with different seed
    // Storage layout: [uint8_t values (dim)] [min_val] [delta] [sum] [sum_squares]
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, false, 5678);

    float baseline =
        test_utils::SQ8_FP32_NotOptimized_L2Sqr(v2_compressed.data(), v1_orig.data(), dim);

    float dist =
        SQ8_FP32_L2Sqr((const void *)v2_compressed.data(), (const void *)v1_orig.data(), dim);

    ASSERT_NEAR(dist, baseline, 0.01) << "SQ8_FP32_L2Sqr failed to match expected distance";
}

TEST_F(SpacesTest, SQ8_FP32_odd_dim_unaligned_metadata_test) {
    for (const size_t dim : {1UL, 5UL, 7UL, 15UL}) {
        const size_t query_size = dim + sq8::query_metadata_count<VecSimMetric_L2>();
        std::vector<float> query(query_size);
        test_utils::populate_sq8_fp32_query(query.data(), dim, false, 1234);

        const size_t storage_size =
            dim + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
        std::vector<uint8_t> allocation(storage_size + alignof(float));
        auto *storage = allocation.data();
        while (reinterpret_cast<std::uintptr_t>(storage) % alignof(float) != 0) {
            ++storage;
        }
        test_utils::populate_float_vec_to_sq8_with_metadata(storage, dim, false, 5678);

        const auto *metadata = storage + dim;
        ASSERT_NE(reinterpret_cast<std::uintptr_t>(metadata) % alignof(float), 0u);

        const float expected_ip =
            test_utils::SQ8_FP32_NotOptimized_InnerProduct(storage, query.data(), dim);
        const float expected_l2 =
            test_utils::SQ8_FP32_NotOptimized_L2Sqr(storage, query.data(), dim);

        EXPECT_NEAR(SQ8_FP32_InnerProduct(storage, query.data(), dim), expected_ip, 0.01)
            << "scalar IP with dim " << dim;
        EXPECT_NEAR(SQ8_FP32_L2Sqr(storage, query.data(), dim), expected_l2, 0.01)
            << "scalar L2 with dim " << dim;
        EXPECT_NEAR(IP_SQ8_FP32_GetDistFunc(dim, nullptr)(storage, query.data(), dim), expected_ip,
                    0.01)
            << "dispatched IP with dim " << dim;
        EXPECT_NEAR(L2_SQ8_FP32_GetDistFunc(dim, nullptr)(storage, query.data(), dim), expected_l2,
                    0.01)
            << "dispatched L2 with dim " << dim;
    }
}

/* ======================== Tests SQ8-FP16 ========================= */

TEST_F(SpacesTest, SQ8_FP16_ip_no_optimization_norm_func_test) {
    size_t dim = 5;

    // Create V1 fp16 query with precomputed sum and sum_squares
    // Query layout: [float16 values (dim)] [sum (float)] [sum_squares (float)]
    // Allocate as std::vector<float16> so v1_query.data() is alignof(float16)-aligned, as
    // required by the SQ8_FP16 production kernels' typed float16* loads. Add extra float16
    // slots to cover the trailing FP32 metadata bytes.
    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v1_query(query_count);
    test_utils::populate_sq8_fp16_query(v1_query.data(), dim, true, 1234);

    // Create V2 as SQ8 quantized vector with different seed
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, true, 5678);

    float baseline =
        test_utils::SQ8_FP16_NotOptimized_InnerProduct(v2_compressed.data(), v1_query.data(), dim);

    float dist = SQ8_FP16_InnerProduct((const void *)v2_compressed.data(),
                                       (const void *)v1_query.data(), dim);

    ASSERT_NEAR(dist, baseline, 0.01) << "SQ8_FP16_InnerProduct failed to match expected distance";
}

TEST_F(SpacesTest, SQ8_FP16_cosine_no_optimization_norm_func_test) {
    size_t dim = 5;

    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v1_query(query_count);
    test_utils::populate_sq8_fp16_query(v1_query.data(), dim, true, 1234);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, true, 5678);

    float baseline =
        test_utils::SQ8_FP16_NotOptimized_Cosine(v2_compressed.data(), v1_query.data(), dim);

    float dist =
        SQ8_FP16_Cosine((const void *)v2_compressed.data(), (const void *)v1_query.data(), dim);

    ASSERT_NEAR(dist, baseline, 0.01) << "SQ8_FP16_Cosine failed to match expected distance";
}

TEST_F(SpacesTest, SQ8_FP16_l2sqr_no_optimization_func_test) {
    size_t dim = 5;

    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v1_query(query_count);
    test_utils::populate_sq8_fp16_query(v1_query.data(), dim, false, 1234);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, false, 5678);

    float baseline =
        test_utils::SQ8_FP16_NotOptimized_L2Sqr(v2_compressed.data(), v1_query.data(), dim);

    float dist =
        SQ8_FP16_L2Sqr((const void *)v2_compressed.data(), (const void *)v1_query.data(), dim);

    ASSERT_NEAR(dist, baseline, 0.01) << "SQ8_FP16_L2Sqr failed to match expected distance";
}

TEST_F(SpacesTest, SQ8_FP16_l2sqr_odd_dim_unaligned_metadata_test) {
    constexpr size_t dim = 5;
    constexpr size_t storage_bytes =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    static_assert(sizeof(float) % sizeof(float16) == 0);
    constexpr size_t query_values_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * sizeof(float) / sizeof(float16);

    alignas(float) std::array<uint8_t, storage_bytes> storage{};
    alignas(float) std::array<float16, query_values_count> query{};

    for (size_t i = 0; i < dim; i++) {
        storage[i] = static_cast<uint8_t>(i + 1);
    }

    auto store_float = [](uint8_t *dst, float value) { std::memcpy(dst, &value, sizeof(value)); };

    constexpr float min_val = 0.0f;
    constexpr float delta = 1.0f;
    constexpr float storage_sum = 15.0f;
    constexpr float storage_sum_squares = 55.0f;
    uint8_t *storage_meta = storage.data() + dim;
    store_float(storage_meta + sq8::MIN_VAL * sizeof(float), min_val);
    store_float(storage_meta + sq8::DELTA * sizeof(float), delta);
    store_float(storage_meta + sq8::SUM * sizeof(float), storage_sum);
    store_float(storage_meta + sq8::SUM_SQUARES * sizeof(float), storage_sum_squares);

    for (size_t i = 0; i < dim; i++) {
        query[i] = vecsim_types::FP32_to_FP16(static_cast<float>(i + 2));
    }

    constexpr float query_sum = 20.0f;
    constexpr float query_sum_squares = 90.0f;
    uint8_t *query_meta = reinterpret_cast<uint8_t *>(query.data() + dim);
    store_float(query_meta + sq8::SUM_QUERY * sizeof(float), query_sum);
    store_float(query_meta + sq8::SUM_SQUARES_QUERY * sizeof(float), query_sum_squares);

    const auto *storage_sum_squares_addr = storage_meta + sq8::SUM_SQUARES * sizeof(float);
    const auto *query_sum_squares_addr = query_meta + sq8::SUM_SQUARES_QUERY * sizeof(float);
    ASSERT_NE(reinterpret_cast<std::uintptr_t>(storage_sum_squares_addr) % alignof(float), 0u);
    ASSERT_NE(reinterpret_cast<std::uintptr_t>(query_sum_squares_addr) % alignof(float), 0u);

    const float dist = SQ8_FP16_L2Sqr(storage.data(), query.data(), dim);

    ASSERT_FLOAT_EQ(dist, 5.0f);
}

/* ======================== Test Getters ======================== */

TEST_F(SpacesTest, GetDistFuncInvalidMetricFP32) {
    EXPECT_THROW(
        (spaces::GetDistFunc<float, float>((VecSimMetric)(VecSimMetric_Cosine + 1), 10, nullptr)),
        std::invalid_argument);
}
TEST_F(SpacesTest, GetDistFuncInvalidMetricFP64) {
    EXPECT_THROW(
        (spaces::GetDistFunc<double, double>((VecSimMetric)(VecSimMetric_Cosine + 1), 10, nullptr)),
        std::invalid_argument);
}
TEST_F(SpacesTest, GetDistFuncInvalidMetricBF16) {
    EXPECT_THROW((spaces::GetDistFunc<bfloat16, float>((VecSimMetric)(VecSimMetric_Cosine + 1), 10,
                                                       nullptr)),
                 std::invalid_argument);
}
TEST_F(SpacesTest, GetDistFuncInvalidMetricFP16) {
    EXPECT_THROW(
        (spaces::GetDistFunc<float16, float>((VecSimMetric)(VecSimMetric_Cosine + 1), 10, nullptr)),
        std::invalid_argument);
}
TEST_F(SpacesTest, GetDistFuncInvalidMetricINT8) {
    EXPECT_THROW(
        (spaces::GetDistFunc<int8_t, float>((VecSimMetric)(VecSimMetric_Cosine + 1), 10, nullptr)),
        std::invalid_argument);
}
TEST_F(SpacesTest, GetDistFuncInvalidMetricUINT8) {
    EXPECT_THROW(
        (spaces::GetDistFunc<uint8_t, float>((VecSimMetric)(VecSimMetric_Cosine + 1), 10, nullptr)),
        std::invalid_argument);
}
TEST_F(SpacesTest, GetDistFuncInvalidMetricSQ8) {
    // SQ8 to SQ8 (symmetric)
    EXPECT_THROW(
        (spaces::GetDistFunc<sq8, float>((VecSimMetric)(VecSimMetric_Cosine + 1), 10, nullptr)),
        std::invalid_argument);
}
TEST_F(SpacesTest, GetDistFuncInvalidMetricSQ8ToFloat) {
    // SQ8 to float (asymmetric)
    EXPECT_THROW((spaces::GetDistFunc<sq8, float, float>((VecSimMetric)(VecSimMetric_Cosine + 1),
                                                         10, nullptr)),
                 std::invalid_argument);
}
TEST_F(SpacesTest, GetDistFuncInvalidMetricSQ8ToFP16) {
    // SQ8 storage with FP16 query (asymmetric)
    EXPECT_THROW((spaces::GetDistFunc<sq8, float, float16>((VecSimMetric)(VecSimMetric_Cosine + 1),
                                                           10, nullptr)),
                 std::invalid_argument);
}

// Positive tests for GetDistFunc - verify correct function is returned
TEST_F(SpacesTest, GetDistFuncSQ8Symmetric) {
    // SQ8 to SQ8 (symmetric) - should return SQ8_SQ8 functions
    size_t dim = 128;
    auto l2_func = spaces::GetDistFunc<sq8, float>(VecSimMetric_L2, dim, nullptr);
    auto ip_func = spaces::GetDistFunc<sq8, float>(VecSimMetric_IP, dim, nullptr);
    auto cosine_func = spaces::GetDistFunc<sq8, float>(VecSimMetric_Cosine, dim, nullptr);
    ASSERT_EQ(l2_func, L2_SQ8_SQ8_GetDistFunc(dim, nullptr));
    ASSERT_EQ(ip_func, IP_SQ8_SQ8_GetDistFunc(dim, nullptr));
    ASSERT_EQ(cosine_func, Cosine_SQ8_SQ8_GetDistFunc(dim, nullptr));
}

TEST_F(SpacesTest, GetDistFuncSQ8Asymmetric) {
    // SQ8 to float (asymmetric) - should return SQ8 functions
    size_t dim = 128;
    auto l2_func = spaces::GetDistFunc<sq8, float, float>(VecSimMetric_L2, dim, nullptr);
    auto ip_func = spaces::GetDistFunc<sq8, float, float>(VecSimMetric_IP, dim, nullptr);
    auto cosine_func = spaces::GetDistFunc<sq8, float, float>(VecSimMetric_Cosine, dim, nullptr);
    ASSERT_EQ(l2_func, L2_SQ8_FP32_GetDistFunc(dim, nullptr));
    ASSERT_EQ(ip_func, IP_SQ8_FP32_GetDistFunc(dim, nullptr));
    ASSERT_EQ(cosine_func, Cosine_SQ8_FP32_GetDistFunc(dim, nullptr));
}

TEST_F(SpacesTest, GetDistFuncSQ8FP16Asymmetric) {
    // SQ8 storage with FP16 query (asymmetric) - should return SQ8_FP16 functions.
    // Per-ISA dispatcher walk coverage lives in the SQ8_FP16 SpacesOptimizationTest below.
    size_t dim = 128;
    auto l2_func = spaces::GetDistFunc<sq8, float, float16>(VecSimMetric_L2, dim, nullptr);
    auto ip_func = spaces::GetDistFunc<sq8, float, float16>(VecSimMetric_IP, dim, nullptr);
    auto cosine_func = spaces::GetDistFunc<sq8, float, float16>(VecSimMetric_Cosine, dim, nullptr);
    ASSERT_EQ(l2_func, L2_SQ8_FP16_GetDistFunc(dim, nullptr));
    ASSERT_EQ(ip_func, IP_SQ8_FP16_GetDistFunc(dim, nullptr));
    ASSERT_EQ(cosine_func, Cosine_SQ8_FP16_GetDistFunc(dim, nullptr));

    // dim < 16 takes the scalar early-return in every SQ8_FP16 dispatcher (no SIMD tier).
    size_t small_dim = 8;
    ASSERT_EQ(L2_SQ8_FP16_GetDistFunc(small_dim, nullptr), SQ8_FP16_L2Sqr);
    ASSERT_EQ(IP_SQ8_FP16_GetDistFunc(small_dim, nullptr), SQ8_FP16_InnerProduct);
    ASSERT_EQ(Cosine_SQ8_FP16_GetDistFunc(small_dim, nullptr), SQ8_FP16_Cosine);
}

#ifdef CPU_FEATURES_ARCH_X86_64
#ifdef OPT_SSE
// Regression test for MOD-16730: the FP32 L2 SSE kernel's residual % 4 == 3 path used
// _mm_loadr_ps (movaps), which faults on non-16-byte-aligned addresses. Vectors are not
// guaranteed such alignment: VecSimAllocator::allocate() returns malloc + 8 (allocation
// header), and the dispatcher sets no alignment hint when dim % 4 != 0. This test feeds the
// kernel buffers at that exact placement (16-aligned base + 8).
TEST_F(SpacesTest, FP32_L2Sqr_SSE_misaligned_residual3) {
    constexpr size_t dim = 19; // dim % 16 == 3 -> residual 3 path
    alignas(16) static char raw1[16 + dim * sizeof(float)];
    alignas(16) static char raw2[16 + dim * sizeof(float)];
    float *v1 = reinterpret_cast<float *>(raw1 + 8); // address == 8 (mod 16)
    float *v2 = reinterpret_cast<float *>(raw2 + 8);
    for (size_t i = 0; i < dim; i++) {
        v1[i] = float(i);
        v2[i] = float(i) + 1.5f;
    }
    float baseline = FP32_L2Sqr(v1, v2, dim);
    dist_func_t<float> arch_opt_func = spaces::Choose_FP32_L2_implementation_SSE(dim);
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim));
}
#endif // OPT_SSE

TEST_F(SpacesTest, smallDimChooser) {
    // Verify that dimensions below each type's SIMD threshold get the no optimization function.
    // FP64 is optimized from dim >= 4.
    for (size_t dim = 1; dim < 4; dim++) {
        ASSERT_EQ(L2_FP64_GetDistFunc(dim), FP64_L2Sqr);
        ASSERT_EQ(IP_FP64_GetDistFunc(dim), FP64_InnerProduct);
    }
    // FP32 and FP16 are optimized from dim >= 8 (FP16 only on machines with F16C; for
    // 8 <= dim < 32 the chosen FP16 function depends on the available features, so we only
    // assert the range that is naive regardless of features).
    for (size_t dim = 1; dim < 8; dim++) {
        ASSERT_EQ(L2_FP32_GetDistFunc(dim), FP32_L2Sqr);
        ASSERT_EQ(IP_FP32_GetDistFunc(dim), FP32_InnerProduct);
        ASSERT_EQ(L2_FP16_GetDistFunc(dim), FP16_L2Sqr);
        ASSERT_EQ(IP_FP16_GetDistFunc(dim), FP16_InnerProduct);
    }
    // BF16, INT8 and UINT8 are optimized from dim >= 32.
    for (size_t dim = 1; dim < 32; dim++) {
        ASSERT_EQ(L2_BF16_GetDistFunc(dim), BF16_L2Sqr_LittleEndian);
        ASSERT_EQ(L2_INT8_GetDistFunc(dim), INT8_L2Sqr);
        ASSERT_EQ(L2_UINT8_GetDistFunc(dim), UINT8_L2Sqr);
        ASSERT_EQ(IP_BF16_GetDistFunc(dim), BF16_InnerProduct_LittleEndian);
        ASSERT_EQ(IP_INT8_GetDistFunc(dim), INT8_InnerProduct);
        ASSERT_EQ(IP_UINT8_GetDistFunc(dim), UINT8_InnerProduct);
        ASSERT_EQ(Cosine_INT8_GetDistFunc(dim), INT8_Cosine);
        ASSERT_EQ(Cosine_UINT8_GetDistFunc(dim), UINT8_Cosine);
    }
}
#endif

/* ======================== Test SIMD Functions ======================== */

// In this following tests we assume that compiler supports all X86 optimizations, so if we have
// some hardware flag enabled, we check that the corresponding optimization function was chosen.

class FP32SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(FP32SpacesOptimizationTest, FP32L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    float v[dim];
    float v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
        v2[i] = (float)(i + 1.5);
    }

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(float) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(float) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = FP32_L2Sqr(v, v2, dim);
// CPU_FEATURES_ARCH_X86_64
#ifdef OPT_AVX512F
    if (optimization.avx512f) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_L2_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        // Unset avx512f flag, so we'll choose the next optimization (AVX).
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX
    if (optimization.avx) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_L2_implementation_AVX(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX with dim " << dim;
        // Unset avx flag as well, so we'll choose the next optimization (SSE).
        optimization.avx = 0;
    }
#endif
#ifdef OPT_SSE
    if (optimization.sse) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_L2_implementation_SSE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(128, dim)) << "SSE with dim " << dim;
        // Unset sse flag as well, so we'll choose the next option (default).
        optimization.sse = 0;
    }
#endif

// CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_L2_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_L2_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_L2_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        optimization.asimd = 0;
    }
#endif

    unsigned char alignment = 0;
    arch_opt_func = L2_FP32_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, FP32_L2Sqr) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(FP32SpacesOptimizationTest, FP32InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    float v[dim];
    float v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
        v2[i] = (float)(i + 1.5);
    }

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(float) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(float) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = FP32_InnerProduct(v, v2, dim);

// CPU_FEATURES_ARCH_X86_64
#ifdef OPT_AVX512F
    if (optimization.avx512f) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_IP_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX
    if (optimization.avx) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_IP_implementation_AVX(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX with dim " << dim;
        optimization.avx = 0;
    }
#endif
#ifdef OPT_SSE
    if (optimization.sse) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_IP_implementation_SSE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(128, dim)) << "SSE with dim " << dim;
        optimization.sse = 0;
    }
#endif

// CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_IP_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_IP_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_IP_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim OPT_NEON " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        optimization.asimd = 0;
    }
#endif

    unsigned char alignment = 0;
    arch_opt_func = IP_FP32_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, FP32_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

INSTANTIATE_TEST_SUITE_P(FP32OptFuncs, FP32SpacesOptimizationTest,
                         testing::Range(8UL, 32 * 2UL + 1));

class FP64SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(FP64SpacesOptimizationTest, FP64L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    double v[dim];
    double v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
        v2[i] = (double)(i + 1.5);
    }

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(double) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(double) : 0;
    };

    dist_func_t<double> arch_opt_func;
    double baseline = FP64_L2Sqr(v, v2, dim);
#ifdef OPT_AVX512F
    if (optimization.avx512f) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(arch_opt_func, Choose_FP64_L2_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX
    if (optimization.avx) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_L2_implementation_AVX(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX with dim " << dim;
        optimization.avx = 0;
    }
#endif
#ifdef OPT_SSE
    if (optimization.sse) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        ASSERT_EQ(arch_opt_func, Choose_FP64_L2_implementation_SSE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(128, dim)) << "SSE with dim " << dim;
        optimization.sse = 0;
    }
#endif

// CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_L2_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        EXPECT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_L2_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        EXPECT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_L2_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim OPT_NEON " << dim;
        EXPECT_EQ(baseline, arch_opt_func(v, v2, dim)) << "NOEN with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        optimization.asimd = 0;
    }
#endif

    unsigned char alignment = 0;
    arch_opt_func = L2_FP64_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, FP64_L2Sqr) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(FP64SpacesOptimizationTest, FP64InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    double v[dim];
    double v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
        v2[i] = (double)(i + 1.5);
    }

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(double) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(double) : 0;
    };

    dist_func_t<double> arch_opt_func;
    double baseline = FP64_InnerProduct(v, v2, dim);
#ifdef OPT_AVX512F
    if (optimization.avx512f) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_IP_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX
    if (optimization.avx) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_IP_implementation_AVX(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX with dim " << dim;
        optimization.avx = 0;
    }
#endif
#ifdef OPT_SSE
    if (optimization.sse) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_IP_implementation_SSE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(128, dim)) << "SSE with dim " << dim;
        optimization.sse = 0;
    }
#endif
// CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_IP_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        EXPECT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_IP_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        EXPECT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP64_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_IP_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim OPT_NEON " << dim;
        EXPECT_EQ(baseline, arch_opt_func(v, v2, dim)) << "NEON with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        optimization.asimd = 0;
    }
#endif

    unsigned char alignment = 0;
    arch_opt_func = IP_FP64_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, FP64_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

INSTANTIATE_TEST_SUITE_P(FP64OptFuncs, FP64SpacesOptimizationTest,
                         testing::Range(4UL, 16 * 2UL + 1));

class BF16SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(BF16SpacesOptimizationTest, BF16InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    bfloat16 v[dim];
    bfloat16 v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = vecsim_types::float_to_bf16((float)i);
        v2[i] = vecsim_types::float_to_bf16(((float)i + 1.5f));
    }

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(bfloat16) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(bfloat16) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = BF16_InnerProduct_LittleEndian(v, v2, dim);
#ifdef OPT_AVX512_BF16_VL
    if (optimization.avx512_bf16 && optimization.avx512vl) {
        unsigned char alignment = 0;
        arch_opt_func = IP_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_IP_implementation_AVX512BF16_VL(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        optimization.avx512_bf16 = optimization.avx512vl = 0;
    }
#endif
#ifdef OPT_AVX512_BW_VBMI2
    if (optimization.avx512bw && optimization.avx512vbmi2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_IP_implementation_AVX512BW_VBMI2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        optimization.avx512bw = optimization.avx512vbmi2 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_IP_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX with dim " << dim;
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE3
    if (optimization.sse3) {
        unsigned char alignment = 0;
        arch_opt_func = IP_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_IP_implementation_SSE3(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(128, dim)) << "SSE with dim " << dim;
        optimization.sse3 = 0;
    }
#endif
#ifdef OPT_SVE_BF16
    if (optimization.svebf16) {
        unsigned char alignment = 0;
        arch_opt_func = IP_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_IP_implementation_SVE_BF16(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SVE_BF16 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "SVE_BF16 with dim " << dim;
        optimization.svebf16 = 0;
    }
#endif
#ifdef OPT_NEON_BF16
    if (optimization.bf16) {
        unsigned char alignment = 0;
        arch_opt_func = IP_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_IP_implementation_NEON_BF16(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "NEON_BF16 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "NEON_BF16 with dim " << dim;
        optimization.bf16 = 0;
    }
#endif
    unsigned char alignment = 0;
    arch_opt_func = IP_BF16_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, BF16_InnerProduct_LittleEndian)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(BF16SpacesOptimizationTest, BF16L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    bfloat16 v[dim];
    bfloat16 v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = vecsim_types::float_to_bf16((float)i);
        v2[i] = vecsim_types::float_to_bf16(((float)i + 1.5f));
    }

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(bfloat16) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(bfloat16) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = BF16_L2Sqr_LittleEndian(v, v2, dim);
#ifdef OPT_AVX512_BW_VBMI2
    if (optimization.avx512bw && optimization.avx512vbmi2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_L2_implementation_AVX512BW_VBMI2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        optimization.avx512bw = optimization.avx512vbmi2 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_L2_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX with dim " << dim;
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE3
    if (optimization.sse3) {
        unsigned char alignment = 0;
        arch_opt_func = L2_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_L2_implementation_SSE3(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(128, dim)) << "SSE with dim " << dim;
        optimization.sse3 = 0;
    }
#endif
#ifdef OPT_SVE_BF16
    if (optimization.svebf16) {
        unsigned char alignment = 0;
        arch_opt_func = L2_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_L2_implementation_SVE_BF16(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SVE_BF16 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "SVE_BF16 with dim " << dim;
        optimization.svebf16 = 0;
    }
#endif
#ifdef OPT_NEON_BF16
    if (optimization.bf16) {
        unsigned char alignment = 0;
        arch_opt_func = L2_BF16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_L2_implementation_NEON_BF16(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "NEON_BF16 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "NEON_BF16 with dim " << dim;
        optimization.bf16 = 0;
    }
#endif
    unsigned char alignment = 0;
    arch_opt_func = L2_BF16_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, BF16_L2Sqr_LittleEndian)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

INSTANTIATE_TEST_SUITE_P(BF16OptFuncs, BF16SpacesOptimizationTest,
                         testing::Range(32UL, 32 * 2UL + 1));

class FP16SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(FP16SpacesOptimizationTest, FP16InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    float16 v1[dim], v2[dim];
    float v1_fp32[dim], v2_fp32[dim];
    for (size_t i = 0; i < dim; i++) {
        v1_fp32[i] = (float)i;
        v1[i] = vecsim_types::FP32_to_FP16(v1_fp32[i]);
        v2_fp32[i] = (float)i + 1.5f;
        v2[i] = vecsim_types::FP32_to_FP16(v2_fp32[i]);
    }

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(float16) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(float16) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = FP16_InnerProduct(v1, v2, dim);
    ASSERT_EQ(baseline, FP32_InnerProduct(v1_fp32, v2_fp32, dim)) << "Baseline check " << dim;
#if defined(CPU_FEATURES_ARCH_X86_64)
    // Turn off advanced fp16 flags. They will be tested in the next test.
    optimization.avx512_fp16 = optimization.avx512vl = 0;
#ifdef OPT_AVX512F
    // The AVX512F FP16 tier requires at least 16 elements; below that the dispatcher falls
    // through to the next tier.
    if (optimization.avx512f && dim >= 16) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_IP_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
    }
    optimization.avx512f = 0;
#endif
#ifdef OPT_F16C
    if (optimization.f16c && optimization.fma3 && optimization.avx) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_IP_implementation_F16C(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "F16C with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "F16C with dim " << dim;
        optimization.f16c = optimization.fma3 = optimization.avx = 0;
    }
#endif
#elif defined(CPU_FEATURES_ARCH_AARCH64)
    // Turn off advanced fp16 flags. They will be tested in the next test.
    optimization.sve = optimization.sve2 = optimization.asimdhp = 0;
#endif
    unsigned char alignment = 0;
    arch_opt_func = IP_FP16_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, FP16_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(FP16SpacesOptimizationTest, FP16L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    float16 v1[dim], v2[dim];
    float v1_fp32[dim], v2_fp32[dim];
    for (size_t i = 0; i < dim; i++) {
        v1_fp32[i] = (float)i;
        v1[i] = vecsim_types::FP32_to_FP16(v1_fp32[i]);
        v2_fp32[i] = (float)i + 1.5f;
        v2[i] = vecsim_types::FP32_to_FP16(v2_fp32[i]);
    }

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(float16) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(float16) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = FP16_L2Sqr(v1, v2, dim);
    ASSERT_EQ(baseline, FP32_L2Sqr(v1_fp32, v2_fp32, dim)) << "Baseline check " << dim;
#if defined(CPU_FEATURES_ARCH_X86_64)
    // Turn off advanced fp16 flags. They will be tested in the next test.
    optimization.avx512_fp16 = optimization.avx512vl = 0;
#ifdef OPT_AVX512F
    // The AVX512F FP16 tier requires at least 16 elements; below that the dispatcher falls
    // through to the next tier.
    if (optimization.avx512f && dim >= 16) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_L2_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
    }
    optimization.avx512f = 0;
#endif
#ifdef OPT_F16C
    if (optimization.f16c && optimization.fma3 && optimization.avx) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_L2_implementation_F16C(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "F16C with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "F16C with dim " << dim;
        optimization.f16c = optimization.fma3 = optimization.avx = 0;
    }
#endif
#elif defined(CPU_FEATURES_ARCH_AARCH64)
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_L2_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment SVE2 with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_L2_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment SVE with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_HP
    if (optimization.asimdhp) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_L2_implementation_NEON_HP(dim))
            << "Unexpected distance function chosen for dim OPT_NEON_HP " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "NEON_HP with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment NEON_HP with dim " << dim;
        optimization.asimdhp = 0;
    }
#endif
#endif
    unsigned char alignment = 0;
    arch_opt_func = L2_FP16_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, FP16_L2Sqr) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "F16C with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

INSTANTIATE_TEST_SUITE_P(FP16OptFuncs, FP16SpacesOptimizationTest,
                         testing::Range(8UL, 32 * 2UL + 1));

/** Since we are handling floats, the order of summation affect on the final result.
 * This is very significant when the entries are half precision floats, since the accumulated
 * error is much higher than in single precision floats.
 * In the following tests the error between the naive calculation to SIMD optimization function
 * is allowed to be up to 1%. If we wanted to be accurate, we could have done the baseline
 * calculations accumulating the results in a SIMD size vector and reduce the final result to float,
 * but this is too complicated for the scope of this test.
 * Special attention should be given to the implementation of the SIMD reduce function for float16,
 * that has different logic than the float32 and float64 reduce functions.
 * For more info, refer to intel's intrinsics guide.
 */
#if defined(OPT_AVX512_FP16_VL) || defined(CPU_FEATURES_ARCH_AARCH64)
class FP16SpacesOptimizationTestAdvanced : public testing::TestWithParam<size_t> {};

TEST_P(FP16SpacesOptimizationTestAdvanced, FP16InnerProductTestAdv) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    float16 v1[dim], v2[dim];

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.99, 0.99);

#if defined(CPU_FEATURES_ARCH_AARCH64) && defined(__GNUC__) && (__GNUC__ < 13)
    // https://github.com/pytorch/executorch/issues/6844
    __fp16 baseline = 0;
#else
    _Float16 baseline = 0;
#endif

    for (size_t i = 0; i < dim; i++) {
        float val1 = (dis(gen));
        float val2 = (dis(gen));
        v1[i] = vecsim_types::FP32_to_FP16((val1));
        v2[i] = vecsim_types::FP32_to_FP16((val2));

        baseline += static_cast<decltype(baseline)>(val1) * static_cast<decltype(baseline)>(val2);
    }
    baseline = decltype(baseline)(1) - baseline;

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(float16) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(float16) : 0;
    };

    dist_func_t<float> arch_opt_func;

#ifdef OPT_AVX512_FP16_VL
    if (optimization.avx512_fp16 && optimization.avx512vl) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_IP_implementation_AVX512FP16_VL(dim))
            << "Unexpected distance function chosen for dim " << dim;
        float dist = arch_opt_func(v1, v2, dim);
        float f_baseline = baseline;
        float error = std::abs((dist / f_baseline) - 1);
        // Alow 1% error
        ASSERT_LE(error, 0.01) << "AVX512 with dim " << dim << ", baseline: " << f_baseline
                               << ", dist: " << dist;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
    }
#endif
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_IP_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        float dist = arch_opt_func(v1, v2, dim);
        float f_baseline = baseline;
        float error = std::abs((dist / f_baseline) - 1);
        // Alow 1% error
        ASSERT_LE(error, 0.01) << "SVE2 with dim " << dim << ", baseline: " << f_baseline
                               << ", dist: " << dist;
        // ASSERT_EQ(alignment, expected_alignment(512, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "SVE2 with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_IP_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        float dist = arch_opt_func(v1, v2, dim);
        float f_baseline = baseline;
        float error = std::abs((dist / f_baseline) - 1);
        // Alow 1% error
        ASSERT_LE(error, 0.01) << "SVE with dim " << dim << ", baseline: " << f_baseline
                               << ", dist: " << dist;
        // ASSERT_EQ(alignment, expected_alignment(512, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "SVE with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_HP
    if (optimization.asimdhp) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_IP_implementation_NEON_HP(dim))
            << "Unexpected distance function chosen for dim " << dim;
        float dist = arch_opt_func(v1, v2, dim);
        float f_baseline = baseline;
        float error = std::abs((dist / f_baseline) - 1);
        // Alow 1% error
        ASSERT_LE(error, 0.01) << "NEON_HP with dim " << dim << ", baseline: " << f_baseline
                               << ", dist: " << dist;
        // ASSERT_EQ(alignment, expected_alignment(512, dim)) << "NEON_HP with dim " << dim;
        ASSERT_EQ(alignment, 0) << "NEON_HP with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.asimdhp = 0;
    }
#endif
}

#ifdef OPT_AVX512_FP16_VL
TEST_P(FP16SpacesOptimizationTestAdvanced, FP16L2SqrTestAdv) {
    auto optimization = cpu_features::GetX86Info().features;
    if (optimization.avx512_fp16 && optimization.avx512vl) {
        size_t dim = GetParam();
        float16 v1[dim], v2[dim];

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-0.99f, 0.99f);

        _Float16 baseline = 0;
        for (size_t i = 0; i < dim; i++) {
            float val1 = (dis(gen));
            float val2 = (dis(gen));
            v1[i] = vecsim_types::FP32_to_FP16((val1));
            v2[i] = vecsim_types::FP32_to_FP16((val2));

            _Float16 diff = static_cast<_Float16>(val1) - static_cast<_Float16>(val2);
            baseline += diff * diff;
        }

        auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
            size_t elements_in_reg = reg_bit_size / sizeof(float16) / 8;
            return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(float16) : 0;
        };

        dist_func_t<float> arch_opt_func;
        unsigned char alignment = 0;
        arch_opt_func = L2_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_L2_implementation_AVX512FP16_VL(dim))
            << "Unexpected distance function chosen for dim " << dim;
        float dist = arch_opt_func(v1, v2, dim);
        float f_baseline = baseline;
        float error = std::abs((dist / f_baseline) - 1);
        // Alow 1% error
        ASSERT_LE(error, 0.01) << "AVX512 with dim " << dim << ", baseline: " << f_baseline
                               << ", dist: " << dist;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
    }
}
#endif

// Start from a 32 multiplier
INSTANTIATE_TEST_SUITE_P(, FP16SpacesOptimizationTestAdvanced,
                         testing::Range(512UL, 512 + 32UL + 1));

#endif // defined(OPT_AVX512_FP16_VL) || defined(CPU_FEATURES_ARCH_AARCH64)

class INT8SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(INT8SpacesOptimizationTest, INT8L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    int8_t v1[dim];
    int8_t v2[dim];
    test_utils::populate_int8_vec(v1, dim, 123);
    test_utils::populate_int8_vec(v2, dim, 1234);

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(int8_t) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(int8_t) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = INT8_L2Sqr(v1, v2, dim);
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vl &&
        optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = L2_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_L2_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX512 with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.avx512f = optimization.avx512bw = optimization.avx512vl =
            optimization.avx512vnni = 0;
    }
#endif
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_L2_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = L2_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_L2_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        arch_opt_func = L2_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_L2_implementation_NEON_DOTPROD(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "NEON_DOTPROD with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(0, dim)) << "NEON_DOTPROD with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = L2_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_L2_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "NEON with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(0, dim)) << "NEON with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.asimd = 0;
    }
#endif
    unsigned char alignment = 0;
    arch_opt_func = L2_INT8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, INT8_L2Sqr) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(INT8SpacesOptimizationTest, INT8InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    int8_t v1[dim];
    int8_t v2[dim];
    test_utils::populate_int8_vec(v1, dim, 123);
    test_utils::populate_int8_vec(v2, dim, 1234);

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(int8_t) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(int8_t) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = INT8_InnerProduct(v1, v2, dim);
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_IP_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = IP_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_IP_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        arch_opt_func = IP_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_IP_implementation_NEON_DOTPROD(dim))
            << "Unexpected distance function chosen for dim OPT_NEON_DOTPROD " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "NEON_DOTPROD with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(0, dim)) << "NEON_DOTPROD with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = IP_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_IP_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim OPT_NEON " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vl &&
        optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = IP_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_IP_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX512 with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.avx512f = optimization.avx512bw = optimization.avx512vl =
            optimization.avx512vnni = 0;
    }
#endif
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_IP_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = IP_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_IP_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
    unsigned char alignment = 0;
    arch_opt_func = IP_INT8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, INT8_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(INT8SpacesOptimizationTest, INT8CosineTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    int8_t v1[dim + sizeof(float)];
    int8_t v2[dim + sizeof(float)];
    test_utils::populate_int8_vec(v1, dim, 123);
    test_utils::populate_int8_vec(v2, dim, 1234);

    // write the norm at the end of the vector
    const float norm_v1 = test_utils::integral_compute_norm(v1, dim);
    const float norm_v2 = test_utils::integral_compute_norm(v2, dim);
    std::memcpy(v1 + dim, &norm_v1, sizeof(norm_v1));
    std::memcpy(v2 + dim, &norm_v2, sizeof(norm_v2));

    dist_func_t<float> arch_opt_func;
    float baseline = INT8_Cosine(v1, v2, dim);
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vl &&
        optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_Cosine_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        // We don't align int8 vectors with cosine distance
        ASSERT_EQ(alignment, 0) << "AVX512 with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.avx512f = optimization.avx512bw = optimization.avx512vl =
            optimization.avx512vnni = 0;
    }
#endif
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_Cosine_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_Cosine_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_Cosine_implementation_NEON_DOTPROD(dim))
            << "Unexpected distance function chosen for dim OPT_NEON_DOTPROD " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "NEON_DOTPROD with dim " << dim;
        ASSERT_EQ(alignment, 0) << "NEON_DOTPROD with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_INT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_INT8_Cosine_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim OPT_NEON " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        optimization.asimd = 0;
    }
#endif
    unsigned char alignment = 0;
    arch_opt_func = Cosine_INT8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, INT8_Cosine) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

INSTANTIATE_TEST_SUITE_P(INT8OptFuncs, INT8SpacesOptimizationTest,
                         testing::Range(32UL, 32 * 2UL + 1));

class UINT8SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(UINT8SpacesOptimizationTest, UINT8L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    uint8_t v1[dim];
    uint8_t v2[dim];
    test_utils::populate_uint8_vec(v1, dim, 123);
    test_utils::populate_uint8_vec(v2, dim, 1234);

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(uint8_t) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(uint8_t) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = UINT8_L2Sqr(v1, v2, dim);
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_L2_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = L2_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_L2_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        arch_opt_func = L2_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_L2_implementation_NEON_DOTPROD(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "NEON with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(0, dim)) << "NEON with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = L2_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_L2_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "NEON with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(0, dim)) << "NEON with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.asimd = 0;
    }
#endif

#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vl &&
        optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = L2_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_L2_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX512 with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.avx512f = optimization.avx512bw = optimization.avx512vl =
            optimization.avx512vnni = 0;
    }
#endif
    unsigned char alignment = 0;
    arch_opt_func = L2_UINT8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, UINT8_L2Sqr) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(UINT8SpacesOptimizationTest, UINT8InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    uint8_t v1[dim];
    uint8_t v2[dim];
    test_utils::populate_uint8_vec(v1, dim, 123);
    test_utils::populate_uint8_vec(v2, dim, 1234);

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(uint8_t) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(uint8_t) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = UINT8_InnerProduct(v1, v2, dim);
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_IP_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = IP_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_IP_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        arch_opt_func = IP_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_IP_implementation_NEON_DOTPROD(dim))
            << "Unexpected distance function chosen for dim NEON_DOTPROD " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "NEON_DOTPROD with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(0, dim)) << "NEON_DOTPROD with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = IP_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_IP_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim OPT_NEON " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vl &&
        optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = IP_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_IP_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX512 with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.avx512f = optimization.avx512bw = optimization.avx512vl =
            optimization.avx512vnni = 0;
    }
#endif
    unsigned char alignment = 0;
    arch_opt_func = IP_UINT8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, UINT8_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(UINT8SpacesOptimizationTest, UINT8CosineTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    uint8_t v1[dim + sizeof(float)];
    uint8_t v2[dim + sizeof(float)];
    test_utils::populate_uint8_vec(v1, dim, 123);
    test_utils::populate_uint8_vec(v2, dim, 1234);

    // write the norm at the end of the vector
    const float norm_v1 = test_utils::integral_compute_norm(v1, dim);
    const float norm_v2 = test_utils::integral_compute_norm(v2, dim);
    std::memcpy(v1 + dim, &norm_v1, sizeof(norm_v1));
    std::memcpy(v2 + dim, &norm_v2, sizeof(norm_v2));

    dist_func_t<float> arch_opt_func;
    float baseline = UINT8_Cosine(v1, v2, dim);
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_Cosine_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_Cosine_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_Cosine_implementation_NEON_DOTPROD(dim))
            << "Unexpected distance function chosen for dim OPT_NEON_DOTPROD " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "NEON_DOTPROD with dim " << dim;
        // We don't align uint8 vectors with cosine distance
        ASSERT_EQ(alignment, 0) << "NEON with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_Cosine_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim OPT_NEON " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vl &&
        optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_UINT8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_UINT8_Cosine_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        // We don't align uint8 vectors with cosine distance
        ASSERT_EQ(alignment, 0) << "AVX512 with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.avx512f = optimization.avx512bw = optimization.avx512vl =
            optimization.avx512vnni = 0;
    }
#endif

    unsigned char alignment = 0;
    arch_opt_func = Cosine_UINT8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, UINT8_Cosine) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}
TEST_P(UINT8SpacesOptimizationTest, UINT8_full_range_test) {
    auto optimization = getCpuOptimizationFeatures();
    constexpr size_t dim = 512;

    uint8_t v1[dim + sizeof(float)];
    uint8_t v2[dim + sizeof(float)];

    // v1: 0..255 followed by 255..0
    for (size_t i = 0; i < 256; i++) {
        v1[i] = static_cast<uint8_t>(i);
        v1[256 + i] = static_cast<uint8_t>(255 - i);
    }

    // v2: 255..0 followed by 0..255
    for (size_t i = 0; i < 256; i++) {
        v2[i] = static_cast<uint8_t>(255 - i);
        v2[256 + i] = static_cast<uint8_t>(i);
    }

    // write the norm at the end of the vector
    const float norm_v1 = test_utils::integral_compute_norm(v1, dim);
    const float norm_v2 = test_utils::integral_compute_norm(v2, dim);
    std::memcpy(v1 + dim, &norm_v1, sizeof(norm_v1));
    std::memcpy(v2 + dim, &norm_v2, sizeof(norm_v2));

    float baseline_l2 = UINT8_L2Sqr(v1, v2, dim);
    float baseline_ip = UINT8_InnerProduct(v1, v2, dim);
    float baseline_cosine = UINT8_Cosine(v1, v2, dim);

    dist_func_t<float> arch_opt_func;

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;

        arch_opt_func = Choose_UINT8_L2_implementation_SVE2(dim);
        ASSERT_EQ(baseline_l2, arch_opt_func(v1, v2, dim)) << "L2 SVE2 with dim " << dim;
        arch_opt_func = Choose_UINT8_IP_implementation_SVE2(dim);
        ASSERT_EQ(baseline_ip, arch_opt_func(v1, v2, dim)) << "IP SVE2 with dim " << dim;
        arch_opt_func = Choose_UINT8_Cosine_implementation_SVE2(dim);
        ASSERT_EQ(baseline_cosine, arch_opt_func(v1, v2, dim)) << "Cosine SVE2 with dim " << dim;

        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;

        arch_opt_func = Choose_UINT8_L2_implementation_SVE(dim);
        ASSERT_EQ(baseline_l2, arch_opt_func(v1, v2, dim)) << "L2 SVE with dim " << dim;
        arch_opt_func = Choose_UINT8_IP_implementation_SVE(dim);
        ASSERT_EQ(baseline_ip, arch_opt_func(v1, v2, dim)) << "IP SVE with dim " << dim;
        arch_opt_func = Choose_UINT8_Cosine_implementation_SVE(dim);
        ASSERT_EQ(baseline_cosine, arch_opt_func(v1, v2, dim)) << "Cosine SVE with dim " << dim;

        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vl &&
        optimization.avx512vnni) {
        unsigned char alignment = 0;

        arch_opt_func = Choose_UINT8_L2_implementation_AVX512F_BW_VL_VNNI(dim);
        ASSERT_EQ(baseline_l2, arch_opt_func(v1, v2, dim)) << "L2 AVX512 with dim " << dim;
        arch_opt_func = Choose_UINT8_IP_implementation_AVX512F_BW_VL_VNNI(dim);
        ASSERT_EQ(baseline_ip, arch_opt_func(v1, v2, dim)) << "IP AVX512 with dim " << dim;
        arch_opt_func = Choose_UINT8_Cosine_implementation_AVX512F_BW_VL_VNNI(dim);
        ASSERT_EQ(baseline_cosine, arch_opt_func(v1, v2, dim)) << "Cosine AVX512 with dim " << dim;

        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.avx512f = optimization.avx512bw = optimization.avx512vl =
            optimization.avx512vnni = 0;
    }
#endif
}

INSTANTIATE_TEST_SUITE_P(UINT8OptFuncs, UINT8SpacesOptimizationTest,
                         testing::Range(32UL, 64 * 2UL + 1));

// The accumulated total is 255 * 255 * dim, which passes INT_MAX from dimension 33,026: the scalar
// path was signed-overflow UB there, and the AVX512 and NEON L2 reduces read their unsigned total
// back as a signed int and went negative. All-255 bytes are the worst case and make the expected
// value an exact integer. The existing UINT8 suites stop at dim 128, which is why this went unseen.
TEST_F(SpacesTest, UINT8_L2Sqr_and_InnerProduct_are_exact_past_int32) {
    for (const size_t dim : {33026UL, 40000UL}) {
        std::vector<uint8_t> v1(dim + sizeof(float), 255);
        std::vector<uint8_t> v2(dim + sizeof(float), 0);

        // L2 between all-255 and all-0 is 255^2 * dim.
        const double expected_l2 = 255.0 * 255.0 * static_cast<double>(dim);
        const float l2 = UINT8_L2Sqr(v1.data(), v2.data(), dim);
        EXPECT_GT(l2, 0.0f) << "dim " << dim << ": squared distance went negative";
        EXPECT_LT(std::abs(static_cast<double>(l2) - expected_l2) / expected_l2, 1e-6)
            << "scalar L2, dim " << dim;

        unsigned char alignment = 0;
        auto dispatched_l2 = L2_UINT8_GetDistFunc(dim, &alignment, nullptr);
        const float l2_simd = dispatched_l2(v1.data(), v2.data(), dim);
        EXPECT_GT(l2_simd, 0.0f) << "dim " << dim << ": SIMD squared distance went negative";
        EXPECT_LT(std::abs(static_cast<double>(l2_simd) - expected_l2) / expected_l2, 1e-6)
            << "dispatched L2, dim " << dim;

        // IP between two all-255 vectors is 255^2 * dim, and the kernel returns 1 - IP.
        const double expected_ip = 1.0 - 255.0 * 255.0 * static_cast<double>(dim);
        const double ip = static_cast<double>(UINT8_InnerProduct(v1.data(), v1.data(), dim));
        EXPECT_LT(std::abs(ip - expected_ip) / std::abs(expected_ip), 1e-6)
            << "scalar IP, dim " << dim;

        auto dispatched_ip = IP_UINT8_GetDistFunc(dim, &alignment, nullptr);
        const double ip_simd = static_cast<double>(dispatched_ip(v1.data(), v1.data(), dim));
        EXPECT_LT(std::abs(ip_simd - expected_ip) / std::abs(expected_ip), 1e-6)
            << "dispatched IP, dim " << dim;
    }
}

// Past spaces::UINT8_CHUNK_ELEMENTS the uint8 SIMD kernels accumulate in chunks: each chunk's total
// still fits the 32-bit accumulators (65025 * 65536 <= UINT32_MAX), and the per-chunk totals are
// folded in 64 bits. The dispatched kernel must therefore agree exactly with the scalar kernel,
// which accumulates the whole vector into a 64-bit ret_t. Exact equality is the right assertion
// because both paths convert the same integer total to float once, at the end.
//
// All-255 against all-0 is the worst case and puts the L2 total past UINT32_MAX from dimension
// 66,052, so the multi-chunk dimensions below genuinely exercise the 64-bit fold. The existing
// UINT8 suites stop at dim 128, which is why the wrap went unseen.
TEST_F(SpacesTest, UINT8_dispatched_kernels_are_exact_across_the_chunk_boundary) {
    // Below the boundary, on it, one past it (whose last chunk is a single 64-element block), an
    // exact multiple of it, and dimensions spanning two and three chunks.
    for (const size_t dim : {65535UL, 65536UL, 65537UL, 65600UL, 131072UL, 131109UL, 200000UL}) {
        // The cosine kernels read a float norm from just past the payload, so size for both.
        std::vector<uint8_t> ones(dim + sizeof(float), 255);
        std::vector<uint8_t> zeros(dim + sizeof(float), 0);
        std::vector<uint8_t> ramp(dim + sizeof(float));
        for (size_t i = 0; i < dim; i++) {
            ramp[i] = static_cast<uint8_t>(i % 256);
        }
        const float norm = std::sqrt(255.0f * 255.0f * static_cast<float>(dim));
        memcpy(ones.data() + dim, &norm, sizeof(float));
        memcpy(ramp.data() + dim, &norm, sizeof(float));

        unsigned char alignment = 0;
        auto l2 = L2_UINT8_GetDistFunc(dim, &alignment, nullptr);
        auto ip = IP_UINT8_GetDistFunc(dim, &alignment, nullptr);
        auto cosine = Cosine_UINT8_GetDistFunc(dim, &alignment, nullptr);

        // Worst case: the largest total the byte range allows.
        EXPECT_EQ(UINT8_L2Sqr(ones.data(), zeros.data(), dim), l2(ones.data(), zeros.data(), dim))
            << "L2 all-255 vs all-0, dim " << dim;
        EXPECT_EQ(UINT8_InnerProduct(ones.data(), ones.data(), dim),
                  ip(ones.data(), ones.data(), dim))
            << "IP all-255, dim " << dim;
        EXPECT_EQ(UINT8_Cosine(ones.data(), ones.data(), dim),
                  cosine(ones.data(), ones.data(), dim))
            << "Cosine all-255, dim " << dim;

        // A varying pattern, so the residual and chunk seams have to line up element for element
        // rather than merely produce the right sum of identical values.
        EXPECT_EQ(UINT8_L2Sqr(ramp.data(), ones.data(), dim), l2(ramp.data(), ones.data(), dim))
            << "L2 ramp vs all-255, dim " << dim;
        EXPECT_EQ(UINT8_InnerProduct(ramp.data(), ones.data(), dim),
                  ip(ramp.data(), ones.data(), dim))
            << "IP ramp vs all-255, dim " << dim;
        EXPECT_EQ(UINT8_Cosine(ramp.data(), ones.data(), dim),
                  cosine(ramp.data(), ones.data(), dim))
            << "Cosine ramp vs all-255, dim " << dim;
    }
}

// The boundary test above samples dimensions; this sweeps every residual instantiation. 65,600 and
// 196,608 are both multiples of 64, so base + r has residual r: one chunk past the boundary, then
// three chunks past it, so the seam between the residual-bearing first chunk and the residual-0
// chunks after it is exercised for all 64 shapes. A ramp against all-255 is position sensitive, so
// a seam that double-counts or skips elements changes the total rather than cancelling out, and the
// total still passes UINT32_MAX (about 32,500 * dim) so the 64-bit fold is under test throughout.
TEST_F(SpacesTest, UINT8_dispatched_kernels_are_exact_at_every_residual_past_the_chunk_boundary) {
    constexpr size_t max_dim = 196608 + 63;
    std::vector<uint8_t> ones(max_dim + sizeof(float), 255);
    std::vector<uint8_t> ramp(max_dim + sizeof(float));
    for (size_t i = 0; i < max_dim; i++) {
        ramp[i] = static_cast<uint8_t>(i % 256);
    }

    for (const size_t base : {65600UL, 196608UL}) {
        for (size_t r = 0; r < 64; r++) {
            const size_t dim = base + r;
            // The cosine kernels read a float norm from just past the payload, which moves with
            // dim.
            const float norm = std::sqrt(255.0f * 255.0f * static_cast<float>(dim));
            memcpy(ones.data() + dim, &norm, sizeof(float));
            memcpy(ramp.data() + dim, &norm, sizeof(float));

            unsigned char alignment = 0;
            const void *a = ramp.data();
            const void *b = ones.data();

            EXPECT_EQ(UINT8_L2Sqr(a, b, dim),
                      L2_UINT8_GetDistFunc(dim, &alignment, nullptr)(a, b, dim))
                << "L2 at dim " << dim << " (residual " << r << ")";
            EXPECT_EQ(UINT8_InnerProduct(a, b, dim),
                      IP_UINT8_GetDistFunc(dim, &alignment, nullptr)(a, b, dim))
                << "IP at dim " << dim << " (residual " << r << ")";
            EXPECT_EQ(UINT8_Cosine(a, b, dim),
                      Cosine_UINT8_GetDistFunc(dim, &alignment, nullptr)(a, b, dim))
                << "Cosine at dim " << dim << " (residual " << r << ")";
        }
    }
}

// Every uint8 SIMD tier this host can actually execute, with its three dispatched kernels. Both
// the per-tier exactness test and the independent-oracle test below iterate this list, so a tier
// cannot be covered by one and silently missed by the other.
struct UInt8TierFuncs {
    const char *name;
    dist_func_t<float> l2;
    dist_func_t<float> ip;
    dist_func_t<float> cosine;
};

static std::vector<UInt8TierFuncs> AvailableUInt8Tiers(size_t dim) {
    std::vector<UInt8TierFuncs> tiers;
    [[maybe_unused]] const auto opt = getCpuOptimizationFeatures();
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (opt.avx512f && opt.avx512bw && opt.avx512vl && opt.avx512vnni) {
        tiers.push_back({"AVX512F_BW_VL_VNNI",
                         Choose_UINT8_L2_implementation_AVX512F_BW_VL_VNNI(dim),
                         Choose_UINT8_IP_implementation_AVX512F_BW_VL_VNNI(dim),
                         Choose_UINT8_Cosine_implementation_AVX512F_BW_VL_VNNI(dim)});
    }
#endif
#ifdef OPT_SVE2
    if (opt.sve2) {
        tiers.push_back({"SVE2", Choose_UINT8_L2_implementation_SVE2(dim),
                         Choose_UINT8_IP_implementation_SVE2(dim),
                         Choose_UINT8_Cosine_implementation_SVE2(dim)});
    }
#endif
#ifdef OPT_SVE
    if (opt.sve) {
        tiers.push_back({"SVE", Choose_UINT8_L2_implementation_SVE(dim),
                         Choose_UINT8_IP_implementation_SVE(dim),
                         Choose_UINT8_Cosine_implementation_SVE(dim)});
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (opt.asimddp) {
        tiers.push_back({"NEON_DOTPROD", Choose_UINT8_L2_implementation_NEON_DOTPROD(dim),
                         Choose_UINT8_IP_implementation_NEON_DOTPROD(dim),
                         Choose_UINT8_Cosine_implementation_NEON_DOTPROD(dim)});
    }
#endif
#ifdef OPT_NEON
    if (opt.asimd) {
        tiers.push_back({"NEON", Choose_UINT8_L2_implementation_NEON(dim),
                         Choose_UINT8_IP_implementation_NEON(dim),
                         Choose_UINT8_Cosine_implementation_NEON(dim)});
    }
#endif
    return tiers;
}

// Worst-case inputs against an oracle computed here in 64-bit integers, rather than by calling the
// scalar kernel.
//
// Why not the scalar kernel: scalar and SIMD share conventions, and this series changed the scalar
// and SIMD inner product epilogues together so they would stay bit-identical. A test asserting only
// scalar == SIMD cannot catch that shared convention being wrong, in either sign or width. Here the
// expectation is derived from the inputs alone, and the scalar kernel is asserted against it on the
// same footing as every SIMD tier.
//
// Why these inputs: all-255 against all-255 puts 65,025 into the inner product accumulator for
// every element, and all-255 against all-0 does the same for L2. Those are the maxima the byte
// range allows, so they are where a 32-bit accumulator wraps first. A ramp against all-255 is
// carried alongside because constant data lets a gap and an overlap of equal size cancel, which a
// position-dependent pattern does not.
//
// Why these dimensions: 65024 is 1016*64, so 65024+r has residual r and stays at or below the
// 65,536 chunk size, exercising the plain kernel right up against the limit of its 32-bit reduce
// (65025 * 65087 is about 4.23e9, just under UINT32_MAX). 131072 is 2048*64, so 131072+r has
// residual r and its total is about 8.5e9, which only a 64-bit fold can carry. Every residual is
// swept at both.
TEST_F(SpacesTest, UINT8_worst_case_matches_an_independent_64bit_oracle) {
    constexpr size_t max_dim = 131072 + 63;
    std::vector<uint8_t> ones(max_dim + sizeof(float), 255);
    std::vector<uint8_t> zeros(max_dim + sizeof(float), 0);
    std::vector<uint8_t> ramp(max_dim + sizeof(float));
    for (size_t i = 0; i < max_dim; i++) {
        ramp[i] = static_cast<uint8_t>(i % 256);
    }

    for (const size_t base : {65024UL, 131072UL}) {
        for (size_t r = 0; r < 64; r++) {
            const size_t dim = base + r;
            SCOPED_TRACE("dim " + std::to_string(dim) + " residual " + std::to_string(r));

            // Norms live just past the payload and move with dim, so rewrite them per dimension.
            const float norm_ones = std::sqrt(255.0f * 255.0f * static_cast<float>(dim));
            float norm_ramp = 0.0f;
            for (size_t i = 0; i < dim; i++) {
                norm_ramp += static_cast<float>(ramp[i]) * static_cast<float>(ramp[i]);
            }
            norm_ramp = std::sqrt(norm_ramp);
            memcpy(ones.data() + dim, &norm_ones, sizeof(float));
            memcpy(ramp.data() + dim, &norm_ramp, sizeof(float));

            struct Pair {
                const char *name;
                const uint8_t *a;
                const uint8_t *b;
                float norm_a;
                float norm_b;
                bool check_cosine;
            };
            const Pair pairs[] = {
                {"all-255 vs all-255", ones.data(), ones.data(), norm_ones, norm_ones, true},
                {"all-255 vs all-0", ones.data(), zeros.data(), norm_ones, 0.0f, false},
                {"ramp vs all-255", ramp.data(), ones.data(), norm_ramp, norm_ones, true},
            };

            for (const auto &pr : pairs) {
                // The oracle: plain 64-bit integer accumulation over the inputs.
                uint64_t ip_total = 0;
                uint64_t l2_total = 0;
                for (size_t i = 0; i < dim; i++) {
                    const uint64_t x = pr.a[i];
                    const uint64_t y = pr.b[i];
                    ip_total += x * y;
                    const int64_t diff = static_cast<int64_t>(x) - static_cast<int64_t>(y);
                    l2_total += static_cast<uint64_t>(diff * diff);
                }
                // Both worst-case pairs must exceed a 32-bit accumulator at the multi-chunk base,
                // otherwise this test would not be reaching the case it exists for.
                if (base == 131072 && pr.b != ramp.data() && pr.a != ramp.data()) {
                    EXPECT_GT(std::max(ip_total, l2_total),
                              static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
                        << "worst case no longer exceeds UINT32_MAX, test is not exercising the "
                           "fold";
                }

                // Expected returns, formed with the same operations the kernels use so the
                // comparison can be exact rather than approximate.
                const float want_ip = static_cast<float>(1 - static_cast<int64_t>(ip_total));
                const float want_l2 = static_cast<float>(l2_total);
                const float want_cos =
                    1.0f - static_cast<float>(ip_total) / (pr.norm_a * pr.norm_b);

                EXPECT_EQ(want_l2, UINT8_L2Sqr(pr.a, pr.b, dim)) << "scalar L2, " << pr.name;
                EXPECT_EQ(want_ip, UINT8_InnerProduct(pr.a, pr.b, dim)) << "scalar IP, " << pr.name;
                if (pr.check_cosine) {
                    EXPECT_EQ(want_cos, UINT8_Cosine(pr.a, pr.b, dim))
                        << "scalar cosine, " << pr.name;
                }

                for (const auto &tier : AvailableUInt8Tiers(dim)) {
                    EXPECT_EQ(want_l2, tier.l2(pr.a, pr.b, dim))
                        << "L2 " << tier.name << ", " << pr.name;
                    EXPECT_EQ(want_ip, tier.ip(pr.a, pr.b, dim))
                        << "IP " << tier.name << ", " << pr.name;
                    if (pr.check_cosine) {
                        EXPECT_EQ(want_cos, tier.cosine(pr.a, pr.b, dim))
                            << "cosine " << tier.name << ", " << pr.name;
                    }
                }
            }
        }
    }
}

// The tests above go through the generic dispatcher, which only ever returns the best tier this
// host supports, so on an ARM machine with SVE the NEON and NEON_DOTPROD chunked kernels are never
// executed. Reach every compiled-in tier directly instead. Each tier is still gated on the CPU
// actually supporting it, since calling an unsupported kernel faults.
TEST_F(SpacesTest, UINT8_every_tier_is_exact_past_the_chunk_boundary) {
    // Boundary (plain family), one past it with residuals 0/1/63, two chunks, and a ragged
    // multiple.
    const std::vector<size_t> dims = {65536, 65600, 65601, 65663, 131072, 200000};
    std::set<std::string> all_tiers;

    constexpr size_t max_dim = 200000;
    std::vector<uint8_t> ones(max_dim + sizeof(float), 255);
    std::vector<uint8_t> ramp(max_dim + sizeof(float));
    for (size_t i = 0; i < max_dim; i++) {
        ramp[i] = static_cast<uint8_t>(i % 256);
    }

    for (const size_t dim : dims) {
        const float norm = std::sqrt(255.0f * 255.0f * static_cast<float>(dim));
        memcpy(ones.data() + dim, &norm, sizeof(float));
        memcpy(ramp.data() + dim, &norm, sizeof(float));
        const void *a = ramp.data();
        const void *b = ones.data();

        const float want_l2 = UINT8_L2Sqr(a, b, dim);
        const float want_ip = UINT8_InnerProduct(a, b, dim);
        const float want_cos = UINT8_Cosine(a, b, dim);

        std::string covered;
        for (const auto &tier : AvailableUInt8Tiers(dim)) {
            EXPECT_EQ(want_l2, tier.l2(a, b, dim)) << "L2 " << tier.name << " dim " << dim;
            EXPECT_EQ(want_ip, tier.ip(a, b, dim)) << "IP " << tier.name << " dim " << dim;
            EXPECT_EQ(want_cos, tier.cosine(a, b, dim)) << "Cosine " << tier.name << " dim " << dim;
            all_tiers.insert(tier.name);
            covered += covered.empty() ? tier.name : std::string(", ") + tier.name;
        }
        RecordProperty("tiers_at_dim_" + std::to_string(dim), covered);
        std::cout << "  dim " << dim << " covered tiers: " << (covered.empty() ? "<none>" : covered)
                  << std::endl;
    }

    // Order matters. A stated requirement must be able to FAIL, so it is checked before the skip:
    // hardware-specific CI sets VECSIM_REQUIRE_UINT8_TIER to the tier that job exists to cover
    // (AVX512F_BW_VL_VNNI, SVE2, SVE, NEON_DOTPROD or NEON), and a mislabeled or silently
    // downgraded runner then fails instead of quietly skipping.
    // An empty value counts as unset. A CI expression that expands to nothing still puts the
    // variable in the environment, so getenv returns a pointer to "" rather than nullptr, and
    // treating that as a requirement fails every job that did not ask for one.
    const char *required = std::getenv("VECSIM_REQUIRE_UINT8_TIER");
    if (required != nullptr && *required != '\0') {
        EXPECT_TRUE(all_tiers.count(required) > 0)
            << "VECSIM_REQUIRE_UINT8_TIER=" << required << " but that tier was not exercised. "
            << "This host reached " << all_tiers.size() << " tier(s), so the run proves nothing "
            << "about " << required;
        return;
    }

    // With no requirement stated, a run that exercised no tier proves nothing. Report it skipped
    // rather than passed, because passing reads as coverage on a host that has none.
    if (all_tiers.empty()) {
        GTEST_SKIP() << "no uint8 SIMD tier on this host, no chunked kernel was executed";
    }
}

// The chooser picks the chunked kernel once per index rather than branching per call, so assert the
// switch actually happens. Both dimensions are a multiple of 64, so they map to the same residual
// instantiation: any difference in the returned pointer can only come from the chunked family being
// chosen. Only meaningful where a uint8 SIMD tier exists, since otherwise both are the scalar
// kernel.
TEST_F(SpacesTest, UINT8_choosers_switch_to_the_chunked_kernel_past_the_chunk_size) {
    const auto features = getCpuOptimizationFeatures();
    const bool has_uint8_simd =
#ifdef CPU_FEATURES_ARCH_X86_64
        features.avx512f && features.avx512bw && features.avx512vl && features.avx512vnni;
#else
        features.sve2 || features.sve || features.asimddp || features.asimd;
#endif
    if (!has_uint8_simd) {
        GTEST_SKIP() << "no uint8 SIMD tier on this host";
    }

    constexpr size_t plain = spaces::UINT8_CHUNK_ELEMENTS;       // on the boundary, not chunked
    constexpr size_t chunked = spaces::UINT8_CHUNK_ELEMENTS * 2; // same residual, chunked
    unsigned char alignment = 0;

    EXPECT_NE(L2_UINT8_GetDistFunc(plain, &alignment, nullptr),
              L2_UINT8_GetDistFunc(chunked, &alignment, nullptr));
    EXPECT_NE(IP_UINT8_GetDistFunc(plain, &alignment, nullptr),
              IP_UINT8_GetDistFunc(chunked, &alignment, nullptr));
    EXPECT_NE(Cosine_UINT8_GetDistFunc(plain, &alignment, nullptr),
              Cosine_UINT8_GetDistFunc(chunked, &alignment, nullptr));

    // And the SIMD kernel is still what gets chosen past the boundary: the chunked variant replaces
    // the plain one, it does not fall back to scalar.
    EXPECT_NE(L2_UINT8_GetDistFunc(chunked, &alignment, nullptr), UINT8_L2Sqr);
    EXPECT_NE(IP_UINT8_GetDistFunc(chunked, &alignment, nullptr), UINT8_InnerProduct);
    EXPECT_NE(Cosine_UINT8_GetDistFunc(chunked, &alignment, nullptr), UINT8_Cosine);
}

class SQ8_FP32_SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(SQ8_FP32_SpacesOptimizationTest, SQ8_FP32_L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    // Create V1 fp32 query with precomputed sum and sum_squares
    // Query layout: [float values (dim)] [sum] [sum_squares]
    size_t query_size = dim + sq8::query_metadata_count<VecSimMetric_L2>();
    std::vector<float> v1_orig(query_size);
    test_utils::populate_sq8_fp32_query(v1_orig.data(), dim, false, 1234);

    // Create V2 as SQ8 quantized vector with different seed
    // Storage layout: [uint8_t values (dim)] [min_val] [delta] [sum] [sum_squares]
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, false, 456);

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(uint8_t) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(uint8_t) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_FP32_L2Sqr(v2_compressed.data(), v1_orig.data(), dim);
// Test different optimizations based on CPU features
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_L2_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        // ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_L2_implementation_AVX2_FMA(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "AVX with dim " << dim;
        // ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_L2_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "AVX with dim " << dim;
        // ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX with dim " << dim;
        // Unset avx flag as well, so we'll choose the next optimization (SSE).
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE4
    if (optimization.sse4_1) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_L2_implementation_SSE4(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "SSE with dim " << dim;
        // ASSERT_EQ(alignment, expected_alignment(128, dim)) << "SSE with dim " << dim;
        // Unset sse flag as well, so we'll choose the next optimization (default).
        optimization.sse4_1 = 0;
    }
#endif

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_L2_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_L2_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_L2_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "NEON with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.asimd = 0;
    }
#endif

    // Test default implementation
    unsigned char alignment = 0;
    arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_FP32_L2Sqr)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(SQ8_FP32_SpacesOptimizationTest, SQ8_FP32_InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    // Create original vectors with precomputed sum and sum_squares
    // Query layout: [float values (dim)] [sum] [sum_squares]
    size_t query_size = dim + sq8::query_metadata_count<VecSimMetric_L2>();
    std::vector<float> v1_orig(query_size);
    test_utils::populate_sq8_fp32_query(v1_orig.data(), dim, true, 1234);
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, true, 456);

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(uint8_t) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(uint8_t) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_FP32_InnerProduct(v2_compressed.data(), v1_orig.data(), dim);

// Test different optimizations based on CPU features
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_IP_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_IP_implementation_AVX2_FMA(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "AVX with dim " << dim;
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_IP_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "AVX with dim " << dim;
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE
    if (optimization.sse4_1) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_IP_implementation_SSE4(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "SSE with dim " << dim;
        optimization.sse4_1 = 0;
    }
#endif
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_IP_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_IP_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_IP_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
            << "NEON with dim " << dim;
        optimization.asimd = 0;
    }
#endif

    // Test default implementation
    unsigned char alignment = 0;
    arch_opt_func = IP_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_FP32_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_orig.data(), dim), 0.01)
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

// Instantiate the test suite with dimensions to test
INSTANTIATE_TEST_SUITE_P(SQ8_FP32_Test, SQ8_FP32_SpacesOptimizationTest,
                         testing::Range(8UL, 32 * 2UL + 1));

TEST_P(SQ8_FP32_SpacesOptimizationTest, SQ8_FP32_CosineTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    // Create original vectors - v1 needs extra space for precomputed sum and sum_squares
    // Query layout: [float values (dim)] [sum] [sum_squares]
    size_t query_size = dim + sq8::query_metadata_count<VecSimMetric_L2>();
    std::vector<float> v1_orig(query_size);
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_quantized(quantized_size);

    test_utils::populate_sq8_fp32_query(v1_orig.data(), dim, true, 1234);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_quantized.data(), dim, false, 456);

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(uint8_t) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(uint8_t) : 0;
    };

    dist_func_t<float> arch_opt_func;
    // Arguments: (SQ8_storage, FP32_query, dim)
    float baseline = SQ8_FP32_Cosine(v2_quantized.data(), v1_orig.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_Cosine_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_quantized.data(), v1_orig.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_Cosine_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_quantized.data(), v1_orig.data(), dim), 0.01)
            << "SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_Cosine_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_quantized.data(), v1_orig.data(), dim), 0.01)
            << "NEON with dim " << dim;
        optimization.asimd = 0;
    }
#endif

// Test different optimizations based on CPU features
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_Cosine_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_quantized.data(), v1_orig.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_Cosine_implementation_AVX2_FMA(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_quantized.data(), v1_orig.data(), dim), 0.01)
            << "AVX with dim " << dim;
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_Cosine_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_quantized.data(), v1_orig.data(), dim), 0.01)
            << "AVX with dim " << dim;
        optimization.avx2 = 0;
    }
#endif

#ifdef OPT_SSE
    if (optimization.sse4_1) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP32_Cosine_implementation_SSE4(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_quantized.data(), v1_orig.data(), dim), 0.01)
            << "SSE with dim " << dim;
        optimization.sse4_1 = 0;
    }
#endif

    // Test default implementation
    unsigned char alignment = 0;
    arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_FP32_Cosine)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v2_quantized.data(), v1_orig.data(), dim), 0.01)
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

// Test self-distance: distance to itself should be 0 for cosine (normalized vectors)
TEST(SQ8_FP32_EdgeCases, SelfDistanceCosine) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;
    // Query layout: [float values (dim)] [sum] [sum_squares]
    size_t query_size = (dim + sq8::query_metadata_count<VecSimMetric_L2>());
    std::vector<float> v_orig(query_size);
    test_utils::populate_sq8_fp32_query(v_orig.data(), dim, true, 1234);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v_quantized.data(), dim, true, 1234);

    // Arguments: (SQ8_storage, FP32_query, dim)
    float baseline = SQ8_FP32_Cosine(v_quantized.data(), v_orig.data(), dim);

    // Self-distance for cosine should be close to 0
    ASSERT_NEAR(baseline, 0.0f, 0.001f) << "Self-distance should be ~0 for cosine";

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE4
    if (optimization.sse4_1) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.sse4_1 = 0;
    }
#endif

    unsigned char alignment = 0;
    auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
    auto result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
    ASSERT_NEAR(baseline, result, 0.00001) << "No optimization self-distance should match baseline";
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

// Test self-distance: distance to itself should be 0 for L2
TEST(SQ8_FP32_EdgeCases, SelfDistanceL2) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;
    // Create fp32 query with precomputed sum and sum_squares
    // Query layout: [float values (dim)] [sum] [sum_squares]
    size_t query_size = (dim + sq8::query_metadata_count<VecSimMetric_L2>());
    std::vector<float> v_orig(query_size);
    test_utils::populate_sq8_fp32_query(v_orig.data(), dim, false, 1234);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v_quantized.data(), dim, false, 1234);

    float baseline = SQ8_FP32_L2Sqr(v_quantized.data(), v_orig.data(), dim);

    // Self-distance for L2 should be close to 0 (due to quantization effects, small errors are
    // expected)
    ASSERT_NEAR(baseline, 0.0f, 0.1f) << "Self-distance should be ~0 for L2";

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE4
    if (optimization.sse4_1) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.sse4_1 = 0;
    }
#endif

    unsigned char alignment = 0;
    auto arch_opt_func = L2_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
    auto result = arch_opt_func(v_quantized.data(), v_orig.data(), dim);
    ASSERT_NEAR(baseline, result, 0.00001) << "No optimization self-distance should match baseline";
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

// Test symmetry: dist(v1, v2) == dist(v2, v1)
// For asymmetric SQ8_FP32, symmetry means: dist(sq8_1, fp32_2) == dist(sq8_2, fp32_1)
TEST(SQ8_FP32_EdgeCases, CosineSymmetryTest) {
    size_t dim = 128;
    auto optimization = getCpuOptimizationFeatures();
    // Query layout: [float values (dim)] [sum] [sum_squares]
    size_t query_size = dim + sq8::query_metadata_count<VecSimMetric_L2>();
    std::vector<float> v1_fp32(query_size);
    test_utils::populate_sq8_fp32_query(v1_fp32.data(), dim, true, 1234);
    std::vector<float> v2_fp32(query_size);
    test_utils::populate_sq8_fp32_query(v2_fp32.data(), dim, true, 456);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v1_quantized.data(), dim, true, 1234);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_quantized.data(), dim, true, 456);
    // Arguments: (SQ8_storage, FP32_query, dim)
    float baseline_1 = SQ8_FP32_Cosine(v2_quantized.data(), v1_fp32.data(), dim);
    float baseline_2 = SQ8_FP32_Cosine(v1_quantized.data(), v2_fp32.data(), dim);
    ASSERT_NEAR(baseline_1, baseline_2, 0.001f) << "Cosine should be symmetric";

    unsigned char alignment = 0;

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float cos_12 = arch_opt_func(v2_quantized.data(), v1_fp32.data(), dim);
        float cos_21 = arch_opt_func(v1_quantized.data(), v2_fp32.data(), dim);
        ASSERT_NEAR(cos_12, cos_21, 0.001f) << "Optimized cosine should be symmetric";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float cos_12 = arch_opt_func(v2_quantized.data(), v1_fp32.data(), dim);
        float cos_21 = arch_opt_func(v1_quantized.data(), v2_fp32.data(), dim);
        ASSERT_NEAR(cos_12, cos_21, 0.001f) << "Optimized cosine should be symmetric";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float cos_12 = arch_opt_func(v2_quantized.data(), v1_fp32.data(), dim);
        float cos_21 = arch_opt_func(v1_quantized.data(), v2_fp32.data(), dim);
        ASSERT_NEAR(cos_12, cos_21, 0.001f) << "Optimized cosine should be symmetric";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float cos_12 = arch_opt_func(v2_quantized.data(), v1_fp32.data(), dim);
        float cos_21 = arch_opt_func(v1_quantized.data(), v2_fp32.data(), dim);
        ASSERT_NEAR(cos_12, cos_21, 0.001f) << "Optimized cosine should be symmetric";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float cos_12 = arch_opt_func(v2_quantized.data(), v1_fp32.data(), dim);
        float cos_21 = arch_opt_func(v1_quantized.data(), v2_fp32.data(), dim);
        ASSERT_NEAR(cos_12, cos_21, 0.001f) << "Optimized cosine should be symmetric";
        optimization.avx512f = 0;
    }
#endif
    auto cosine_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, nullptr);
    float cos_12 = cosine_func(v2_quantized.data(), v1_fp32.data(), dim);
    float cos_21 = cosine_func(v1_quantized.data(), v2_fp32.data(), dim);
    ASSERT_NEAR(cos_12, cos_21, 0.001f) << "Cosine should be symmetric";
}

// Test with zero vector
TEST(SQ8_FP32_EdgeCases, CosineZeroVectorTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;
    size_t query_size = dim + sq8::query_metadata_count<VecSimMetric_L2>();
    std::vector<float> v_zero(query_size, 0.0f);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_nonzero_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v_nonzero_quantized.data(), dim, true);

    // Arguments: (SQ8_storage, FP32_query, dim)
    float baseline = SQ8_FP32_Cosine(v_nonzero_quantized.data(), v_zero.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_nonzero_quantized.data(), v_zero.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector IP should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_nonzero_quantized.data(), v_zero.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector IP should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_nonzero_quantized.data(), v_zero.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector IP should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_nonzero_quantized.data(), v_zero.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector IP should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_nonzero_quantized.data(), v_zero.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector IP should match baseline";
        optimization.avx512f = 0;
    }
#endif
    unsigned char alignment = 0;
    auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, nullptr);
    float result = arch_opt_func(v_nonzero_quantized.data(), v_zero.data(), dim);

    ASSERT_EQ(result, baseline) << "Zero vector Cosine should match baseline";
}

// Test with constant quantized vector (all same values - edge case where delta = 0)
TEST(SQ8_FP32_EdgeCases, CosineConstantVectorTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;

    // Create a random query vector (preprocessed for FP32->SQ8 cosine)
    // Query layout: [float values (dim)] [sum] [sum_squares]
    size_t query_size = dim + sq8::query_metadata_count<VecSimMetric_L2>();
    std::vector<float> v_query(query_size);
    test_utils::populate_float_vec(v_query.data(), dim);
    test_utils::preprocess_sq8_fp32_query(v_query.data(), dim);

    // Create a constant quantized vector (all same values)
    // This tests the edge case where delta = 0 (or set to 1.0 to avoid division by zero)
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_const_quantized(quantized_size);
    std::vector<float> v_const(dim, 0.5f);
    spaces::GetNormalizeFunc<float>()(v_const.data(), dim);
    test_utils::quantize_float_vec_to_sq8_with_metadata(v_const.data(), dim,
                                                        v_const_quantized.data());

    // Arguments: (SQ8_storage, FP32_query, dim)
    float baseline = SQ8_FP32_Cosine(v_const_quantized.data(), v_query.data(), dim);
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_query.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector Cosine should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_query.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector Cosine should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_query.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector Cosine should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_query.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector Cosine should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_query.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector Cosine should match baseline";
        optimization.avx512f = 0;
    }
#endif
    unsigned char alignment = 0;
    auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, nullptr);
    float result = arch_opt_func(v_const_quantized.data(), v_query.data(), dim);

    ASSERT_NEAR(result, baseline, 0.01f)
        << "Constant quantized vector Cosine should match baseline";
}

// Test with extreme values (-1 and 1 only)
TEST(SQ8_FP32_EdgeCases, CosineExtremeValuesTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;
    // Query layout: [float values (dim)] [sum] [sum_squares]
    size_t query_size = dim + sq8::query_metadata_count<VecSimMetric_L2>();
    std::vector<float> v1(query_size), v2(dim);

    // Alternating extreme values
    for (size_t i = 0; i < dim; i++) {
        v1[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        v2[i] = (i % 3 == 0) ? 1.0f : -1.0f;
    }
    test_utils::preprocess_sq8_fp32_query(v1.data(), dim);
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::quantize_float_vec_to_sq8_with_metadata(v2.data(), dim, v2_quantized.data());

    // Arguments: (SQ8_storage, FP32_query, dim)
    float baseline = SQ8_FP32_Cosine(v2_quantized.data(), v1.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v2_quantized.data(), v1.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized extreme values Cosine should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v2_quantized.data(), v1.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized extreme values Cosine should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v2_quantized.data(), v1.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized extreme values Cosine should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v2_quantized.data(), v1.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized extreme values Cosine should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v2_quantized.data(), v1.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized extreme values Cosine should match baseline";
        optimization.avx512f = 0;
    }
#endif
    unsigned char alignment = 0;
    auto arch_opt_func = Cosine_SQ8_FP32_GetDistFunc(dim, &alignment, nullptr);
    float result = arch_opt_func(v2_quantized.data(), v1.data(), dim);

    ASSERT_NEAR(result, baseline, 0.01f) << "Extreme values Cosine should match baseline";
}

/* ======================== Tests SQ8_FP16 (parameterized) ========================= */

// Parameterized tests that verify the scalar SQ8_FP16 kernels against the not-optimized
// baseline across multiple dimensions, including odd dimensions and SIMD-boundary residues.
// The SIMD-tier dispatcher coverage lives in SQ8_FP16_SpacesOptimizationTest below; this
// suite intentionally exercises the scalar reference directly to keep it as a fixed baseline
// the SIMD tiers are compared against.
class SQ8_FP16_NoOptimizationSpacesTest : public testing::TestWithParam<size_t> {};

TEST_P(SQ8_FP16_NoOptimizationSpacesTest, SQ8_FP16_L2SqrTest) {
    size_t dim = GetParam();

    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v1_query(query_count);
    test_utils::populate_sq8_fp16_query(v1_query.data(), dim, false, 1234);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, false, 5678);

    float baseline =
        test_utils::SQ8_FP16_NotOptimized_L2Sqr(v2_compressed.data(), v1_query.data(), dim);
    float dist = SQ8_FP16_L2Sqr(v2_compressed.data(), v1_query.data(), dim);

    ASSERT_NEAR(dist, baseline, 0.01f) << "SQ8_FP16_L2Sqr mismatch for dim " << dim;
}

TEST_P(SQ8_FP16_NoOptimizationSpacesTest, SQ8_FP16_InnerProductTest) {
    size_t dim = GetParam();

    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v1_query(query_count);
    test_utils::populate_sq8_fp16_query(v1_query.data(), dim, true, 1234);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, true, 5678);

    float baseline =
        test_utils::SQ8_FP16_NotOptimized_InnerProduct(v2_compressed.data(), v1_query.data(), dim);
    float dist = SQ8_FP16_InnerProduct(v2_compressed.data(), v1_query.data(), dim);

    ASSERT_NEAR(dist, baseline, 0.01f) << "SQ8_FP16_InnerProduct mismatch for dim " << dim;
}

TEST_P(SQ8_FP16_NoOptimizationSpacesTest, SQ8_FP16_CosineTest) {
    size_t dim = GetParam();

    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v1_query(query_count);
    test_utils::populate_sq8_fp16_query(v1_query.data(), dim, true, 1234);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, true, 5678);

    float baseline =
        test_utils::SQ8_FP16_NotOptimized_Cosine(v2_compressed.data(), v1_query.data(), dim);
    float dist = SQ8_FP16_Cosine(v2_compressed.data(), v1_query.data(), dim);

    ASSERT_NEAR(dist, baseline, 0.01f) << "SQ8_FP16_Cosine mismatch for dim " << dim;
}

// Cover small dims, odd dims, SIMD-boundary residues for upcoming AVX2 / AVX512 / SVE / NEON
// register widths (8/16/32/64 elements per register for SQ8 storage).
INSTANTIATE_TEST_SUITE_P(SQ8_FP16_NoOpt, SQ8_FP16_NoOptimizationSpacesTest,
                         testing::Values(1, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64,
                                         65, 127, 128));

/* ======================== SQ8_FP16 SIMD optimisation tests ========================= */

// Walks down the x86 ISA tiers (AVX-512 → AVX2+FMA → AVX2 → SSE4 → scalar) and asserts
// that {IP,Cosine,L2}_SQ8_FP16_GetDistFunc returns the expected Choose_* symbol and that
// its output matches the scalar baseline within 0.01.
class SQ8_FP16_SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(SQ8_FP16_SpacesOptimizationTest, SQ8_FP16_L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v1_query(query_count);
    test_utils::populate_sq8_fp16_query(v1_query.data(), dim, false, 1234);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, false, 5678);

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_FP16_L2Sqr(v2_compressed.data(), v1_query.data(), dim);

#ifdef OPT_AVX512F
    if (optimization.avx512f) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif
    // F16C is required by every non-AVX-512 SQ8↔FP16 tier (vcvtph2ps), so the guard is hoisted
    // around all three — matches the dispatcher layout in L2_space.cpp.
#ifdef OPT_F16C
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3 && optimization.f16c) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_AVX2_FMA(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "AVX2+FMA with dim " << dim;
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2 && optimization.f16c) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "AVX2 with dim " << dim;
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE4
    if (optimization.sse4_1 && optimization.f16c && optimization.avx) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_SSE4(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SSE4 with dim " << dim;
        optimization.sse4_1 = 0;
    }
#endif
#endif // OPT_F16C

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_HP
    if (optimization.asimdfhm) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_NEON_FHM(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "NEON_FHM with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment NEON_FHM with dim " << dim;
        optimization.asimdfhm = 0;
    }
    if (optimization.asimdhp) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_NEON_HP(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "NEON_HP with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment NEON_HP with dim " << dim;
        optimization.asimdhp = 0;
    }
#endif
#endif // CPU_FEATURES_ARCH_AARCH64

    unsigned char alignment = 0;
    arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_FP16_L2Sqr)
        << "Unexpected scalar fallback function for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
        << "Scalar fallback with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(SQ8_FP16_SpacesOptimizationTest, SQ8_FP16_InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v1_query(query_count);
    test_utils::populate_sq8_fp16_query(v1_query.data(), dim, true, 1234);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, true, 5678);

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_FP16_InnerProduct(v2_compressed.data(), v1_query.data(), dim);

#ifdef OPT_AVX512F
    if (optimization.avx512f) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_IP_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif
    // F16C is required by every non-AVX-512 SQ8↔FP16 tier (vcvtph2ps), so the guard is hoisted
    // around all three — matches the dispatcher layout in IP_space.cpp.
#ifdef OPT_F16C
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3 && optimization.f16c) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_IP_implementation_AVX2_FMA(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "AVX2+FMA with dim " << dim;
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2 && optimization.f16c) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_IP_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "AVX2 with dim " << dim;
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE4
    if (optimization.sse4_1 && optimization.f16c && optimization.avx) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_IP_implementation_SSE4(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SSE4 with dim " << dim;
        optimization.sse4_1 = 0;
    }
#endif
#endif // OPT_F16C

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_IP_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_IP_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_HP
    if (optimization.asimdfhm) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_IP_implementation_NEON_FHM(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "NEON_FHM with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment NEON_FHM with dim " << dim;
        optimization.asimdfhm = 0;
    }
    if (optimization.asimdhp) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_IP_implementation_NEON_HP(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "NEON_HP with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment NEON_HP with dim " << dim;
        optimization.asimdhp = 0;
    }
#endif
#endif // CPU_FEATURES_ARCH_AARCH64

    unsigned char alignment = 0;
    arch_opt_func = IP_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_FP16_InnerProduct)
        << "Unexpected scalar fallback function for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
        << "Scalar fallback with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(SQ8_FP16_SpacesOptimizationTest, SQ8_FP16_CosineTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v1_query(query_count);
    test_utils::populate_sq8_fp16_query(v1_query.data(), dim, true, 1234);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v2_compressed(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_compressed.data(), dim, true, 5678);

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_FP16_Cosine(v2_compressed.data(), v1_query.data(), dim);

#ifdef OPT_AVX512F
    if (optimization.avx512f) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_Cosine_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif
    // F16C is required by every non-AVX-512 SQ8↔FP16 tier (vcvtph2ps), so the guard is hoisted
    // around all three — matches the dispatcher layout in IP_space.cpp.
#ifdef OPT_F16C
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3 && optimization.f16c) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_Cosine_implementation_AVX2_FMA(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "AVX2+FMA with dim " << dim;
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2 && optimization.f16c) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_Cosine_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "AVX2 with dim " << dim;
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE4
    if (optimization.sse4_1 && optimization.f16c && optimization.avx) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_Cosine_implementation_SSE4(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SSE4 with dim " << dim;
        optimization.sse4_1 = 0;
    }
#endif
#endif // OPT_F16C

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_Cosine_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_Cosine_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_HP
    if (optimization.asimdfhm) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_Cosine_implementation_NEON_FHM(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "NEON_FHM with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment NEON_FHM with dim " << dim;
        optimization.asimdfhm = 0;
    }
    if (optimization.asimdhp) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_Cosine_implementation_NEON_HP(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "NEON_HP with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No alignment NEON_HP with dim " << dim;
        optimization.asimdhp = 0;
    }
#endif
#endif // CPU_FEATURES_ARCH_AARCH64

    unsigned char alignment = 0;
    arch_opt_func = Cosine_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_FP16_Cosine)
        << "Unexpected scalar fallback function for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
        << "Scalar fallback with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

// Dim range [16, 32] covers every residual class for the 16-element chunk used by every tier.
INSTANTIATE_TEST_SUITE_P(SQ8_FP16_SIMD, SQ8_FP16_SpacesOptimizationTest,
                         testing::Range(16UL, 16 * 2UL + 1));

// Higher dimensions surface multi-iteration loop bugs (pointer stride, do-while termination
// off-by-one) that the [16, 32] range does not exercise because the AVX-512 inner loop runs at
// most twice in that range. 48 and 112 specifically hit the AVX-512 three-chunk tail
// (remaining == 48, i.e. (dim / 16) % 4 == 3): 48 with zero main-loop iterations, 112 with one.
INSTANTIATE_TEST_SUITE_P(SQ8_FP16_SIMD_HighDim, SQ8_FP16_SpacesOptimizationTest,
                         testing::Values(48UL, 64UL, 112UL, 128UL, 256UL, 512UL, 1024UL));

/* ======================== Tests SQ8_FP16 (edge cases) ========================= */

// Zero FP16 query against a non-zero SQ8 storage. IP must be exactly 1.0 (1 - 0),
// L2² must equal Σ dequantized². Math correctness on adversarial inputs is verified
// against the scalar reference; SIMD tier coverage with branchless kernels is provided
// separately by SQ8_FP16_SpacesOptimizationTest.
TEST(SQ8_FP16_EdgeCases, ZeroQueryTest) {
    size_t dim = 64;

    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v_zero_query(query_count, float16{0});
    // Metadata bits are zero (sum = 0, sum_squares = 0); FP16 zero is bit-pattern 0.

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_nonzero_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v_nonzero_quantized.data(), dim, false,
                                                        1234);

    float ip_baseline = test_utils::SQ8_FP16_NotOptimized_InnerProduct(v_nonzero_quantized.data(),
                                                                       v_zero_query.data(), dim);
    float ip = SQ8_FP16_InnerProduct(v_nonzero_quantized.data(), v_zero_query.data(), dim);
    ASSERT_NEAR(ip, ip_baseline, 0.01f) << "Zero-query SQ8_FP16_InnerProduct mismatch";
    ASSERT_NEAR(ip, 1.0f, 0.01f) << "Zero-query IP must equal 1.0 (1 - 0)";

    float l2_baseline = test_utils::SQ8_FP16_NotOptimized_L2Sqr(v_nonzero_quantized.data(),
                                                                v_zero_query.data(), dim);
    float l2 = SQ8_FP16_L2Sqr(v_nonzero_quantized.data(), v_zero_query.data(), dim);
    ASSERT_NEAR(l2, l2_baseline, 0.01f) << "Zero-query SQ8_FP16_L2Sqr mismatch";
}

// Constant SQ8 storage (all values identical => delta = 0). Storage quantizer sets delta to 1.0
// to avoid div-by-zero, so verify the kernels still match the dequantization baseline.
TEST(SQ8_FP16_EdgeCases, ConstantStorageTest) {
    size_t dim = 64;

    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v_query(query_count);
    test_utils::populate_sq8_fp16_query(v_query.data(), dim, false, 4321);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_const_quantized(quantized_size);
    std::vector<float> v_const(dim, 0.5f);
    test_utils::quantize_float_vec_to_sq8_with_metadata(v_const.data(), dim,
                                                        v_const_quantized.data());

    float ip_baseline = test_utils::SQ8_FP16_NotOptimized_InnerProduct(v_const_quantized.data(),
                                                                       v_query.data(), dim);
    float ip = SQ8_FP16_InnerProduct(v_const_quantized.data(), v_query.data(), dim);
    ASSERT_NEAR(ip, ip_baseline, 0.01f) << "Constant-storage SQ8_FP16_InnerProduct mismatch";

    float l2_baseline =
        test_utils::SQ8_FP16_NotOptimized_L2Sqr(v_const_quantized.data(), v_query.data(), dim);
    float l2 = SQ8_FP16_L2Sqr(v_const_quantized.data(), v_query.data(), dim);
    ASSERT_NEAR(l2, l2_baseline, 0.01f) << "Constant-storage SQ8_FP16_L2Sqr mismatch";
}

// Mixed-sign FP16 query (alternating positive/negative values) verifies sign handling
// in the FP16->FP32 widening path and in the algebraic identity used by the kernels.
TEST(SQ8_FP16_EdgeCases, MixedSignQueryTest) {
    size_t dim = 64;

    // Build an alternating +0.75 / -0.75 FP16 query manually so we don't depend on RNG sign mix.
    // Allocated as std::vector<float16> so v_query.data() is alignof(float16)-aligned for
    // the SQ8_FP16 production kernel.
    size_t query_count =
        dim + sq8::query_metadata_count<VecSimMetric_L2>() * (sizeof(float) / sizeof(float16));
    std::vector<float16> v_query(query_count);
    for (size_t i = 0; i < dim; i++) {
        v_query[i] = vecsim_types::FP32_to_FP16((i % 2 == 0) ? 0.75f : -0.75f);
    }
    test_utils::preprocess_sq8_fp16_query(v_query.data(), dim);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v_quantized.data(), dim, false, 9876);

    float ip_baseline =
        test_utils::SQ8_FP16_NotOptimized_InnerProduct(v_quantized.data(), v_query.data(), dim);
    float ip = SQ8_FP16_InnerProduct(v_quantized.data(), v_query.data(), dim);
    ASSERT_NEAR(ip, ip_baseline, 0.01f) << "Mixed-sign SQ8_FP16_InnerProduct mismatch";

    float cos_baseline =
        test_utils::SQ8_FP16_NotOptimized_Cosine(v_quantized.data(), v_query.data(), dim);
    float cos = SQ8_FP16_Cosine(v_quantized.data(), v_query.data(), dim);
    ASSERT_NEAR(cos, cos_baseline, 0.01f) << "Mixed-sign SQ8_FP16_Cosine mismatch";

    float l2_baseline =
        test_utils::SQ8_FP16_NotOptimized_L2Sqr(v_quantized.data(), v_query.data(), dim);
    float l2 = SQ8_FP16_L2Sqr(v_quantized.data(), v_query.data(), dim);
    ASSERT_NEAR(l2, l2_baseline, 0.01f) << "Mixed-sign SQ8_FP16_L2Sqr mismatch";
}

/* ======================== Tests SQ8_SQ8 ========================= */

TEST_F(SpacesTest, SQ8_SQ8_ip_no_optimization_func_test) {
    size_t dim = 5;

    // Create SQ8 quantized versions of both vectors
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v1_quantized.data(), dim, true, 1234);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_quantized.data(), dim, true, 5678);

    float baseline = test_utils::SQ8_SQ8_NotOptimized_InnerProduct(v1_quantized.data(),
                                                                   v2_quantized.data(), dim);

    unsigned char alignment = 0;
#ifdef CPU_FEATURES_ARCH_AARCH64
    // Make sure we don't use any optimization (because there is no size optimization for arm)
    auto optimization = getCpuOptimizationFeatures();
    optimization.sve = optimization.sve2 = 0;
    auto arch_opt_func = IP_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
#else
    auto arch_opt_func = IP_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
#endif
    ASSERT_EQ(arch_opt_func, SQ8_SQ8_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    // Checks that the function with the optimized math equivalence returns similar result.
    // Use ASSERT_NEAR due to floating-point differences between naive and algebraic formulas.
    ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.001)
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_F(SpacesTest, SQ8_SQ8_Cosine_no_optimization_func_test) {
    size_t dim = 5;

    // Create SQ8 quantized versions of both vectors
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v1_quantized.data(), dim, true, 1234);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_quantized.data(), dim, true, 5678);

    float baseline =
        test_utils::SQ8_SQ8_NotOptimized_Cosine(v1_quantized.data(), v2_quantized.data(), dim);

    unsigned char alignment = 0;
#ifdef CPU_FEATURES_ARCH_AARCH64
    // Make sure we don't use any optimization (because there is no size optimization for arm)
    auto optimization = getCpuOptimizationFeatures();
    optimization.sve = optimization.sve2 = 0;
    auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
#else
    auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
#endif
    ASSERT_EQ(arch_opt_func, SQ8_SQ8_Cosine)
        << "Unexpected distance function chosen for dim " << dim;
    // Checks that the function with the optimized math equivalence returns the same result.
    // min1*sum2 + min2*sum1 + delta1*delta2*Σ(q1[i]*q2[i]) - dim*min1*min2
    ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.001)
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_F(SpacesTest, SQ8_SQ8_L2_no_optimization_func_test) {
    size_t dim = 5;

    // Create SQ8 quantized versions of both vectors
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v1_quantized.data(), dim, false, 1234);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_quantized.data(), dim, false, 5678);

    float baseline =
        test_utils::SQ8_SQ8_NotOptimized_L2Sqr(v1_quantized.data(), v2_quantized.data(), dim);
    unsigned char alignment = 0;
#ifdef CPU_FEATURES_ARCH_AARCH64
    // Make sure we don't use any optimization (because there is no size optimization for arm)
    auto optimization = getCpuOptimizationFeatures();
    optimization.sve = optimization.sve2 = 0;
    auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
#else
    // Get distance function with nullptr alignment to cover that code path
    auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
#endif
    ASSERT_EQ(arch_opt_func, SQ8_SQ8_L2Sqr)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.001f)
        << "SQ8_SQ8_L2Sqr failed to match expected distance";
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

class SQ8_SQ8_SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(SQ8_SQ8_SpacesOptimizationTest, SQ8_SQ8_InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    // Create SQ8 quantized versions of both vectors
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v1_quantized.data(), dim, true, 1234);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_quantized.data(), dim, true, 5678);

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_SQ8_InnerProduct(v1_quantized.data(), v2_quantized.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_IP_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_IP_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.01)
            << "SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp && dim >= 64) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_IP_implementation_NEON_DOTPROD(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.01)
            << "NEON_DOTPROD with dim " << dim;
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_IP_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.01)
            << "NEON with dim " << dim;
        optimization.asimd = 0;
    }
#endif

#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_IP_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif

    // Test default implementation
    unsigned char alignment = 0;
    arch_opt_func = IP_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_SQ8_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim))
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(SQ8_SQ8_SpacesOptimizationTest, SQ8_SQ8_CosineTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    // Create quantized vectors
    // Size: dim (uint8_t) + min_val (float) + delta (float) + sum (float) + sum_squares (float)
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v1_quantized.data(), dim, true, 1234);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_quantized.data(), dim, true, 5678);

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_SQ8_Cosine(v1_quantized.data(), v2_quantized.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_Cosine_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_Cosine_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.01)
            << "SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp && dim >= 64) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_Cosine_implementation_NEON_DOTPROD(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.01)
            << "NEON_DOTPROD with dim " << dim;
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_Cosine_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.01)
            << "NEON with dim " << dim;
        optimization.asimd = 0;
    }
#endif

#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_Cosine_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif

    // Test default implementation
    unsigned char alignment = 0;
    arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_SQ8_Cosine)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim))
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(SQ8_SQ8_SpacesOptimizationTest, SQ8_SQ8_L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    // Create SQ8 quantized versions of both vectors
    // Layout: [uint8_t values (dim)] [min_val] [delta] [sum] [sum_of_squares]
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v1_quantized.data(), dim, false, 1234);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_quantized.data(), dim, false, 5678);

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_SQ8_L2Sqr(v1_quantized.data(), v2_quantized.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_L2_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.02)
            << "SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_L2_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.02)
            << "SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp && dim >= 64) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_L2_implementation_NEON_DOTPROD(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.02)
            << "NEON_DOTPROD with dim " << dim;
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_L2_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.02)
            << "NEON with dim " << dim;
        optimization.asimd = 0;
    }
#endif

#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_SQ8_L2_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.02)
            << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif

    // Test default implementation
    unsigned char alignment = 0;
    arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_SQ8_L2Sqr)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.02)
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

// Note: This suite intentionally uses a larger dimension range (64–128) than SQ8OptFuncs.
// It is designed to exercise SQ8–SQ8 cosine implementations, including SIMD paths
// that are only enabled or meaningfully stressed for dimensions >= 64.
INSTANTIATE_TEST_SUITE_P(SQ8_SQ8OptFuncs, SQ8_SQ8_SpacesOptimizationTest,
                         testing::Range(64UL, 64 * 2UL + 1));

// Test self-distance: distance to itself should be 0 for cosine (normalized vectors)
TEST(SQ8_SQ8_EdgeCases, SelfDistanceCosine) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v_quantized.data(), dim, true);

    float baseline = SQ8_SQ8_Cosine(v_quantized.data(), v_quantized.data(), dim);

    // Self-distance for cosine should be close to 0
    ASSERT_NEAR(baseline, 0.0f, 0.001f) << "Self-distance should be ~0 for cosine";

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.avx512f = 0;
    }
#endif

    unsigned char alignment = 0;
    auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(baseline, arch_opt_func(v_quantized.data(), v_quantized.data(), dim))
        << "No optimization self-distance should match baseline";
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

// Test symmetry: dist(v1, v2) == dist(v2, v1)
TEST(SQ8_SQ8_EdgeCases, CosineSymmetryTest) {
    size_t dim = 128;
    auto optimization = getCpuOptimizationFeatures();
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v1_quantized.data(), dim, true, 456, -1.0f,
                                                        1.0f);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_quantized.data(), dim, true, 123, -1.0f,
                                                        1.0f);

    unsigned char alignment = 0;

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float cos_12 = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        float cos_21 = arch_opt_func(v2_quantized.data(), v1_quantized.data(), dim);
        ASSERT_EQ(cos_12, cos_21) << "Optimized cosine should be symmetric";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float cos_12 = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        float cos_21 = arch_opt_func(v2_quantized.data(), v1_quantized.data(), dim);
        ASSERT_EQ(cos_12, cos_21) << "Optimized cosine should be symmetric";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float cos_12 = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        float cos_21 = arch_opt_func(v2_quantized.data(), v1_quantized.data(), dim);
        ASSERT_EQ(cos_12, cos_21) << "Optimized cosine should be symmetric";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float cos_12 = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        float cos_21 = arch_opt_func(v2_quantized.data(), v1_quantized.data(), dim);
        ASSERT_EQ(cos_12, cos_21) << "Optimized cosine should be symmetric";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float cos_12 = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        float cos_21 = arch_opt_func(v2_quantized.data(), v1_quantized.data(), dim);
        ASSERT_EQ(cos_12, cos_21) << "Optimized cosine should be symmetric";
        optimization.avx512f = 0;
    }
#endif
    auto cosine_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
    float cos_12 = cosine_func(v1_quantized.data(), v2_quantized.data(), dim);
    float cos_21 = cosine_func(v2_quantized.data(), v1_quantized.data(), dim);
    ASSERT_EQ(cos_12, cos_21) << "Cosine should be symmetric";
}

// Test with zero vector
TEST(SQ8_SQ8_EdgeCases, CosineZeroVectorTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;
    std::vector<float> v_zero(dim, 0.0f);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_zero_quantized(quantized_size);
    std::vector<uint8_t> v_nonzero_quantized(quantized_size);
    test_utils::quantize_float_vec_to_sq8_with_metadata(v_zero.data(), dim,
                                                        v_zero_quantized.data());
    test_utils::populate_float_vec_to_sq8_with_metadata(v_nonzero_quantized.data(), dim, true);

    float baseline = SQ8_SQ8_Cosine(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector IP should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector IP should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector IP should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector IP should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector IP should match baseline";
        optimization.avx512f = 0;
    }
#endif
    unsigned char alignment = 0;
    auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
    float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);

    ASSERT_NEAR(result, baseline, 0.01f) << "Zero vector Cosine should match baseline";
}

// Test with constant vector (all same values)
TEST(SQ8_SQ8_EdgeCases, CosineConstantVectorTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;
    std::vector<float> v_const(dim, 0.5f);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_const_quantized(quantized_size);
    std::vector<uint8_t> v_random_quantized(quantized_size);
    spaces::GetNormalizeFunc<float>()(v_const.data(), dim);
    test_utils::quantize_float_vec_to_sq8_with_metadata(v_const.data(), dim,
                                                        v_const_quantized.data());
    test_utils::populate_float_vec_to_sq8_with_metadata(v_random_quantized.data(), dim, true);

    float baseline = SQ8_SQ8_Cosine(v_const_quantized.data(), v_random_quantized.data(), dim);
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector Cosine should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector Cosine should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector Cosine should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector Cosine should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector Cosine should match baseline";
        optimization.avx512f = 0;
    }
#endif
    unsigned char alignment = 0;
    auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
    float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);

    ASSERT_NEAR(result, baseline, 0.01f) << "Constant vector Cosine should match baseline";
}

// Test with extreme values (-1 and 1 only)
TEST(SQ8_SQ8_EdgeCases, CosineExtremeValuesTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;
    std::vector<float> v1(dim), v2(dim);

    // Alternating extreme values
    for (size_t i = 0; i < dim; i++) {
        v1[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        v2[i] = (i % 3 == 0) ? 1.0f : -1.0f;
    }

    spaces::GetNormalizeFunc<float>()(v1.data(), dim);
    spaces::GetNormalizeFunc<float>()(v2.data(), dim);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::quantize_float_vec_to_sq8_with_metadata(v1.data(), dim, v1_quantized.data());
    test_utils::quantize_float_vec_to_sq8_with_metadata(v2.data(), dim, v2_quantized.data());

    float baseline = SQ8_SQ8_Cosine(v1_quantized.data(), v2_quantized.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized extreme values Cosine should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized extreme values Cosine should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized extreme values Cosine should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized extreme values Cosine should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized extreme values Cosine should match baseline";
        optimization.avx512f = 0;
    }
#endif
    unsigned char alignment = 0;
    auto arch_opt_func = Cosine_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
    float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);

    ASSERT_NEAR(result, baseline, 0.01f) << "Extreme values Cosine should match baseline";
}

// Test self-distance: distance to itself should be 0 for L2
TEST(SQ8_SQ8_EdgeCases, SelfDistanceL2) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v_quantized.data(), dim, false);

    float baseline = SQ8_SQ8_L2Sqr(v_quantized.data(), v_quantized.data(), dim);

    // Self-distance for L2 should be close to 0 (due to quantization effects, small errors are
    // expected)
    ASSERT_NEAR(baseline, 0.0f, 0.1f) << "Self-distance should be ~0 for L2";

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_quantized.data(), v_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized self-distance should match baseline";
        optimization.avx512f = 0;
    }
#endif

    unsigned char alignment = 0;
    auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(baseline, arch_opt_func(v_quantized.data(), v_quantized.data(), dim))
        << "No optimization self-distance should match baseline";
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

// Test symmetry: dist(v1, v2) == dist(v2, v1) for L2
TEST(SQ8_SQ8_EdgeCases, L2SymmetryTest) {
    size_t dim = 128;
    auto optimization = getCpuOptimizationFeatures();
    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::populate_float_vec_to_sq8_with_metadata(v1_quantized.data(), dim, false, 456, -1.0f,
                                                        1.0f);
    test_utils::populate_float_vec_to_sq8_with_metadata(v2_quantized.data(), dim, false, 123, -1.0f,
                                                        1.0f);

    unsigned char alignment = 0;

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float l2_12 = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        float l2_21 = arch_opt_func(v2_quantized.data(), v1_quantized.data(), dim);
        ASSERT_EQ(l2_12, l2_21) << "Optimized L2 should be symmetric";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float l2_12 = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        float l2_21 = arch_opt_func(v2_quantized.data(), v1_quantized.data(), dim);
        ASSERT_EQ(l2_12, l2_21) << "Optimized L2 should be symmetric";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float l2_12 = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        float l2_21 = arch_opt_func(v2_quantized.data(), v1_quantized.data(), dim);
        ASSERT_EQ(l2_12, l2_21) << "Optimized L2 should be symmetric";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float l2_12 = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        float l2_21 = arch_opt_func(v2_quantized.data(), v1_quantized.data(), dim);
        ASSERT_EQ(l2_12, l2_21) << "Optimized L2 should be symmetric";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float l2_12 = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        float l2_21 = arch_opt_func(v2_quantized.data(), v1_quantized.data(), dim);
        ASSERT_EQ(l2_12, l2_21) << "Optimized L2 should be symmetric";
        optimization.avx512f = 0;
    }
#endif
    auto l2_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
    float l2_12 = l2_func(v1_quantized.data(), v2_quantized.data(), dim);
    float l2_21 = l2_func(v2_quantized.data(), v1_quantized.data(), dim);
    ASSERT_EQ(l2_12, l2_21) << "L2 should be symmetric";
}

// Test with zero vector for L2
TEST(SQ8_SQ8_EdgeCases, L2ZeroVectorTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;
    std::vector<float> v_zero(dim, 0.0f);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_zero_quantized(quantized_size);
    std::vector<uint8_t> v_nonzero_quantized(quantized_size);
    test_utils::quantize_float_vec_to_sq8_with_metadata(v_zero.data(), dim,
                                                        v_zero_quantized.data());
    test_utils::populate_float_vec_to_sq8_with_metadata(v_nonzero_quantized.data(), dim, false);

    float baseline = SQ8_SQ8_L2Sqr(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector L2 should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector L2 should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector L2 should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector L2 should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized zero vector L2 should match baseline";
        optimization.avx512f = 0;
    }
#endif
    unsigned char alignment = 0;
    auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
    float result = arch_opt_func(v_zero_quantized.data(), v_nonzero_quantized.data(), dim);

    ASSERT_NEAR(result, baseline, 0.01f) << "Zero vector L2 should match baseline";
}

// Test with constant vector (all same values) for L2
TEST(SQ8_SQ8_EdgeCases, L2ConstantVectorTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;
    std::vector<float> v_const(dim, 0.5f);

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v_const_quantized(quantized_size);
    std::vector<uint8_t> v_random_quantized(quantized_size);
    test_utils::quantize_float_vec_to_sq8_with_metadata(v_const.data(), dim,
                                                        v_const_quantized.data());
    test_utils::populate_float_vec_to_sq8_with_metadata(v_random_quantized.data(), dim, false);

    float baseline = SQ8_SQ8_L2Sqr(v_const_quantized.data(), v_random_quantized.data(), dim);
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector L2 should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector L2 should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector L2 should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector L2 should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f)
            << "Optimized constant vector L2 should match baseline";
        optimization.avx512f = 0;
    }
#endif
    unsigned char alignment = 0;
    auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
    float result = arch_opt_func(v_const_quantized.data(), v_random_quantized.data(), dim);

    ASSERT_NEAR(result, baseline, 0.01f) << "Constant vector L2 should match baseline";
}

// Test with extreme values (-1 and 1 only) for L2
TEST(SQ8_SQ8_EdgeCases, L2ExtremeValuesTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = 128;
    std::vector<float> v1(dim), v2(dim);

    // Alternating extreme values
    for (size_t i = 0; i < dim; i++) {
        v1[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        v2[i] = (i % 3 == 0) ? 1.0f : -1.0f;
    }

    size_t quantized_size =
        dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
    std::vector<uint8_t> v1_quantized(quantized_size);
    std::vector<uint8_t> v2_quantized(quantized_size);
    test_utils::quantize_float_vec_to_sq8_with_metadata(v1.data(), dim, v1_quantized.data());
    test_utils::quantize_float_vec_to_sq8_with_metadata(v2.data(), dim, v2_quantized.data());

    float baseline = SQ8_SQ8_L2Sqr(v1_quantized.data(), v2_quantized.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized extreme values L2 should match baseline";
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized extreme values L2 should match baseline";
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (optimization.asimddp) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized extreme values L2 should match baseline";
        optimization.asimddp = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized extreme values L2 should match baseline";
        optimization.asimd = 0;
    }
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, &optimization);
        float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);
        ASSERT_NEAR(result, baseline, 0.01f) << "Optimized extreme values L2 should match baseline";
        optimization.avx512f = 0;
    }
#endif
    unsigned char alignment = 0;
    auto arch_opt_func = L2_SQ8_SQ8_GetDistFunc(dim, &alignment, nullptr);
    float result = arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim);

    ASSERT_NEAR(result, baseline, 0.01f) << "Extreme values L2 should match baseline";
}

// spaces::uint8_chunked_total (uint8_chunking.h) is the chunked-accumulation driver shared by
// every uint8 SIMD kernel: it tiles a vector into segments no larger than UINT8_CHUNK_ELEMENTS
// so each segment's 32-bit SIMD partial sum stays exact, then folds the per-segment totals into
// a 64-bit scalar. The driver only ever calls Kernel::granule/first/rest, so it can be exercised
// directly with a mock kernel instead of a real SIMD kernel, which means this test needs no
// architecture-specific build flag and runs the same way on every host: an x86 box without
// AVX512 and an ARM box exercise identical logic here, closing the gap where this driver was
// previously only reachable through whichever hardware-specific kernel happened to be present.
// Real coverage of "did the driver visit every element exactly once, correctly" comes from the
// value check below: the mock's first()/rest() return the sum of the bytes in the slice they
// were handed (read from a position-dependent, non-constant fill), and the total the driver
// returns is compared against an independent, trivially-correct sum over the whole buffer. A
// skipped element, a double-counted element, or a mis-sized chunk changes that sum; it cannot
// cancel out the way it could with a constant fill or a return value of 0. The mock also records
// the (offset, length) of every call; the tiling check on those recordings does not by itself
// prove the driver visited the right elements (offset is derived from the same length the driver
// just advanced its pointer by, so "no gap/overlap" holds by construction), but it does guard the
// coupling between the length passed to the kernel and the distance the pointer is advanced, plus
// the length-shape properties below (granule multiples, chunk-size cap, congruence). Together the
// two checks cover a sweep of granules (64, 128, 192, 256 and 1024, standing in for fixed-width
// kernels and SVE vector lengths of 32/48/64/256 bytes) and dimensions chosen to cover every
// residue class modulo the granule across one-, two- and three-segment cases, plus the boundary
// around UINT8_CHUNK_ELEMENTS itself.
TEST_F(SpacesTest, UINT8_chunked_driver_tiles_the_vector_exactly) {
    struct RecordedCall {
        size_t offset;
        size_t length;
    };

    // Local mock adapter matching the Kernel contract from uint8_chunking.h. All state lives in
    // function-local statics reached through static member functions (a local class cannot have
    // static data members), so `reset` must be called before each driver invocation.
    struct RecordingKernel {
        static size_t granule() { return granule_ref(); }

        // Returns the sum of the bytes in [v1, v1 + length), not 0: combined with a
        // position-dependent fill, this makes the driver's return value an end-to-end proof
        // that every element was visited exactly once, not just a coupling check on lengths.
        static uint32_t first(const uint8_t *v1, const uint8_t *, size_t length) {
            record(v1, length);
            return sum_of(v1, length);
        }

        static uint32_t rest(const uint8_t *v1, const uint8_t *, size_t length) {
            record(v1, length);
            return sum_of(v1, length);
        }

        static void reset(const uint8_t *base, size_t granule) {
            calls_ref().clear();
            base_ref() = base;
            granule_ref() = granule;
        }

        static const std::vector<RecordedCall> &calls_seen() { return calls_ref(); }

    private:
        static uint32_t sum_of(const uint8_t *v1, size_t length) {
            uint32_t sum = 0;
            for (size_t i = 0; i < length; i++) {
                sum += v1[i];
            }
            return sum;
        }

        static void record(const uint8_t *v1, size_t length) {
            calls_ref().push_back({static_cast<size_t>(v1 - base_ref()), length});
        }

        static std::vector<RecordedCall> &calls_ref() {
            static std::vector<RecordedCall> calls;
            return calls;
        }

        static const uint8_t *&base_ref() {
            static const uint8_t *base = nullptr;
            return base;
        }

        static size_t &granule_ref() {
            static size_t granule = 0;
            return granule;
        }
    };

    constexpr size_t chunk = spaces::UINT8_CHUNK_ELEMENTS;
    // Large enough for the biggest dimension exercised below (first segment plus two full
    // max-size segments, bounded by chunk), with slack.
    constexpr size_t buffer_size = 3 * chunk + 4096;
    // v1 is filled with a position-dependent, non-constant pattern so a skipped, duplicated or
    // mis-sized element changes the summed value rather than cancelling out (a constant fill, or
    // returning 0 from the mock, would not catch that). Values stay under 251 and dimensions
    // stay well under 600,000, so the accumulated uint64_t sum cannot overflow. v2 is unused by
    // the mock kernel and left zero-filled.
    std::vector<uint8_t> v1(buffer_size);
    for (size_t i = 0; i < buffer_size; i++) {
        v1[i] = static_cast<uint8_t>((i * 31 + 7) % 251);
    }
    std::vector<uint8_t> v2(buffer_size, 0);

    const std::array<size_t, 5> granules = {64, 128, 192, 256, 1024};

    for (size_t granule : granules) {
        const size_t max_step = (chunk / granule) * granule;
        auto first_chunk_for = [&](size_t tail) {
            return tail + ((chunk - tail) / granule) * granule;
        };

        std::vector<size_t> dims;
        for (size_t r = 0; r < granule; r++) {
            const size_t fc = first_chunk_for(r);
            dims.push_back(r == 0 ? granule : r); // one segment: dim <= chunk
            dims.push_back(fc + max_step);        // two segments
            dims.push_back(fc + 2 * max_step);    // three segments
        }
        dims.push_back(chunk - 1);
        dims.push_back(chunk);
        dims.push_back(chunk + 1);

        for (size_t dimension : dims) {
            ASSERT_LE(dimension, buffer_size)
                << "granule=" << granule << " dimension=" << dimension;

            RecordingKernel::reset(v1.data(), granule);
            const uint64_t total =
                spaces::uint8_chunked_total<RecordingKernel>(v1.data(), v2.data(), dimension);

            SCOPED_TRACE("granule=" + std::to_string(granule) +
                         " dimension=" + std::to_string(dimension));

            // Value check: independently sum the same byte range the driver was asked to cover.
            // This is what actually proves every element was visited exactly once (a skipped,
            // duplicated or mis-sized chunk changes this sum); the tiling check below only
            // proves the length passed to the kernel matches how far the pointer advanced.
            uint64_t expected = 0;
            for (size_t i = 0; i < dimension; i++) {
                expected += v1[i];
            }
            EXPECT_EQ(total, expected) << "driver total does not match independent byte sum";

            const auto &calls = RecordingKernel::calls_seen();
            ASSERT_FALSE(calls.empty());

            size_t sum = 0;
            size_t expected_offset = 0;
            for (size_t i = 0; i < calls.size(); i++) {
                EXPECT_EQ(calls[i].offset, expected_offset)
                    << "call " << i << " does not tile contiguously (gap or overlap)";
                EXPECT_LE(calls[i].length, chunk)
                    << "call " << i << " exceeds UINT8_CHUNK_ELEMENTS";
                if (i > 0) {
                    EXPECT_EQ(calls[i].length % granule, 0u)
                        << "call " << i << " length is not a whole multiple of granule";
                }
                sum += calls[i].length;
                expected_offset += calls[i].length;
            }
            EXPECT_EQ(sum, dimension) << "recorded lengths do not sum to the dimension";
            EXPECT_EQ(calls[0].length % granule, dimension % granule)
                << "first call length is not congruent to dimension modulo granule";
            if (dimension <= chunk) {
                EXPECT_EQ(calls.size(), 1u)
                    << "dimension at or below UINT8_CHUNK_ELEMENTS should need exactly one call";
            }
        }
    }
}

// Assert the exact alignment-hint values published by the SQ8 distance dispatchers.
// The hint refers to the SQ8 (first / storage) operand per the GetDistFunc contract documented
// in spaces/spaces.h. These tests guard against silent regressions of the per-kernel hints used
// by the preprocessor pipeline to align the storage blob.
#ifdef CPU_FEATURES_ARCH_X86_64
TEST_F(SpacesTest, SQ8_FP32_DispatcherAlignmentHints) {
    // dim divisible by 16 (and therefore 8 and 4) so every x86 path sets a non-zero hint.
    constexpr size_t dim = 64;
    auto features = getCpuOptimizationFeatures();

    auto check = [&](const char *kind,
                     spaces::dist_func_t<float> (*get)(size_t, unsigned char *, const void *)) {
        auto opt = features;
#ifdef OPT_AVX512_F_BW_VL_VNNI
        if (opt.avx512f && opt.avx512bw && opt.avx512vnni) {
            unsigned char alignment = 0;
            (void)get(dim, &alignment, &opt);
            ASSERT_EQ(alignment, 16u) << kind << ": AVX512 SQ8_FP32 hint should be 16";
            opt.avx512f = 0;
        }
#endif
#ifdef OPT_AVX2_FMA
        if (opt.avx2 && opt.fma3) {
            unsigned char alignment = 0;
            (void)get(dim, &alignment, &opt);
            ASSERT_EQ(alignment, 8u) << kind << ": AVX2_FMA SQ8_FP32 hint should be 8";
            opt.fma3 = 0;
        }
#endif
#ifdef OPT_AVX2
        if (opt.avx2) {
            unsigned char alignment = 0;
            (void)get(dim, &alignment, &opt);
            ASSERT_EQ(alignment, 8u) << kind << ": AVX2 SQ8_FP32 hint should be 8";
            opt.avx2 = 0;
        }
#endif
#ifdef OPT_SSE4
        if (opt.sse4_1) {
            unsigned char alignment = 0;
            (void)get(dim, &alignment, &opt);
            ASSERT_EQ(alignment, 4u) << kind << ": SSE4 SQ8_FP32 hint should be 4";
            opt.sse4_1 = 0;
        }
#endif
        // No-optimization path must leave the hint at 0.
        unsigned char alignment = 0;
        (void)get(dim, &alignment, &opt);
        ASSERT_EQ(alignment, 0u) << kind << ": no-optimization hint should be 0";
    };

    check("IP", &spaces::IP_SQ8_FP32_GetDistFunc);
    check("L2", &spaces::L2_SQ8_FP32_GetDistFunc);
    check("Cosine", &spaces::Cosine_SQ8_FP32_GetDistFunc);
}

TEST_F(SpacesTest, SQ8_SQ8_DispatcherAlignmentHints) {
    // dim divisible by 32 so the AVX512 SQ8_SQ8 path sets the hint (otherwise it stays at 0).
    constexpr size_t dim = 64;
    auto features = getCpuOptimizationFeatures();

    auto check = [&](const char *kind,
                     spaces::dist_func_t<float> (*get)(size_t, unsigned char *, const void *)) {
        auto opt = features;
#ifdef OPT_AVX512_F_BW_VL_VNNI
        if (opt.avx512f && opt.avx512bw && opt.avx512vnni) {
            unsigned char alignment = 0;
            (void)get(dim, &alignment, &opt);
            ASSERT_EQ(alignment, 32u) << kind << ": AVX512 SQ8_SQ8 hint should be 32";
            opt.avx512f = 0;
        }
#endif
        // No-optimization path must leave the hint at 0.
        unsigned char alignment = 0;
        (void)get(dim, &alignment, &opt);
        ASSERT_EQ(alignment, 0u) << kind << ": no-optimization hint should be 0";
    };

    check("IP", &spaces::IP_SQ8_SQ8_GetDistFunc);
    check("L2", &spaces::L2_SQ8_SQ8_GetDistFunc);
    check("Cosine", &spaces::Cosine_SQ8_SQ8_GetDistFunc);
}
#endif // CPU_FEATURES_ARCH_X86_64
