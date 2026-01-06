/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include <utility>
#include <random>
#include <cmath>

#include "gtest/gtest.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/types/float16.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512BF16_VL.h"
#include "VecSim/spaces/functions/AVX512FP16_VL.h"
#include "VecSim/spaces/functions/AVX512F_BW_VL_VNNI.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/AVX2_FMA.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/SSE4.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/NEON.h"
#include "VecSim/spaces/functions/NEON_DOTPROD.h"
#include "VecSim/spaces/functions/NEON_HP.h"
#include "VecSim/spaces/functions/NEON_BF16.h"
#include "VecSim/spaces/functions/SVE.h"
#include "VecSim/spaces/functions/SVE_BF16.h"
#include "VecSim/spaces/functions/SVE2.h"
#include "tests_utils.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;
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
    *(float *)(v1 + dim) = test_utils::integral_compute_norm(v1, dim);
    *(float *)(v2 + dim) = test_utils::integral_compute_norm(v2, dim);

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
    *(float *)(v1 + dim) = test_utils::integral_compute_norm(v1, dim);
    *(float *)(v2 + dim) = test_utils::integral_compute_norm(v2, dim);

    float dist = UINT8_Cosine((const void *)v1, (const void *)v2, dim);
    ASSERT_NEAR(dist, 0.0, 0.000001);
}

void common_ip_sq8(bool should_normalize, float expected_dist) {

    size_t dim = 5;

    // Create original vectors
    float v1_orig[dim], v2_orig[dim];
    for (size_t i = 0; i < dim; i++) {
        v1_orig[i] = float(i + 1.5);
        v2_orig[i] = float(i + 1.5);
    }

    // Create SQ8 compressed version of v2
    // Size: dim (uint8_t) + min_val (float) + delta (float) + sum (float) + sum_squares (float)
    size_t compressed_size = dim * sizeof(uint8_t) + 3 * sizeof(float);
    if (should_normalize) {
        spaces::GetNormalizeFunc<float>()(v1_orig, dim);
        spaces::GetNormalizeFunc<float>()(v2_orig, dim);
    }

    // Find min and max for quantization
    float min_val = v2_orig[0];
    float max_val = v2_orig[0];
    float sum = v2_orig[0];
    for (size_t i = 1; i < dim; i++) {
        min_val = std::min(min_val, v2_orig[i]);
        max_val = std::max(max_val, v2_orig[i]);
        sum += v2_orig[i];
    }

    // Calculate delta
    float delta = (max_val - min_val) / 255.0f;
    if (delta == 0)
        delta = 1.0f; // Avoid division by zero

    std::vector<uint8_t> v2_compressed(compressed_size);

    // Quantize v2
    uint8_t *quant_values = reinterpret_cast<uint8_t *>(v2_compressed.data());
    float *params = reinterpret_cast<float *>(quant_values + dim);

    // Store parameters
    params[0] = min_val;
    params[1] = delta;
    params[2] = sum;

    // Quantize each value
    for (size_t i = 0; i < dim; i++) {
        float quantized = (v2_orig[i] - min_val) / delta;
        quantized = std::max(0.0f, std::min(255.0f, quantized));
        quant_values[i] = static_cast<uint8_t>(std::round(quantized));
    }

    float dist = SQ8_InnerProduct((const void *)v1_orig, (const void *)v2_compressed.data(), dim);

    // Since we're comparing identical vectors, the inner product distance should be close to
    // expected
    ASSERT_NEAR(dist, expected_dist, 0.01) << "SQ8_InnerProduct failed to match expected distance";
}

/* ======================== Tests SQ8 ========================= */
TEST_F(SpacesTest, SQ8_ip_no_optimization_func_test) {
    float expected_dist = -70.2147f; // Expected distance for identical vectors
    common_ip_sq8(false, expected_dist);
}

TEST_F(SpacesTest, SQ8_ip_no_optimization_norm_func_test) { common_ip_sq8(true, 0.0f); }

TEST_F(SpacesTest, SQ8_Cosine_no_optimization_func_test) {
    // create a vector with extra space for the norm
    size_t dim = 5;

    // Create original vectors
    float v1_orig[dim], v2_orig[dim];
    for (size_t i = 0; i < dim; i++) {
        v1_orig[i] = float(i + 1.5);
        v2_orig[i] = float(i + 1.5);
    }

    // Size: dim (uint8_t) + min_val (float) + delta (float) + inv_norm (float)
    size_t compressed_size = dim * sizeof(uint8_t) + 3 * sizeof(float);
    spaces::GetNormalizeFunc<float>()(v1_orig, dim);
    spaces::GetNormalizeFunc<float>()(v2_orig, dim);
    // Find min and max for quantization
    float min_val = v2_orig[0];
    float max_val = v2_orig[0];
    float sum = v2_orig[0];
    for (size_t i = 1; i < dim; i++) {
        min_val = std::min(min_val, v2_orig[i]);
        max_val = std::max(max_val, v2_orig[i]);
        sum += v2_orig[i];
    }
    // Calculate delta and inverse norm
    float delta = (max_val - min_val) / 255.0f;
    if (delta == 0)
        delta = 1.0f; // Avoid division by zero

    // Compress v2
    std::vector<uint8_t> v2_compressed(compressed_size);
    uint8_t *quant_values = reinterpret_cast<uint8_t *>(v2_compressed.data());
    float *params = reinterpret_cast<float *>(quant_values + dim);

    // Quantize each value
    for (size_t i = 0; i < dim; i++) {
        float normalized = (v2_orig[i] - min_val) / delta;
        normalized = std::max(0.0f, std::min(255.0f, normalized));
        quant_values[i] = static_cast<uint8_t>(std::round(normalized));
    }
    // Store parameters
    params[0] = min_val;
    params[1] = delta;
    params[2] = sum;

    float dist = SQ8_Cosine((const void *)v1_orig, (const void *)v2_compressed.data(), dim);
    ASSERT_NEAR(dist, 0.0f, 0.001f) << "SQ8_Cosine failed to match expected distance";
}
TEST_F(SpacesTest, SQ8_l2sqr_no_optimization_func_test) {
    // create a vector with extra space for the norm
    size_t dim = 5;

    // Create original vectors
    float v1_orig[dim], v2_orig[dim];
    for (size_t i = 0; i < dim; i++) {
        v1_orig[i] = float(i + 1.5);
        v2_orig[i] = float(i + 1.5);
    }

    // Size: dim (uint8_t) + min_val (float) + delta (float) + inv_norm (float)
    size_t compressed_size = dim * sizeof(uint8_t) + 3 * sizeof(float);
    spaces::GetNormalizeFunc<float>()(v1_orig, dim);
    spaces::GetNormalizeFunc<float>()(v2_orig, dim);
    // Find min and max for quantization
    float min_val = v2_orig[0];
    float max_val = v2_orig[0];
    for (size_t i = 1; i < dim; i++) {
        min_val = std::min(min_val, v2_orig[i]);
        max_val = std::max(max_val, v2_orig[i]);
    }
    // Calculate delta and inverse norm
    float delta = (max_val - min_val) / 255.0f;
    if (delta == 0)
        delta = 1.0f; // Avoid division by zero

    // Compress v2
    std::vector<uint8_t> v2_compressed(compressed_size);
    uint8_t *quant_values = reinterpret_cast<uint8_t *>(v2_compressed.data());
    float *params = reinterpret_cast<float *>(quant_values + dim);

    // Quantize each value
    for (size_t i = 0; i < dim; i++) {
        float normalized = (v2_orig[i] - min_val) / delta;
        normalized = std::max(0.0f, std::min(255.0f, normalized));
        quant_values[i] = static_cast<uint8_t>(std::round(normalized));
    }
    // Calculate inverse norm from decompressed values
    float inv_norm = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float decompressed_value = min_val + quant_values[i] * delta;
        inv_norm += decompressed_value * decompressed_value;
    }
    inv_norm = 1.0f / std::sqrt(inv_norm);
    // Store parameters
    params[0] = min_val;
    params[1] = delta;
    params[2] = inv_norm;

    float dist = SQ8_L2Sqr((const void *)v1_orig, (const void *)v2_compressed.data(), dim);
    ASSERT_NEAR(dist, 0.0f, 0.00001f) << "SQ8_Cosine failed to match expected distance";
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
#ifdef CPU_FEATURES_ARCH_X86_64
TEST_F(SpacesTest, smallDimChooser) {
    // Verify that small dimensions gets the no optimization function.
    for (size_t dim = 1; dim < 8; dim++) {
        ASSERT_EQ(L2_FP32_GetDistFunc(dim), FP32_L2Sqr);
        ASSERT_EQ(L2_FP64_GetDistFunc(dim), FP64_L2Sqr);
        ASSERT_EQ(L2_BF16_GetDistFunc(dim), BF16_L2Sqr_LittleEndian);
        ASSERT_EQ(L2_FP16_GetDistFunc(dim), FP16_L2Sqr);
        ASSERT_EQ(L2_INT8_GetDistFunc(dim), INT8_L2Sqr);
        ASSERT_EQ(L2_UINT8_GetDistFunc(dim), UINT8_L2Sqr);
        ASSERT_EQ(IP_FP32_GetDistFunc(dim), FP32_InnerProduct);
        ASSERT_EQ(IP_FP64_GetDistFunc(dim), FP64_InnerProduct);
        ASSERT_EQ(IP_BF16_GetDistFunc(dim), BF16_InnerProduct_LittleEndian);
        ASSERT_EQ(IP_FP16_GetDistFunc(dim), FP16_InnerProduct);
        ASSERT_EQ(IP_INT8_GetDistFunc(dim), INT8_InnerProduct);
        ASSERT_EQ(IP_UINT8_GetDistFunc(dim), UINT8_InnerProduct);
        ASSERT_EQ(Cosine_INT8_GetDistFunc(dim), INT8_Cosine);
        ASSERT_EQ(Cosine_UINT8_GetDistFunc(dim), UINT8_Cosine);
    }
    for (size_t dim = 8; dim < 16; dim++) {
        ASSERT_EQ(L2_FP32_GetDistFunc(dim), FP32_L2Sqr);
        ASSERT_EQ(L2_BF16_GetDistFunc(dim), BF16_L2Sqr_LittleEndian);
        ASSERT_EQ(L2_FP16_GetDistFunc(dim), FP16_L2Sqr);
        ASSERT_EQ(L2_INT8_GetDistFunc(dim), INT8_L2Sqr);
        ASSERT_EQ(L2_UINT8_GetDistFunc(dim), UINT8_L2Sqr);
        ASSERT_EQ(IP_FP32_GetDistFunc(dim), FP32_InnerProduct);
        ASSERT_EQ(IP_BF16_GetDistFunc(dim), BF16_InnerProduct_LittleEndian);
        ASSERT_EQ(IP_FP16_GetDistFunc(dim), FP16_InnerProduct);
        ASSERT_EQ(IP_INT8_GetDistFunc(dim), INT8_InnerProduct);
        ASSERT_EQ(IP_UINT8_GetDistFunc(dim), UINT8_InnerProduct);
        ASSERT_EQ(Cosine_INT8_GetDistFunc(dim), INT8_Cosine);
        ASSERT_EQ(Cosine_UINT8_GetDistFunc(dim), UINT8_Cosine);
    }
    for (size_t dim = 16; dim < 32; dim++) {
        ASSERT_EQ(L2_BF16_GetDistFunc(dim), BF16_L2Sqr_LittleEndian);
        ASSERT_EQ(L2_FP16_GetDistFunc(dim), FP16_L2Sqr);
        ASSERT_EQ(L2_INT8_GetDistFunc(dim), INT8_L2Sqr);
        ASSERT_EQ(L2_UINT8_GetDistFunc(dim), UINT8_L2Sqr);
        ASSERT_EQ(IP_BF16_GetDistFunc(dim), BF16_InnerProduct_LittleEndian);
        ASSERT_EQ(IP_FP16_GetDistFunc(dim), FP16_InnerProduct);
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
                         testing::Range(16UL, 16 * 2UL + 1));

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
                         testing::Range(8UL, 8 * 2UL + 1));

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
    if (optimization.avx512f) {
        unsigned char alignment = 0;
        arch_opt_func = IP_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_IP_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
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
    if (optimization.avx512f) {
        unsigned char alignment = 0;
        arch_opt_func = L2_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_L2_implementation_AVX512F(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
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
                         testing::Range(32UL, 32 * 2UL + 1));

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
    *(float *)(v1 + dim) = test_utils::integral_compute_norm(v1, dim);
    *(float *)(v2 + dim) = test_utils::integral_compute_norm(v2, dim);

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
    *(float *)(v1 + dim) = test_utils::integral_compute_norm(v1, dim);
    *(float *)(v2 + dim) = test_utils::integral_compute_norm(v2, dim);

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
    *(float *)(v1 + dim) = test_utils::integral_compute_norm(v1, dim);
    *(float *)(v2 + dim) = test_utils::integral_compute_norm(v2, dim);

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

// Helper function to create SQ8 compressed vector
std::vector<uint8_t> CreateSQ8CompressedVector(const float *original, size_t dim) {
    // Create a copy of the original vector that we can modify
    std::vector<float> vec_copy(original, original + dim);

    // Size: dim (uint8_t) + min_val (float) + delta (float) + norm (float)
    size_t compressed_size = dim * sizeof(uint8_t) + 3 * sizeof(float);
    std::vector<uint8_t> compressed(compressed_size);

    // Find min and max for quantization
    float min_val = vec_copy[0];
    float max_val = vec_copy[0];
    for (size_t i = 1; i < dim; i++) {
        min_val = std::min(min_val, vec_copy[i]);
        max_val = std::max(max_val, vec_copy[i]);
    }

    // Calculate delta
    float delta = (max_val - min_val) / 255.0f;
    if (delta == 0)
        delta = 1.0f; // Avoid division by zero

    // Quantize vector
    uint8_t *quant_values = compressed.data();
    float norm = 0.0f;
    // Quantize each value
    for (size_t i = 0; i < dim; i++) {
        float normalized = (vec_copy[i] - min_val) / delta;
        normalized = std::max(0.0f, std::min(255.0f, normalized));
        quant_values[i] = static_cast<uint8_t>(std::round(normalized));
        norm += (quant_values[i] * delta + min_val) * (quant_values[i] * delta + min_val);
    }

    float inv_norm = 1.0f / std::sqrt(norm);
    // Store parameters
    float *params = reinterpret_cast<float *>(quant_values + dim);
    params[0] = min_val;
    params[1] = delta;
    params[2] = inv_norm;

    return compressed;
}

class SQ8SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(SQ8SpacesOptimizationTest, SQ8L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    // Create original vectors
    std::vector<float> v1_orig(dim);
    std::vector<float> v2_orig(dim);
    for (size_t i = 0; i < dim; i++) {
        v1_orig[i] = float(i + 1.5);
        v2_orig[i] = float(i * 0.75 + 1.0);
    }

    // Create SQ8 compressed version of v2
    std::vector<uint8_t> v2_compressed = CreateSQ8CompressedVector(v2_orig.data(), dim);

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(uint8_t) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(uint8_t) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_L2Sqr(v1_orig.data(), v2_compressed.data(), dim);
// Test different optimizations based on CPU features
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_L2_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        // ASSERT_EQ(alignment, expected_alignment(512, dim)) << "AVX512 with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_L2_implementation_AVX2_FMA(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "AVX with dim " << dim;
        // ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_L2_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "AVX with dim " << dim;
        // ASSERT_EQ(alignment, expected_alignment(256, dim)) << "AVX with dim " << dim;
        // Unset avx flag as well, so we'll choose the next optimization (SSE).
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE4
    if (optimization.sse4_1) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_L2_implementation_SSE4(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "SSE with dim " << dim;
        // ASSERT_EQ(alignment, expected_alignment(128, dim)) << "SSE with dim " << dim;
        // Unset sse flag as well, so we'll choose the next optimization (default).
        optimization.sse4_1 = 0;
    }
#endif

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_L2_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve2 flag as well, so we'll choose the next option (default).
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_L2_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "SVE with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset sve flag as well, so we'll choose the next option (default).
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_L2_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "NEON with dim " << dim;
        ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
        // Unset optimizations flag, so we'll choose the next optimization.
        optimization.asimd = 0;
    }
#endif

    // Test default implementation
    unsigned char alignment = 0;
    arch_opt_func = L2_SQ8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_L2Sqr) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

TEST_P(SQ8SpacesOptimizationTest, SQ8InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    // Create original vectors
    std::vector<float> v1_orig(dim);
    std::vector<float> v2_orig(dim);
    for (size_t i = 0; i < dim; i++) {
        v1_orig[i] = float(i + 1.5);
        v2_orig[i] = float(i * 0.75 + 1.0);
    }
    spaces::GetNormalizeFunc<float>()(v1_orig.data(), dim);

    // Create SQ8 compressed version of v2
    std::vector<uint8_t> v2_compressed = CreateSQ8CompressedVector(v2_orig.data(), dim);
    // print min and delta
    float *params = reinterpret_cast<float *>(v2_compressed.data() + dim);

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(uint8_t) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(uint8_t) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_InnerProduct(v1_orig.data(), v2_compressed.data(), dim);

// Test different optimizations based on CPU features
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_IP_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_IP_implementation_AVX2_FMA(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "AVX with dim " << dim;
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_IP_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "AVX with dim " << dim;
        optimization.avx2 = 0;
    }
#endif
#ifdef OPT_SSE
    if (optimization.sse4_1) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_IP_implementation_SSE4(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "SSE with dim " << dim;
        optimization.sse4_1 = 0;
    }
#endif
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_IP_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_IP_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = IP_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_IP_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "NEON with dim " << dim;
        optimization.asimd = 0;
    }
#endif

    // Test default implementation
    unsigned char alignment = 0;
    arch_opt_func = IP_SQ8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

// Instantiate the test suite with dimensions to test
INSTANTIATE_TEST_SUITE_P(SQ8InnerProductTest, SQ8SpacesOptimizationTest,
                         testing::Range(16UL, 16 * 2UL + 1));

TEST_P(SQ8SpacesOptimizationTest, SQ8CosineTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    // Create original vectors
    std::vector<float> v1_orig(dim);
    std::vector<float> v2_orig(dim);
    for (size_t i = 0; i < dim; i++) {
        v1_orig[i] = float(i + 1.5);
        v2_orig[i] = float(i * 0.75 + 1.0);
    }

    // Normalize v1
    spaces::GetNormalizeFunc<float>()(v1_orig.data(), dim);
    spaces::GetNormalizeFunc<float>()(v2_orig.data(), dim);

    // Create SQ8 compressed version of v2 (with normalization)
    std::vector<uint8_t> v2_compressed = CreateSQ8CompressedVector(v2_orig.data(), dim);

    auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
        size_t elements_in_reg = reg_bit_size / sizeof(uint8_t) / 8;
        return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(uint8_t) : 0;
    };

    dist_func_t<float> arch_opt_func;
    float baseline = SQ8_Cosine(v1_orig.data(), v2_compressed.data(), dim);

#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_Cosine_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_Cosine_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON
    if (optimization.asimd) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_Cosine_implementation_NEON(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "NEON with dim " << dim;
        optimization.asimd = 0;
    }
#endif

// Test different optimizations based on CPU features
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (optimization.avx512f && optimization.avx512bw && optimization.avx512vnni) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_Cosine_implementation_AVX512F_BW_VL_VNNI(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
#endif
#ifdef OPT_AVX2_FMA
    if (optimization.avx2 && optimization.fma3) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_Cosine_implementation_AVX2_FMA(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "AVX with dim " << dim;
        optimization.fma3 = 0;
    }
#endif
#ifdef OPT_AVX2
    if (optimization.avx2) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_Cosine_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "AVX with dim " << dim;
        optimization.avx2 = 0;
    }
#endif

#ifdef OPT_SSE
    if (optimization.sse4_1) {
        unsigned char alignment = 0;
        arch_opt_func = Cosine_SQ8_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_Cosine_implementation_SSE4(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
            << "SSE with dim " << dim;
        optimization.sse4_1 = 0;
    }
#endif

    // Test default implementation
    unsigned char alignment = 0;
    arch_opt_func = Cosine_SQ8_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, SQ8_Cosine) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v1_orig.data(), v2_compressed.data(), dim), 0.01)
        << "No optimization with dim " << dim;
    ASSERT_EQ(alignment, 0) << "No optimization with dim " << dim;
}

/* ======================== Tests SQ8_SQ8 ========================= */

TEST_F(SpacesTest, SQ8_SQ8_ip_no_optimization_func_test) {
    size_t dim = 5;

    // Create SQ8 quantized versions of both vectors
    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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
    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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
    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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
    ASSERT_EQ(arch_opt_func, SQ8_SQ8_L2Sqr) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_NEAR(baseline, arch_opt_func(v1_quantized.data(), v2_quantized.data(), dim), 0.001f)
        << "SQ8_SQ8_L2Sqr failed to match expected distance";
}

class SQ8_SQ8_SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(SQ8_SQ8_SpacesOptimizationTest, SQ8_SQ8_InnerProductTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();

    // Create SQ8 quantized versions of both vectors
    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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
    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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
    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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

    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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
    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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

    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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

    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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

    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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

    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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
    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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

    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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

    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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

    size_t quantized_size = dim * sizeof(uint8_t) + 4 * sizeof(float);
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
