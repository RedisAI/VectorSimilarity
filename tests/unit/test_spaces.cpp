/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <utility>

#include "gtest/gtest.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/types/float16.h"
#include "VecSim/spaces/functions/AVX512.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/AVX512BW_VL.h"
#include "VecSim/spaces/functions/F16C.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

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
    for (size_t i = 0; i < dim; i++) {
        ASSERT_EQ(vecsim_types::bfloat16_to_float32(a[i]), sanity_a[i])
            << " bf16 normalization failed for i = " << i;
        ASSERT_EQ(vecsim_types::bfloat16_to_float32(a[i]), 0.5)
            << " bf16 normalization failed for i = " << i;
    }
}

TEST_F(SpacesTest, bf16_ip_no_optimization_func_test2) {
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

using namespace spaces;

TEST_F(SpacesTest, smallDimChooser) {
    // Verify that small dimensions gets the no optimization function.
    for (size_t dim = 1; dim < 8; dim++) {
        ASSERT_EQ(L2_FP32_GetDistFunc(dim), FP32_L2Sqr);
        ASSERT_EQ(L2_FP64_GetDistFunc(dim), FP64_L2Sqr);
        ASSERT_EQ(L2_BF16_GetDistFunc(dim), BF16_L2Sqr_LittleEndian);
        ASSERT_EQ(L2_FP16_GetDistFunc(dim), FP16_L2Sqr);
        ASSERT_EQ(IP_FP32_GetDistFunc(dim), FP32_InnerProduct);
        ASSERT_EQ(IP_FP64_GetDistFunc(dim), FP64_InnerProduct);
        ASSERT_EQ(IP_BF16_GetDistFunc(dim), BF16_InnerProduct_LittleEndian);
        ASSERT_EQ(IP_FP16_GetDistFunc(dim), FP16_InnerProduct);
    }
    for (size_t dim = 8; dim < 16; dim++) {
        ASSERT_EQ(L2_FP32_GetDistFunc(dim), FP32_L2Sqr);
        ASSERT_EQ(L2_BF16_GetDistFunc(dim), BF16_L2Sqr_LittleEndian);
        ASSERT_EQ(L2_FP16_GetDistFunc(dim), FP16_L2Sqr);
        ASSERT_EQ(IP_FP32_GetDistFunc(dim), FP32_InnerProduct);
        ASSERT_EQ(IP_BF16_GetDistFunc(dim), BF16_InnerProduct_LittleEndian);
        ASSERT_EQ(IP_FP16_GetDistFunc(dim), FP16_InnerProduct);
    }
    for (size_t dim = 16; dim < 32; dim++) {
        ASSERT_EQ(L2_BF16_GetDistFunc(dim), BF16_L2Sqr_LittleEndian);
        ASSERT_EQ(L2_FP16_GetDistFunc(dim), FP16_L2Sqr);
        ASSERT_EQ(IP_BF16_GetDistFunc(dim), BF16_InnerProduct_LittleEndian);
        ASSERT_EQ(IP_FP16_GetDistFunc(dim), FP16_InnerProduct);
    }
}

// In this following tests we assume that compiler supports all X86 optimizations, so if we have
// some hardware flag enabled, we check that the corresponding optimization function was chosen.
#ifdef CPU_FEATURES_ARCH_X86_64

class FP32SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(FP32SpacesOptimizationTest, FP32L2SqrTest) {
    auto optimization = cpu_features::GetX86Info().features;
    size_t dim = GetParam();
    float v[dim];
    float v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
        v2[i] = (float)(i + 1.5);
    }

    dist_func_t<float> arch_opt_func;
    float baseline = FP32_L2Sqr(v, v2, dim);
    if (optimization.avx512f) {
        arch_opt_func = L2_FP32_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_L2_implementation_AVX512(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        // Unset avx512f flag, so we'll choose the next optimization (AVX).
        optimization.avx512f = 0;
    }
    if (optimization.avx) {
        arch_opt_func = L2_FP32_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_L2_implementation_AVX(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        // Unset avx flag as well, so we'll choose the next optimization (SSE).
        optimization.avx = 0;
    }
    if (optimization.sse) {
        arch_opt_func = L2_FP32_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_L2_implementation_SSE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        // Unset sse flag as well, so we'll choose the next option (default).
        optimization.sse = 0;
    }
    arch_opt_func = L2_FP32_GetDistFunc(dim, &optimization);
    ASSERT_EQ(arch_opt_func, FP32_L2Sqr) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
}

TEST_P(FP32SpacesOptimizationTest, FP32InnerProductTest) {
    auto optimization = cpu_features::GetX86Info().features;
    size_t dim = GetParam();
    float v[dim];
    float v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
        v2[i] = (float)(i + 1.5);
    }

    dist_func_t<float> arch_opt_func;
    float baseline = FP32_InnerProduct(v, v2, dim);
    if (optimization.avx512f) {
        arch_opt_func = IP_FP32_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_IP_implementation_AVX512(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
    if (optimization.avx) {
        arch_opt_func = IP_FP32_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_IP_implementation_AVX(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        optimization.avx = 0;
    }
    if (optimization.sse) {
        arch_opt_func = IP_FP32_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP32_IP_implementation_SSE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        optimization.sse = 0;
    }
    arch_opt_func = IP_FP32_GetDistFunc(dim, &optimization);
    ASSERT_EQ(arch_opt_func, FP32_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
}

INSTANTIATE_TEST_SUITE_P(FP32OptFuncs, FP32SpacesOptimizationTest, testing::Range(16UL, 16 * 2UL));

class FP64SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(FP64SpacesOptimizationTest, FP64L2SqrTest) {
    auto optimization = cpu_features::GetX86Info().features;
    size_t dim = GetParam();
    double v[dim];
    double v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
        v2[i] = (double)(i + 1.5);
    }

    dist_func_t<double> arch_opt_func;
    double baseline = FP64_L2Sqr(v, v2, dim);
    if (optimization.avx512f) {
        arch_opt_func = L2_FP64_GetDistFunc(dim, &optimization);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        ASSERT_EQ(arch_opt_func, Choose_FP64_L2_implementation_AVX512(dim))
            << "Unexpected distance function chosen for dim " << dim;
        optimization.avx512f = 0;
    }
    if (optimization.avx) {
        arch_opt_func = L2_FP64_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_L2_implementation_AVX(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        optimization.avx = 0;
    }
    if (optimization.sse) {
        arch_opt_func = L2_FP64_GetDistFunc(dim, &optimization);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        ASSERT_EQ(arch_opt_func, Choose_FP64_L2_implementation_SSE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        optimization.sse = 0;
    }
    arch_opt_func = L2_FP64_GetDistFunc(dim, &optimization);
    ASSERT_EQ(arch_opt_func, FP64_L2Sqr) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
}

TEST_P(FP64SpacesOptimizationTest, FP64InnerProductTest) {
    auto optimization = cpu_features::GetX86Info().features;
    size_t dim = GetParam();
    double v[dim];
    double v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
        v2[i] = (double)(i + 1.5);
    }

    dist_func_t<double> arch_opt_func;
    double baseline = FP64_InnerProduct(v, v2, dim);
    if (optimization.avx512f) {
        arch_opt_func = IP_FP64_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_IP_implementation_AVX512(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        optimization.avx512f = 0;
    }
    if (optimization.avx) {
        arch_opt_func = IP_FP64_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_IP_implementation_AVX(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        optimization.avx = 0;
    }
    if (optimization.sse) {
        arch_opt_func = IP_FP64_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP64_IP_implementation_SSE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        optimization.sse = 0;
    }
    arch_opt_func = IP_FP64_GetDistFunc(dim, &optimization);
    ASSERT_EQ(arch_opt_func, FP64_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
}

INSTANTIATE_TEST_SUITE_P(FP64OptFuncs, FP64SpacesOptimizationTest, testing::Range(8UL, 8 * 2UL));

class BF16SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(BF16SpacesOptimizationTest, BF16InnerProductTest) {
    auto optimization = cpu_features::GetX86Info().features;
    size_t dim = GetParam();
    bfloat16 v[dim];
    bfloat16 v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = vecsim_types::float_to_bf16((float)i);
        v2[i] = vecsim_types::float_to_bf16(((float)i + 1.5f));
    }

    dist_func_t<float> arch_opt_func;
    float baseline = BF16_InnerProduct_LittleEndian(v, v2, dim);
    if (optimization.avx512bw && optimization.avx512vbmi2) {
        arch_opt_func = IP_BF16_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_IP_implementation_AVX512BW_VBMI2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        optimization.avx512bw = optimization.avx512vbmi2 = 0;
    }
    if (optimization.avx2) {
        arch_opt_func = IP_BF16_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_IP_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        optimization.avx2 = 0;
    }
    if (optimization.sse3) {
        arch_opt_func = IP_BF16_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_IP_implementation_SSE3(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        optimization.sse3 = 0;
    }
    arch_opt_func = IP_BF16_GetDistFunc(dim, &optimization);
    ASSERT_EQ(arch_opt_func, BF16_InnerProduct_LittleEndian)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
}

TEST_P(BF16SpacesOptimizationTest, BF16L2SqrTest) {
    auto optimization = cpu_features::GetX86Info().features;
    size_t dim = GetParam();
    bfloat16 v[dim];
    bfloat16 v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = vecsim_types::float_to_bf16((float)i);
        v2[i] = vecsim_types::float_to_bf16(((float)i + 1.5f));
    }

    dist_func_t<float> arch_opt_func;
    float baseline = BF16_L2Sqr_LittleEndian(v, v2, dim);

    if (optimization.avx512bw && optimization.avx512vbmi2) {
        arch_opt_func = L2_BF16_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_L2_implementation_AVX512BW_VBMI2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
        optimization.avx512bw = optimization.avx512vbmi2 = 0;
    }
    if (optimization.avx2) {
        arch_opt_func = L2_BF16_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_L2_implementation_AVX2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
        optimization.avx2 = 0;
    }
    if (optimization.sse3) {
        arch_opt_func = L2_BF16_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_BF16_L2_implementation_SSE3(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
        optimization.sse3 = 0;
    }
    arch_opt_func = L2_BF16_GetDistFunc(dim, &optimization);
    ASSERT_EQ(arch_opt_func, BF16_L2Sqr_LittleEndian)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "No optimization with dim " << dim;
}

INSTANTIATE_TEST_SUITE_P(BF16OptFuncs, BF16SpacesOptimizationTest, testing::Range(32UL, 32 * 2UL));

class FP16SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(FP16SpacesOptimizationTest, FP16InnerProductTest) {
    auto optimization = cpu_features::GetX86Info().features;
    size_t dim = GetParam();
    float16 v1[dim], v2[dim];
    float v1_fp32[dim], v2_fp32[dim];
    for (size_t i = 0; i < dim; i++) {
        v1_fp32[i] = (float)i;
        v1[i] = vecsim_types::FP32_to_FP16(v1_fp32[i]);
        v2_fp32[i] = (float)i + 1.5f;
        v2[i] = vecsim_types::FP32_to_FP16(v2_fp32[i]);
    }

    dist_func_t<float> arch_opt_func;
    float baseline = FP16_InnerProduct(v1, v2, dim);
    ASSERT_EQ(baseline, FP32_InnerProduct(v1_fp32, v2_fp32, dim)) << "Baseline check " << dim;

    if (optimization.avx512bw && optimization.avx512vl) {
        arch_opt_func = IP_FP16_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_IP_implementation_AVX512BW_VL(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        optimization.avx512bw = optimization.avx512vl = 0;
    }
    if (optimization.f16c && optimization.fma3 && optimization.avx) {
        arch_opt_func = IP_FP16_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_IP_implementation_F16C(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "F16C with dim " << dim;
        optimization.f16c = optimization.fma3 = optimization.avx = 0;
    }
    arch_opt_func = IP_FP16_GetDistFunc(dim, &optimization);
    ASSERT_EQ(arch_opt_func, FP16_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "F16C with dim " << dim;
}

TEST_P(FP16SpacesOptimizationTest, FP16L2SqrTest) {
    auto optimization = cpu_features::GetX86Info().features;
    size_t dim = GetParam();
    float16 v1[dim], v2[dim];
    float v1_fp32[dim], v2_fp32[dim];
    for (size_t i = 0; i < dim; i++) {
        v1_fp32[i] = (float)i;
        v1[i] = vecsim_types::FP32_to_FP16(v1_fp32[i]);
        v2_fp32[i] = (float)i + 1.5f;
        v2[i] = vecsim_types::FP32_to_FP16(v2_fp32[i]);
    }

    dist_func_t<float> arch_opt_func;
    float baseline = FP16_L2Sqr(v1, v2, dim);
    ASSERT_EQ(baseline, FP32_L2Sqr(v1_fp32, v2_fp32, dim)) << "Baseline check " << dim;

    if (optimization.avx512bw && optimization.avx512vl) {
        arch_opt_func = L2_FP16_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_L2_implementation_AVX512BW_VL(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "AVX512 with dim " << dim;
        optimization.avx512bw = optimization.avx512vl = 0;
    }
    if (optimization.f16c && optimization.fma3 && optimization.avx) {
        arch_opt_func = L2_FP16_GetDistFunc(dim, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_FP16_L2_implementation_F16C(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "F16C with dim " << dim;
        optimization.f16c = optimization.fma3 = optimization.avx = 0;
    }
    arch_opt_func = L2_FP16_GetDistFunc(dim, &optimization);
    ASSERT_EQ(arch_opt_func, FP16_L2Sqr) << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "F16C with dim " << dim;
}

INSTANTIATE_TEST_SUITE_P(FP16OptFuncs, FP16SpacesOptimizationTest, testing::Range(32UL, 32 * 2UL));

#endif // CPU_FEATURES_ARCH_X86_64
