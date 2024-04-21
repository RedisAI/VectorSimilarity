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
#include "VecSim/spaces/functions/AVX512.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/SSE3.h"

using bfloat16 = vecsim_types::bfloat16;

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

TEST_F(SpacesTest, float_ip_no_optimization_func_test) {
    size_t dim = 5;

    float a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = float(i + 1.5);
        b[i] = float(i + 1.5);
    }

    normalizeVector(a, dim);
    normalizeVector(b, dim);

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

    normalizeVector(a, dim);
    normalizeVector(b, dim);

    double dist = FP64_InnerProduct((const void *)a, (const void *)b, dim);
    ASSERT_NEAR(dist, 0.0, 0.00000001);
}

using namespace spaces;

TEST_F(SpacesTest, smallDimChooser) {
    // Verify that small dimensions gets the no optimization function.
    for (size_t dim = 1; dim < 8; dim++) {
        ASSERT_EQ(L2_FP32_GetDistFunc(dim), FP32_L2Sqr);
        ASSERT_EQ(L2_FP64_GetDistFunc(dim), FP64_L2Sqr);
        ASSERT_EQ(L2_BF16_GetDistFunc(dim), BF16_L2Sqr_LittleEndian);
        ASSERT_EQ(IP_FP32_GetDistFunc(dim), FP32_InnerProduct);
        ASSERT_EQ(IP_FP64_GetDistFunc(dim), FP64_InnerProduct);
        ASSERT_EQ(IP_BF16_GetDistFunc(dim), BF16_InnerProduct_LittleEndian);
    }
    for (size_t dim = 8; dim < 16; dim++) {
        ASSERT_EQ(L2_FP32_GetDistFunc(dim), FP32_L2Sqr);
        ASSERT_EQ(L2_BF16_GetDistFunc(dim), BF16_L2Sqr_LittleEndian);
        ASSERT_EQ(IP_FP32_GetDistFunc(dim), FP32_InnerProduct);
        ASSERT_EQ(IP_BF16_GetDistFunc(dim), BF16_InnerProduct_LittleEndian);
    }
    for (size_t dim = 16; dim < 32; dim++) {
        ASSERT_EQ(L2_BF16_GetDistFunc(dim), BF16_L2Sqr_LittleEndian);
        ASSERT_EQ(IP_BF16_GetDistFunc(dim), BF16_InnerProduct_LittleEndian);
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
        // Unet avx512f flag, so we'll choose the next optimization (AVX).
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
        v2[i] = vecsim_types::float_to_bf16(((float)i + 1.5));
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
        v2[i] = vecsim_types::float_to_bf16(((float)i + 1.5));
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

#endif // CPU_FEATURES_ARCH_X86_64
