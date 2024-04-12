/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <utility>

#include "gtest/gtest.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"

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

#if defined(M1)

#elif defined(__x86_64)
#include "cpu_features_macros.h"
#ifdef CPU_FEATURES_ARCH_X86_64

using namespace spaces;

TEST_F(SpacesTest, smallDimChooser) {
    size_t dim = 5;

    // Verify that small dimensions gets the no optimization function
    ASSERT_EQ(L2_FP32_GetDistFunc(dim, ARCH_OPT_AVX512_F), FP32_L2Sqr);
    ASSERT_EQ(L2_FP64_GetDistFunc(dim, ARCH_OPT_AVX512_F), FP64_L2Sqr);
    ASSERT_EQ(IP_FP32_GetDistFunc(dim, ARCH_OPT_AVX512_F), FP32_InnerProduct);
    ASSERT_EQ(IP_FP64_GetDistFunc(dim, ARCH_OPT_AVX512_F), FP64_InnerProduct);
}

class FP32SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(FP32SpacesOptimizationTest, FP32L2SqrTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = GetParam();
    float v[dim];
    float v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
        v2[i] = (float)(i + 1.5);
    }

    dist_func_t<float> arch_opt_func;
    float baseline = FP32_L2Sqr(v, v2, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512_BW_VBMI2:
    case ARCH_OPT_AVX512_F:
        arch_opt_func = L2_FP32_GetDistFunc(dim, ARCH_OPT_AVX512_F);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
    case ARCH_OPT_AVX2:
    case ARCH_OPT_AVX:
        arch_opt_func = L2_FP32_GetDistFunc(dim, ARCH_OPT_AVX);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
    case ARCH_OPT_SSE3:
    case ARCH_OPT_SSE:
        arch_opt_func = L2_FP32_GetDistFunc(dim, ARCH_OPT_SSE);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
    case ARCH_OPT_NONE:
        arch_opt_func = L2_FP32_GetDistFunc(dim, ARCH_OPT_NONE);
        ASSERT_EQ(FP32_L2Sqr, arch_opt_func);
        break;
    default:
        FAIL();
    }
}

TEST_P(FP32SpacesOptimizationTest, FP32InnerProductTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = GetParam();
    float v[dim];
    float v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
        v2[i] = (float)(i + 1.5);
    }

    dist_func_t<float> arch_opt_func;
    float baseline = FP32_InnerProduct(v, v2, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512_BW_VBMI2:
    case ARCH_OPT_AVX512_F:
        arch_opt_func = IP_FP32_GetDistFunc(dim, ARCH_OPT_AVX512_F);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
    case ARCH_OPT_AVX2:
    case ARCH_OPT_AVX:
        arch_opt_func = IP_FP32_GetDistFunc(dim, ARCH_OPT_AVX);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
    case ARCH_OPT_SSE3:
    case ARCH_OPT_SSE:
        arch_opt_func = IP_FP32_GetDistFunc(dim, ARCH_OPT_SSE);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
    case ARCH_OPT_NONE:
        arch_opt_func = IP_FP32_GetDistFunc(dim, ARCH_OPT_NONE);
        ASSERT_EQ(FP32_InnerProduct, arch_opt_func);
        break;
    default:
        FAIL();
    }
}

INSTANTIATE_TEST_SUITE_P(FP32OptFuncs, FP32SpacesOptimizationTest, testing::Range(1UL, 16 * 2UL));

class FP64SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

TEST_P(FP64SpacesOptimizationTest, FP64L2SqrTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = GetParam();
    double v[dim];
    double v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
        v2[i] = (double)(i + 1.5);
    }

    dist_func_t<double> arch_opt_func;
    double baseline = FP64_L2Sqr(v, v2, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512_BW_VBMI2:
    case ARCH_OPT_AVX512_F:
        arch_opt_func = L2_FP64_GetDistFunc(dim, ARCH_OPT_AVX512_F);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
    case ARCH_OPT_AVX2:
    case ARCH_OPT_AVX:
        arch_opt_func = L2_FP64_GetDistFunc(dim, ARCH_OPT_AVX);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
    case ARCH_OPT_SSE3:
    case ARCH_OPT_SSE:
        arch_opt_func = L2_FP64_GetDistFunc(dim, ARCH_OPT_SSE);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
    case ARCH_OPT_NONE:
        arch_opt_func = L2_FP64_GetDistFunc(dim, ARCH_OPT_NONE);
        ASSERT_EQ(FP64_L2Sqr, arch_opt_func);
        break;
    default:
        FAIL();
    }
}

TEST_P(FP64SpacesOptimizationTest, FP64InnerProductTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = GetParam();
    double v[dim];
    double v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
        v2[i] = (double)(i + 1.5);
    }

    dist_func_t<double> arch_opt_func;
    double baseline = FP64_InnerProduct(v, v2, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512_BW_VBMI2:
    case ARCH_OPT_AVX512_F:
        arch_opt_func = IP_FP64_GetDistFunc(dim, ARCH_OPT_AVX512_F);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
    case ARCH_OPT_AVX2:
    case ARCH_OPT_AVX:
        arch_opt_func = IP_FP64_GetDistFunc(dim, ARCH_OPT_AVX);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
    case ARCH_OPT_SSE3:
    case ARCH_OPT_SSE:
        arch_opt_func = IP_FP64_GetDistFunc(dim, ARCH_OPT_SSE);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
    case ARCH_OPT_NONE:
        arch_opt_func = IP_FP64_GetDistFunc(dim, ARCH_OPT_NONE);
        ASSERT_EQ(FP64_InnerProduct, arch_opt_func);
        break;
    default:
        FAIL();
    }
}

INSTANTIATE_TEST_SUITE_P(FP64OptFuncs, FP64SpacesOptimizationTest, testing::Range(1UL, 8 * 2UL));

class BF16SpacesOptimizationTest : public testing::TestWithParam<size_t> {};

static float sanity_calc_IP(const bfloat16 *v1, const bfloat16 *v2, size_t dim) {
    float sanity_dist = 0;
    for (size_t i = 0; i < dim; i++) {
        float mul = bf16_to_float(v1[i]) * bf16_to_float(v2[i]);
        sanity_dist += mul;
    }
    return 1.0f - sanity_dist;
}

TEST_P(BF16SpacesOptimizationTest, BF16InnerProductTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = GetParam();
    bfloat16 v[dim];
    bfloat16 v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = float_to_bf16((float)i);
        v2[i] = float_to_bf16(((float)i + 1.5));
    }

    dist_func_t<float> arch_opt_func;
    float baseline = BF16_InnerProduct_LittleEndian(v, v2, dim);

    switch (optimization) {
    case ARCH_OPT_AVX512_BW_VBMI2:
        arch_opt_func = IP_BF16_GetDistFunc(dim, ARCH_OPT_AVX512_BW_VBMI2);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
    case ARCH_OPT_AVX512_F:
    case ARCH_OPT_AVX2:
        arch_opt_func = IP_BF16_GetDistFunc(dim, ARCH_OPT_AVX2);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
    case ARCH_OPT_AVX:
    case ARCH_OPT_SSE3:
        arch_opt_func = IP_BF16_GetDistFunc(dim, ARCH_OPT_SSE3);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
    case ARCH_OPT_SSE:
    case ARCH_OPT_NONE:
        arch_opt_func = IP_BF16_GetDistFunc(dim, ARCH_OPT_NONE);
        ASSERT_EQ(BF16_InnerProduct_LittleEndian, arch_opt_func);
        break;
    default:
        FAIL();
    }
}

static float sanity_calc_L2(const bfloat16 *v1, const bfloat16 *v2, size_t dim) {
    float sanity_dist = 0;
    for (size_t i = 0; i < dim; i++) {
        float diff = bf16_to_float(v1[i]) - bf16_to_float(v2[i]);
        sanity_dist += diff * diff;
    }
    return sanity_dist;
}

TEST_P(BF16SpacesOptimizationTest, BF16L2SqrTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = GetParam();
    bfloat16 v[dim];
    bfloat16 v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = float_to_bf16((float)i);
        v2[i] = float_to_bf16(((float)i + 1.5));
    }

    dist_func_t<float> arch_opt_func;
    float baseline = BF16_L2Sqr_LittleEndian(v, v2, dim);

    switch (optimization) {
    case ARCH_OPT_AVX512_BW_VBMI2:
        arch_opt_func = L2_BF16_GetDistFunc(dim, ARCH_OPT_AVX512_BW_VBMI2);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX512 with dim " << dim;
    case ARCH_OPT_AVX512_F:
    case ARCH_OPT_AVX2:
        arch_opt_func = L2_BF16_GetDistFunc(dim, ARCH_OPT_AVX2);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "AVX with dim " << dim;
    case ARCH_OPT_AVX:
    case ARCH_OPT_SSE3:
        arch_opt_func = L2_BF16_GetDistFunc(dim, ARCH_OPT_SSE3);
        ASSERT_EQ(baseline, arch_opt_func(v, v2, dim)) << "SSE with dim " << dim;
    case ARCH_OPT_SSE:
    case ARCH_OPT_NONE:
        arch_opt_func = L2_BF16_GetDistFunc(dim, ARCH_OPT_NONE);
        ASSERT_EQ(BF16_L2Sqr_LittleEndian, arch_opt_func);
        break;
    default:
        FAIL();
    }
}

INSTANTIATE_TEST_SUITE_P(BF16OptFuncs, BF16SpacesOptimizationTest, testing::Range(1UL, 32 * 2UL));

#endif // CPU_FEATURES_ARCH_X86_64

#endif // M1/X86_64
