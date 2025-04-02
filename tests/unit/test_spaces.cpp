/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <utility>

#include "gtest/gtest.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/IP/IP_SSE.h"
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX512DQ.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/L2/L2_SSE.h"
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/L2/L2_AVX512DQ.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/functions/NEON.h"
#include "VecSim/spaces/functions/SVE.h"
#include "VecSim/spaces/functions/SVE2.h"

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

using spaces::dist_func_t;
namespace spaces_test {
// Each array contains all the possible architecture optimization related to one dimension
// optimization. For example: L2_dist_funcs_16Ext[ARCH_OPT_NONE = 0] = FP32_L2Sqr,
// L2_dist_funcs_16Ext[ARCH_OPT_SSE = 1] = FP32_L2SqrSIMD16Ext_SSE, etc.

#ifdef CPU_FEATURES_ARCH_X86_64
// Functions for dimension % 16 == 0 for each optimization.
static dist_func_t<float> L2_dist_funcs_16Ext[] = {
    FP32_L2Sqr, FP32_L2SqrSIMD16Ext_SSE, FP32_L2SqrSIMD16Ext_AVX, FP32_L2SqrSIMD16Ext_AVX512};
static dist_func_t<float> IP_dist_funcs_16Ext[] = {
    FP32_InnerProduct, FP32_InnerProductSIMD16Ext_SSE, FP32_InnerProductSIMD16Ext_AVX,
    FP32_InnerProductSIMD16Ext_AVX512};
// Functions for dimension % 4 == 0 for each optimization.
static dist_func_t<float> L2_dist_funcs_4Ext[] = {
    FP32_L2Sqr, FP32_L2SqrSIMD4Ext_SSE, FP32_L2SqrSIMD4Ext_AVX, FP32_L2SqrSIMD4Ext_AVX512};
static dist_func_t<float> IP_dist_funcs_4Ext[] = {FP32_InnerProduct, FP32_InnerProductSIMD4Ext_SSE,
                                                  FP32_InnerProductSIMD4Ext_AVX,
                                                  FP32_InnerProductSIMD4Ext_AVX512};
// Function for dimension > 16, for each optimization.
static dist_func_t<float> L2_dist_funcs_16ExtResiduals[] = {
    FP32_L2Sqr, FP32_L2SqrSIMD16ExtResiduals_SSE, FP32_L2SqrSIMD16ExtResiduals_AVX,
    FP32_L2SqrSIMD16ExtResiduals_AVX512};
static dist_func_t<float> IP_dist_funcs_16ExtResiduals[] = {
    FP32_InnerProduct, FP32_InnerProductSIMD16ExtResiduals_SSE,
    FP32_InnerProductSIMD16ExtResiduals_AVX, FP32_InnerProductSIMD16ExtResiduals_AVX512};
// Function for dimension < 16, for each optimization.
static dist_func_t<float> L2_dist_funcs_4ExtResiduals[] = {
    FP32_L2Sqr, FP32_L2SqrSIMD4ExtResiduals_SSE, FP32_L2SqrSIMD4ExtResiduals_AVX,
    FP32_L2SqrSIMD4ExtResiduals_AVX512};
static dist_func_t<float> IP_dist_funcs_4ExtResiduals[] = {
    FP32_InnerProduct, FP32_InnerProductSIMD4ExtResiduals_SSE,
    FP32_InnerProductSIMD4ExtResiduals_AVX, FP32_InnerProductSIMD4ExtResiduals_AVX512};

// Functions for dimension % 8 == 0 for each optimization.
static dist_func_t<double> L2_dist_funcs_8Ext[] = {
    FP64_L2Sqr, FP64_L2SqrSIMD8Ext_SSE, FP64_L2SqrSIMD8Ext_AVX, FP64_L2SqrSIMD8Ext_AVX512,
    FP64_L2SqrSIMD8Ext_AVX512};
static dist_func_t<double> IP_dist_funcs_8Ext[] = {
    FP64_InnerProduct, FP64_InnerProductSIMD8Ext_SSE, FP64_InnerProductSIMD8Ext_AVX,
    FP64_InnerProductSIMD8Ext_AVX512, FP64_InnerProductSIMD8Ext_AVX512};
// Functions for dimension % 2 == 0 for each optimization.
static dist_func_t<double> L2_dist_funcs_2Ext[] = {
    FP64_L2Sqr, FP64_L2SqrSIMD2Ext_SSE, FP64_L2SqrSIMD2Ext_AVX, FP64_L2SqrSIMD2Ext_AVX512_noDQ,
    FP64_L2SqrSIMD2Ext_AVX512};
static dist_func_t<double> IP_dist_funcs_2Ext[] = {
    FP64_InnerProduct, FP64_InnerProductSIMD2Ext_SSE, FP64_InnerProductSIMD2Ext_AVX,
    FP64_InnerProductSIMD2Ext_AVX512_noDQ, FP64_InnerProductSIMD2Ext_AVX512};
// Function for dim > 8  && dim % 8 < 2, for each optimization.
static dist_func_t<double> L2_dist_funcs_8ExtResiduals[] = {
    FP64_L2Sqr, FP64_L2SqrSIMD8ExtResiduals_SSE, FP64_L2SqrSIMD8ExtResiduals_AVX,
    FP64_L2SqrSIMD8ExtResiduals_AVX512, FP64_L2SqrSIMD8ExtResiduals_AVX512};
static dist_func_t<double> IP_dist_funcs_8ExtResiduals[] = {
    FP64_InnerProduct, FP64_InnerProductSIMD8ExtResiduals_SSE,
    FP64_InnerProductSIMD8ExtResiduals_AVX, FP64_InnerProductSIMD8ExtResiduals_AVX512,
    FP64_InnerProductSIMD8ExtResiduals_AVX512};
// Function for 2 < dimension < 8, dim %2 != 0 for each optimization.
static dist_func_t<double> L2_dist_funcs_2ExtResiduals[] = {
    FP64_L2Sqr, FP64_L2SqrSIMD2ExtResiduals_SSE, FP64_L2SqrSIMD2ExtResiduals_AVX,
    FP64_L2SqrSIMD2ExtResiduals_AVX512_noDQ, FP64_L2SqrSIMD2ExtResiduals_AVX512};
static dist_func_t<double> IP_dist_funcs_2ExtResiduals[] = {
    FP64_InnerProduct, FP64_InnerProductSIMD2ExtResiduals_SSE,
    FP64_InnerProductSIMD2ExtResiduals_AVX, FP64_InnerProductSIMD2ExtResiduals_AVX512_noDQ,
    FP64_InnerProductSIMD2ExtResiduals_AVX512};
#endif

#ifdef CPU_FEATURES_ARCH_AARCH64
static dist_func_t<float> L2_dist_funcs_arm16[] = {FP32_L2Sqr,
#ifdef OPT_NEON
                                                   spaces::Choose_FP32_L2_implementation_NEON(16),
#endif
#ifdef OPT_SVE
                                                   spaces::Choose_FP32_L2_implementation_SVE(16),
#endif
#ifdef OPT_SVE2
                                                   spaces::Choose_FP32_L2_implementation_SVE2(16)
#endif
};
static dist_func_t<float> IP_dist_funcs_arm16[] = {FP32_InnerProduct,
#ifdef OPT_NEON
                                                   spaces::Choose_FP32_IP_implementation_NEON(16),
#endif
#ifdef OPT_SVE
                                                   spaces::Choose_FP32_IP_implementation_SVE(16),
#endif
#ifdef OPT_SVE2
                                                   spaces::Choose_FP32_IP_implementation_SVE2(16)
#endif
};
static dist_func_t<float> L2_dist_funcs_arm8[] = {FP32_L2Sqr,
#ifdef OPT_NEON
                                                  spaces::Choose_FP32_L2_implementation_NEON(8),
#endif
#ifdef OPT_SVE
                                                  spaces::Choose_FP32_L2_implementation_SVE(8),
#endif
#ifdef OPT_SVE2
                                                  spaces::Choose_FP32_L2_implementation_SVE2(8)
#endif
};
static dist_func_t<float> IP_dist_funcs_arm8[] = {FP32_InnerProduct,
#ifdef OPT_NEON
                                                  spaces::Choose_FP32_IP_implementation_NEON(8),
#endif
#ifdef OPT_SVE
                                                  spaces::Choose_FP32_IP_implementation_SVE(8),
#endif
#ifdef OPT_SVE2
                                                  spaces::Choose_FP32_IP_implementation_SVE2(8)
#endif

};
static dist_func_t<float> L2_dist_funcs_arm4[] = {FP32_L2Sqr,
#ifdef OPT_NEON
                                                  spaces::Choose_FP32_L2_implementation_NEON(4),
#endif
#ifdef OPT_SVE
                                                  spaces::Choose_FP32_L2_implementation_SVE(4),
#endif
#ifdef OPT_SVE2
                                                  spaces::Choose_FP32_L2_implementation_SVE2(4)
#endif
};
static dist_func_t<float> IP_dist_funcs_arm4[] = {FP32_InnerProduct,
#ifdef OPT_NEON
                                                  spaces::Choose_FP32_IP_implementation_NEON(4),
#endif
#ifdef OPT_SVE
                                                  spaces::Choose_FP32_IP_implementation_SVE(4),
#endif
#ifdef OPT_SVE2
                                                  spaces::Choose_FP32_IP_implementation_SVE2(4)
#endif
};

#endif

} // namespace spaces_test

class FP32SpacesOptimizationTest
    : public testing::TestWithParam<std::pair<size_t, dist_func_t<float> *>> {};

TEST_P(FP32SpacesOptimizationTest, FP32DistanceFunctionTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = GetParam().first;
TEST_P(FP32SpacesOptimizationTest, FP32L2SqrTest) {
    auto optimization = getCpuOptimizationFeatures();
    size_t dim = GetParam();
    float v[dim];
    float v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
        v2[i] = (float)(i + 1.5);
    }

    dist_func_t<float> *arch_opt_funcs = GetParam().second;
    float baseline = arch_opt_funcs[ARCH_OPT_NONE](v, v2, dim);
    switch (optimization) {
#ifdef CPU_FEATURES_ARCH_X86_64
    case ARCH_OPT_AVX512_DQ:
    case ARCH_OPT_AVX512_F:
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_AVX512_F](v, v2, dim));
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_AVX](v, v2, dim));
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_SSE](v, v2, dim));
        break;
#endif
#ifdef CPU_FEATURES_ARCH_AARCH64
    case ARCH_OPT_SVE2:
#ifdef OPT_SVE2
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_SVE2](v, v2, dim));
#endif
    case ARCH_OPT_SVE:
#ifdef OPT_SVE
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_SVE](v, v2, dim));
#endif
    case ARCH_OPT_NEON:
#ifdef OPT_NEON
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_NEON](v, v2, dim));
#endif
        break;
#endif
    default:
        ASSERT_TRUE(false);
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

#ifdef CPU_FEATURES_ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    FP32DimNOptFuncs, FP32SpacesOptimizationTest,
    testing::Values(std::make_pair(16, spaces_test::L2_dist_funcs_16Ext),
                    std::make_pair(16, spaces_test::IP_dist_funcs_16Ext),
                    std::make_pair(20, spaces_test::L2_dist_funcs_4Ext),
                    std::make_pair(20, spaces_test::IP_dist_funcs_4Ext),
                    std::make_pair(17, spaces_test::L2_dist_funcs_16ExtResiduals),
                    std::make_pair(17, spaces_test::IP_dist_funcs_16ExtResiduals),
                    std::make_pair(9, spaces_test::L2_dist_funcs_4ExtResiduals),
                    std::make_pair(9, spaces_test::IP_dist_funcs_4ExtResiduals)));
#endif

#ifdef CPU_FEATURES_ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(FP32DimNOptFuncs, FP32SpacesOptimizationTest,
                         testing::Values(std::make_pair(16, spaces_test::L2_dist_funcs_arm16),
                                         std::make_pair(16, spaces_test::IP_dist_funcs_arm16),
                                         std::make_pair(8, spaces_test::L2_dist_funcs_arm8),
                                         std::make_pair(8, spaces_test::IP_dist_funcs_arm8),
                                         std::make_pair(4, spaces_test::L2_dist_funcs_arm4),
                                         std::make_pair(4, spaces_test::IP_dist_funcs_arm4)));
#endif

#ifdef CPU_FEATURES_ARCH_X86_64
class FP64SpacesOptimizationTest
    : public testing::TestWithParam<std::pair<size_t, dist_func_t<double> *>> {};

TEST_P(FP64SpacesOptimizationTest, FP64DistanceFunctionTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = GetParam().first;
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

    dist_func_t<double> *arch_opt_funcs = GetParam().second;
    double baseline = arch_opt_funcs[ARCH_OPT_NONE](v, v2, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512_DQ:
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_AVX512_DQ](v, v2, dim));
    case ARCH_OPT_AVX512_F:
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_AVX512_F](v, v2, dim));
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_AVX](v, v2, dim));
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_SSE](v, v2, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}
INSTANTIATE_TEST_SUITE_P(
    FP64DimNOptFuncs, FP64SpacesOptimizationTest,
    testing::Values(std::make_pair(8, spaces_test::L2_dist_funcs_8Ext),
                    std::make_pair(8, spaces_test::IP_dist_funcs_8Ext),
                    std::make_pair(10, spaces_test::L2_dist_funcs_2Ext),
                    std::make_pair(10, spaces_test::IP_dist_funcs_2Ext),
                    std::make_pair(17, spaces_test::L2_dist_funcs_8ExtResiduals),
                    std::make_pair(17, spaces_test::IP_dist_funcs_8ExtResiduals),
                    std::make_pair(7, spaces_test::L2_dist_funcs_2ExtResiduals),
                    std::make_pair(7, spaces_test::IP_dist_funcs_2ExtResiduals)));

#endif // CPU_FEATURES_ARCH_X86_64
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
    // Turn off advanced fp16 flags. They will be tested in the next test.
#if defined(CPU_FEATURES_ARCH_X86_64)
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
#endif
    unsigned char alignment = 0;
    arch_opt_func = IP_FP16_GetDistFunc(dim, &alignment, &optimization);
    ASSERT_EQ(arch_opt_func, FP16_InnerProduct)
        << "Unexpected distance function chosen for dim " << dim;
    ASSERT_EQ(baseline, arch_opt_func(v1, v2, dim)) << "F16C with dim " << dim;
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
#ifdef OPT_AVX512_FP16_VL
class FP16SpacesOptimizationTestAdvanced : public testing::TestWithParam<size_t> {};

TEST_P(FP16SpacesOptimizationTestAdvanced, FP16InnerProductTestAdv) {
    auto optimization = cpu_features::GetX86Info().features;
    if (optimization.avx512_fp16 && optimization.avx512vl) {
        size_t dim = GetParam();
        float16 v1[dim], v2[dim];

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-0.99, 0.99);

        _Float16 baseline = 0;
        for (size_t i = 0; i < dim; i++) {
            float val1 = (dis(gen));
            float val2 = (dis(gen));
            v1[i] = vecsim_types::FP32_to_FP16((val1));
            v2[i] = vecsim_types::FP32_to_FP16((val2));

            baseline += static_cast<_Float16>(val1) * static_cast<_Float16>(val2);
        }
        baseline = _Float16(1) - baseline;

        auto expected_alignment = [](size_t reg_bit_size, size_t dim) {
            size_t elements_in_reg = reg_bit_size / sizeof(float16) / 8;
            return (dim % elements_in_reg == 0) ? elements_in_reg * sizeof(float16) : 0;
        };

        dist_func_t<float> arch_opt_func;
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
}

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

// Start from a 32 multiplier
INSTANTIATE_TEST_SUITE_P(, FP16SpacesOptimizationTestAdvanced,
                         testing::Range(512UL, 512 + 32UL + 1));

#endif

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

INSTANTIATE_TEST_SUITE_P(UINT8OptFuncs, UINT8SpacesOptimizationTest,
                         testing::Range(32UL, 64 * 2UL + 1));
