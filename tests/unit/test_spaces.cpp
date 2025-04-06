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
