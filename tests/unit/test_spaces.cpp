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
static dist_func_t<float> *build_arm_funcs_array_fp32(size_t dim, bool is_ip) {
    static dist_func_t<float> funcs[ARCH_OPT_SIZE] = {nullptr};
    cpu_features::Aarch64Features features = cpu_features::GetAarch64Info().features;
    // Always add baseline implementation
    funcs[ARCH_OPT_NONE] = is_ip ? FP32_InnerProduct : FP32_L2Sqr;

// Add NEON implementation if available
#ifdef OPT_NEON
    if (features.asimd) {
        funcs[ARCH_OPT_NEON] = is_ip ? spaces::Choose_FP32_IP_implementation_NEON(dim)
                                     : spaces::Choose_FP32_L2_implementation_NEON(dim);
    }
#endif

// Add SVE implementation if available
#ifdef OPT_SVE
    if (features.sve) {
        funcs[ARCH_OPT_SVE] = is_ip ? spaces::Choose_FP32_IP_implementation_SVE(dim)
                                    : spaces::Choose_FP32_L2_implementation_SVE(dim);
    }
#endif

// Add SVE2 implementation if available
#ifdef OPT_SVE2
    if (features.sve2) {
        funcs[ARCH_OPT_SVE2] = is_ip ? spaces::Choose_FP32_IP_implementation_SVE2(dim)
                                     : spaces::Choose_FP32_L2_implementation_SVE2(dim);
    }
#endif

    return funcs;
}

static dist_func_t<double> *build_arm_funcs_array_fp64(size_t dim, bool is_ip) {
    static dist_func_t<double> funcs[ARCH_OPT_SIZE] = {nullptr};
    cpu_features::Aarch64Features features = cpu_features::GetAarch64Info().features;
    // Always add baseline implementation
    funcs[ARCH_OPT_NONE] = is_ip ? FP64_InnerProduct : FP64_L2Sqr;

// Add NEON implementation if available
#ifdef OPT_NEON
    if (features.asimd) {
        funcs[ARCH_OPT_NEON] = is_ip ? spaces::Choose_FP64_IP_implementation_NEON(dim)
                                     : spaces::Choose_FP64_L2_implementation_NEON(dim);
    }
#endif

// Add SVE implementation if available
#ifdef OPT_SVE
    if (features.sve) {
        funcs[ARCH_OPT_SVE] = is_ip ? spaces::Choose_FP64_IP_implementation_SVE(dim)
                                    : spaces::Choose_FP64_L2_implementation_SVE(dim);
    }
#endif

// Add SVE2 implementation if available
#ifdef OPT_SVE2
    if (features.sve2) {
        funcs[ARCH_OPT_SVE2] = is_ip ? spaces::Choose_FP64_IP_implementation_SVE2(dim)
                                     : spaces::Choose_FP64_L2_implementation_SVE2(dim);
    }
#endif

    return funcs;
}

#endif

} // namespace spaces_test

class FP32SpacesOptimizationTest
#ifdef CPU_FEATURES_ARCH_X86_64
    : public testing::TestWithParam<std::pair<size_t, dist_func_t<float> *>> {
};
#elif defined(CPU_FEATURES_ARCH_AARCH64)
    : public testing::TestWithParam<std::pair<size_t, bool>> {
};
#endif

TEST_P(FP32SpacesOptimizationTest, FP32DistanceFunctionTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = GetParam().first;
    float v[dim];
    float v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
        v2[i] = (float)(i + 1.5);
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    dist_func_t<float> *arch_opt_funcs = GetParam().second;
#elif defined(CPU_FEATURES_ARCH_AARCH64)
    dist_func_t<float> *arch_opt_funcs =
        spaces_test::build_arm_funcs_array_fp32(dim, GetParam().second);
#endif
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
                         testing::Values(std::make_pair(16, true), // is_ip = true
                                         std::make_pair(16, false), std::make_pair(8, true),
                                         std::make_pair(8, false), std::make_pair(4, true),
                                         std::make_pair(4, false)));
#endif

#ifdef CPU_FEATURES_ARCH_X86_64
class FP64SpacesOptimizationTest
    : public testing::TestWithParam<std::pair<size_t, dist_func_t<double> *>> {};

TEST_P(FP64SpacesOptimizationTest, FP64DistanceFunctionTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = GetParam().first;
    double v[dim];
    double v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
        v2[i] = (double)(i + 1.5);
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    dist_func_t<double> *arch_opt_funcs = GetParam().second;
#elif defined(CPU_FEATURES_ARCH_AARCH64)
    dist_func_t<double> *arch_opt_funcs =
        spaces_test::build_arm_funcs_array_fp64(dim, GetParam().second);
#endif
    double baseline = arch_opt_funcs[ARCH_OPT_NONE](v, v2, dim);
    switch (optimization) {
#ifdef CPU_FEATURES_ARCH_X86_64

    case ARCH_OPT_AVX512_DQ:
        ASSERT_EQ(baseline, arch_opt_funcs[ARCH_OPT_AVX512_DQ](v, v2, dim));
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
