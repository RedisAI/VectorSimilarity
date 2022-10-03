#include "gtest/gtest.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/IP/IP_SSE.h"
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/L2/L2_SSE.h"
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/vec_utils.h"

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
#include <tuple>

using spaces::dist_func_t;
namespace spaces_test {
// Functions for dimension % 16 == 0 for each optimization.
static dist_func_t<float> L2_dist_funcs_16Ext[] = {
    FP32_L2SqrSIMD16Ext_AVX512, FP32_L2SqrSIMD16Ext_AVX, FP32_L2SqrSIMD16Ext_SSE, FP32_L2Sqr};
static dist_func_t<float> IP_dist_funcs_16Ext[] = {
    FP32_InnerProductSIMD16Ext_AVX512, FP32_InnerProductSIMD16Ext_AVX,
    FP32_InnerProductSIMD16Ext_SSE, FP32_InnerProduct};
// Functions for dimension % 4 == 0 for each optimization.
static dist_func_t<float> L2_dist_funcs_4Ext[] = {FP32_L2SqrSIMD4Ext_AVX512, FP32_L2SqrSIMD4Ext_AVX,
                                                  FP32_L2SqrSIMD4Ext_SSE, FP32_L2Sqr};
static dist_func_t<float> IP_dist_funcs_4Ext[] = {FP32_InnerProductSIMD4Ext_AVX512,
                                                  FP32_InnerProductSIMD4Ext_AVX,
                                                  FP32_InnerProductSIMD4Ext_SSE, FP32_InnerProduct};
// Function for dimension > 16, for each optimization.
static dist_func_t<float> L2_dist_funcs_16ExtResiduals[] = {
    FP32_L2SqrSIMD16ExtResiduals_AVX512, FP32_L2SqrSIMD16ExtResiduals_AVX,
    FP32_L2SqrSIMD16ExtResiduals_SSE, FP32_L2Sqr};
static dist_func_t<float> IP_dist_funcs_16ExtResiduals[] = {
    FP32_InnerProductSIMD16ExtResiduals_AVX512, FP32_InnerProductSIMD16ExtResiduals_AVX,
    FP32_InnerProductSIMD16ExtResiduals_SSE, FP32_InnerProduct};
// Function for dimension < 16, for each optimization.
static dist_func_t<float> L2_dist_funcs_4ExtResiduals[] = {
    FP32_L2SqrSIMD4ExtResiduals_AVX512, FP32_L2SqrSIMD4ExtResiduals_AVX,
    FP32_L2SqrSIMD4ExtResiduals_SSE, FP32_L2Sqr};
static dist_func_t<float> IP_dist_funcs_4ExtResiduals[] = {
    FP32_InnerProductSIMD4ExtResiduals_AVX512, FP32_InnerProductSIMD4ExtResiduals_AVX,
    FP32_InnerProductSIMD4ExtResiduals_SSE, FP32_InnerProduct};
} // namespace spaces_test

class FP32SpacesOptimizationTest
    : public testing::TestWithParam<std::tuple<size_t, dist_func_t<float> *>> {};

TEST_P(FP32SpacesOptimizationTest, FP32DistanceFunctionTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = std::get<0>(GetParam());
    float v[dim];
    float v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
        v2[i] = (float)(i + 1.5);
    }

    dist_func_t<float> *arch_opt_funcs = std::get<1>(GetParam());
    float baseline = arch_opt_funcs[3](v, v2, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, arch_opt_funcs[0](v, v2, dim));
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, arch_opt_funcs[1](v, v2, dim));
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, arch_opt_funcs[2](v, v2, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}
INSTANTIATE_TEST_SUITE_P(
    FP32DimNOptFuncs, FP32SpacesOptimizationTest,
    testing::Values(std::make_tuple(16, spaces_test::L2_dist_funcs_16Ext),
                    std::make_tuple(16, spaces_test::IP_dist_funcs_16Ext),
                    std::make_tuple(20, spaces_test::L2_dist_funcs_4Ext),
                    std::make_tuple(20, spaces_test::IP_dist_funcs_4Ext),
                    std::make_tuple(17, spaces_test::L2_dist_funcs_16ExtResiduals),
                    std::make_tuple(17, spaces_test::IP_dist_funcs_16ExtResiduals),
                    std::make_tuple(9, spaces_test::L2_dist_funcs_4ExtResiduals),
                    std::make_tuple(9, spaces_test::IP_dist_funcs_4ExtResiduals)));

namespace spaces_test {
// Functions for dimension % 8 == 0 for each optimization.
static dist_func_t<double> L2_dist_funcs_8Ext[] = {NULL, FP64_L2SqrSIMD8Ext_AVX,
                                                   FP64_L2SqrSIMD8Ext_SSE, FP64_L2Sqr};
static dist_func_t<double> IP_dist_funcs_8Ext[] = {NULL, FP64_InnerProductSIMD8Ext_AVX, FP64_InnerProductSIMD8Ext_SSE,
                                                   FP64_InnerProduct};
// Functions for dimension % 2 == 0 for each optimization.
static dist_func_t<double> L2_dist_funcs_2Ext[] = {NULL, FP64_L2SqrSIMD2Ext_AVX,
                                                   FP64_L2SqrSIMD2Ext_SSE, FP64_L2Sqr};
static dist_func_t<double> IP_dist_funcs_2Ext[] = {NULL, FP64_InnerProductSIMD2Ext_AVX, FP64_InnerProductSIMD2Ext_SSE,
                                                   FP64_InnerProduct};
// Function for dim > 8  && dim % 8 < 2, for each optimization.
static dist_func_t<double> L2_dist_funcs_8ExtResiduals[] = {
    NULL, FP64_L2SqrSIMD8ExtResiduals_AVX, FP64_L2SqrSIMD8ExtResiduals_SSE, FP64_L2Sqr};
static dist_func_t<double> IP_dist_funcs_8ExtResiduals[] = {
    NULL, FP64_InnerProductSIMD8ExtResiduals_AVX, FP64_InnerProductSIMD8ExtResiduals_SSE, FP64_InnerProduct};
// Function for 2 < dimension < 8, dim %2 != 0 for each optimization.
static dist_func_t<double> L2_dist_funcs_2ExtResiduals[] = {
    NULL, FP64_L2SqrSIMD2ExtResiduals_AVX, FP64_L2SqrSIMD2ExtResiduals_SSE, FP64_L2Sqr};
static dist_func_t<double> IP_dist_funcs_2ExtResiduals[] = {
    NULL, FP64_InnerProductSIMD2ExtResiduals_AVX, FP64_InnerProductSIMD2ExtResiduals_SSE, FP64_InnerProduct};
} // namespace spaces_test

class FP64SpacesOptimizationTest
    : public testing::TestWithParam<std::tuple<size_t, dist_func_t<double> *>> {};

TEST_P(FP64SpacesOptimizationTest, FP64DistanceFunctionTest) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = std::get<0>(GetParam());
    double v[dim];
    double v2[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
        v2[i] = (double)(i + 1.5);
    }

    dist_func_t<double> *arch_opt_funcs = std::get<1>(GetParam());
    double baseline = arch_opt_funcs[3](v, v2, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512: // TODO remove condition when all optimization are implemented
        if (arch_opt_funcs[0]) {
            ASSERT_EQ(baseline, arch_opt_funcs[0](v, v2, dim));
        }
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, arch_opt_funcs[1](v, v2, dim));
        
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, arch_opt_funcs[2](v, v2, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}
INSTANTIATE_TEST_SUITE_P(
    FP64DimNOptFuncs, FP64SpacesOptimizationTest,
    testing::Values(std::make_tuple(8, spaces_test::L2_dist_funcs_8Ext),
                    std::make_tuple(8, spaces_test::IP_dist_funcs_8Ext),
                    std::make_tuple(10, spaces_test::L2_dist_funcs_2Ext),
                    std::make_tuple(10, spaces_test::IP_dist_funcs_2Ext),
                    std::make_tuple(17, spaces_test::L2_dist_funcs_8ExtResiduals),
                    std::make_tuple(17, spaces_test::IP_dist_funcs_8ExtResiduals),
                    std::make_tuple(7, spaces_test::L2_dist_funcs_2ExtResiduals),
                    std::make_tuple(7, spaces_test::IP_dist_funcs_2ExtResiduals)));

#endif // CPU_FEATURES_ARCH_X86_64

#endif // M1/X86_64
