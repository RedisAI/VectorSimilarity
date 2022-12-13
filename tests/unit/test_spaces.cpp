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

using spaces::dist_func_t;

namespace spaces_test {
// Each array contains all the possible architecture optimization related to one dimension
// optimization. For example: L2_dist_funcs_16Ext[ARCH_OPT_NONE = 0] = FP32_L2Sqr,
// L2_dist_funcs_16Ext[ARCH_OPT_SSE = 1] = FP32_L2SqrSIMD16Ext_SSE, etc.

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
} // namespace spaces_test

// In the 4 next tests we test #space_#data_type_GetDistFunc for each combination of data type and
// space. For a dimension that can't be optimized (dimensions[0]), GetDistFunc always returns the
// naive calculation function, regardless of the supported architecture optimization.
// fp32_L2_dist_funcs_array[i] is associated with the architecture optimizations that match
// dimension[i]. Instead of defining an array that all of its entries would be the same
// naive function and put it in fp32_L2_dist_funcs_array[0],
// we manually match GetDistFunc(dim = NO_OPTIMIZATION, arch = any) to the naive function,
// and start to iterate dimensions array from a dimension that can be splitted to optimize the
// calculations, according to the hardware capabilities.
TEST_F(SpacesTest, fp32_L2_GetDistFunc) {
    using namespace spaces_test;
    using namespace spaces;
    // fp32_L2_dist_funcs_array[i][j] is a pointer to a function that match CalculationGuideline[i]
    // and architecture optimization Arch_Optimization[j]
    static dist_func_t<float> *fp32_L2_dist_funcs_array[] = {
        nullptr, L2_dist_funcs_16Ext, L2_dist_funcs_4Ext, L2_dist_funcs_16ExtResiduals,
        L2_dist_funcs_4ExtResiduals};
    static size_t dimensions[] = {
        3,  // NO_OPTIMIZATION
        16, // % 16 ==0,
        20, //% 4 == 0,
        17, //>16, % 16 < 4 ,
        5,  //> 4  ]
    };
    /* enum Arch_Optimization {
    ARCH_OPT_NONE,
    ARCH_OPT_SSE,
    ARCH_OPT_AVX,
    ARCH_OPT_AVX512_F,
    ARCH_OPT_AVX512_DQ // Relevant only to fp64
}; */
    // For each arch
    for (int arch = ARCH_OPT_NONE; arch <= ARCH_OPT_AVX512_F; ++arch) {
        ASSERT_EQ(L2_FP32_GetDistFunc(dimensions[NO_OPTIMIZATION], (Arch_Optimization)arch),
                  FP32_L2Sqr);
        // For each dimension optimization
        for (int dim_opt = SPLIT_TO_512_BITS; dim_opt <= SPLIT_TO_512_128_BITS_WITH_RESIDUALS;
             ++dim_opt) {
            // Check we choose the correct function
            ASSERT_EQ(L2_FP32_GetDistFunc(dimensions[dim_opt], (Arch_Optimization)arch),
                      fp32_L2_dist_funcs_array[dim_opt][arch]);
        }
    }
}

TEST_F(SpacesTest, fp32_InnerProduct_GetDistFunc) {
    using namespace spaces_test;
    using namespace spaces;
    // fp32_IP_dist_funcs_array[i][j] is a pointer to a function that match CalculationGuideline[i]
    // and architecture optimization Arch_Optimization[j]
    static dist_func_t<float> *fp32_IP_dist_funcs_array[] = {
        nullptr, IP_dist_funcs_16Ext, IP_dist_funcs_4Ext, IP_dist_funcs_16ExtResiduals,
        IP_dist_funcs_4ExtResiduals};
    static size_t dimensions[] = {
        3,  // NO_OPTIMIZATION
        16, // % 16 ==0,
        20, //% 4 == 0,
        17, //>16, % 16 < 4 ,
        5,  //> 4  ]
    };
    /* enum Arch_Optimization {
    ARCH_OPT_NONE,
    ARCH_OPT_SSE,
    ARCH_OPT_AVX,
    ARCH_OPT_AVX512_F,
    ARCH_OPT_AVX512_DQ // Relevant only to fp64
}; */
    // For each arch
    for (int arch = ARCH_OPT_NONE; arch <= ARCH_OPT_AVX512_F; ++arch) {
        ASSERT_EQ(IP_FP32_GetDistFunc(dimensions[NO_OPTIMIZATION], (Arch_Optimization)arch),
                  FP32_InnerProduct);
        // For each dimension optimization
        for (int dim_opt = SPLIT_TO_512_BITS; dim_opt <= SPLIT_TO_512_128_BITS_WITH_RESIDUALS;
             ++dim_opt) {
            // Check we choose the correct function
            ASSERT_EQ(IP_FP32_GetDistFunc(dimensions[dim_opt], (Arch_Optimization)arch),
                      fp32_IP_dist_funcs_array[dim_opt][arch]);
        }
    }
}

TEST_F(SpacesTest, fp64_L2_GetDistFunc) {
    using namespace spaces_test;
    using namespace spaces;
    // fp64_L2_dist_funcs_array[i][j] is a pointer to a function that match CalculationGuideline[i]
    // and architecture optimization Arch_Optimization[j]
    static dist_func_t<double> *fp64_L2_dist_funcs_array[] = {
        nullptr, L2_dist_funcs_8Ext, L2_dist_funcs_2Ext, L2_dist_funcs_8ExtResiduals,
        L2_dist_funcs_2ExtResiduals};
    static size_t dimensions[] = {
        1,  // NO_OPTIMIZATION
        16, // % 8 ==0,
        20, //% 2 == 0,
        17, //>8, % 8 < 2 ,
        5,  //> 2  ]
    };
    /* enum Arch_Optimization {
    ARCH_OPT_NONE,
    ARCH_OPT_SSE,
    ARCH_OPT_AVX,
    ARCH_OPT_AVX512_F,
    ARCH_OPT_AVX512_DQ
}; */
    // For each arch
    for (int arch = ARCH_OPT_NONE; arch <= ARCH_OPT_AVX512_DQ; ++arch) {
        ASSERT_EQ(L2_FP64_GetDistFunc(dimensions[NO_OPTIMIZATION], (Arch_Optimization)arch),
                  FP64_L2Sqr);
        // For each dimension optimization
        for (int dim_opt = SPLIT_TO_512_BITS; dim_opt <= SPLIT_TO_512_128_BITS_WITH_RESIDUALS;
             ++dim_opt) {
            // Check we choose the correct function
            ASSERT_EQ(L2_FP64_GetDistFunc(dimensions[dim_opt], (Arch_Optimization)arch),
                      fp64_L2_dist_funcs_array[dim_opt][arch]);
        }
    }
}

TEST_F(SpacesTest, fp64_InnerProduct_GetDistFunc) {
    using namespace spaces_test;
    using namespace spaces;
    // fp64_IP_dist_funcs_array[i][j] is a pointer to a function that match CalculationGuideline[i]
    // and architecture optimization Arch_Optimization[j]
    static dist_func_t<double> *fp64_IP_dist_funcs_array[] = {
        nullptr, IP_dist_funcs_8Ext, IP_dist_funcs_2Ext, IP_dist_funcs_8ExtResiduals,
        IP_dist_funcs_2ExtResiduals};
    static size_t dimensions[] = {
        1,  // NO_OPTIMIZATION
        16, // % 8 ==0,
        20, //% 2 == 0,
        17, //>8, % 8 < 2 ,
        5,  //> 2  ]
    };
    /* enum Arch_Optimization {
    ARCH_OPT_NONE,
    ARCH_OPT_SSE,
    ARCH_OPT_AVX,
    ARCH_OPT_AVX512_F,
    ARCH_OPT_AVX512_DQ
}; */
    // For each arch
    for (int arch = ARCH_OPT_NONE; arch <= ARCH_OPT_AVX512_DQ; ++arch) {
        ASSERT_EQ(IP_FP64_GetDistFunc(dimensions[NO_OPTIMIZATION], (Arch_Optimization)arch),
                  FP64_InnerProduct);
        // For each dimension optimization
        for (int dim_opt = SPLIT_TO_512_BITS; dim_opt <= SPLIT_TO_512_128_BITS_WITH_RESIDUALS;
             ++dim_opt) {
            // Check we choose the correct function
            ASSERT_EQ(IP_FP64_GetDistFunc(dimensions[dim_opt], (Arch_Optimization)arch),
                      fp64_IP_dist_funcs_array[dim_opt][arch]);
        }
    }
}

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
    case ARCH_OPT_AVX512_DQ:
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
    FP32DimNOptFuncs, FP32SpacesOptimizationTest,
    testing::Values(std::make_pair(16, spaces_test::L2_dist_funcs_16Ext),
                    std::make_pair(16, spaces_test::IP_dist_funcs_16Ext),
                    std::make_pair(20, spaces_test::L2_dist_funcs_4Ext),
                    std::make_pair(20, spaces_test::IP_dist_funcs_4Ext),
                    std::make_pair(17, spaces_test::L2_dist_funcs_16ExtResiduals),
                    std::make_pair(17, spaces_test::IP_dist_funcs_16ExtResiduals),
                    std::make_pair(9, spaces_test::L2_dist_funcs_4ExtResiduals),
                    std::make_pair(9, spaces_test::IP_dist_funcs_4ExtResiduals)));

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

#endif // M1/X86_64
