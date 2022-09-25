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

using spaces::dist_func_t;
/****** templates distance function tests suite ******/
template <typename DistType>
class DistFuncTest : public testing::Test {
public:
    dist_func_t<DistType> dist_func;
};

using DistTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DistFuncTest, DistTypes);

/****** no optimization function tests suite ******/

TYPED_TEST(DistFuncTest, l2_no_optimization_func_test) {
    // Choose a dim that has no optimizations
    size_t dim = 5;

    TypeParam a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = TypeParam(i + 1.0);
        b[i] = TypeParam(i + 1.0);
    }

    TypeParam dist = L2Sqr(a, b, dim);
    ASSERT_EQ(dist, 0.0);
}

TYPED_TEST(DistFuncTest, ip_no_optimization_func_test) {
    // Choose a dim that has no optimizations
    size_t dim = 5;

    TypeParam a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = TypeParam(i + 1.5);
        b[i] = TypeParam(i + 1.5);
    }

    normalizeVector(a, dim);
    normalizeVector(b, dim);

    TypeParam dist = InnerProduct(a, b, dim);
    ASSERT_NEAR(dist, 0.0, 0.00000001);
}

class SpacesTest : public ::testing::Test {

protected:
    SpacesTest() {}

    ~SpacesTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};
#if defined(M1)

#elif defined(__x86_64)
#include "cpu_features_macros.h"
#ifdef CPU_FEATURES_ARCH_X86_64

/* template <size_t dim, typename DataType, typename DistType = DataType>
struct DistFuncParams {
    static VecSimType get_dim() { return dim; }
    typedef DataType data_t;
    typedef DistType dist_t;
};
template <typename dist_func_params_t>
class OptDistFuncTest : public testing::Test {
public:
};

#define TEST_DATA_T typename DistFuncParams::data_t
#define TEST_DIST_T typename DistFuncParams::dist_t
using DataTypeSet = ::testing::Types<DistFuncParams<16, float>, DistFuncParams<16, double>,
                                     DistFuncParams<20, float>, DistFuncParams<10, double>,
                                     DistFuncParams<17, float>, DistFuncParams<17, double>,
                                     DistFuncParams<7, float>, DistFuncParams<7, double>,
                                     >;

TYPED_TEST_CASE(DistFuncTest, DataTypeSet); */
// This test will trigger the "Residuals" function for dimension > 16, for each optimization.
TEST_F(SpacesTest, l2_17) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 17;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD16ExtResiduals_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD16ExtResiduals_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, L2SqrSIMD16ExtResiduals_SSE(v, v, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the "Residuals" function for dimension < 16, for each optimization.
TEST_F(SpacesTest, l2_9) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 9;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD4ExtResiduals_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD4ExtResiduals_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, L2SqrSIMD4ExtResiduals_SSE(v, v, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the "Residuals" function for dimension > 16, for each optimization.
TEST_F(SpacesTest, ip_17) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 17;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = InnerProduct(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD16ExtResiduals_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD16ExtResiduals_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD16ExtResiduals_SSE(v, v, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the "Residuals" function for dimension < 16, for each optimization.
TEST_F(SpacesTest, ip_9) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 9;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = InnerProduct(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD4ExtResiduals_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD4ExtResiduals_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD4ExtResiduals_SSE(v, v, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the function for dimension % 16 == 0 for each optimization.
TYPED_TEST(DistFuncTest, ip_16) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 16;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = InnerProduct(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD16Ext_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD16Ext_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD16Ext_SSE(v, v, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the function for dimension % 16 == 0 for each optimization.
TYPED_TEST(DistFuncTest, l2_16) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 16;
    TypeParam v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (TypeParam)i;
    }

    TypeParam baseline = L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:

    case ARCH_OPT_AVX:

    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, L2SqrSIMDsplit512Ext_SSE(v, v, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the function for dimension % 4 == 0 for each optimization.
TEST_F(SpacesTest, ip_20) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 20;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = InnerProduct(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD4Ext_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD4Ext_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, FP32_InnerProductSIMD4Ext_SSE(v, v, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the function for dimension % 4 == 0 for each optimization.
TEST_F(SpacesTest, l2_20) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 20;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD4Ext_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD4Ext_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, L2SqrSIMD4Ext_SSE(v, v, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the function for dimension % 2 == 0 for each optimization.
TEST_F(SpacesTest, l2_10_double) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 10;
    double v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
    }

    ASSERT_EQ(spaces::FP64_GetCalculationGuideline(dim), spaces::SPLIT_TO_512_128_BITS);
    double baseline = L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512: // TODO: add comparison when AVX and AVX512 is implemented
    case ARCH_OPT_AVX:
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, L2SqrSIMD2Ext_SSE(v, v, dim));
    case ARCH_OPT_NONE:
        break;
    default:
        ASSERT_TRUE(false);
    }
}
// This test will trigger the function for dim > 8  && dim % 8 < 2 for each optimization.
TEST_F(SpacesTest, l2_17_double) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 17;
    double v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
    }

    ASSERT_EQ(spaces::FP64_GetCalculationGuideline(dim), spaces::SPLIT_TO_512_BITS_WITH_RESIDUALS);
    double baseline = L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512: // TODO: add comparison when AVX and AVX512 is implemented
    case ARCH_OPT_AVX:
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, L2SqrSIMD8ExtResiduals_SSE(v, v, dim));
    case ARCH_OPT_NONE:
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the "Residuals" function for 2 < dimension < 8, dim %2 != 0 for each
// optimization.
TEST_F(SpacesTest, l2_7_double) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 7;
    double v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (double)i;
    }

    ASSERT_EQ(spaces::FP64_GetCalculationGuideline(dim),
              spaces::SPLIT_TO_512_128_BITS_WITH_RESIDUALS);
    double baseline = L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512: // TODO: add comparison when AVX and AVX512 is implemented
    case ARCH_OPT_AVX:
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, L2SqrSIMD2ExtResiduals_SSE(v, v, dim));
    case ARCH_OPT_NONE:
        break;
    default:
        ASSERT_TRUE(false);
    }
}

#endif // CPU_FEATURES_ARCH_X86_64

#endif // M1/X86_64
