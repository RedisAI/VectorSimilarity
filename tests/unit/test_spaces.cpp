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

/****** Get distance function tests  ******/

// No Optimization distance function.
TYPED_TEST(DistFuncTest, l2_no_optimization_get_func_test) {
    // Choose a dim that has no optimizations
    size_t dim = 1;
    VecSimMetric metric = VecSimMetric_L2;

    spaces::SetDistFunc(metric, dim, &(this->dist_func));

    // Casting to void * so both sides of the comparison would have the
    // same type.
    if (std::is_same<TypeParam, float>::value) {
        ASSERT_EQ(spaces::GetCalculationGuideline(dim), spaces::NO_OPTIMIZATION);
        ASSERT_EQ((void *)(this->dist_func), (void *)(FP32_L2Sqr));
    } else if (std::is_same<TypeParam, double>::value) {
        ASSERT_EQ((void *)this->dist_func, (void *)FP64_L2Sqr);
    }
}

// No Optimization distance function.
TYPED_TEST(DistFuncTest, ip_no_optimization_get_func_test) {
    // Choose a dim that has no optimizations
    size_t dim = 1;
    VecSimMetric metric = VecSimMetric_IP;

    spaces::SetDistFunc(metric, dim, &(this->dist_func));

    // Casting to void * so both sides of the comparison would have the
    // same type.
    if (std::is_same<TypeParam, float>::value) {
        ASSERT_EQ((void *)(this->dist_func), (void *)(FP32_InnerProduct));
    } else if (std::is_same<TypeParam, double>::value) {
        ASSERT_EQ((void *)this->dist_func, (void *)FP64_InnerProduct);
    }
}

/****** no optimization function tests suite ******/

TYPED_TEST(DistFuncTest, l2_no_optimization_func_test) {
    // Choose a dim that has no optimizations
    size_t dim = 3;
    VecSimMetric metric = VecSimMetric_L2;

    spaces::SetDistFunc(metric, dim, &(this->dist_func));

    TypeParam a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = TypeParam(i + 1.0);
        b[i] = TypeParam(i + 1.0);
    }

    TypeParam dist = this->dist_func((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, 0.0);
}

TYPED_TEST(DistFuncTest, IP_no_optimization_func_test) {
    // Choose a dim that has no optimizations
    size_t dim = 3;
    VecSimMetric metric = VecSimMetric_IP;

    spaces::SetDistFunc(metric, dim, &(this->dist_func));

    TypeParam a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = TypeParam(i + 1.5);
        b[i] = TypeParam(i + 1.5);
    }

    normalizeVector(a, dim);
    normalizeVector(b, dim);

    TypeParam dist = this->dist_func((const void *)a, (const void *)b, dim);
    ASSERT_NEAR(dist, 0.0, 0.00000001);
}
class SpacesTest : public ::testing::Test {

protected:
    SpacesTest() {}

    ~SpacesTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(SpacesTest, float_l2_no_optimization_func_test) {
    // Choose a dim that has no optimizations
    size_t dim = 5;

    float a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = float(i + 1.0);
        b[i] = float(i + 1.0);
    }

    float dist = FP32_L2Sqr((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, 0.0);
}

TEST_F(SpacesTest, double_l2_no_optimization_func_test) {
    // Choose a dim that has no optimizations
    size_t dim = 5;

    double a[dim], b[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = double(i + 1.0);
        b[i] = double(i + 1.0);
    }

    double dist = FP64_L2Sqr((const void *)a, (const void *)b, dim);
    ASSERT_EQ(dist, 0.0);
}

TEST_F(SpacesTest, float_ip_no_optimization_func_test) {
    // Choose a dim that has no optimizations
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
    // Choose a dim that has no optimizations
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
// This test will trigger the "Residuals" function for dimension > 16, for each optimization.
TEST_F(SpacesTest, l2_17) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 17;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = FP32_L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD16ExtResiduals_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD16ExtResiduals_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD16ExtResiduals_SSE(v, v, dim));
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

    float baseline = FP32_L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD4ExtResiduals_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD4ExtResiduals_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD4ExtResiduals_SSE(v, v, dim));
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

    float baseline = FP32_InnerProduct(v, v, dim);
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

    float baseline = FP32_InnerProduct(v, v, dim);
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
TEST_F(SpacesTest, ip_16) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 16;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = FP32_InnerProduct(v, v, dim);
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
TEST_F(SpacesTest, l2_16) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 16;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = FP32_L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD16Ext_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD16Ext_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD16Ext_SSE(v, v, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the function for dimension % 4 == 0 for each optimization.
// optimization.
TEST_F(SpacesTest, ip_20) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 20;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = FP32_InnerProduct(v, v, dim);
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

    float baseline = FP32_L2Sqr(v, v, dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD4Ext_AVX512(v, v, dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD4Ext_AVX(v, v, dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, FP32_L2SqrSIMD4Ext_SSE(v, v, dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}
#endif // CPU_FEATURES_ARCH_X86_64

#endif // M1/X86_64
