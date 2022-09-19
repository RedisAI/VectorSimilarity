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

#include <tuple>

using DistTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(DistFuncTest, DistTypes);

template <typename DistType, VecSimMetric metric>
struct space_class {
    static VecSimMetric get_metric() { return metric; }
    dist_func_t<DistType> dist_func;
};
template <typename SpaceType>
class GetDistFuncTest : public ::testing::Test {

public:
    SpaceType space_m;
    decltype(SpaceType::dist_func) get_dist_func() { return space_m.dist_func; }
    void set_dist_func(size_t dim) {
        spaces::SetDistFunc(SpaceType::get_metric(), dim, &(space_m.dist_func));
    }
};

using SpacesTypes =
    ::testing::Types<space_class<float, VecSimMetric_IP>, space_class<float, VecSimMetric_L2>,
                     space_class<double, VecSimMetric_IP>, space_class<double, VecSimMetric_L2>>;

TYPED_TEST_SUITE(GetDistFuncTest, SpacesTypes);
// INSTANTIATE_TEST_SUITE_P(My, GetDistFuncTest<double>,testing::Values(VecSimMetric_Cosine,
// VecSimMetric_IP, VecSimMetric_L2), test_get_space);
TYPED_TEST(GetDistFuncTest, test_get_space) {
    size_t dim = 3;
    this->set_dist_func(dim);
    // Casting to void * so both sides of the comparison would have the
    // same type.
    if (std::is_same<TypeParam, float>::value) {
        ASSERT_EQ((void *)(this->get_dist_func()), (void *)(FP32_L2Sqr));
    } else if (std::is_same<TypeParam, double>::value) {
        ASSERT_EQ((void *)this->get_dist_func(), (void *)FP64_L2Sqr);
    }
}
/****** no optimization function tests suite ******/

TYPED_TEST(DistFuncTest, l2_no_optimization_func_test) {
    // Choose a dim that has no optimizations
    size_t dim = 3;
    VecSimMetric metric = VecSimMetric_L2;

    this->set_dist_func(dim);
    // Casting to void * so both sides of the comparison would have the
    // same type.
    if (std::is_same<TypeParam, float>::value) {
        ASSERT_EQ((void *)(this->dist_func), (void *)(FP32_L2Sqr));
    } else if (std::is_same<TypeParam, double>::value) {
        ASSERT_EQ((void *)this->dist_func, (void *)FP64_L2Sqr);
    }

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

    this->set_dist_func(dim);

    // Casting to void * so both sides of the comparison would have the
    // same type.
    if (std::is_same<TypeParam, float>::value) {
        ASSERT_EQ((void *)(this->dist_func), (void *)(FP32_InnerProduct));
    } else if (std::is_same<TypeParam, double>::value) {
        ASSERT_EQ((void *)this->dist_func, (void *)FP64_InnerProduct);
    }

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
