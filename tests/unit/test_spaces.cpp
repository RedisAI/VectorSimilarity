#include "gtest/gtest.h"
#include "cpu_features_macros.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/IP/IP_SSE.h"
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/L2/L2_SSE.h"
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/L2/L2_AVX512.h"

class SpacesTest : public ::testing::Test {

protected:
    SpacesTest() {}

    ~SpacesTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

#ifdef CPU_FEATURES_ARCH_X86_64
// This test will trigger the "Residuals" function for dimension > 16, for each optimization.
TEST_F(SpacesTest, l2_17) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 17;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = L2Sqr(v, v, &dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, L2SqrSIMD16ExtResiduals_AVX512(v, v, &dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, L2SqrSIMD16ExtResiduals_AVX(v, v, &dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, L2SqrSIMD16ExtResiduals_SSE(v, v, &dim));
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

    float baseline = L2Sqr(v, v, &dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, L2SqrSIMD16ExtResiduals_AVX512(v, v, &dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, L2SqrSIMD16ExtResiduals_AVX(v, v, &dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, L2SqrSIMD4ExtResiduals_SSE(v, v, &dim));
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

    float baseline = InnerProduct(v, v, &dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, InnerProductSIMD16ExtResiduals_AVX512(v, v, &dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, InnerProductSIMD16ExtResiduals_AVX(v, v, &dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, InnerProductSIMD16ExtResiduals_SSE(v, v, &dim));
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

    float baseline = InnerProduct(v, v, &dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, InnerProductSIMD16ExtResiduals_AVX512(v, v, &dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, InnerProductSIMD4ExtResiduals_AVX(v, v, &dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, InnerProductSIMD4ExtResiduals_SSE(v, v, &dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}

// This test will trigger the function for dimension % 16 == 0 and dimension % 4 == 0, for each
// optimization.
TEST_F(SpacesTest, ip_16) {
    Arch_Optimization optimization = getArchitectureOptimization();
    size_t dim = 16;
    float v[dim];
    for (size_t i = 0; i < dim; i++) {
        v[i] = (float)i;
    }

    float baseline = InnerProduct(v, v, &dim);
    switch (optimization) {
    case ARCH_OPT_AVX512:
        ASSERT_EQ(baseline, InnerProductSIMD16Ext_AVX512(v, v, &dim));
        optimization = ARCH_OPT_AVX;
    case ARCH_OPT_AVX:
        ASSERT_EQ(baseline, InnerProductSIMD16Ext_AVX(v, v, &dim));
        ASSERT_EQ(baseline, InnerProductSIMD4Ext_AVX(v, v, &dim));
        optimization = ARCH_OPT_SSE;
    case ARCH_OPT_SSE:
        ASSERT_EQ(baseline, InnerProductSIMD16Ext_SSE(v, v, &dim));
        ASSERT_EQ(baseline, InnerProductSIMD4Ext_SSE(v, v, &dim));
        break;
    default:
        ASSERT_TRUE(false);
    }
}
#endif // CPU_FEATURES_ARCH_X86_64
