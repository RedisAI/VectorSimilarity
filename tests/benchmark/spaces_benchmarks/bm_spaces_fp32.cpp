/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#define DATA_TYPE float
#include "bm_spaces.h"

// A wrapper of benchmark definition for fp32
#define BENCHMARK_DISTANCE_F_FP32(arch, settings, func)                                            \
    BENCHMARK_DISTANCE_F(FP32, arch, settings, func)

#ifdef CPU_FEATURES_ARCH_AARCH64
// NEON implementation for ARMv8-a
#ifdef OPT_NEON
INITIALIZE_EXACT_BM(FP32, NEON, L2, 16, spaces::Choose_FP32_L2_implementation_NEON(16));
INITIALIZE_EXACT_BM(FP32, NEON, L2, 4, spaces::Choose_FP32_L2_implementation_NEON(4));
INITIALIZE_RESIDUAL_BM(FP32, NEON, L2, 16_Residuals,
                       spaces::Choose_FP32_L2_implementation_NEON(16));
INITIALIZE_RESIDUAL_BM(FP32, NEON, L2, 4_Residuals, spaces::Choose_FP32_L2_implementation_NEON(4));

INITIALIZE_EXACT_BM(FP32, NEON, IP, 16, spaces::Choose_FP32_L2_implementation_NEON(16));
INITIALIZE_EXACT_BM(FP32, NEON, IP, 4, spaces::Choose_FP32_L2_implementation_NEON(4));
INITIALIZE_RESIDUAL_BM(FP32, NEON, IP, 16_Residuals,
                       spaces::Choose_FP32_L2_implementation_NEON(16));
INITIALIZE_RESIDUAL_BM(FP32, NEON, IP, 4_Residuals, spaces::Choose_FP32_L2_implementation_NEON(4));
#endif
// SVE implementation
#ifdef OPT_SVE
INITIALIZE_EXACT_BM(FP32, SVE, L2, 16, spaces::Choose_FP32_L2_implementation_SVE(16));
INITIALIZE_EXACT_BM(FP32, SVE, L2, 4, spaces::Choose_FP32_L2_implementation_SVE(4));
INITIALIZE_RESIDUAL_BM(FP32, SVE, L2, 16_Residuals, spaces::Choose_FP32_L2_implementation_SVE(16));
INITIALIZE_RESIDUAL_BM(FP32, SVE, L2, 4_Residuals, spaces::Choose_FP32_L2_implementation_SVE(4));

INITIALIZE_EXACT_BM(FP32, SVE, IP, 16, spaces::Choose_FP32_L2_implementation_SVE(16));
INITIALIZE_EXACT_BM(FP32, SVE, IP, 4, spaces::Choose_FP32_L2_implementation_SVE(4));
INITIALIZE_RESIDUAL_BM(FP32, SVE, IP, 16_Residuals, spaces::Choose_FP32_L2_implementation_SVE(16));
INITIALIZE_RESIDUAL_BM(FP32, SVE, IP, 4_Residuals, spaces::Choose_FP32_L2_implementation_SVE(4));
#endif
// SVE2 implementation
#ifdef OPT_SVE2
INITIALIZE_EXACT_BM(FP32, SVE2, L2, 16, spaces::Choose_FP32_L2_implementation_SVE2(16));
INITIALIZE_EXACT_BM(FP32, SVE2, L2, 4, spaces::Choose_FP32_L2_implementation_SVE2(4));
INITIALIZE_RESIDUAL_BM(FP32, SVE2, L2, 16_Residuals,
                       spaces::Choose_FP32_L2_implementation_SVE2(16));
INITIALIZE_RESIDUAL_BM(FP32, SVE2, L2, 4_Residuals, spaces::Choose_FP32_L2_implementation_SVE2(4));
INITIALIZE_EXACT_BM(FP32, SVE2, IP, 16, spaces::Choose_FP32_IP_implementation_SVE2(16));
INITIALIZE_EXACT_BM(FP32, SVE2, IP, 4, spaces::Choose_FP32_IP_implementation_SVE2(4));
INITIALIZE_RESIDUAL_BM(FP32, SVE2, IP, 16_Residuals,
                       spaces::Choose_FP32_IP_implementation_SVE2(16));
INITIALIZE_RESIDUAL_BM(FP32, SVE2, IP, 4_Residuals, spaces::Choose_FP32_IP_implementation_SVE2(4));

#endif
#endif // AARCH64

#ifdef CPU_FEATURES_ARCH_X86_64

// AVX512 functions
#ifdef OPT_AVX512F
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX512.h"

INITIALIZE_EXACT_BM(FP32, AVX512_F, L2, 16, FP32_L2SqrSIMD16Ext_AVX512);
INITIALIZE_EXACT_BM(FP32, AVX512_F, L2, 4, FP32_L2SqrSIMD4Ext_AVX512);
INITIALIZE_RESIDUAL_BM(FP32, AVX512_F, L2, 16_Residuals, FP32_L2SqrSIMD16ExtResiduals_AVX512);
INITIALIZE_RESIDUAL_BM(FP32, AVX512_F, L2, 4_Residuals, FP32_L2SqrSIMD4ExtResiduals_AVX512);

INITIALIZE_EXACT_BM(FP32, AVX512_F, IP, 16, FP32_InnerProductSIMD16Ext_AVX512);
INITIALIZE_EXACT_BM(FP32, AVX512_F, IP, 4, FP32_InnerProductSIMD4Ext_AVX512);
INITIALIZE_RESIDUAL_BM(FP32, AVX512_F, IP, 16_Residuals,
                       FP32_InnerProductSIMD16ExtResiduals_AVX512);
INITIALIZE_RESIDUAL_BM(FP32, AVX512_F, IP, 4_Residuals, FP32_InnerProductSIMD4ExtResiduals_AVX512);
#endif // AVX512F

// AVX functions
#ifdef OPT_AVX
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/IP/IP_AVX.h"

INITIALIZE_EXACT_BM(FP32, AVX, L2, 16, FP32_L2SqrSIMD16Ext_AVX);
INITIALIZE_EXACT_BM(FP32, AVX, L2, 4, FP32_L2SqrSIMD4Ext_AVX);
INITIALIZE_RESIDUAL_BM(FP32, AVX, L2, 16_Residuals, FP32_L2SqrSIMD16ExtResiduals_AVX);
INITIALIZE_RESIDUAL_BM(FP32, AVX, L2, 4_Residuals, FP32_L2SqrSIMD4ExtResiduals_AVX);

INITIALIZE_EXACT_BM(FP32, AVX, IP, 16, FP32_InnerProductSIMD16Ext_AVX);
INITIALIZE_EXACT_BM(FP32, AVX, IP, 4, FP32_InnerProductSIMD4Ext_AVX);
INITIALIZE_RESIDUAL_BM(FP32, AVX, IP, 16_Residuals, FP32_InnerProductSIMD16ExtResiduals_AVX);
INITIALIZE_RESIDUAL_BM(FP32, AVX, IP, 4_Residuals, FP32_InnerProductSIMD4ExtResiduals_AVX);
#endif // AVX

// SSE functions
#ifdef OPT_SSE
#include "VecSim/spaces/L2/L2_SSE.h"
#include "VecSim/spaces/IP/IP_SSE.h"

INITIALIZE_EXACT_BM(FP32, SSE, L2, 16, FP32_L2SqrSIMD16Ext_SSE);
INITIALIZE_EXACT_BM(FP32, SSE, L2, 4, FP32_L2SqrSIMD4Ext_SSE);
INITIALIZE_RESIDUAL_BM(FP32, SSE, L2, 16_Residuals, FP32_L2SqrSIMD16ExtResiduals_SSE);
INITIALIZE_RESIDUAL_BM(FP32, SSE, L2, 4_Residuals, FP32_L2SqrSIMD4ExtResiduals_SSE);

INITIALIZE_EXACT_BM(FP32, SSE, IP, 16, FP32_InnerProductSIMD16Ext_SSE);
INITIALIZE_EXACT_BM(FP32, SSE, IP, 4, FP32_InnerProductSIMD4Ext_SSE);
INITIALIZE_RESIDUAL_BM(FP32, SSE, IP, 16_Residuals, FP32_InnerProductSIMD16ExtResiduals_SSE);
INITIALIZE_RESIDUAL_BM(FP32, SSE, IP, 4_Residuals, FP32_InnerProductSIMD4ExtResiduals_SSE);

#endif // SSE
#endif // CPU_FEATURES_ARCH_X86_64

// Naive algorithms

INITIALIZE_NAIVE_BM(FP32, InnerProduct)->EXACT_PARAMS_MODULU16DIM->RESIDUAL_PARAMS;
INITIALIZE_NAIVE_BM(FP32, L2Sqr)->EXACT_PARAMS_MODULU16DIM->RESIDUAL_PARAMS;

// Naive

BENCHMARK_MAIN();
