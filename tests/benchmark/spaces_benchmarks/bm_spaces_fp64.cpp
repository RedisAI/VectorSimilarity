/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#define DATA_TYPE double
#include "bm_spaces.h"

// A wrapper of benchmark definition for fp64
#define BENCHMARK_DISTANCE_F_FP64(arch, settings, func)                                            \
    BENCHMARK_DISTANCE_F(FP64, arch, settings, func)

#define EXACT_PARAMS_MODULU8DIM EXACT_PARAMS_MODULU16DIM

#define EXACT_PARAMS_MODULU2DIM EXACT_PARAMS_MODULU4DIM

#ifdef CPU_FEATURES_ARCH_AARCH64
// NEON implementation for ARMv8-a
#ifdef OPT_NEON
INITIALIZE_EXACT_BM(FP64, NEON, L2, 8, spaces::Choose_FP64_L2_implementation_NEON(8));
INITIALIZE_EXACT_BM(FP64, NEON, L2, 2, spaces::Choose_FP64_L2_implementation_NEON(2));
INITIALIZE_RESIDUAL_BM(FP64, NEON, L2, 8_Residuals, spaces::Choose_FP64_L2_implementation_NEON(8));
INITIALIZE_RESIDUAL_BM(FP64, NEON, L2, 2_Residuals, spaces::Choose_FP64_L2_implementation_NEON(2));

INITIALIZE_EXACT_BM(FP64, NEON, IP, 8, spaces::Choose_FP64_L2_implementation_NEON(8));
INITIALIZE_EXACT_BM(FP64, NEON, IP, 2, spaces::Choose_FP64_L2_implementation_NEON(2));
INITIALIZE_RESIDUAL_BM(FP64, NEON, IP, 8_Residuals, spaces::Choose_FP64_L2_implementation_NEON(8));
INITIALIZE_RESIDUAL_BM(FP64, NEON, IP, 2_Residuals, spaces::Choose_FP64_L2_implementation_NEON(2));
#endif
// SVE implementation
#ifdef OPT_SVE
INITIALIZE_EXACT_BM(FP64, SVE, L2, 8, spaces::Choose_FP64_L2_implementation_SVE(8));
INITIALIZE_EXACT_BM(FP64, SVE, L2, 2, spaces::Choose_FP64_L2_implementation_SVE(2));
INITIALIZE_RESIDUAL_BM(FP64, SVE, L2, 8_Residuals, spaces::Choose_FP64_L2_implementation_SVE(8));
INITIALIZE_RESIDUAL_BM(FP64, SVE, L2, 2_Residuals, spaces::Choose_FP64_L2_implementation_SVE(2));

INITIALIZE_EXACT_BM(FP64, SVE, IP, 8, spaces::Choose_FP64_L2_implementation_SVE(8));
INITIALIZE_EXACT_BM(FP64, SVE, IP, 2, spaces::Choose_FP64_L2_implementation_SVE(2));
INITIALIZE_RESIDUAL_BM(FP64, SVE, IP, 8_Residuals, spaces::Choose_FP64_L2_implementation_SVE(8));
INITIALIZE_RESIDUAL_BM(FP64, SVE, IP, 2_Residuals, spaces::Choose_FP64_L2_implementation_SVE(2));
#endif
// SVE2 implementation
#ifdef OPT_SVE2
INITIALIZE_EXACT_BM(FP64, SVE2, L2, 8, spaces::Choose_FP64_L2_implementation_SVE2(8));
INITIALIZE_EXACT_BM(FP64, SVE2, L2, 2, spaces::Choose_FP64_L2_implementation_SVE2(2));
INITIALIZE_RESIDUAL_BM(FP64, SVE2, L2, 8_Residuals, spaces::Choose_FP64_L2_implementation_SVE2(8));
INITIALIZE_RESIDUAL_BM(FP64, SVE2, L2, 2_Residuals, spaces::Choose_FP64_L2_implementation_SVE2(2));
INITIALIZE_EXACT_BM(FP64, SVE2, IP, 8, spaces::Choose_FP64_IP_implementation_SVE2(8));
INITIALIZE_EXACT_BM(FP64, SVE2, IP, 2, spaces::Choose_FP64_IP_implementation_SVE2(2));
INITIALIZE_RESIDUAL_BM(FP64, SVE2, IP, 8_Residuals, spaces::Choose_FP64_IP_implementation_SVE2(8));
INITIALIZE_RESIDUAL_BM(FP64, SVE2, IP, 2_Residuals, spaces::Choose_FP64_IP_implementation_SVE2(2));

#endif
#endif // AARCH64

// AVX512 functions
#ifdef OPT_AVX512F
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX512.h"

INITIALIZE_EXACT_BM(FP64, AVX512_F, L2, 8, FP64_L2SqrSIMD8Ext_AVX512);
INITIALIZE_EXACT_BM(FP64, AVX512_F, L2, 2, FP64_L2SqrSIMD2Ext_AVX512_noDQ);
INITIALIZE_RESIDUAL_BM(FP64, AVX512_F, L2, 8_Residuals, FP64_L2SqrSIMD8ExtResiduals_AVX512);
INITIALIZE_RESIDUAL_BM(FP64, AVX512_F, L2, 2_Residuals, FP64_L2SqrSIMD2ExtResiduals_AVX512_noDQ);

INITIALIZE_EXACT_BM(FP64, AVX512_F, IP, 8, FP64_InnerProductSIMD8Ext_AVX512);
INITIALIZE_EXACT_BM(FP64, AVX512_F, IP, 2, FP64_InnerProductSIMD2Ext_AVX512_noDQ);
INITIALIZE_RESIDUAL_BM(FP64, AVX512_F, IP, 8_Residuals, FP64_InnerProductSIMD8ExtResiduals_AVX512);
INITIALIZE_RESIDUAL_BM(FP64, AVX512_F, IP, 2_Residuals,
                       FP64_InnerProductSIMD2ExtResiduals_AVX512_noDQ);

#endif // AVX512F

#ifdef OPT_AVX512DQ
#include "VecSim/spaces/L2/L2_AVX512DQ.h"
#include "VecSim/spaces/IP/IP_AVX512DQ.h"

INITIALIZE_EXACT_BM(FP64, AVX512_DQ, L2, 2, FP64_L2SqrSIMD2Ext_AVX512);
INITIALIZE_RESIDUAL_BM(FP64, AVX512_DQ, L2, 2_Residuals, FP64_L2SqrSIMD2ExtResiduals_AVX512);

INITIALIZE_EXACT_BM(FP64, AVX512_DQ, IP, 2, FP64_InnerProductSIMD2Ext_AVX512);
INITIALIZE_RESIDUAL_BM(FP64, AVX512_DQ, IP, 2_Residuals, FP64_InnerProductSIMD2ExtResiduals_AVX512);
#endif // AVX512DQ

// AVX functions
#ifdef OPT_AVX
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/IP/IP_AVX.h"

INITIALIZE_EXACT_BM(FP64, AVX, L2, 8, FP64_L2SqrSIMD8Ext_AVX);
INITIALIZE_EXACT_BM(FP64, AVX, L2, 2, FP64_L2SqrSIMD2Ext_AVX);
INITIALIZE_RESIDUAL_BM(FP64, AVX, L2, 8_Residuals, FP64_L2SqrSIMD8ExtResiduals_AVX);
INITIALIZE_RESIDUAL_BM(FP64, AVX, L2, 2_Residuals, FP64_L2SqrSIMD2ExtResiduals_AVX);

INITIALIZE_EXACT_BM(FP64, AVX, IP, 8, FP64_InnerProductSIMD8Ext_AVX);
INITIALIZE_EXACT_BM(FP64, AVX, IP, 2, FP64_InnerProductSIMD2Ext_AVX);
INITIALIZE_RESIDUAL_BM(FP64, AVX, IP, 8_Residuals, FP64_InnerProductSIMD8ExtResiduals_AVX);
INITIALIZE_RESIDUAL_BM(FP64, AVX, IP, 2_Residuals, FP64_InnerProductSIMD2ExtResiduals_AVX);
#endif // AVX

// SSE functions
#ifdef OPT_SSE
#include "VecSim/spaces/L2/L2_SSE.h"
#include "VecSim/spaces/IP/IP_SSE.h"

INITIALIZE_EXACT_BM(FP64, SSE, L2, 8, FP64_L2SqrSIMD8Ext_SSE);
INITIALIZE_EXACT_BM(FP64, SSE, L2, 2, FP64_L2SqrSIMD2Ext_SSE);
INITIALIZE_RESIDUAL_BM(FP64, SSE, L2, 8_Residuals, FP64_L2SqrSIMD8ExtResiduals_SSE);
INITIALIZE_RESIDUAL_BM(FP64, SSE, L2, 2_Residuals, FP64_L2SqrSIMD2ExtResiduals_SSE);

INITIALIZE_EXACT_BM(FP64, SSE, IP, 8, FP64_InnerProductSIMD8Ext_SSE);
INITIALIZE_EXACT_BM(FP64, SSE, IP, 2, FP64_InnerProductSIMD2Ext_SSE);
INITIALIZE_RESIDUAL_BM(FP64, SSE, IP, 8_Residuals, FP64_InnerProductSIMD8ExtResiduals_SSE);
INITIALIZE_RESIDUAL_BM(FP64, SSE, IP, 2_Residuals, FP64_InnerProductSIMD2ExtResiduals_SSE);

#endif // SSE
// Naive algorithms

#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/IP/IP.h"

INITIALIZE_NAIVE_BM(FP64, InnerProduct)->EXACT_PARAMS_MODULU8DIM->RESIDUAL_PARAMS;
INITIALIZE_NAIVE_BM(FP64, L2Sqr)->EXACT_PARAMS_MODULU8DIM->RESIDUAL_PARAMS;

// Naive

BENCHMARK_MAIN();
