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

// AVX512 functions
#ifdef __AVX512F__
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

#ifdef __AVX512DQ__
#include "VecSim/spaces/L2/L2_AVX512DQ.h"
#include "VecSim/spaces/IP/IP_AVX512DQ.h"

INITIALIZE_EXACT_BM(FP64, AVX512_DQ, L2, 2, FP64_L2SqrSIMD2Ext_AVX512);
INITIALIZE_RESIDUAL_BM(FP64, AVX512_DQ, L2, 2_Residuals, FP64_L2SqrSIMD2ExtResiduals_AVX512);

INITIALIZE_EXACT_BM(FP64, AVX512_DQ, IP, 2, FP64_InnerProductSIMD2Ext_AVX512);
INITIALIZE_RESIDUAL_BM(FP64, AVX512_DQ, IP, 2_Residuals, FP64_InnerProductSIMD2ExtResiduals_AVX512);
#endif // AVX512DQ

// AVX functions
#ifdef __AVX__
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
#ifdef __SSE__
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
