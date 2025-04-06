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
class BM_VecSimSpaces_FP32 : public BM_VecSimSpaces<float> {};

#ifdef CPU_FEATURES_ARCH_AARCH64
cpu_features::Aarch64Features opt = cpu_features::GetAarch64Info().features;

// NEON implementation for ARMv8-a
#ifdef OPT_NEON
bool neon_supported = opt.asimd; // ARMv8-a always supports NEON
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_FP32, FP32, NEON, 16, neon_supported);
#endif
// SVE implementation
#ifdef OPT_SVE
bool sve_supported = opt.sve; // Check for SVE support
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_FP32, FP32, SVE, 16, sve_supported);
#endif
// SVE2 implementation
#ifdef OPT_SVE2
bool sve2_supported = opt.sve2; // Check for SVE2 support
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_FP32, FP32, SVE2, 16, sve2_supported);
#endif
#endif // AARCH64

#ifdef CPU_FEATURES_ARCH_X86_64
cpu_features::X86Features opt = cpu_features::GetX86Info().features;

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

// Naive algorithms

INITIALIZE_NAIVE_BM(FP32, InnerProduct)->EXACT_PARAMS_MODULU16DIM->RESIDUAL_PARAMS;
INITIALIZE_NAIVE_BM(FP32, L2Sqr)->EXACT_PARAMS_MODULU16DIM->RESIDUAL_PARAMS;

// Naive

BENCHMARK_MAIN();
