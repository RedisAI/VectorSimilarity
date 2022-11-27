/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/space_aux.h"

#include "bm_spaces_class.h"

// Defining the generic benchmark flow: if there is support for the optimization, benchmark the
// function.
#define BENCHMARK_DISTANCE_F(type, arch, settings, func)                                           \
    BENCHMARK_DEFINE_F(BM_VecSimSpaces_##type, arch##_##settings)(benchmark::State & st) {         \
        if (opt < ARCH_OPT_##arch) {                                                               \
            st.SkipWithError("This benchmark requires " #arch ", which is not available");         \
            return;                                                                                \
        }                                                                                          \
        for (auto _ : st) {                                                                        \
            func(v1, v2, dim);                                                                     \
        }                                                                                          \
    }

// AVX512 functions
#ifdef __AVX512F__
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX512.h"

BENCHMARK_DISTANCE_F(FP32, AVX512_F, L2_16, FP32_L2SqrSIMD16Ext_AVX512)
BENCHMARK_DISTANCE_F(FP32, AVX512_F, L2_4, FP32_L2SqrSIMD4Ext_AVX512)
BENCHMARK_DISTANCE_F(FP32, AVX512_F, L2_16_Residuals, FP32_L2SqrSIMD16ExtResiduals_AVX512)
BENCHMARK_DISTANCE_F(FP32, AVX512_F, L2_4_Residuals, FP32_L2SqrSIMD4ExtResiduals_AVX512)

BENCHMARK_DISTANCE_F(FP32, AVX512_F, IP_16, FP32_InnerProductSIMD16Ext_AVX512)
BENCHMARK_DISTANCE_F(FP32, AVX512_F, IP_4, FP32_InnerProductSIMD4Ext_AVX512)
BENCHMARK_DISTANCE_F(FP32, AVX512_F, IP_16_Residuals, FP32_InnerProductSIMD16ExtResiduals_AVX512)
BENCHMARK_DISTANCE_F(FP32, AVX512_F, IP_4_Residuals, FP32_InnerProductSIMD4ExtResiduals_AVX512)

BENCHMARK_DISTANCE_F(FP64, AVX512_F, IP_2, FP64_InnerProductSIMD2Ext_AVX512_noDQ)
BENCHMARK_DISTANCE_F(FP64, AVX512_F, IP_2_Residuals, FP64_InnerProductSIMD2ExtResiduals_AVX512_noDQ)

BENCHMARK_DISTANCE_F(FP64, AVX512_F, L2_2, FP64_L2SqrSIMD2Ext_AVX512_noDQ)
BENCHMARK_DISTANCE_F(FP64, AVX512_F, L2_2_Residuals, FP64_L2SqrSIMD2ExtResiduals_AVX512_noDQ)
#endif // AVX512F

#ifdef __AVX512DQ__
#include "VecSim/spaces/L2/L2_AVX512DQ.h"
#include "VecSim/spaces/IP/IP_AVX512DQ.h"

BENCHMARK_DISTANCE_F(FP64, AVX512_DQ, IP_2, FP64_InnerProductSIMD2Ext_AVX512)
BENCHMARK_DISTANCE_F(FP64, AVX512_DQ, IP_2_Residuals, FP64_InnerProductSIMD2ExtResiduals_AVX512)

BENCHMARK_DISTANCE_F(FP64, AVX512_DQ, L2_2, FP64_L2SqrSIMD2Ext_AVX512)
BENCHMARK_DISTANCE_F(FP64, AVX512_DQ, L2_2_Residuals, FP64_L2SqrSIMD2ExtResiduals_AVX512)
#endif // AVX512DQ

// AVX functions
#ifdef __AVX__
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/IP/IP_AVX.h"

BENCHMARK_DISTANCE_F(FP32, AVX, L2_16, FP32_L2SqrSIMD16Ext_AVX)
BENCHMARK_DISTANCE_F(FP32, AVX, L2_4, FP32_L2SqrSIMD4Ext_AVX)
BENCHMARK_DISTANCE_F(FP32, AVX, L2_16_Residuals, FP32_L2SqrSIMD16ExtResiduals_AVX)
BENCHMARK_DISTANCE_F(FP32, AVX, L2_4_Residuals, FP32_L2SqrSIMD4ExtResiduals_AVX)

BENCHMARK_DISTANCE_F(FP32, AVX, IP_16, FP32_InnerProductSIMD16Ext_AVX)
BENCHMARK_DISTANCE_F(FP32, AVX, IP_4, FP32_InnerProductSIMD4Ext_AVX)
BENCHMARK_DISTANCE_F(FP32, AVX, IP_16_Residuals, FP32_InnerProductSIMD16ExtResiduals_AVX)
BENCHMARK_DISTANCE_F(FP32, AVX, IP_4_Residuals, FP32_InnerProductSIMD4ExtResiduals_AVX)
#endif // AVX

// SSE functions
#ifdef __SSE__
#include "VecSim/spaces/L2/L2_SSE.h"
#include "VecSim/spaces/IP/IP_SSE.h"

BENCHMARK_DISTANCE_F(FP32, SSE, L2_16, FP32_L2SqrSIMD16Ext_SSE)
BENCHMARK_DISTANCE_F(FP32, SSE, L2_4, FP32_L2SqrSIMD4Ext_SSE)
BENCHMARK_DISTANCE_F(FP32, SSE, L2_16_Residuals, FP32_L2SqrSIMD16ExtResiduals_SSE)
BENCHMARK_DISTANCE_F(FP32, SSE, L2_4_Residuals, FP32_L2SqrSIMD4ExtResiduals_SSE)

BENCHMARK_DISTANCE_F(FP32, SSE, IP_16, FP32_InnerProductSIMD16Ext_SSE)
BENCHMARK_DISTANCE_F(FP32, SSE, IP_4, FP32_InnerProductSIMD4Ext_SSE)
BENCHMARK_DISTANCE_F(FP32, SSE, IP_16_Residuals, FP32_InnerProductSIMD16ExtResiduals_SSE)
BENCHMARK_DISTANCE_F(FP32, SSE, IP_4_Residuals, FP32_InnerProductSIMD4ExtResiduals_SSE)
#endif // SSE

// Naive algorithms

#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/IP/IP.h"

BENCHMARK_DEFINE_F(BM_VecSimSpaces_FP32, NAIVE_IP)(benchmark::State &st) {
    for (auto _ : st) {
        FP32_InnerProduct(v1, v2, dim);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimSpaces_FP32, NAIVE_L2)(benchmark::State &st) {
    for (auto _ : st) {
        FP32_L2Sqr(v1, v2, dim);
    }
}

// Register the function as a benchmark

// For exact functions, taking dimensions that are 16 multiplications.
#define EXACT_PARAMS                                                                               \
    ->Arg(16)->Arg(128)->Arg(400)->ArgName("Dimension")->Unit(benchmark::kNanosecond)

// For residual functions, taking dimensions that are 16 multiplications +-1, to show which of
// 16_residual and 4_residual is better in which case.
#define RESIDUAL_PARAMS                                                                            \
    ->Arg(16 - 1)                                                                                  \
        ->Arg(16 + 1)                                                                              \
        ->Arg(128 - 1)                                                                             \
        ->Arg(128 + 1)                                                                             \
        ->Arg(400 - 1)                                                                             \
        ->Arg(400 + 1)                                                                             \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)

#ifdef __AVX512F__
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX512_F_L2_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX512_F_L2_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX512_F_L2_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX512_F_L2_4_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX512_F_IP_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX512_F_IP_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX512_F_IP_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX512_F_IP_4_Residuals) RESIDUAL_PARAMS;

BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP64, AVX512_F_IP_2) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP64, AVX512_F_IP_2_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP64, AVX512_F_L2_2) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP64, AVX512_F_L2_2_Residuals) RESIDUAL_PARAMS;
// TODO: add the rest of FP64 variants
#endif

#ifdef __AVX512DQ__
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP64, AVX512_DQ_IP_2) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP64, AVX512_DQ_IP_2_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP64, AVX512_DQ_L2_2) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP64, AVX512_DQ_L2_2_Residuals) RESIDUAL_PARAMS;
#endif

#ifdef __AVX__
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX_L2_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX_L2_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX_L2_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX_L2_4_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX_IP_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX_IP_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX_IP_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, AVX_IP_4_Residuals) RESIDUAL_PARAMS;
#endif

#ifdef __SSE__
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, SSE_L2_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, SSE_L2_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, SSE_L2_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, SSE_L2_4_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, SSE_IP_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, SSE_IP_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, SSE_IP_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, SSE_IP_4_Residuals) RESIDUAL_PARAMS;
#endif

// Naive
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, NAIVE_L2) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, NAIVE_L2) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, NAIVE_IP) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, NAIVE_IP) RESIDUAL_PARAMS;

BENCHMARK_MAIN();
