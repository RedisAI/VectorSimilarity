#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/space_aux.h"

#include "bm_spaces_class.h"

// Defining the generic benchmark flow: if there is support for the optimization, benchmark the
// function.
#define BENCHMARK_DISTANCE_F(arch, settings, func)                                                 \
    BENCHMARK_DEFINE_F(BM_VecSimSpaces, arch##_##settings)(benchmark::State & st) {                \
        if (opt < ARCH_OPT_##arch) {                                                               \
            st.SkipWithError("This benchmark requires " #arch ", which is not available");         \
            return;                                                                                \
        }                                                                                          \
        for (auto _ : st) {                                                                        \
            func(v1, v2, &dim);                                                                    \
        }                                                                                          \
    }

// AVX512 functions
#ifdef __AVX512F__
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX512.h"

BENCHMARK_DISTANCE_F(AVX512, L2_16, f_L2SqrSIMD16Ext_AVX512)
BENCHMARK_DISTANCE_F(AVX512, L2_4, f_L2SqrSIMD4Ext_AVX512)
BENCHMARK_DISTANCE_F(AVX512, L2_16_Residuals, f_L2SqrSIMD16ExtResiduals_AVX512)
BENCHMARK_DISTANCE_F(AVX512, L2_4_Residuals, f_L2SqrSIMD4ExtResiduals_AVX512)

BENCHMARK_DISTANCE_F(AVX512, IP_16, f_InnerProductSIMD16Ext_AVX512)
BENCHMARK_DISTANCE_F(AVX512, IP_4, f_InnerProductSIMD4Ext_AVX512)
BENCHMARK_DISTANCE_F(AVX512, IP_16_Residuals, f_InnerProductSIMD16ExtResiduals_AVX512)
BENCHMARK_DISTANCE_F(AVX512, IP_4_Residuals, f_InnerProductSIMD4ExtResiduals_AVX512)
#endif // AVX512F

// AVX functions
#ifdef __AVX__
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/IP/IP_AVX.h"

BENCHMARK_DISTANCE_F(AVX, L2_16, f_L2SqrSIMD16Ext_AVX)
BENCHMARK_DISTANCE_F(AVX, L2_4, f_L2SqrSIMD4Ext_AVX)
BENCHMARK_DISTANCE_F(AVX, L2_16_Residuals, f_L2SqrSIMD16ExtResiduals_AVX)
BENCHMARK_DISTANCE_F(AVX, L2_4_Residuals, f_L2SqrSIMD4ExtResiduals_AVX)

BENCHMARK_DISTANCE_F(AVX, IP_16, f_InnerProductSIMD16Ext_AVX)
BENCHMARK_DISTANCE_F(AVX, IP_4, f_InnerProductSIMD4Ext_AVX)
BENCHMARK_DISTANCE_F(AVX, IP_16_Residuals, f_InnerProductSIMD16ExtResiduals_AVX)
BENCHMARK_DISTANCE_F(AVX, IP_4_Residuals, f_InnerProductSIMD4ExtResiduals_AVX)
#endif // AVX

// SSE functions
#ifdef __SSE__
#include "VecSim/spaces/L2/L2_SSE.h"
#include "VecSim/spaces/IP/IP_SSE.h"

BENCHMARK_DISTANCE_F(SSE, L2_16, f_L2SqrSIMD16Ext_SSE)
BENCHMARK_DISTANCE_F(SSE, L2_4, f_L2SqrSIMD4Ext_SSE)
BENCHMARK_DISTANCE_F(SSE, L2_16_Residuals, f_L2SqrSIMD16ExtResiduals_SSE)
BENCHMARK_DISTANCE_F(SSE, L2_4_Residuals, f_L2SqrSIMD4ExtResiduals_SSE)

BENCHMARK_DISTANCE_F(SSE, IP_16, f_InnerProductSIMD16Ext_SSE)
BENCHMARK_DISTANCE_F(SSE, IP_4, f_InnerProductSIMD4Ext_SSE)
BENCHMARK_DISTANCE_F(SSE, IP_16_Residuals, f_InnerProductSIMD16ExtResiduals_SSE)
BENCHMARK_DISTANCE_F(SSE, IP_4_Residuals, f_InnerProductSIMD4ExtResiduals_SSE)
#endif // SSE

// Naive algorithms

#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/IP/IP.h"

BENCHMARK_DEFINE_F(BM_VecSimSpaces, NAIVE_IP)(benchmark::State &st) {
    for (auto _ : st) {
        f_InnerProduct(v1, v2, &dim);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimSpaces, NAIVE_L2)(benchmark::State &st) {
    for (auto _ : st) {
        f_L2Sqr(v1, v2, &dim);
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
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_L2_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_L2_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_L2_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_L2_4_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_IP_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_IP_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_IP_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_IP_4_Residuals) RESIDUAL_PARAMS;
#endif

#ifdef __AVX__
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_L2_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_L2_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_L2_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_L2_4_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_IP_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_IP_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_IP_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_IP_4_Residuals) RESIDUAL_PARAMS;
#endif

#ifdef __SSE__
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_L2_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_L2_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_L2_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_L2_4_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_IP_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_IP_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_IP_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_IP_4_Residuals) RESIDUAL_PARAMS;
#endif

// Naive
BENCHMARK_REGISTER_F(BM_VecSimSpaces, NAIVE_L2) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, NAIVE_L2) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, NAIVE_IP) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, NAIVE_IP) RESIDUAL_PARAMS;

BENCHMARK_MAIN();
