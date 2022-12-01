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
#define BENCHMARK_DISTANCE_F(type_prefix, arch, settings, func, data_type)                         \
    BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSpaces, type_prefix##_##arch##_##settings, data_type)     \
    (benchmark::State & st) {                                                                      \
        if (opt < ARCH_OPT_##arch) {                                                               \
            st.SkipWithError("This benchmark requires " #arch ", which is not available");         \
            return;                                                                                \
        }                                                                                          \
        for (auto _ : st) {                                                                        \
            func(v1, v2, dim);                                                                     \
        }                                                                                          \
    }

// A wrapper of benchmark definition for fp32
#define BENCHMARK_DISTANCE_F_FP32(arch, settings, func)                                            \
    BENCHMARK_DISTANCE_F(FP32, arch, settings, func, float)

// A wrapper of benchmark definition for fp64
#define BENCHMARK_DISTANCE_F_FP64(arch, settings, func)                                            \
    BENCHMARK_DISTANCE_F(FP64, arch, settings, func, double)


// Dimensions for functions that satisfy optimizations on dim % 16 == 0 (fp32) or dim % 8 == 0
// (fp64)
#define EXACT_PARAMS_MULT16DIM Arg(16)->Arg(128)->Arg(400)

#define EXACT_PARAMS_MULT8DIM EXACT_PARAMS_MULT16DIM

// Dimensions for functions that satisfy optimizations on dim % 4 == 0 (fp32) or dim % 2 == 0 (fp64)
#define EXACT_PARAMS_MULT4DIM Arg(28)->Arg(140)->Arg(412)

#define EXACT_PARAMS_MULT2DIM EXACT_PARAMS_MULT4DIM

// For residual functions, taking dimensions that are 16 multiplications +-1, to show which of
// 16_residual (for fp64 - 8_residual) and 4_residual (for fp64 - 2_residual) is better in which
// case.
#define RESIDUAL_PARAMS                                                                            \
    Arg(16 - 1)->Arg(16 + 1)->Arg(128 - 1)->Arg(128 + 1)->Arg(400 - 1)->Arg(400 + 1)

#define INITIALIZE_BM(type_prefix, arch, metric, dim_opt, func)                                    \
    BENCHMARK_DISTANCE_F_##type_prefix(arch, metric##_##dim_opt, func)                             \
        BENCHMARK_REGISTER_F(BM_VecSimSpaces, type_prefix##_##arch##_##metric##_##dim_opt)         \
            ->ArgName("Dimension")                                                                 \
            ->Unit(benchmark::kNanosecond)

#define INITIALIZE_EXACT_BM(type_prefix, arch, metric, dim_opt, func)                              \
    INITIALIZE_BM(type_prefix, arch, metric, dim_opt, func)->EXACT_PARAMS_MULT##dim_opt##DIM

#define INITIALIZE_RES_BM(type_prefix, arch, metric, dim_opt, func)                                \
    INITIALIZE_BM(type_prefix, arch, metric, dim_opt, func)->RESIDUAL_PARAMS

// AVX512 functions
#ifdef __AVX512F__
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX512.h"

INITIALIZE_EXACT_BM(FP32, AVX512_F, L2, 16, FP32_L2SqrSIMD16Ext_AVX512);
INITIALIZE_EXACT_BM(FP32, AVX512_F, L2, 4, FP32_L2SqrSIMD4Ext_AVX512);
INITIALIZE_RES_BM(FP32, AVX512_F, L2, 16_Residuals, FP32_L2SqrSIMD16ExtResiduals_AVX512);
INITIALIZE_RES_BM(FP32, AVX512_F, L2, 4_Residuals, FP32_L2SqrSIMD4ExtResiduals_AVX512);

INITIALIZE_EXACT_BM(FP32, AVX512_F, IP, 16, FP32_InnerProductSIMD16Ext_AVX512);
INITIALIZE_EXACT_BM(FP32, AVX512_F, IP, 4, FP32_InnerProductSIMD4Ext_AVX512);
INITIALIZE_RES_BM(FP32, AVX512_F, IP, 16_Residuals, FP32_InnerProductSIMD16ExtResiduals_AVX512);
INITIALIZE_RES_BM(FP32, AVX512_F, IP, 4_Residuals, FP32_InnerProductSIMD4ExtResiduals_AVX512);

// Register FP64
INITIALIZE_EXACT_BM(FP64, AVX512_F, L2, 8, FP64_L2SqrSIMD8Ext_AVX512);
INITIALIZE_EXACT_BM(FP64, AVX512_F, L2, 2, FP64_L2SqrSIMD2Ext_AVX512_noDQ);
INITIALIZE_RES_BM(FP64, AVX512_F, L2, 8_Residuals, FP64_L2SqrSIMD8ExtResiduals_AVX512);
INITIALIZE_RES_BM(FP64, AVX512_F, L2, 2_Residuals, FP64_L2SqrSIMD2ExtResiduals_AVX512_noDQ);

INITIALIZE_EXACT_BM(FP64, AVX512_F, IP, 8, FP64_InnerProductSIMD8Ext_AVX512);
INITIALIZE_EXACT_BM(FP64, AVX512_F, IP, 2, FP64_InnerProductSIMD2Ext_AVX512_noDQ);
INITIALIZE_RES_BM(FP64, AVX512_F, IP, 8_Residuals, FP64_InnerProductSIMD8ExtResiduals_AVX512);
INITIALIZE_RES_BM(FP64, AVX512_F, IP, 2_Residuals, FP64_InnerProductSIMD2ExtResiduals_AVX512_noDQ);

#endif // AVX512F

#ifdef __AVX512DQ__
#include "VecSim/spaces/L2/L2_AVX512DQ.h"
#include "VecSim/spaces/IP/IP_AVX512DQ.h"

INITIALIZE_EXACT_BM(FP64, AVX512_DQ, L2, 2, FP64_L2SqrSIMD2Ext_AVX512);
INITIALIZE_RES_BM(FP64, AVX512_DQ, L2, 2_Residuals, FP64_L2SqrSIMD2ExtResiduals_AVX512);

INITIALIZE_EXACT_BM(FP64, AVX512_DQ, IP, 2, FP64_InnerProductSIMD2Ext_AVX512);
INITIALIZE_RES_BM(FP64, AVX512_DQ, IP, 2_Residuals, FP64_InnerProductSIMD2ExtResiduals_AVX512);
#endif // AVX512DQ

// AVX functions
#ifdef __AVX__
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/IP/IP_AVX.h"

INITIALIZE_EXACT_BM(FP32, AVX, L2, 16, FP32_L2SqrSIMD16Ext_AVX);
INITIALIZE_EXACT_BM(FP32, AVX, L2, 4, FP32_L2SqrSIMD4Ext_AVX);
INITIALIZE_RES_BM(FP32, AVX, L2, 16_Residuals, FP32_L2SqrSIMD16ExtResiduals_AVX);
INITIALIZE_RES_BM(FP32, AVX, L2, 4_Residuals, FP32_L2SqrSIMD4ExtResiduals_AVX);

INITIALIZE_EXACT_BM(FP32, AVX, IP, 16, FP32_InnerProductSIMD16Ext_AVX);
INITIALIZE_EXACT_BM(FP32, AVX, IP, 4, FP32_InnerProductSIMD4Ext_AVX);
INITIALIZE_RES_BM(FP32, AVX, IP, 16_Residuals, FP32_InnerProductSIMD16ExtResiduals_AVX);
INITIALIZE_RES_BM(FP32, AVX, IP, 4_Residuals, FP32_InnerProductSIMD4ExtResiduals_AVX);

// Register FP64
INITIALIZE_EXACT_BM(FP64, AVX, L2, 8, FP64_L2SqrSIMD8Ext_AVX);
INITIALIZE_EXACT_BM(FP64, AVX, L2, 2, FP64_L2SqrSIMD2Ext_AVX);
INITIALIZE_RES_BM(FP64, AVX, L2, 8_Residuals, FP64_L2SqrSIMD8ExtResiduals_AVX);
INITIALIZE_RES_BM(FP64, AVX, L2, 2_Residuals, FP64_L2SqrSIMD2ExtResiduals_AVX);

INITIALIZE_EXACT_BM(FP64, AVX, IP, 8, FP64_InnerProductSIMD8Ext_AVX);
INITIALIZE_EXACT_BM(FP64, AVX, IP, 2, FP64_InnerProductSIMD2Ext_AVX);
INITIALIZE_RES_BM(FP64, AVX, IP, 8_Residuals, FP64_InnerProductSIMD8ExtResiduals_AVX);
INITIALIZE_RES_BM(FP64, AVX, IP, 2_Residuals, FP64_InnerProductSIMD2ExtResiduals_AVX);
#endif // AVX

// SSE functions
#ifdef __SSE__
#include "VecSim/spaces/L2/L2_SSE.h"
#include "VecSim/spaces/IP/IP_SSE.h"

INITIALIZE_EXACT_BM(FP32, SSE, L2, 16, FP32_L2SqrSIMD16Ext_SSE);
INITIALIZE_EXACT_BM(FP32, SSE, L2, 4, FP32_L2SqrSIMD4Ext_SSE);
INITIALIZE_RES_BM(FP32, SSE, L2, 16_Residuals, FP32_L2SqrSIMD16ExtResiduals_SSE);
INITIALIZE_RES_BM(FP32, SSE, L2, 4_Residuals, FP32_L2SqrSIMD4ExtResiduals_SSE);

INITIALIZE_EXACT_BM(FP32, SSE, IP, 16, FP32_InnerProductSIMD16Ext_SSE);
INITIALIZE_EXACT_BM(FP32, SSE, IP, 4, FP32_InnerProductSIMD4Ext_SSE);
INITIALIZE_RES_BM(FP32, SSE, IP, 16_Residuals, FP32_InnerProductSIMD16ExtResiduals_SSE);
INITIALIZE_RES_BM(FP32, SSE, IP, 4_Residuals, FP32_InnerProductSIMD4ExtResiduals_SSE);

// Register FP64
INITIALIZE_EXACT_BM(FP64, SSE, L2, 8, FP64_L2SqrSIMD8Ext_SSE);
INITIALIZE_EXACT_BM(FP64, SSE, L2, 2, FP64_L2SqrSIMD2Ext_SSE);
INITIALIZE_RES_BM(FP64, SSE, L2, 8_Residuals, FP64_L2SqrSIMD8ExtResiduals_SSE);
INITIALIZE_RES_BM(FP64, SSE, L2, 2_Residuals, FP64_L2SqrSIMD2ExtResiduals_SSE);

INITIALIZE_EXACT_BM(FP64, SSE, IP, 8, FP64_InnerProductSIMD8Ext_SSE);
INITIALIZE_EXACT_BM(FP64, SSE, IP, 2, FP64_InnerProductSIMD2Ext_SSE);
INITIALIZE_RES_BM(FP64, SSE, IP, 8_Residuals, FP64_InnerProductSIMD8ExtResiduals_SSE);
INITIALIZE_RES_BM(FP64, SSE, IP, 2_Residuals, FP64_InnerProductSIMD2ExtResiduals_SSE);

#endif // SSE
// Naive algorithms

#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/IP/IP.h"

#define BENCHMARK_DEFINE_NAIVE(type_prefix, metric, data_type)                                     \
    BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSpaces, type_prefix##_NAIVE_##metric, data_type)          \
    (benchmark::State & st) {                                                                      \
        for (auto _ : st) {                                                                        \
            type_prefix##_##metric(v1, v2, dim);                                                   \
        }                                                                                          \
    }

// A wrapper of benchmark definition for fp32
#define BENCHMARK_DEFINE_NAIVE_FP32(metric) BENCHMARK_DEFINE_NAIVE(FP32, metric, float)

// A wrapper of benchmark definition for fp64
#define BENCHMARK_DEFINE_NAIVE_FP64(metric) BENCHMARK_DEFINE_NAIVE(FP64, metric, double)

#define INITIALIZE_NAIVE_BM(type_prefix, metric)                                                   \
    BENCHMARK_DEFINE_NAIVE_##type_prefix(metric)                                                   \
        BENCHMARK_REGISTER_F(BM_VecSimSpaces, type_prefix##_NAIVE_##metric)                        \
            ->ArgName("Dimension")                                                                 \
            ->Unit(benchmark::kNanosecond)

INITIALIZE_NAIVE_BM(FP32, InnerProduct)->EXACT_PARAMS_MULT16DIM->RESIDUAL_PARAMS;
INITIALIZE_NAIVE_BM(FP32, L2Sqr)->EXACT_PARAMS_MULT16DIM->RESIDUAL_PARAMS;

INITIALIZE_NAIVE_BM(FP64, InnerProduct)->EXACT_PARAMS_MULT16DIM->RESIDUAL_PARAMS;
INITIALIZE_NAIVE_BM(FP64, L2Sqr)->EXACT_PARAMS_MULT16DIM->RESIDUAL_PARAMS;

// Naive

BENCHMARK_MAIN();
