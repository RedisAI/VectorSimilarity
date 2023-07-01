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
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/L2/L2.h"

#include "bm_spaces_class.h"

// Defining the generic benchmark flow: if there is support for the optimization, benchmark the
// function.
#define BENCHMARK_DISTANCE_F(type_prefix, arch, metric, dim_opt)                                   \
    BENCHMARK_DEFINE_F(BM_VecSimSpaces, type_prefix##_##arch##_##metric##_##dim_opt)               \
    (benchmark::State & st) {                                                                      \
        if (opt < ARCH_OPT_##arch) {                                                               \
            st.SkipWithError("This benchmark requires " #arch ", which is not available");         \
            return;                                                                                \
        }                                                                                          \
        auto func = spaces::metric##_##type_prefix##_GetDistFunc(dim, ARCH_OPT_##arch);            \
        for (auto _ : st) {                                                                        \
            func(v1, v2, dim);                                                                     \
        }                                                                                          \
    }

// Dimensions for functions that satisfy optimizations on dim % 16 == 0 (fp32) or dim % 8 == 0
// (fp64)
#define EXACT_512BIT_PARAMS Arg(16)->Arg(128)->Arg(400)

// Dimensions for functions that satisfy optimizations on dim % 4 == 0 (fp32) or dim % 2 == 0 (fp64)
#define EXACT_128BIT_PARAMS Arg(28)->Arg(140)->Arg(412)

// For residual functions, taking dimensions that are 16 multiplications +-1, to show which of
// 16_residual (for fp64 - 8_residual) and 4_residual (for fp64 - 2_residual) is better in which
// case.
#define RESIDUAL_PARAMS                                                                            \
    Arg(16 - 1)->Arg(16 + 1)->Arg(128 - 1)->Arg(128 + 1)->Arg(400 - 1)->Arg(400 + 1)

#define INITIALIZE_BM(type_prefix, arch, metric, dim_opt)                                          \
    BENCHMARK_DISTANCE_F(type_prefix, arch, metric, dim_opt)                                       \
    BENCHMARK_REGISTER_F(BM_VecSimSpaces, type_prefix##_##arch##_##metric##_##dim_opt)             \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)

#define INITIALIZE_EXACT_128BIT_BM(type_prefix, arch, metric, dim_opt)                             \
    INITIALIZE_BM(type_prefix, arch, metric, dim_opt)->EXACT_128BIT_PARAMS

#define INITIALIZE_EXACT_512BIT_BM(type_prefix, arch, metric, dim_opt)                             \
    INITIALIZE_BM(type_prefix, arch, metric, dim_opt)->EXACT_512BIT_PARAMS

#define INITIALIZE_RESIDUAL_BM(type_prefix, arch, metric, dim_opt)                                 \
    INITIALIZE_BM(type_prefix, arch, metric, dim_opt)->RESIDUAL_PARAMS

// Naive algorithms

#define BENCHMARK_DEFINE_NAIVE(type_prefix, metric)                                                \
    BENCHMARK_DEFINE_F(BM_VecSimSpaces, type_prefix##_NAIVE_##metric)                              \
    (benchmark::State & st) {                                                                      \
        for (auto _ : st) {                                                                        \
            type_prefix##_##metric(v1, v2, dim);                                                   \
        }                                                                                          \
    }

#define INITIALIZE_NAIVE_BM(type_prefix, metric)                                                   \
    BENCHMARK_DEFINE_NAIVE(type_prefix, metric)                                                    \
    BENCHMARK_REGISTER_F(BM_VecSimSpaces, type_prefix##_NAIVE_##metric)                            \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)
