/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/AVX512FP16_VL.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512BF16_VL.h"
#include "VecSim/spaces/functions/AVX512F_BW_VL_VNNI.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/SSE.h"

#include "bm_spaces_class.h"

#define EXPAND(x) x
#define EXPAND2(x) EXPAND(x)
// Helper for raw concatenation with varying arguments
#define BM_FUNC_NAME_HELPER1_2(a, b) a ## _ ## b
#define BM_FUNC_NAME_HELPER1_3(a, b, c) a ## _ ## b ## _ ## c
#define BM_FUNC_NAME_HELPER1_4(a, b, c, d) a ## _ ## b ## _ ## c ## _ ## d
#define BM_FUNC_NAME_HELPER1_5(a, b, c, d, e) a ## _ ## b ## _ ## c ## _ ## d ## _ ## e

// Force expansion of macro arguments
#define BM_FUNC_NAME_HELPER_2(a, b) BM_FUNC_NAME_HELPER1_2(a, b)
#define BM_FUNC_NAME_HELPER_3(a, b, c) BM_FUNC_NAME_HELPER1_3(a, b, c)
#define BM_FUNC_NAME_HELPER_4(a, b, c, d) BM_FUNC_NAME_HELPER1_4(a, b, c, d)
#define BM_FUNC_NAME_HELPER_5(a, b, c, d, e) BM_FUNC_NAME_HELPER1_5(a, b, c, d, e)

// Determine the number of arguments and select the appropriate helper
#define COUNT_ARGS(...) COUNT_ARGS_(__VA_ARGS__, 6, 5, 4, 3, 2, 1)
#define COUNT_ARGS_(_1, _2, _3, _4, _5, _6, N, ...) N

// Concatenate BM_FUNC_NAME_HELPER with the number of arguments
#define CONCAT_HELPER(a, b) a ## _ ## b
#define CONCAT(a, b) CONCAT_HELPER(a, b)

// Main macro that selects the appropriate helper based on argument count
#define CONCAT_WITH_UNDERSCORE(...) EXPAND2(CONCAT(BM_FUNC_NAME_HELPER, EXPAND2(COUNT_ARGS(__VA_ARGS__)))(__VA_ARGS__))

#ifdef CPU_FEATURES_ARCH_X86_64
    #define BENCHMARK_ARCH x86_64
#elif defined(CPU_FEATURES_ARCH_AARCH64)
    #if (__ARM_ARCH >= 9)
        #define BENCHMARK_ARCH arm_v9
    #else
        #define BENCHMARK_ARCH arm_v8
    #endif
#endif
        
// Modify this macro to account for the extra BENCHMARK_ARCH parameter
#define CONCAT_WITH_UNDERSCORE_ARCH(...) CONCAT_WITH_UNDERSCORE(__VA_ARGS__, BENCHMARK_ARCH)


// Defining the generic benchmark flow: if there is support for the optimization, benchmark the
// function.
#define BENCHMARK_DISTANCE_F(bm_class, type_prefix, arch, metric, bm_name, arch_supported)         \
    BENCHMARK_DEFINE_F(bm_class, CONCAT_WITH_UNDERSCORE_ARCH(type_prefix, arch, metric, bm_name)) \
    (benchmark::State & st) {                                                                      \
        if (!arch_supported) {                                                                     \
            st.SkipWithError("This benchmark requires " #arch ", which is not available");         \
            return;                                                                                \
        }                                                                                          \
        auto func = spaces::Choose_##type_prefix##_##metric##_implementation_##arch(dim);          \
        for (auto _ : st) {                                                                        \
            func(v1, v2, dim);                                                                     \
        }                                                                                          \
    }

#define INITIALIZE_BM(bm_class, type_prefix, arch, metric, bm_name, arch_supported)                \
    BENCHMARK_DISTANCE_F(bm_class, type_prefix, arch, metric, bm_name, arch_supported)             \
    BENCHMARK_REGISTER_F(bm_class, CONCAT_WITH_UNDERSCORE_ARCH(type_prefix, arch, metric, bm_name)) \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)

/**
 * A number that is divisible by 32 to ensure that we have at least one full 512 bits iteration in
 * all types
 */
static constexpr size_t min_no_res_th_dim = 512;

/**
 * @param dim_opt: Number of elements in 512 bits.
 */

/**
 * @param dim_opt is also, the smallest dimension to satisfy:
 * dim % num_elements_in_512_bits == 0.
 * We use it to start this set of BM from the smallest dimension that satisfies the above condition.
 * RangeMultiplier(val)->Range(start, end) generates powers of `val` in the range [start, end],
 * including `start` and `end`.
 */
#define INITIALIZE_EXACT_512BIT_BM(bm_class, type_prefix, arch, metric, dim_opt, arch_supported)   \
    INITIALIZE_BM(bm_class, type_prefix, arch, metric, 512_bit_chunks, arch_supported)             \
        ->RangeMultiplier(4)                                                                       \
        ->Range(dim_opt, 1024)

/** for `start` = min_no_res_th_dim (defined above) we run bm for all dimensions
 * in the following range: (start, start + 1, start + 2, start + 3, ... start + dim_opt)
 * to test all possible residual cases.
 */
static constexpr size_t start = min_no_res_th_dim;
#define INITIALIZE_RESIDUAL_BM(bm_class, type_prefix, arch, metric, dim_opt, arch_supported)       \
    INITIALIZE_BM(bm_class, type_prefix, arch, metric, residual, arch_supported)                   \
        ->DenseRange(start + 1, start + dim_opt - 1, 1)

/** Test high dim
 * This range satisfies at least one full 512 bits iteration in all types.
 */
#define INITIALIZE_HIGH_DIM(bm_class, type_prefix, arch, metric, arch_supported)                   \
    INITIALIZE_BM(bm_class, type_prefix, arch, metric, high_dim, arch_supported)                   \
        ->DenseRange(900, 1000, 15)

/** Test low dim
 * This range satisfies at least one full 512-bit iteration in all types (160).
 */
#define INITIALIZE_LOW_DIM(bm_class, type_prefix, arch, metric, arch_supported)                    \
    INITIALIZE_BM(bm_class, type_prefix, arch, metric, low_dim, arch_supported)                    \
        ->DenseRange(55, 200, 15)

/* Naive algorithms */
#define BENCHMARK_DEFINE_NAIVE(bm_class, type_prefix, metric)                                      \
    BENCHMARK_DEFINE_F(bm_class, CONCAT_WITH_UNDERSCORE_ARCH(type_prefix, NAIVE, metric))                                     \
    (benchmark::State & st) {                                                                      \
        for (auto _ : st) {                                                                        \
            type_prefix##_##metric(v1, v2, dim);                                                   \
        }                                                                                          \
    }

#define INITIALIZE_NAIVE_BM(bm_class, type_prefix, metric, dim_opt)                                \
    BENCHMARK_DEFINE_NAIVE(bm_class, type_prefix, metric)                                          \
    BENCHMARK_REGISTER_F(bm_class, CONCAT_WITH_UNDERSCORE_ARCH(type_prefix, NAIVE, metric)) \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)                                                             \
        ->Arg(100)                                                                                 \
        ->Arg(dim_opt)                                                                             \
        ->Arg(dim_opt + dim_opt / 4)                                                               \
        ->Arg(dim_opt - 1)

#define INITIALIZE_BENCHMARKS_SET_L2(bm_class, type_prefix, arch, dim_opt, arch_supported)         \
    INITIALIZE_HIGH_DIM(bm_class, type_prefix, arch, L2, arch_supported);                          \
    INITIALIZE_LOW_DIM(bm_class, type_prefix, arch, L2, arch_supported);                           \
    INITIALIZE_EXACT_512BIT_BM(bm_class, type_prefix, arch, L2, dim_opt, arch_supported);          \
    INITIALIZE_RESIDUAL_BM(bm_class, type_prefix, arch, L2, dim_opt, arch_supported);

#define INITIALIZE_BENCHMARKS_SET_IP(bm_class, type_prefix, arch, dim_opt, arch_supported)         \
    INITIALIZE_HIGH_DIM(bm_class, type_prefix, arch, IP, arch_supported);                          \
    INITIALIZE_LOW_DIM(bm_class, type_prefix, arch, IP, arch_supported);                           \
    INITIALIZE_EXACT_512BIT_BM(bm_class, type_prefix, arch, IP, dim_opt, arch_supported);          \
    INITIALIZE_RESIDUAL_BM(bm_class, type_prefix, arch, IP, dim_opt, arch_supported);

#define INITIALIZE_BENCHMARKS_SET_Cosine(bm_class, type_prefix, arch, dim_opt, arch_supported)     \
    INITIALIZE_HIGH_DIM(bm_class, type_prefix, arch, Cosine, arch_supported);                      \
    INITIALIZE_LOW_DIM(bm_class, type_prefix, arch, Cosine, arch_supported);                       \
    INITIALIZE_EXACT_512BIT_BM(bm_class, type_prefix, arch, Cosine, dim_opt, arch_supported);      \
    INITIALIZE_RESIDUAL_BM(bm_class, type_prefix, arch, Cosine, dim_opt, arch_supported);

#define INITIALIZE_BENCHMARKS_SET_L2_IP(bm_class, type_prefix, arch, dim_opt, arch_supported)      \
    INITIALIZE_BENCHMARKS_SET_L2(bm_class, type_prefix, arch, dim_opt, arch_supported)             \
    INITIALIZE_BENCHMARKS_SET_IP(bm_class, type_prefix, arch, dim_opt, arch_supported)
