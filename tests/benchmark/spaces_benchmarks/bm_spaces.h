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
#include "VecSim/spaces/functions/AVX512FP16.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512BF16_VL.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/SSE.h"

#include "bm_spaces_class.h"

// Defining the generic benchmark flow: if there is support for the optimization, benchmark the
// function.
#define BENCHMARK_DISTANCE_F(bm_class, type_prefix, arch, metric, bm_name, arch_supported)         \
    BENCHMARK_DEFINE_F(bm_class, type_prefix##_##arch##_##metric##_##bm_name)                      \
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
    BENCHMARK_REGISTER_F(bm_class, type_prefix##_##arch##_##metric##_##bm_name)                    \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)

/**
 * @param dim_opt: smallest dimension with no residual.
 * i.e smallest dimension to satisfy:
 * dim % num_elements_in_512_bits == 0.
 * @param min_no_res_th_dim: a number that is both divisible by 32 to ensure that we have at least
 * one full 512 bits iteration in all types, and higher than the minimum dimension requires to
 * choose all possible optimizations. (currently it's 500 for IP with AVX512_FP16)
 */
static constexpr size_t min_no_res_th_dim = 512;

/**
 * RangeMultiplier(val)->Range(start, end) generates powers of `val` in the range [start, end],
 * including `start` and `end`.
 */
#define INITIALIZE_EXACT_512BIT_BM(bm_class, type_prefix, arch, metric, dim_opt, arch_supported)   \
    INITIALIZE_BM(bm_class, type_prefix, arch, metric, 512_bit_chunks, arch_supported)             \
        ->RangeMultiplier(4)                                                                       \
        ->Range(dim_opt, 1024)

/** for `start` = min_no_res_th_dim, we run bm for all dimensions
 * in the following range: (start, start + 1, start + 2, start + 3, ... start + dim_opt)
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
        ->DenseRange(100, 200, 15)

/* Naive algorithms */
#define BENCHMARK_DEFINE_NAIVE(bm_class, type_prefix, metric)                                      \
    BENCHMARK_DEFINE_F(bm_class, type_prefix##_NAIVE_##metric)                                     \
    (benchmark::State & st) {                                                                      \
        for (auto _ : st) {                                                                        \
            type_prefix##_##metric(v1, v2, dim);                                                   \
        }                                                                                          \
    }

#define INITIALIZE_NAIVE_BM(bm_class, type_prefix, metric, dim_opt)                                \
    BENCHMARK_DEFINE_NAIVE(bm_class, type_prefix, metric)                                          \
    BENCHMARK_REGISTER_F(bm_class, type_prefix##_NAIVE_##metric)                                   \
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

#define INITIALIZE_BENCHMARKS_SET(bm_class, type_prefix, arch, dim_opt, arch_supported)            \
    INITIALIZE_BENCHMARKS_SET_L2(bm_class, type_prefix, arch, dim_opt, arch_supported)             \
    INITIALIZE_BENCHMARKS_SET_IP(bm_class, type_prefix, arch, dim_opt, arch_supported)
