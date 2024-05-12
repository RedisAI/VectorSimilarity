/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/functions/AVX512.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/SSE.h"

#include "bm_spaces_class.h"

// Defining the generic benchmark flow: if there is support for the optimization, benchmark the
// function.
#define BENCHMARK_DISTANCE_F(type_prefix, arch, metric, bm_name, arch_supported)                   \
    BENCHMARK_DEFINE_F(BM_VecSimSpaces, type_prefix##_##arch##_##metric##_##bm_name)               \
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

// dim_opt:   number of elements in 512 bits.
//            i.e. 512/sizeof(type): FP32 = 16, FP64 = 8, BF16 = 32 ...
//            The is the number of elements calculated in each distance function loop,
//            regardless of the arch optimization type.
// Run each function for {1, 4, 16} iterations.
#define EXACT_512BIT_PARAMS(dim_opt) RangeMultiplier(4)->Range(dim_opt, 400)

// elem_per_128_bits:   dim_opt / 4.  FP32 = 4, FP64 = 2, BF16 = 8...
// Dimensions to test 128 bit chunks.
// Run each function at least one full 512 bits iteration + 1/2/3 iterations of 128 bit chunks.
#define EXACT_128BIT_PARAMS(elem_per_128_bits)                                                     \
    DenseRange(128 + elem_per_128_bits, 128 + 3 * elem_per_128_bits, elem_per_128_bits)

// Run each function at least one full 512 bits iteration + (1 * elements : elem_per_128_bits *
// elements) FP32 = residual = 1,2,3, FP64 = residual = 1, BF16 = residual = 1,2,3,4,5,6,7...
#define RESIDUAL_PARAMS(elem_per_128_bits) DenseRange(128 + 1, 128 + elem_per_128_bits - 1, 1)

#define INITIALIZE_BM(type_prefix, arch, metric, bm_name, arch_supported)                          \
    BENCHMARK_DISTANCE_F(type_prefix, arch, metric, bm_name, arch_supported)                       \
    BENCHMARK_REGISTER_F(BM_VecSimSpaces, type_prefix##_##arch##_##metric##_##bm_name)             \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)

#define INITIALIZE_EXACT_512BIT_BM(type_prefix, arch, metric, dim_opt, arch_supported)             \
    INITIALIZE_BM(type_prefix, arch, metric, 512_bit_chunks, arch_supported)                       \
        ->EXACT_512BIT_PARAMS(dim_opt)

#define INITIALIZE_EXACT_128BIT_BM(type_prefix, arch, metric, dim_opt, arch_supported)             \
    INITIALIZE_BM(type_prefix, arch, metric, 128_bit_chunks, arch_supported)                       \
        ->EXACT_128BIT_PARAMS(dim_opt / 4)

#define INITIALIZE_RESIDUAL_BM(type_prefix, arch, metric, dim_opt, arch_supported)                 \
    INITIALIZE_BM(type_prefix, arch, metric, residual, arch_supported)->RESIDUAL_PARAMS(dim_opt / 4)

// Naive algorithms

#define BENCHMARK_DEFINE_NAIVE(type_prefix, metric)                                                \
    BENCHMARK_DEFINE_F(BM_VecSimSpaces, type_prefix##_NAIVE_##metric)                              \
    (benchmark::State & st) {                                                                      \
        for (auto _ : st) {                                                                        \
            type_prefix##_##metric(v1, v2, dim);                                                   \
        }                                                                                          \
    }

#define INITIALIZE_NAIVE_BM(type_prefix, metric, dim_opt)                                          \
    BENCHMARK_DEFINE_NAIVE(type_prefix, metric)                                                    \
    BENCHMARK_REGISTER_F(BM_VecSimSpaces, type_prefix##_NAIVE_##metric)                            \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)                                                             \
        ->Arg(100)                                                                                 \
        ->Arg(dim_opt)                                                                             \
        ->Arg(dim_opt + dim_opt / 4)                                                               \
        ->Arg(dim_opt - 1)

#define INITIALIZE_BENCHMARKS_SET(type_prefix, arch, dim_opt, arch_supported)                      \
    INITIALIZE_EXACT_128BIT_BM(type_prefix, arch, L2, dim_opt, arch_supported);                    \
    INITIALIZE_EXACT_512BIT_BM(type_prefix, arch, L2, dim_opt, arch_supported);                    \
    INITIALIZE_RESIDUAL_BM(type_prefix, arch, L2, dim_opt, arch_supported);                        \
                                                                                                   \
    INITIALIZE_EXACT_128BIT_BM(type_prefix, arch, IP, dim_opt, arch_supported);                    \
    INITIALIZE_EXACT_512BIT_BM(type_prefix, arch, IP, dim_opt, arch_supported);                    \
    INITIALIZE_RESIDUAL_BM(type_prefix, arch, IP, dim_opt, arch_supported);
