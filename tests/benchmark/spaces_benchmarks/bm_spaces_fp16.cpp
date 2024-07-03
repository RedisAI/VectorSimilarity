/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/types/float16.h"
#include <benchmark/benchmark.h>
#include <random>

template <typename DATA_TYPE>
class BM_VecSimSpaces_Inter : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    size_t dim;
    DATA_TYPE *v1, *v2;

public:
    BM_VecSimSpaces_Inter() { rng.seed(47); }
    ~BM_VecSimSpaces_Inter() = default;
    virtual DATA_TYPE Convert(double val) { return val; }
    virtual void SetUp(const ::benchmark::State &state) {
        dim = state.range(0);
        v1 = new DATA_TYPE[dim];
        v2 = new DATA_TYPE[dim];
        std::uniform_real_distribution<double> distrib(-1.0, 1.0);
        for (size_t i = 0; i < dim; i++) {
            v1[i] = Convert(distrib(rng));
            v2[i] = Convert(distrib(rng));
        }
    }
    virtual void TearDown(const ::benchmark::State &state) {
        delete v1;
        delete v2;
    }
};

class BM_VecSimSpaces_fp16 : public BM_VecSimSpaces_Inter<vecsim_types::float16> {
    virtual vecsim_types::float16 Convert(double val) { return vecsim_types::FP32_to_FP16(val); }
};

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512BF16_VL.h"
#include "VecSim/spaces/functions/AVX512FP16.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/SSE.h"

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

// dim_opt:   number of elements in 512 bits.
//            i.e. 512/sizeof(type): FP32 = 16, FP64 = 8, BF16 = 32 ...
//            The is the number of elements calculated in each distance function loop,
//            regardless of the arch optimization type.
// Run each function for {1, 4, 16} iterations.
#define EXACT_512BIT_PARAMS(dim_opt) RangeMultiplier(4)->Range(dim_opt, 1500)

// elem_per_128_bits:   dim_opt / 4.  FP32 = 4, FP64 = 2, BF16 = 8...
// Dimensions to test 128 bit chunks.
// Run each function at least one full 512 bits iteration + 1/2/3 iterations of 128 bit chunks.
#define EXACT_128BIT_PARAMS(elem_per_128_bits)                                                     \
    DenseRange(4 * 128 + elem_per_128_bits, 4 * 128 + 6 * elem_per_128_bits, elem_per_128_bits)

// Run each function at least one full 512 bits iteration + (1 * elements : elem_per_128_bits *
// elements) FP32 = residual = 1,2,3, FP64 = residual = 1, BF16 = residual = 1,2,3,4,5,6,7...
#define RESIDUAL_PARAMS(elem_per_128_bits)                                                         \
    DenseRange(4 * 128 + 1, 4 * 128 + elem_per_128_bits - 1, 1)

#define INITIALIZE_BM(bm_class, type_prefix, arch, metric, bm_name, arch_supported)                \
    BENCHMARK_DISTANCE_F(bm_class, type_prefix, arch, metric, bm_name, arch_supported)             \
    BENCHMARK_REGISTER_F(bm_class, type_prefix##_##arch##_##metric##_##bm_name)                    \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)

#define INITIALIZE_EXACT_512BIT_BM(bm_class, type_prefix, arch, metric, dim_opt, arch_supported)   \
    INITIALIZE_BM(bm_class, type_prefix, arch, metric, 512_bit_chunks, arch_supported)             \
        ->EXACT_512BIT_PARAMS(dim_opt)

#define INITIALIZE_EXACT_128BIT_BM(bm_class, type_prefix, arch, metric, dim_opt, arch_supported)   \
    INITIALIZE_BM(bm_class, type_prefix, arch, metric, 128_bit_chunks, arch_supported)             \
        ->EXACT_128BIT_PARAMS(dim_opt / 4)

#define INITIALIZE_RESIDUAL_BM(bm_class, type_prefix, arch, metric, dim_opt, arch_supported)       \
    INITIALIZE_BM(bm_class, type_prefix, arch, metric, residual, arch_supported)                   \
        ->RESIDUAL_PARAMS(dim_opt / 4)

#define INITIALIZE_HIGH_DIM(bm_class, type_prefix, arch, metric, arch_supported)                   \
    INITIALIZE_BM(bm_class, type_prefix, arch, metric, high_dim, arch_supported)                   \
        ->DenseRange(900, 1000, 5)

// Naive algorithms

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
    INITIALIZE_EXACT_128BIT_BM(bm_class, type_prefix, arch, L2, dim_opt, arch_supported);          \
    INITIALIZE_EXACT_512BIT_BM(bm_class, type_prefix, arch, L2, dim_opt, arch_supported);          \
    INITIALIZE_RESIDUAL_BM(bm_class, type_prefix, arch, L2, dim_opt, arch_supported);

#define INITIALIZE_BENCHMARKS_SET_IP(bm_class, type_prefix, arch, dim_opt, arch_supported)         \
    INITIALIZE_HIGH_DIM(bm_class, type_prefix, arch, IP, arch_supported);                          \
    INITIALIZE_EXACT_128BIT_BM(bm_class, type_prefix, arch, IP, dim_opt, arch_supported);          \
    INITIALIZE_EXACT_512BIT_BM(bm_class, type_prefix, arch, IP, dim_opt, arch_supported);          \
    INITIALIZE_RESIDUAL_BM(bm_class, type_prefix, arch, IP, dim_opt, arch_supported);

#define INITIALIZE_BENCHMARKS_SET(bm_class, type_prefix, arch, dim_opt, arch_supported)            \
    INITIALIZE_BENCHMARKS_SET_L2(bm_class, type_prefix, arch, dim_opt, arch_supported)             \
    INITIALIZE_BENCHMARKS_SET_IP(bm_class, type_prefix, arch, dim_opt, arch_supported)

#ifdef CPU_FEATURES_ARCH_X86_64
cpu_features::X86Features opt = cpu_features::GetX86Info().features;

// OPT_AVX512FP16 functions
#ifdef OPT_AVX512_FP16

class BM_VecSimSpaces_fp16_adv : public BM_VecSimSpaces_Inter<_Float16> {};

bool avx512fp16_supported = opt.avx512_fp16;
INITIALIZE_BENCHMARKS_SET(BM_VecSimSpaces_fp16_adv, FP16, AVX512FP16, 32, avx512fp16_supported);

INITIALIZE_NAIVE_BM(BM_VecSimSpaces_fp16_adv, FP16, InnerProduct, 32);
INITIALIZE_NAIVE_BM(BM_VecSimSpaces_fp16_adv, FP16, L2Sqr, 32);
#endif // OPT_AVX512_FP16

// OPT_AVX512F functions
#ifdef OPT_AVX512F
bool avx512f_supported = opt.avx512f;
INITIALIZE_BENCHMARKS_SET(BM_VecSimSpaces_fp16, FP16, AVX512F, 32, avx512f_supported);
#endif // OPT_AVX512F
// AVX functions
#ifdef OPT_F16C
bool avx512_bw_f16c_supported = opt.f16c && opt.fma3 && opt.avx;
INITIALIZE_BENCHMARKS_SET(BM_VecSimSpaces_fp16, FP16, F16C, 32, avx512_bw_f16c_supported);
#endif // OPT_F16C

#endif // x86_64

INITIALIZE_NAIVE_BM(BM_VecSimSpaces_fp16, FP16, InnerProduct, 32);
INITIALIZE_NAIVE_BM(BM_VecSimSpaces_fp16, FP16, L2Sqr, 32);
BENCHMARK_MAIN();
