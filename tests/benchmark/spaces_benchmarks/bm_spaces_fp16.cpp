/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/types/float16.h"
#include "bm_spaces.h"

class BM_VecSimSpaces_FP16 : public BM_VecSimSpaces<vecsim_types::float16> {
    vecsim_types::float16 DoubleToType(double val) override {
        return vecsim_types::FP32_to_FP16(val);
    }
};

#ifdef CPU_FEATURES_ARCH_X86_64
cpu_features::X86Features opt = cpu_features::GetX86Info().features;

bool avx512f_supported = opt.avx512f;
BENCHMARK_DEFINE_F(BM_VecSimSpaces_FP16, FP16_InnerProductSIMD32_AVX512F)
(benchmark::State &st) {
    if (!avx512f_supported) {
        st.SkipWithError("This benchmark requires AVX512FP16, which is not available");
        return;
    }
    auto func = spaces::Choose_FP16_IP_implementation_AVX512F(dim);
    for (auto _ : st) {
        func(v1, v2, dim);
    }
}

BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP16, FP16_InnerProductSIMD32_AVX512F)
    ->ArgName("Dimension")
    ->Unit(benchmark::kNanosecond)
    ->Arg(400);

// OPT_AVX512FP16 functions
#ifdef OPT_AVX512_FP16

class BM_VecSimSpaces_FP16_adv : public BM_VecSimSpaces<_Float16> {};

bool avx512fp16_supported = opt.avx512_fp16;
BENCHMARK_DEFINE_F(BM_VecSimSpaces_FP16_adv, FP16_InnerProductSIMD32_AVX512FP16_512Reduce)
(benchmark::State &st) {
    if (!avx512fp16_supported) {
        st.SkipWithError("This benchmark requires AVX512FP16, which is not available");
        return;
    }
    auto func = spaces::Choose_FP16_IP_implementation_AVX512FP16(dim);
    for (auto _ : st) {
        func(v1, v2, dim);
    }
}

BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP16_adv, FP16_InnerProductSIMD32_AVX512FP16_512Reduce)
    ->ArgName("Dimension")
    ->Unit(benchmark::kNanosecond)
    ->Arg(400);
#endif // OPT_AVX512_FP16

#endif // x86_64

BENCHMARK_MAIN();
