/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#include "bm_spaces.h"

class BM_VecSimSpaces_FP32 : public BM_VecSimSpaces<float> {};

#ifdef CPU_FEATURES_ARCH_X86_64
cpu_features::X86Features opt = cpu_features::GetX86Info().features;

// AVX512 functions
#ifdef OPT_AVX512F
bool avx512f_supported = opt.avx512f;
BENCHMARK_DEFINE_F(BM_VecSimSpaces_FP32, FP32_InnerProductSIMD32_AVX512F)
(benchmark::State &st) {
    if (!avx512f_supported) {
        st.SkipWithError("This benchmark requires AVX512F, which is not available");
        return;
    }
    auto func = spaces::Choose_FP32_IP_implementation_AVX512F(dim);
    for (auto _ : st) {
        func(v1, v2, dim);
    }
}

BENCHMARK_REGISTER_F(BM_VecSimSpaces_FP32, FP32_InnerProductSIMD32_AVX512F)
    ->ArgName("Dimension")
    ->Unit(benchmark::kNanosecond)
    ->Arg(400);

#endif // OPT_AVX512F
#endif // x86_64

BENCHMARK_MAIN();
