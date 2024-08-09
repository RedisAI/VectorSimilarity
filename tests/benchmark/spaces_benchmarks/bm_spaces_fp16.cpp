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

// OPT_AVX512FP16 functions
#ifdef OPT_AVX512_FP16


class BM_VecSimSpaces_FP16_adv : public BM_VecSimSpaces<_Float16> {};

bool avx512fp16_supported = opt.avx512_fp16;
// INITIALIZE_BM(BM_VecSimSpaces_FP16_adv, FP16, AVX512FP16,<metric>, <bm_name>, avx512fp16_supported)                 \
//         ->Arg(<dim>)
INITIALIZE_BM(BM_VecSimSpaces_FP16_adv, FP16, AVX512FP16,<metric>, <bm_name>, avx512fp16_supported) \
        ->Arg(<dim>);
#endif // OPT_AVX512_FP16

#endif // x86_64

BENCHMARK_MAIN();
