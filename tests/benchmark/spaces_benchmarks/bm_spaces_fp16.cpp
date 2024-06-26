/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/types/float16.h"
#define DATA_TYPE vecsim_types::float16
#include "bm_spaces.h"

#ifdef CPU_FEATURES_ARCH_X86_64
cpu_features::X86Features opt = cpu_features::GetX86Info().features;

// AVX512_BW_VL functions
#ifdef OPT_AVX512F
bool avx512f_supported = opt.avx512f;
INITIALIZE_BENCHMARKS_SET(FP16, AVX512F, 32, avx512f_supported);
#endif // OPT_AVX512F

// AVX functions
#ifdef OPT_F16C
bool avx512_bw_f16c_supported = opt.f16c && opt.fma3 && opt.avx;
INITIALIZE_BENCHMARKS_SET(FP16, F16C, 32, avx512_bw_f16c_supported);
#endif // OPT_F16C

#endif // x86_64

INITIALIZE_NAIVE_BM(FP16, InnerProduct, 32);
INITIALIZE_NAIVE_BM(FP16, L2Sqr, 32);

BENCHMARK_MAIN();
