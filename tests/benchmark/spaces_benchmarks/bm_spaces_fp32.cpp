/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#define DATA_TYPE float
#include "bm_spaces.h"

#ifdef CPU_FEATURES_ARCH_X86_64
cpu_features::X86Features opt = cpu_features::GetX86Info().features;

// AVX512 functions
#ifdef OPT_AVX512F
bool avx512_supported = opt.avx512f;
INITIALIZE_BENCHMARKS_SET(FP32, AVX512, 16, avx512_supported);
#endif // AVX512F

// AVX functions
#ifdef OPT_AVX
bool avx_supported = opt.avx;
INITIALIZE_BENCHMARKS_SET(FP32, AVX, 16, avx_supported);
#endif // AVX

// SSE functions
#ifdef OPT_SSE
bool sse_supported = opt.sse;
INITIALIZE_BENCHMARKS_SET(FP32, SSE, 16, sse_supported);
#endif // SSE

#endif // x86_64

// Naive algorithms

INITIALIZE_NAIVE_BM(FP32, InnerProduct, 16);
INITIALIZE_NAIVE_BM(FP32, L2Sqr, 16);

// Naive

BENCHMARK_MAIN();
