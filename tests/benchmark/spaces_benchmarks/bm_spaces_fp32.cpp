/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#include "bm_spaces.h"

class BM_VecSimSpaces_FP32 : public BM_VecSimSpaces<float> {};

#ifdef CPU_FEATURES_ARCH_AARCH64
cpu_features::Aarch64Features opt = cpu_features::GetAarch64Info().features;

// ARMPL NEON implementation for ARMv8-a
#ifdef OPT_NEON
bool neon_supported = true; // ARMv8-a always supports NEON
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_FP32, FP32, ARMPL_NEON, 16, neon_supported);
#endif

// ARMPL SVE2 implementation 
#ifdef OPT_SVE
bool sve2_supported = opt.sve; // Check for SVE2 support
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_FP32, FP32, ARMPL_SVE2, 16, sve2_supported);
#endif
#endif // AARCH64

#ifdef CPU_FEATURES_ARCH_X86_64
cpu_features::X86Features opt = cpu_features::GetX86Info().features;

// AVX512 functions
#ifdef OPT_AVX512F
bool avx512f_supported = opt.avx512f;
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_FP32, FP32, AVX512F, 16, avx512f_supported);
#endif // AVX512F

// AVX functions
#ifdef OPT_AVX
bool avx_supported = opt.avx;
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_FP32, FP32, AVX, 16, avx_supported);
#endif // AVX

// SSE functions
#ifdef OPT_SSE
bool sse_supported = opt.sse;
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_FP32, FP32, SSE, 16, sse_supported);
#endif // SSE

#endif // x86_64

// Naive algorithms

INITIALIZE_NAIVE_BM(BM_VecSimSpaces_FP32, FP32, InnerProduct, 16);
INITIALIZE_NAIVE_BM(BM_VecSimSpaces_FP32, FP32, L2Sqr, 16);

// Naive

BENCHMARK_MAIN();
