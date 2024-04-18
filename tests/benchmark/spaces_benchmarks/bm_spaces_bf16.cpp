/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/types/bfloat16.h"
#define DATA_TYPE vecsim_types::bfloat16
#include "bm_spaces.h"

spaces::Arch_Optimization BM_VecSimSpaces::opt = spaces::getArchitectureOptimization();

// AVX512 functions
#ifdef OPT_AVX512_BW_VBMI2
bool avx512_bw_vbmi2_supported =
    BM_VecSimSpaces::opt.features.avx512bw && BM_VecSimSpaces::opt.features.avx512vbmi2;
INITIALIZE_BENCHMARKS_SET(BF16, AVX512BW_VBMI2, 32, avx512_bw_vbmi2_supported);
#endif // AVX512F

// AVX functions
#ifdef OPT_AVX2
bool avx2_supported = BM_VecSimSpaces::opt.features.avx2;
INITIALIZE_BENCHMARKS_SET(BF16, AVX2, 32, avx2_supported);
#endif // AVX

// SSE functions
#ifdef OPT_SSE3
bool sse3_supported = BM_VecSimSpaces::opt.features.sse3;
INITIALIZE_BENCHMARKS_SET(BF16, SSE3, 32, sse3_supported);
#endif // SSE

INITIALIZE_NAIVE_BM(BF16, InnerProduct_LittleEndian, 32);
INITIALIZE_NAIVE_BM(BF16, L2Sqr_LittleEndian, 32);
BENCHMARK_MAIN();
