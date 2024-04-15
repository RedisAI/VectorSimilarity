/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#define DATA_TYPE double
#include "bm_spaces.h"

// AVX512 functions
#ifdef OPT_AVX512F

INITIALIZE_BENCHMARKS_SET(FP64, AVX512_F, 8);

#endif // AVX512F

// AVX functions
#ifdef OPT_AVX

INITIALIZE_BENCHMARKS_SET(FP64, AVX, 8);
#endif // AVX

// SSE functions
#ifdef OPT_SSE

INITIALIZE_BENCHMARKS_SET(FP64, SSE, 8);

#endif // SSE
// Naive algorithms
INITIALIZE_NAIVE_BM(FP64, InnerProduct, 8);
INITIALIZE_NAIVE_BM(FP64, L2Sqr, 8);
// Naive

BENCHMARK_MAIN();
