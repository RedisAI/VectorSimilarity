/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#define DATA_TYPE double
#include "bm_spaces.h"

// AVX512 functions
#ifdef OPT_AVX512F

INITIALIZE_EXACT_512BIT_BM(FP64, AVX512_F, L2, 8);
INITIALIZE_EXACT_128BIT_BM(FP64, AVX512_F, L2, 2);
INITIALIZE_RESIDUAL_BM(FP64, AVX512_F, L2, 8_Residuals);

INITIALIZE_EXACT_512BIT_BM(FP64, AVX512_F, IP, 8);
INITIALIZE_EXACT_128BIT_BM(FP64, AVX512_F, IP, 2);
INITIALIZE_RESIDUAL_BM(FP64, AVX512_F, IP, 8_Residuals);

#endif // AVX512F

// AVX functions
#ifdef OPT_AVX

INITIALIZE_EXACT_512BIT_BM(FP64, AVX, L2, 8);
INITIALIZE_EXACT_128BIT_BM(FP64, AVX, L2, 2);
INITIALIZE_RESIDUAL_BM(FP64, AVX, L2, 8_Residuals);

INITIALIZE_EXACT_512BIT_BM(FP64, AVX, IP, 8);
INITIALIZE_EXACT_128BIT_BM(FP64, AVX, IP, 2);
INITIALIZE_RESIDUAL_BM(FP64, AVX, IP, 8_Residuals);
#endif // AVX

// SSE functions
#ifdef OPT_SSE

INITIALIZE_EXACT_512BIT_BM(FP64, SSE, L2, 8);
INITIALIZE_EXACT_128BIT_BM(FP64, SSE, L2, 2);
INITIALIZE_RESIDUAL_BM(FP64, SSE, L2, 8_Residuals);

INITIALIZE_EXACT_512BIT_BM(FP64, SSE, IP, 8);
INITIALIZE_EXACT_128BIT_BM(FP64, SSE, IP, 2);
INITIALIZE_RESIDUAL_BM(FP64, SSE, IP, 8_Residuals);

#endif // SSE
// Naive algorithms

INITIALIZE_NAIVE_BM(FP64, InnerProduct)->EXACT_512BIT_PARAMS->RESIDUAL_PARAMS;
INITIALIZE_NAIVE_BM(FP64, L2Sqr)->EXACT_512BIT_PARAMS->RESIDUAL_PARAMS;

// Naive

BENCHMARK_MAIN();
