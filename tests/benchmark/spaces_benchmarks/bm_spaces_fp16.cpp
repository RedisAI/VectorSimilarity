/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/types/float16.h"
#define DATA_TYPE vecsim_types::float16
#include "bm_spaces.h"

// AVX512_BW_VL functions
#ifdef OPT_AVX512_BW_VL

INITIALIZE_BENCHMARKS_SET(FP16, AVX512_BW_VL, 32);
#endif // OPT_AVX512_BW_VL

// AVX functions
#ifdef OPT_F16C

INITIALIZE_BENCHMARKS_SET(FP16, F16C, 32);
#endif // OPT_F16C

INITIALIZE_NAIVE_BM(FP16, InnerProduct, 32);
INITIALIZE_NAIVE_BM(FP16, L2Sqr, 32);

BENCHMARK_MAIN();
