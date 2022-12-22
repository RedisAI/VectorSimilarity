/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#include "VecSim/spaces/conveters/bf16/BF16_converter_AVX512.h"
#include "VecSim/vec_sim_common.h"
#include <immintrin.h>
#include <memory.h>


void FP32_to_BF16_AVX512_SIMD16(const void *pVect1v, void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    bf16 *pVect2 = (bf16 *)pVect2v;

    const float *pEnd1 = pVect1 + qty;

    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m256bh v2 = _mm512_cvtneps_pbh (v1);
        memcpy(pVect2, &v2, 16*sizeof(bf16));
        
        pVect1++;
        pVect2++;
    }
}
