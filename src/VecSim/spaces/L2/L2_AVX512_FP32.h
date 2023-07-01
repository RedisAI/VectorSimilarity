/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void L2SqrStep(float *&pVect1, float *&pVect2, __m512 &sum) {
    __m512 v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    __m512 v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    __m512 diff = _mm512_sub_ps(v1, v2);
    // sum = _mm512_fmadd_ps(diff, diff, sum);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
}

template <unsigned char residual> // 0..15
float FP32_L2SqrSIMD16Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + qty;

    __m512 sum = _mm512_setzero_ps();

    if (residual) {
        __mmask16 constexpr mask = (1 << residual) - 1;
        __m512 v1 = _mm512_maskz_loadu_ps(mask, pVect1);
        pVect1 += residual;
        __m512 v2 = _mm512_maskz_loadu_ps(mask, pVect2);
        pVect2 += residual;
        __m512 diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_mul_ps(diff, diff);
    }

    // In each iteration we calculate 16 floats = 512 bits.
    while (pVect1 < pEnd1) {
        L2SqrStep(pVect1, pVect2, sum);
    }

    return _mm512_reduce_add_ps(sum);
}
