/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void InnerProductStep(float *&pVect1, float *&pVect2, __m512 &sum512) {
    __m512 v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    __m512 v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
}

template <unsigned char residual> // 0..15
float FP32_InnerProductSIMD16_AVX512(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    __m512 sum512 = _mm512_setzero_ps();

    // Deal with remainder first. `dim` is more than 16, so we have at least one 16-float block,
    // so mask loading is guaranteed to be safe
    if (residual) {
        __mmask16 constexpr mask = (1 << residual) - 1;
        __m512 v1 = _mm512_maskz_loadu_ps(mask, pVect1);
        pVect1 += residual;
        __m512 v2 = _mm512_maskz_loadu_ps(mask, pVect2);
        pVect2 += residual;
        sum512 = _mm512_mul_ps(v1, v2);
    }

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    do {
        InnerProductStep(pVect1, pVect2, sum512);
    } while (pVect1 < pEnd1);

    return 1.0f - _mm512_reduce_add_ps(sum512);
}
