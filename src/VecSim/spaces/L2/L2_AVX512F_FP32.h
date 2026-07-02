/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"

static inline void L2SqrStep(float *&pVect1, float *&pVect2, __m512 &sum) {
    __m512 v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    __m512 v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    __m512 diff = _mm512_sub_ps(v1, v2);

    sum = _mm512_fmadd_ps(diff, diff, sum);
}

template <unsigned char residual> // 0..15
float FP32_L2SqrSIMD16_AVX512(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    // Four accumulators break the FMA dependency chain, letting more FMAs be in flight at once.
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    // Deal with remainder first. AVX-512 masked loads suppress faults on masked-out lanes, so
    // this is safe for any dimension.
    if constexpr (residual) {
        __mmask16 constexpr mask = (1 << residual) - 1;
        __m512 v1 = _mm512_maskz_loadu_ps(mask, pVect1);
        pVect1 += residual;
        __m512 v2 = _mm512_maskz_loadu_ps(mask, pVect2);
        pVect2 += residual;
        __m512 diff = _mm512_sub_ps(v1, v2);
        sum0 = _mm512_mul_ps(diff, diff);
    }

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    // The main loop handles 64 floats per iteration; leftover blocks of 16/32/48 floats are
    // handled after it. The loops may run zero times (dim can be as small as 8).
    while (pVect1 + 64 <= pEnd1) {
        L2SqrStep(pVect1, pVect2, sum0);
        L2SqrStep(pVect1, pVect2, sum1);
        L2SqrStep(pVect1, pVect2, sum2);
        L2SqrStep(pVect1, pVect2, sum3);
    }
    if (pVect1 + 32 <= pEnd1) {
        L2SqrStep(pVect1, pVect2, sum1);
        L2SqrStep(pVect1, pVect2, sum2);
    }
    if (pVect1 < pEnd1) {
        L2SqrStep(pVect1, pVect2, sum3);
    }

    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    return _mm512_reduce_add_ps(sum);
}
