/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"

static inline void InnerProductStep(double *&pVect1, double *&pVect2, __m512d &sum512) {
    __m512d v1 = _mm512_loadu_pd(pVect1);
    pVect1 += 8;
    __m512d v2 = _mm512_loadu_pd(pVect2);
    pVect2 += 8;
    sum512 = _mm512_fmadd_pd(v1, v2, sum512);
}

template <unsigned char residual> // 0..31
double FP64_InnerProductSIMD8_AVX512(const void *pVect1v, const void *pVect2v, size_t dimension) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    const double *pEnd1 = pVect1 + dimension;

    // Four accumulators break the FMA dependency chain, letting more FMAs be in flight at once.
    __m512d sum0 = _mm512_setzero_pd();
    __m512d sum1 = _mm512_setzero_pd();
    __m512d sum2 = _mm512_setzero_pd();
    __m512d sum3 = _mm512_setzero_pd();

    // Deal with the sub-8 remainder first. AVX-512 masked loads suppress faults on masked-out
    // lanes, so this is safe for any dimension.
    if constexpr (residual % 8) {
        __mmask8 constexpr mask = (1 << (residual % 8)) - 1;
        __m512d v1 = _mm512_maskz_loadu_pd(mask, pVect1);
        pVect1 += residual % 8;
        __m512d v2 = _mm512_maskz_loadu_pd(mask, pVect2);
        pVect2 += residual % 8;
        sum0 = _mm512_mul_pd(v1, v2);
    }

    // Handle the remaining full 8-double blocks of the residual (compile-time resolved).
    if constexpr (residual >= 8) {
        InnerProductStep(pVect1, pVect2, sum1);
    }
    if constexpr (residual >= 16) {
        InnerProductStep(pVect1, pVect2, sum2);
    }
    if constexpr (residual >= 24) {
        InnerProductStep(pVect1, pVect2, sum3);
    }

    // We dealt with the residual part. We are left with some multiple of 32 doubles.
    // In each iteration we calculate 32 doubles = 4 chunks of 512 bits. The loop may run zero
    // times (dim can be as small as 4).
    while (pVect1 < pEnd1) {
        InnerProductStep(pVect1, pVect2, sum0);
        InnerProductStep(pVect1, pVect2, sum1);
        InnerProductStep(pVect1, pVect2, sum2);
        InnerProductStep(pVect1, pVect2, sum3);
    }

    __m512d sum512 = _mm512_add_pd(_mm512_add_pd(sum0, sum1), _mm512_add_pd(sum2, sum3));
    return 1.0 - _mm512_reduce_add_pd(sum512);
}
