/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void InnerProductStep(double *&pVect1, double *&pVect2, __m512d &sum512) {
    __m512d v1 = _mm512_loadu_pd(pVect1);
    pVect1 += 8;
    __m512d v2 = _mm512_loadu_pd(pVect2);
    pVect2 += 8;
    sum512 = _mm512_add_pd(sum512, _mm512_mul_pd(v1, v2));
}

template <__mmask8 mask>
double FP64_InnerProductSIMD8Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    const double *pEnd1 = pVect1 + qty - 8;

    __m512d sum512 = _mm512_setzero_pd();

    while (pVect1 <= pEnd1) {
        InnerProductStep(pVect1, pVect2, sum512);
    }

    if (mask != 0) {
        __m512d v1 = _mm512_maskz_loadu_pd(mask, pVect1);
        __m512d v2 = _mm512_maskz_loadu_pd(mask, pVect2);
        sum512 = _mm512_add_pd(sum512, _mm512_mul_pd(v1, v2));
    }

    return 1.0 - _mm512_reduce_add_pd(sum512);
}
