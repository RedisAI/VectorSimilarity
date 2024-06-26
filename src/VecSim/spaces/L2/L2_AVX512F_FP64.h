/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void L2SqrStep(double *&pVect1, double *&pVect2, __m512d &sum) {
    __m512d v1 = _mm512_loadu_pd(pVect1);
    pVect1 += 8;
    __m512d v2 = _mm512_loadu_pd(pVect2);
    pVect2 += 8;
    __m512d diff = _mm512_sub_pd(v1, v2);
    // sum = _mm512_fmadd_pd(diff, diff, sum);
    sum = _mm512_add_pd(sum, _mm512_mul_pd(diff, diff));
}

template <unsigned char residual> // 0..7
double FP64_L2SqrSIMD8_AVX512(const void *pVect1v, const void *pVect2v, size_t dimension) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    const double *pEnd1 = pVect1 + dimension;

    __m512d sum = _mm512_setzero_pd();

    // Deal with remainder first. `dim` is more than 8, so we have at least one 8-double block,
    // so mask loading is guaranteed to be safe
    if (residual) {
        __mmask8 constexpr mask = (1 << residual) - 1;
        __m512d v1 = _mm512_maskz_loadu_pd(mask, pVect1);
        pVect1 += residual;
        __m512d v2 = _mm512_maskz_loadu_pd(mask, pVect2);
        pVect2 += residual;
        __m512d diff = _mm512_sub_pd(v1, v2);
        sum = _mm512_mul_pd(diff, diff);
    }

    // We dealt with the residual part. We are left with some multiple of 8 doubles.
    do {
        L2SqrStep(pVect1, pVect2, sum);
    } while (pVect1 < pEnd1);

    return _mm512_reduce_add_pd(sum);
}
