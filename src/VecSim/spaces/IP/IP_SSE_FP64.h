/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void InnerProductStep(double *&pVect1, double *&pVect2, __m128d &sum_prod) {
    __m128d v1 = _mm_loadu_pd(pVect1);
    pVect1 += 2;
    __m128d v2 = _mm_loadu_pd(pVect2);
    pVect2 += 2;
    sum_prod = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));
}

template <unsigned char residual> // 0..7
double FP64_InnerProductSIMD8Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {

    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    const double *pEnd1 = pVect1 + qty;

    __m128d sum_prod = _mm_setzero_pd();

    if (residual % 2 == 1) {
        // TODO: simple multiplication is faster than load+mul?
        __m128d v1 = _mm_load_sd(pVect1);
        pVect1++;
        __m128d v2 = _mm_load_sd(pVect2);
        pVect2++;
        sum_prod = _mm_mul_pd(v1, v2);
    }

    if (residual >= 6)
        InnerProductStep(pVect1, pVect2, sum_prod);
    if (residual >= 4)
        InnerProductStep(pVect1, pVect2, sum_prod);
    if (residual >= 2)
        InnerProductStep(pVect1, pVect2, sum_prod);

    // In each iteration we calculate 8 doubles = 512 bits in total.
    while (pVect1 < pEnd1) {
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
    }

    double PORTABLE_ALIGN16 TmpRes[2];
    _mm_store_pd(TmpRes, sum_prod);
    double sum = TmpRes[0] + TmpRes[1];

    return 1.0 - sum;
}
