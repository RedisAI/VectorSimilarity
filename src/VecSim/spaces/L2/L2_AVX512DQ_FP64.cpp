/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "L2_AVX512DQ.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/L2/L2.h"

double FP64_L2SqrSIMD2Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    size_t qty8 = qty >> 3 << 3;

    const double *pEnd1 = pVect1 + qty8;
    const double *pEnd2 = pVect1 + qty;

    __m512d diff512, v1_512, v2_512;
    __m512d sum512 = _mm512_set1_pd(0);

    // In each iteration we calculate 8 doubles = 512 bits.
    while (pVect1 < pEnd1) {
        v1_512 = _mm512_loadu_pd(pVect1);
        pVect1 += 8;
        v2_512 = _mm512_loadu_pd(pVect2);
        pVect2 += 8;
        diff512 = _mm512_sub_pd(v1_512, v2_512);
        sum512 = _mm512_add_pd(sum512, _mm512_mul_pd(diff512, diff512));
    }

    __m128d diff, v1, v2;
    __m128d sum = _mm512_extractf64x2_pd(sum512, 0) + _mm512_extractf64x2_pd(sum512, 1) +
                  _mm512_extractf64x2_pd(sum512, 2) + _mm512_extractf64x2_pd(sum512, 3);

    // In each iteration we calculate 2 doubles = 128 bits.
    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_pd(pVect1);
        pVect1 += 2;
        v2 = _mm_loadu_pd(pVect2);
        pVect2 += 2;
        diff = _mm_sub_pd(v1, v2);
        sum = _mm_add_pd(sum, _mm_mul_pd(diff, diff));
    }

    double PORTABLE_ALIGN16 TmpRes[2];
    _mm_store_pd(TmpRes, sum);
    return TmpRes[0] + TmpRes[1];
}

double FP64_L2SqrSIMD2ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v, size_t qty) {
    // Calculate how many doubles we can calculate using 128 bits iterations.
    size_t qty4 = qty >> 1 << 1;
    double res = FP64_L2SqrSIMD2Ext_AVX512(pVect1v, pVect2v, qty4);
    double *pVect1 = (double *)pVect1v + qty4;
    double *pVect2 = (double *)pVect2v + qty4;

    // Calculate the rest using the basic function
    size_t qty_left = qty - qty4;
    double res_tail = FP64_L2Sqr(pVect1, pVect2, qty_left);
    return (res + res_tail);
}
