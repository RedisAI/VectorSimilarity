/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <iostream>
#include "L2_AVX512.h"
#include "L2.h"
#include "VecSim/spaces/space_includes.h"

double FP64_L2SqrSIMD8Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    const double *pEnd1 = pVect1 + qty;

    __m512d diff, v1, v2;
    __m512d sum = _mm512_set1_pd(0);

    // In each iteration we calculate 8 doubles = 512 bits.
    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_pd(pVect1);
        pVect1 += 8;
        v2 = _mm512_loadu_pd(pVect2);
        pVect2 += 8;
        diff = _mm512_sub_pd(v1, v2);
        sum = _mm512_add_pd(sum, _mm512_mul_pd(diff, diff));
    }

    return _mm512_reduce_add_pd(sum);
}

double FP64_L2SqrSIMD8ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v, size_t qty) {
    // Calculate how many doubles we can calculate using 512 bits iterations.
    size_t qty8 = qty >> 3 << 3;
    double res = FP64_L2SqrSIMD8Ext_AVX512(pVect1v, pVect2v, qty8);
    double *pVect1 = (double *)pVect1v + qty8;
    double *pVect2 = (double *)pVect2v + qty8;

    // Calculate the rest using the basic function
    size_t qty_left = qty - qty8;
    double res_tail = FP64_L2Sqr(pVect1, pVect2, qty_left);
    return (res + res_tail);
}

double FP64_L2SqrSIMD2Ext_AVX512_noDQ(const void *pVect1v, const void *pVect2v, size_t qty) {
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

    // Store the res for the first qty / 8 of the vectors.
    double resHead = _mm512_reduce_add_pd(sum512);

    __m128d v1, v2, diff;
    __m128d sum = _mm_set1_pd(0);
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
    return TmpRes[0] + TmpRes[1] + resHead;
}

double FP64_L2SqrSIMD2ExtResiduals_AVX512_noDQ(const void *pVect1v, const void *pVect2v,
                                               size_t qty) {
    size_t qty2 = qty >> 1 << 1;
    double res = FP64_L2SqrSIMD2Ext_AVX512_noDQ(pVect1v, pVect2v, qty2);
    double *pVect1 = (double *)pVect1v + qty2;
    double *pVect2 = (double *)pVect2v + qty2;

    size_t qty_left = qty - qty2;
    double res_tail = FP64_L2Sqr(pVect1, pVect2, qty_left);
    return res + res_tail;
}
