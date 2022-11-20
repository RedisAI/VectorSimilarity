/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "IP_SSE.h"
#include "IP.h"
#include "VecSim/spaces/space_includes.h"

double FP64_InnerProductSIMD8Ext_SSE_impl(const void *pVect1v, const void *pVect2v, size_t qty) {

    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    const double *pEnd1 = pVect1 + qty;

    __m128d v1, v2;
    __m128d sum_prod = _mm_set1_pd(0);

    // In each iteration we calculate 8 doubles = 512 bits in total.
    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_pd(pVect1);
        pVect1 += 2;
        v2 = _mm_loadu_pd(pVect2);
        pVect2 += 2;
        sum_prod = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));

        v1 = _mm_loadu_pd(pVect1);
        pVect1 += 2;
        v2 = _mm_loadu_pd(pVect2);
        pVect2 += 2;
        sum_prod = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));

        v1 = _mm_loadu_pd(pVect1);
        pVect1 += 2;
        v2 = _mm_loadu_pd(pVect2);
        pVect2 += 2;
        sum_prod = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));

        v1 = _mm_loadu_pd(pVect1);
        pVect1 += 2;
        v2 = _mm_loadu_pd(pVect2);
        pVect2 += 2;
        sum_prod = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));
    }

    double PORTABLE_ALIGN16 TmpRes[2];
    _mm_store_pd(TmpRes, sum_prod);

    return TmpRes[0] + TmpRes[1];
}

double FP64_InnerProductSIMD8Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {
    return 1.0 - FP64_InnerProductSIMD8Ext_SSE_impl(pVect1v, pVect2v, qty);
}

double FP64_InnerProductSIMD8ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                              size_t qty) {
    size_t qty8 = qty >> 3 << 3;
    double res = FP64_InnerProductSIMD8Ext_SSE_impl(pVect1v, pVect2v, qty8);
    double *pVect1 = (double *)pVect1v + qty8;
    double *pVect2 = (double *)pVect2v + qty8;

    size_t qty_left = qty - qty8;
    double res_tail = FP64_InnerProduct_impl(pVect1, pVect2, qty_left);
    return 1.0 - (res + res_tail);
}

double FP64_InnerProductSIMD2Ext_SSE_impl(const void *pVect1v, const void *pVect2v, size_t qty) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    // Calculate how many doubles we can calculate using 512 bits iterations.
    size_t qty8 = qty >> 3 << 3;
    const double *pEnd1 = pVect1 + qty8;

    const double *pEnd2 = pVect1 + qty;

    __m128d v1, v2;
    __m128d sum_prod = _mm_set1_pd(0);

    // In each iteration we calculate 8 doubles = 512 bits in total.
    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_pd(pVect1);
        pVect1 += 2;
        v2 = _mm_loadu_pd(pVect2);
        pVect2 += 2;
        sum_prod = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));

        v1 = _mm_loadu_pd(pVect1);
        pVect1 += 2;
        v2 = _mm_loadu_pd(pVect2);
        pVect2 += 2;
        sum_prod = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));

        v1 = _mm_loadu_pd(pVect1);
        pVect1 += 2;
        v2 = _mm_loadu_pd(pVect2);
        pVect2 += 2;
        sum_prod = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));

        v1 = _mm_loadu_pd(pVect1);
        pVect1 += 2;
        v2 = _mm_loadu_pd(pVect2);
        pVect2 += 2;
        sum_prod = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));
    }

    // Calculate the rest using 128 bits iterations.
    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_pd(pVect1);
        pVect1 += 2;
        v2 = _mm_loadu_pd(pVect2);
        pVect2 += 2;
        sum_prod = _mm_add_pd(sum_prod, _mm_mul_pd(v1, v2));
    }

    // TmpRes must be 16 bytes aligned
    double PORTABLE_ALIGN16 TmpRes[2];
    _mm_store_pd(TmpRes, sum_prod);

    return TmpRes[0] + TmpRes[1];
}

double FP64_InnerProductSIMD2Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {
    return 1.0 - FP64_InnerProductSIMD2Ext_SSE_impl(pVect1v, pVect2v, qty);
}

double FP64_InnerProductSIMD2ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                              size_t qty) {
    // Calculate how many doubles we can calculate using 128 bits iterations.
    size_t qty2 = qty >> 1 << 1;

    double res = FP64_InnerProductSIMD2Ext_SSE_impl(pVect1v, pVect2v, qty2);
    double *pVect1 = (double *)pVect1v + qty2;
    double *pVect2 = (double *)pVect2v + qty2;

    // Calculate the rest using the basic function.
    size_t qty_left = qty - qty2;
    double res_tail = FP64_InnerProduct_impl(pVect1, pVect2, qty_left);

    return 1.0 - (res + res_tail);
}
