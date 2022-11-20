/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "L2_AVX.h"
#include "L2.h"
#include "VecSim/spaces/space_includes.h"

double FP64_L2SqrSIMD8Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    const double *pEnd1 = pVect1 + qty;

    __m256d diff, v1, v2;
    __m256d sum = _mm256_set1_pd(0);

    // In each iteration we calculate 8 doubles = 512 bits in total.
    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_pd(pVect1);
        pVect1 += 4;
        v2 = _mm256_loadu_pd(pVect2);
        pVect2 += 4;
        diff = _mm256_sub_pd(v1, v2);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(diff, diff));

        v1 = _mm256_loadu_pd(pVect1);
        pVect1 += 4;
        v2 = _mm256_loadu_pd(pVect2);
        pVect2 += 4;
        diff = _mm256_sub_pd(v1, v2);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(diff, diff));
    }

    double PORTABLE_ALIGN32 TmpRes[4];
    _mm256_store_pd(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

double FP64_L2SqrSIMD2Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    // Calculate how many doubles we can calculate using 512 bits iterations.
    size_t qty8 = qty >> 3 << 3;

    const double *pEnd1 = pVect1 + qty8;
    const double *pEnd2 = pVect1 + qty;

    __m256d diff256, v1_256, v2_256;
    __m256d sum256 = _mm256_set1_pd(0);

    // In each iteration we calculate 8 doubles = 512 bits in total.
    while (pVect1 < pEnd1) {
        v1_256 = _mm256_loadu_pd(pVect1);
        pVect1 += 4;
        v2_256 = _mm256_loadu_pd(pVect2);
        pVect2 += 4;
        diff256 = _mm256_sub_pd(v1_256, v2_256);
        sum256 = _mm256_add_pd(sum256, _mm256_mul_pd(diff256, diff256));

        v1_256 = _mm256_loadu_pd(pVect1);
        pVect1 += 4;
        v2_256 = _mm256_loadu_pd(pVect2);
        pVect2 += 4;
        diff256 = _mm256_sub_pd(v1_256, v2_256);
        sum256 = _mm256_add_pd(sum256, _mm256_mul_pd(diff256, diff256));
    }

    __m128d diff, v1, v2;
    __m128d sum = _mm_add_pd(_mm256_extractf128_pd(sum256, 0), _mm256_extractf128_pd(sum256, 1));

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

double FP64_L2SqrSIMD8ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    // Calculate how many doubles we can calculate using 512 bits iterations.
    size_t qty8 = qty >> 3 << 3;
    double res = FP64_L2SqrSIMD8Ext_AVX(pVect1v, pVect2v, qty8);
    double *pVect1 = (double *)pVect1v + qty8;
    double *pVect2 = (double *)pVect2v + qty8;

    // Calculate the rest using the basic function
    size_t qty_left = qty - qty8;
    double res_tail = FP64_L2Sqr(pVect1, pVect2, qty_left);
    return (res + res_tail);
}

double FP64_L2SqrSIMD2ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    // Calculate how many doubles we can calculate using 512 bits iterations.
    size_t qty2 = qty >> 1 << 1;
    double res = FP64_L2SqrSIMD2Ext_AVX(pVect1v, pVect2v, qty2);
    double *pVect1 = (double *)pVect1v + qty2;
    double *pVect2 = (double *)pVect2v + qty2;

    size_t qty_left = qty - qty2;
    double res_tail = FP64_L2Sqr(pVect1, pVect2, qty_left);
    return (res + res_tail);
}
