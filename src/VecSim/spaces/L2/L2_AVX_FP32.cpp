/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "L2_AVX.h"
#include "L2.h"
#include "VecSim/spaces/space_includes.h"

float FP32_L2SqrSIMD16Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + qty;

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    // In each iteration we calculate 16 floats = 512 bits.
    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    float PORTABLE_ALIGN32 TmpRes[8];
    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
           TmpRes[7];
}

float FP32_L2SqrSIMD4Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    // Calculate how many floats we can calculate using 512 bits iterations.
    size_t qty16 = qty >> 4 << 4;

    const float *pEnd1 = pVect1 + qty16;
    const float *pEnd2 = pVect1 + qty;

    __m256 diff256, v1_256, v2_256;
    __m256 sum256 = _mm256_set1_ps(0);

    // In each iteration we calculate 16 floats = 512 bits.
    while (pVect1 < pEnd1) {
        v1_256 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2_256 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff256 = _mm256_sub_ps(v1_256, v2_256);
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff256, diff256));

        v1_256 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2_256 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff256 = _mm256_sub_ps(v1_256, v2_256);
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff256, diff256));
    }

    __m128 diff, v1, v2;
    __m128 sum = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    // In each iteration we calculate 4 floats = 128 bits.
    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    float PORTABLE_ALIGN16 TmpRes[4];
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

float FP32_L2SqrSIMD16ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    // Calculate how many floats we can calculate using 512 bits iterations.
    size_t qty16 = qty >> 4 << 4;
    float res = FP32_L2SqrSIMD16Ext_AVX(pVect1v, pVect2v, qty16);
    float *pVect1 = (float *)pVect1v + qty16;
    float *pVect2 = (float *)pVect2v + qty16;

    // Calculate the rest using the basic function
    size_t qty_left = qty - qty16;
    float res_tail = FP32_L2Sqr(pVect1, pVect2, qty_left);
    return (res + res_tail);
}

float FP32_L2SqrSIMD4ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    // Calculate how many floats we can calculate using 128 bits iterations.
    size_t qty4 = qty >> 2 << 2;
    float res = FP32_L2SqrSIMD4Ext_AVX(pVect1v, pVect2v, qty4);
    float *pVect1 = (float *)pVect1v + qty4;
    float *pVect2 = (float *)pVect2v + qty4;

    // Calculate the rest using the basic function
    size_t qty_left = qty - qty4;
    float res_tail = FP32_L2Sqr(pVect1, pVect2, qty_left);
    return (res + res_tail);
}
