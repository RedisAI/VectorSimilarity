#include "L2_AVX.h"
#include "L2.h"
#include "VecSim/spaces/space_includes.h"
#include <stddef.h>

float L2SqrSIMD16Ext_AVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4 << 4;

    const float *pEnd1 = pVect1 + qty16;

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

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

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
           TmpRes[7];
}

float L2SqrSIMD4Ext_AVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN16 TmpRes[4];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    size_t qty16 = qty >> 4 << 4;
    size_t qty4 = qty >> 2 << 2;

    const float *pEnd1 = pVect1 + qty16;
    const float *pEnd2 = pVect1 + qty4;

    __m256 diff256, v1_256, v2_256;
    __m256 sum256 = _mm256_set1_ps(0);

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

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

float L2SqrSIMD16ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext_AVX(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *)pVect1v + qty16;
    float *pVect2 = (float *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}

float L2SqrSIMD4ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> 2 << 2;
    float res = L2SqrSIMD4Ext_AVX(pVect1v, pVect2v, &qty4);
    float *pVect1 = (float *)pVect1v + qty4;
    float *pVect2 = (float *)pVect2v + qty4;

    size_t qty_left = qty - qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}
