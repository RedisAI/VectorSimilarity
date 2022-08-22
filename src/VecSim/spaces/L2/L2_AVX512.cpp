#include "L2_AVX512.h"
#include "L2.h"
#include "VecSim/spaces/space_includes.h"

float FP32_L2SqrSIMD16Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty16 = qty >> 4 << 4;

    const float *pEnd1 = pVect1 + qty16;

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    return _mm512_reduce_add_ps(sum);
}

float FP32_L2SqrSIMD4Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty) {
    float PORTABLE_ALIGN16 TmpRes[4];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    size_t qty16 = qty >> 4 << 4;
    size_t qty4 = qty >> 2 << 2;

    const float *pEnd1 = pVect1 + qty16;
    const float *pEnd2 = pVect1 + qty4;

    __m512 diff512, v1_512, v2_512;
    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1_512 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2_512 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff512 = _mm512_sub_ps(v1_512, v2_512);
        // sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(diff512, diff512));
    }

    __m128 diff, v1, v2;
    __m128 sum = _mm512_extractf32x4_ps(sum512, 0) + _mm512_extractf32x4_ps(sum512, 1) +
                 _mm512_extractf32x4_ps(sum512, 2) + _mm512_extractf32x4_ps(sum512, 3);

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

float FP32_L2SqrSIMD16ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v, size_t qty) {
    size_t qty16 = qty >> 4 << 4;
    float res = FP32_L2SqrSIMD16Ext_AVX512(pVect1v, pVect2v, qty16);
    float *pVect1 = (float *)pVect1v + qty16;
    float *pVect2 = (float *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = FP32_L2Sqr(pVect1, pVect2, qty_left);
    return (res + res_tail);
}

float FP32_FP32_L2SqrSIMD4ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v,
                                         size_t qty) {
    size_t qty4 = qty >> 2 << 2;
    float res = FP32_L2SqrSIMD4Ext_AVX512(pVect1v, pVect2v, qty4);
    float *pVect1 = (float *)pVect1v + qty4;
    float *pVect2 = (float *)pVect2v + qty4;

    size_t qty_left = qty - qty4;
    float res_tail = FP32_L2Sqr(pVect1, pVect2, qty_left);
    return (res + res_tail);
}
