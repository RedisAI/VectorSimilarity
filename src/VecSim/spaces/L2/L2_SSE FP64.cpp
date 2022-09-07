#include "L2_SSE.h"
#include "L2.h"
#include "VecSim/spaces/space_includes.h"

float FP32_L2SqrSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    float PORTABLE_ALIGN16 TmpRes[4];

    const float *pEnd1 = pVect1 + qty;

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    // In each iteration we calculate 16 floats = 512 bits.
    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

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

float FP32_L2SqrSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {
    // Calc how many doubles we can calc using 512 bits iterations.
    size_t qty16 = qty >> 4 << 4;
    float res = FP32_L2SqrSIMD16Ext_SSE(pVect1v, pVect2v, qty16);
    float *pVect1 = (float *)pVect1v + qty16;
    float *pVect2 = (float *)pVect2v + qty16;

    // Calc the rest using a brute force function
    size_t qty_left = qty - qty16;
    float res_tail = FP32_L2Sqr(pVect1, pVect2, qty_left);
    return (res + res_tail);
}

float FP32_L2SqrSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {
    float PORTABLE_ALIGN16 TmpRes[4];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + qty;

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    // In each iteration we calculate 4 floats = 128 bits.
    while (pVect1 < pEnd1) {
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

float FP32_L2SqrSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {
    // Calc how many doubles we can calc using 512 bits iterations.
    size_t qty4 = qty >> 2 << 2;

    float res = FP32_L2SqrSIMD4Ext_SSE(pVect1v, pVect2v, qty4);
    float *pVect1 = (float *)pVect1v + qty4;
    float *pVect2 = (float *)pVect2v + qty4;
    
    // Calc the rest using a brute force function
    size_t qty_left = qty - qty4;
    float res_tail = FP32_L2Sqr(pVect1, pVect2, qty_left);

    return (res + res_tail);
}
