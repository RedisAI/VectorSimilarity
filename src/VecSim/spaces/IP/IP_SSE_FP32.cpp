#include "IP_SSE.h"
#include "IP.h"
#include "VecSim/spaces/space_includes.h"


float FP32_InnerProductSIMD16Ext_SSE_impl(const void *pVect1v, const void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + qty;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    // In each iteration we calculate 16 floats = 512 bits.
    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    float PORTABLE_ALIGN32 TmpRes[8];
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

float FP32_InnerProductSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {
    return 1.0f - FP32_InnerProductSIMD16Ext_SSE_impl(pVect1v, pVect2v, qty);
}

float FP32_InnerProductSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                              size_t qty) {
    // Calc how many floats we can calc using 512 bits iterations.
    size_t qty16 = qty >> 4 << 4;
    float res = FP32_InnerProductSIMD16Ext_SSE_impl(pVect1v, pVect2v, qty16);
    float *pVect1 = (float *)pVect1v + qty16;
    float *pVect2 = (float *)pVect2v + qty16;

    // Calc the rest using a brute force function
    size_t qty_left = qty - qty16;
    float res_tail = FP32_InnerProduct_impl(pVect1, pVect2, qty_left);
    return 1.0f - (res + res_tail);
}

float FP32_InnerProductSIMD4Ext_SSE_impl(const void *pVect1v, const void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    // Calc how many floats we can calc using 512 bits iterations.
    size_t qty16 = qty >> 4 << 4;
    const float *pEnd1 = pVect1 + qty16;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    // Calc the rest using 128 bits iterations.
    const float *pEnd2 = pVect1 + qty;
    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    float PORTABLE_ALIGN16 TmpRes[4];
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

float FP32_InnerProductSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {
    return 1.0f - FP32_InnerProductSIMD4Ext_SSE_impl(pVect1v, pVect2v, qty);
}


float FP32_InnerProductSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {
    // Calc how many floats we can calc using 128 bits iterations. TODO change in L2
    size_t qty4 = qty >> 2 << 2;

    float res = FP32_InnerProductSIMD4Ext_SSE_impl(pVect1v, pVect2v, qty4);
    float *pVect1 = (float *)pVect1v + qty4;
    float *pVect2 = (float *)pVect2v + qty4;

    // Calc the rest using a brute force function
    size_t qty_left = qty - qty4;
    float res_tail = FP32_InnerProduct_impl(pVect1, pVect2, qty_left);

    return 1.0f - (res + res_tail);
}
