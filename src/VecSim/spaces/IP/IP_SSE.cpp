
#include "IP_SSE.h"
#include "IP.h"
#include "VecSim/spaces/space_includes.h"

#include <stdlib.h>

float InnerProductSIMD4Ext_SSE_impl(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[4];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    size_t qty16 = qty >> 4 << 4;
    size_t qty4 = qty >> 2 << 2;

    const float *pEnd1 = pVect1 + qty16;
    const float *pEnd2 = pVect1 + qty4;

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

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

float InnerProductSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4Ext_SSE_impl(pVect1v, pVect2v, qty_ptr);
}

float InnerProductSIMD16Ext_SSE_impl(const void *pVect1v, const void *pVect2v,
                                     const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

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
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

float InnerProductSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16Ext_SSE_impl(pVect1v, pVect2v, qty_ptr);
}

float InnerProductSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                         const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext_SSE_impl(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *)pVect1v + qty16;
    float *pVect2 = (float *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct_impl(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

float InnerProductSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                        const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductSIMD4Ext_SSE_impl(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *)pVect1v + qty4;
    float *pVect2 = (float *)pVect2v + qty4;
    float res_tail = InnerProduct_impl(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}
