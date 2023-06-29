/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

#define DIM_MOD_4_IS(mask, val)                                                                    \
    (((mask & 0x000F) == (((1 << val) - 1) << 0)) ||                                               \
     ((mask & 0x00F0) == (((1 << val) - 1) << 4)) ||                                               \
     ((mask & 0x0F00) == (((1 << val) - 1) << 8)) ||                                               \
     ((mask & 0xF000) == (((1 << val) - 1) << 12)))

static inline void InnerProductStep(float *&pVect1, float *&pVect2, __m128 &sum_prod) {
    __m128 v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    __m128 v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
}

template <__mmask16 mask>
float FP32_InnerProductSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + qty - (mask ? 16 : 0);

    __m128 sum_prod = _mm_setzero_ps();

    // In each iteration we calculate 16 floats = 512 bits.
    while (pVect1 < pEnd1) {
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
    }

    if (mask > ((1 << 12) - 1))
        InnerProductStep(pVect1, pVect2, sum_prod);
    if (mask > ((1 << 8) - 1))
        InnerProductStep(pVect1, pVect2, sum_prod);
    if (mask > ((1 << 4) - 1))
        InnerProductStep(pVect1, pVect2, sum_prod);

    if (DIM_MOD_4_IS(mask, 3)) {
        __m128 v1 = _mm_load_ss(pVect1);
        __m128 v2 = _mm_load_ss(pVect2);
        v1 = _mm_loadh_pi(v1, (__m64 *)(pVect1 + 1));
        v2 = _mm_loadh_pi(v2, (__m64 *)(pVect2 + 1));
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    } else if (DIM_MOD_4_IS(mask, 2)) {
        __m128 v1 = _mm_loadh_pi(_mm_setzero_ps(), (__m64 *)pVect1);
        __m128 v2 = _mm_loadh_pi(_mm_setzero_ps(), (__m64 *)pVect2);
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    } else if (DIM_MOD_4_IS(mask, 1)) {
        __m128 v1 = _mm_load_ss(pVect1);
        __m128 v2 = _mm_load_ss(pVect2);
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    // TmpRes must be 16 bytes aligned.
    float PORTABLE_ALIGN16 TmpRes[4];
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return 1.0f - sum;
}
