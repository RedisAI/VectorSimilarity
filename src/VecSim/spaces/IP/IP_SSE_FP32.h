/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void InnerProductStep(float *&pVect1, float *&pVect2, __m128 &sum_prod) {
    __m128 v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    __m128 v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
}

template <unsigned char residual> // 0..15
float FP32_InnerProductSIMD16_SSE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    __m128 sum_prod = _mm_setzero_ps();

    // Deal with %4 remainder first. `dim` is >16, so we have at least one 16-float block,
    // so loading 4 floats and then masking them is safe.
    if (residual % 4) {
        __m128 v1, v2;
        if (residual % 4 == 3) {
            // Load 3 floats and set the last one to 0
            v1 = _mm_load_ps(pVect1); // load 4 floats
            v2 = _mm_load_ps(pVect2);
            v1 = _mm_blend_ps(_mm_setzero_ps(), v1, 7); // set the last one to 0
            v2 = _mm_blend_ps(_mm_setzero_ps(), v2, 7);
        } else if (residual % 4 == 2) {
            // Load 2 floats and set the last two to 0
            v1 = _mm_loadh_pi(_mm_setzero_ps(), (__m64 *)pVect1);
            v2 = _mm_loadh_pi(_mm_setzero_ps(), (__m64 *)pVect2);
        } else if (residual % 4 == 1) {
            // Load 1 float and set the last three to 0
            v1 = _mm_load_ss(pVect1);
            v2 = _mm_load_ss(pVect2);
        }
        pVect1 += residual % 4;
        pVect2 += residual % 4;
        sum_prod = _mm_mul_ps(v1, v2);
    }

    // have another 1, 2 or 3 4-float steps according to residual
    if (residual >= 12)
        InnerProductStep(pVect1, pVect2, sum_prod);
    if (residual >= 8)
        InnerProductStep(pVect1, pVect2, sum_prod);
    if (residual >= 4)
        InnerProductStep(pVect1, pVect2, sum_prod);

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    // In each iteration we calculate 16 floats = 512 bits.
    do {
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
    } while (pVect1 < pEnd1);

    // TmpRes must be 16 bytes aligned.
    float PORTABLE_ALIGN16 TmpRes[4];
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return 1.0f - sum;
}
