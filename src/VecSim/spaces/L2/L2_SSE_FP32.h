/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void L2SqrStep(float *&pVect1, float *&pVect2, __m128 &sum) {
    __m128 v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    __m128 v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    __m128 diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
}

template <unsigned char residual> // 0..15
float FP32_L2SqrSIMD16_SSE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    __m128 sum = _mm_setzero_ps();

    // Deal with %4 remainder first. `dim` is >16, so we have at least one 16-float block,
    // so loading 4 floats and then masking them is safe.
    if (residual % 4) {
        __m128 v1, v2, diff;
        if (residual % 4 == 3) {
            // Load 3 floats and set the last one to 0
            v1 = _mm_load_ps(pVect1); // load 4 floats
            v2 = _mm_load_ps(pVect2);
            // sets the last float of v1 to the last of v2, so the diff is 0.
            v1 = _mm_blend_ps(v2, v1, 7);
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
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_mul_ps(diff, diff);
    }

    // have another 1, 2 or 3 4-floats steps according to residual
    if (residual >= 12)
        L2SqrStep(pVect1, pVect2, sum);
    if (residual >= 8)
        L2SqrStep(pVect1, pVect2, sum);
    if (residual >= 4)
        L2SqrStep(pVect1, pVect2, sum);

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    // In each iteration we calculate 16 floats = 512 bits.
    do {
        L2SqrStep(pVect1, pVect2, sum);
        L2SqrStep(pVect1, pVect2, sum);
        L2SqrStep(pVect1, pVect2, sum);
        L2SqrStep(pVect1, pVect2, sum);
    } while (pVect1 < pEnd1);

    // TmpRes must be 16 bytes aligned
    float PORTABLE_ALIGN16 TmpRes[4];
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
