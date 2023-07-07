/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"

static inline void L2SqrStep(float *&pVect1, float *&pVect2, __m256 &sum) {
    __m256 v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    __m256 v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    __m256 diff = _mm256_sub_ps(v1, v2);
    // sum = _mm256_fmadd_ps(diff, diff, sum);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
}

template <unsigned char residual> // 0..15
float FP32_L2SqrSIMD16_AVX(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    __m256 sum = _mm256_setzero_ps();

    // Deal with 1-7 floats with mask loading, if needed
    if (residual % 8) {
        __mmask8 constexpr mask8 = (1 << (residual % 8)) - 1;
        __m256 v1 = my_mm256_maskz_loadu_ps<mask8>(pVect1);
        pVect1 += residual % 8;
        __m256 v2 = my_mm256_maskz_loadu_ps<mask8>(pVect2);
        pVect2 += residual % 8;
        __m256 diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_mul_ps(diff, diff);
    }

    // If the reminder is >=8, have another step of 8 floats
    if (residual >= 8) {
        L2SqrStep(pVect1, pVect2, sum);
    }

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    // In each iteration we calculate 16 floats = 512 bits.
    do {
        L2SqrStep(pVect1, pVect2, sum);
        L2SqrStep(pVect1, pVect2, sum);
    } while (pVect1 < pEnd1);

    float PORTABLE_ALIGN32 TmpRes[8];
    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
           TmpRes[7];
}
