/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void InnerProductStep(float *&pVect1, float *&pVect2, __m256 &sum256) {
    __m256 v1 = _mm256_loadu_ps(pVect1); // AVX
    pVect1 += 8;
    __m256 v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    sum256 = _mm256_fmadd_ps(v1, v2, sum256); // FMA
}

template <unsigned char residual> // 0..15
float FP32_InnerProductSIMD16_AVX512(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    __m256 sum256 = _mm256_setzero_ps(); // AVX

    // Deal with remainder first. `dim` is more than 16, so we have at least one 16-float block,
    // so mask loading is guaranteed to be safe
    if constexpr (residual % 8) {
        __mmask8 constexpr mask = (1 << (residual % 8)) - 1;
        __m256 v1 = _mm256_maskz_loadu_ps(mask, pVect1); // AVX512F + AVX512VL
        pVect1 += residual % 8;
        __m256 v2 = _mm256_maskz_loadu_ps(mask, pVect2);
        pVect2 += residual % 8;
        sum256 = _mm256_mul_ps(v1, v2);
    }
    // If the reminder is >=8, have another step of 8 floats
    if constexpr (residual >= 8) {
        InnerProductStep(pVect1, pVect2, sum256);
    }

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    do {
        InnerProductStep(pVect1, pVect2, sum256);
        InnerProductStep(pVect1, pVect2, sum256);
    } while (pVect1 < pEnd1);

    float PORTABLE_ALIGN32 TmpRes[8];
    _mm256_store_ps(TmpRes, sum256); // AVX
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
           TmpRes[7];
    return 1.0f - res;
}
