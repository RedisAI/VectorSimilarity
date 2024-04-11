/*
*Copyright Redis Ltd. 2021 - present
*Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
*the Server Side Public License v1 (SSPLv1).
*/

#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"

static inline void InnerProductStep(uint16_t *&pVect1, uint16_t *&pVect2, __m256 &sum256) {
    // Convert the 8 half-floats into floats and store them in a 256-bit register.
    __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect1));
    __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect2));
    sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

    pVect1 += 8;
    pVect2 += 8;
}

template <unsigned char residual> // 0..31
float FP16_InnerProductSIMD16_AVX(const void *pVect1v, const void *pVect2v, size_t dimension) {
    auto *pVect1 = (uint16_t *)pVect1v;
    auto *pVect2 = (uint16_t *)pVect2v;

    const uint16_t *pEnd1 = pVect1 + dimension;

    __m256 sum256 = _mm256_setzero_ps();

    // Deal with 1-15 floats with mask loading, if needed. `dim` is > 32, so we have at least one
    // 32 FP16 block, so mask loading is guaranteed to be safe.
    if (residual % 16) {
        __mmask16 constexpr mask = (1 << (residual % 16)) - 1;
        // Convert the 8 half-floats into floats and store them in a 256-bit register.
        __m256 v1 = _mm256_blend_ps(_mm256_setzero_ps(),
                                    _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect1)),
                                    mask);
        __m256 v2 = _mm256_blend_ps(_mm256_setzero_ps(),
                                    _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect2)),
                                    mask);
        sum256 = _mm256_mul_ps(v1, v2);

        pVect1 += residual % 16;
        pVect2 += residual % 16;
    }

    // We dealt with the residual part. We are left with some multiple of 32 FP16.
    // In each iteration we calculate 8 FP16 = 256 bits.
    do {
        InnerProductStep(pVect1, pVect2, sum256);
        InnerProductStep(pVect1, pVect2, sum256);
    } while (pVect1 < pEnd1);

    float PORTABLE_ALIGN32 TmpRes[8];
    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7];

    return 1.0f - sum;
}
