/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static void InnerProductStep(uint16_t *&pVect1, uint16_t *&pVect2, __m512 &sum512) {
    // To be used with AVX512_FP16 optimizations:
    // __m256i packed_half_floats = _mm256_castph_si256(_mm256_loadu_ph(pVect1));
    // __m512 v1 = _mm512_cvtph_ps(packed_half_floats);

    // Convert the first 8 half-floats into floats and store them in the lower 256 bits of a
    // 512-bit register.
    __m512 v1_first_half = _mm512_castps256_ps512(
        _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect1)));
    __m512 v2_first_half = _mm512_castps256_ps512(
        _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect2)));

    // Convert the next 8 half-floats into floats and store them in the lower 256 bits of a
    // 512-bit register.
    __m512 v1_second_half = _mm512_castps256_ps512(
        _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)(pVect1 + 8))));
    __m512 v2_second_half = _mm512_castps256_ps512(
        _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)(pVect2 + 8))));

    // The following loads v*_first_half into the lower bits of the target and v*_second_half
    // into the upper bits of the target, using this mask.
    __mmask16 constexpr mask = 0b1111111100000000;
    __m512 v1 = _mm512_mask_expand_ps(v1_first_half, mask, v1_second_half);
    __m512 v2 = _mm512_mask_expand_ps(v2_first_half, mask, v2_second_half);

    sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));

    pVect1 += 32;
    pVect2 += 32;
}

template <unsigned short residual> // 0..31
float FP16_InnerProductSIMD16_AVX512(const void *pVect1v, const void *pVect2v, size_t dimension) {
    auto *pVect1 = (uint16_t *)pVect1v;
    auto *pVect2 = (uint16_t *)pVect2v;

    const uint16_t *pEnd1 = pVect1 + dimension;

    __m512 sum512 = _mm512_setzero_ps();

    // Deal with remainder first. `dim` is more than 32, so we have at least one block of 32 16-bit
    // float so mask loading is guaranteed to be safe.
    __mmask16 constexpr residuals_mask = (1 << (residual % 16)) - 1;

    if (residual >= 16) {
        // Convert the first 8 half-floats into floats and store them in the lower 256 bits of a
        // 512-bit register.
        __m512 v1_first_half =
            _mm512_castps256_ps512(_mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect1)));
        __m512 v2_first_half =
            _mm512_castps256_ps512(_mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect2)));

        // Convert the next 8 half-floats into floats and store them in the lower 256 bits of a
        // 512-bit register, where the floats in the positions corresponding to residuals are zeros.
        __m512 v1_second_half = _mm512_castps256_ps512(
            _mm256_blend_ps(_mm256_setzero_ps(),
                            _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)(pVect1 + 8))),
                            residuals_mask));
        __m512 v2_second_half = _mm512_castps256_ps512(
            _mm256_blend_ps(_mm256_setzero_ps(),
                            _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)(pVect2 + 8))),
                            residuals_mask));

        // The following loads v*_first_half into the lower bits of the target and v*_second_half
        // into the upper bits of the target, using this mask.
        __mmask16 constexpr mask = 0b1111111100000000;
        __m512 v1 = _mm512_mask_expand_ps(v1_first_half, mask, v1_second_half);
        __m512 v2 = _mm512_mask_expand_ps(v2_first_half, mask, v2_second_half);
        sum512 = _mm512_mul_ps(v1, v2);

    } else if (residual) {  // 0 < residual < 16
        // Convert the first 8 half-floats into floats and store them in the lower 256 bits of a
        // 512-bit register, where the floats in the positions corresponding to residuals are zeros.
        // Also, the upper 256 bits in the target will be zeroed as well.
        __m512 v1 = _mm512_zextps256_ps512(
            _mm256_blend_ps(_mm256_setzero_ps(),
                            _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect1)),
                            residuals_mask));
        __m512 v2 = _mm512_zextps256_ps512(
            _mm256_blend_ps(_mm256_setzero_ps(),
                            _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect2)),
                            residuals_mask));
        sum512 = _mm512_mul_ps(v1, v2);
    }

    pVect1 += residual;
    pVect2 += residual;

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    do {
        InnerProductStep(pVect1, pVect2, sum512);
    } while (pVect1 < pEnd1);

    return 1.0f - _mm512_reduce_add_ps(sum512);
}
