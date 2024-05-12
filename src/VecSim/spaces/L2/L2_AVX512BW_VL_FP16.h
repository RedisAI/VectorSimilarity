/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <cstdint>
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"
#include "VecSim/types/float16.h"

using float16 = vecsim_types::float16;

static void L2SqrStep(float16 *&pVect1, float16 *&pVect2, __m512 &sum) {
    // Convert 16 half-floats into floats and store them in 512 bits register.
    auto v1 = _mm512_cvtph_ps(_mm256_loadu_epi16(pVect1));
    auto v2 = _mm512_cvtph_ps(_mm256_loadu_epi16(pVect2));

    // sum = (v1 - v2)^2 + sum
    auto c = _mm512_sub_ps(v1, v2);
    sum = _mm512_fmadd_ps(c, c, sum);
    pVect1 += 16;
    pVect2 += 16;
}

template <unsigned short residual> // 0..31
float FP16_L2SqrSIMD32_AVX512BW_VL(const void *pVect1v, const void *pVect2v, size_t dimension) {
    auto *pVect1 = (float16 *)pVect1v;
    auto *pVect2 = (float16 *)pVect2v;

    const float16 *pEnd1 = pVect1 + dimension;

    auto sum = _mm512_setzero_ps();

    if (residual % 16) {
        // Deal with remainder first. `dim` is more than 32, so we have at least one block of 32
        // 16-bit float so mask loading is guaranteed to be safe.
        __mmask16 constexpr residuals_mask = (1 << (residual % 16)) - 1;
        // Convert the first half-floats in the residual positions into floats and store them
        // 512 bits register, where the floats in the positions corresponding to the non-residuals
        // positions are zeros.
        auto v1 = _mm512_maskz_mov_ps(residuals_mask, _mm512_cvtph_ps(_mm256_loadu_epi16(pVect1)));
        auto v2 = _mm512_maskz_mov_ps(residuals_mask, _mm512_cvtph_ps(_mm256_loadu_epi16(pVect2)));
        auto c = _mm512_sub_ps(v1, v2);
        sum = _mm512_mul_ps(c, c);
        pVect1 += residual % 16;
        pVect2 += residual % 16;
    }
    if (residual >= 16) {
        L2SqrStep(pVect1, pVect2, sum);
    }

    // We dealt with the residual part. We are left with some multiple of 32 16-bit floats.
    // In every iteration we process 2 chunk of 256bit (32 FP16)
    do {
        L2SqrStep(pVect1, pVect2, sum);
        L2SqrStep(pVect1, pVect2, sum);
    } while (pVect1 < pEnd1);

    return _mm512_reduce_add_ps(sum);
}
