/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include <cstdint>
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"
#include "VecSim/types/float16.h"

using float16 = vecsim_types::float16;

static void InnerProductStep(float16 *&pVect1, float16 *&pVect2, __m256 &sum) {
    // Convert 8 half-floats into floats and store them in 256 bits register.
    auto v1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)(pVect1)));
    auto v2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)(pVect2)));

    // sum = v1 * v2 + sum
    sum = _mm256_fmadd_ps(v1, v2, sum);
    pVect1 += 8;
    pVect2 += 8;
}

template <unsigned short residual> // 0..31
float FP16_InnerProductSIMD32_F16C(const void *pVect1v, const void *pVect2v, size_t dimension) {
    auto *pVect1 = (float16 *)pVect1v;
    auto *pVect2 = (float16 *)pVect2v;

    const float16 *pEnd1 = pVect1 + dimension;

    auto sum = _mm256_setzero_ps();

    if constexpr (residual % 8) {
        // Deal with remainder first. `dim` is more than 32, so we have at least one block of 32
        // 16-bit float so mask loading is guaranteed to be safe.
        __mmask16 constexpr residuals_mask = (1 << (residual % 8)) - 1;
        // Convert the first 8 half-floats into floats and store them 256 bits register,
        // where the floats in the positions corresponding to residuals are zeros.
        auto v1 = _mm256_blend_ps(_mm256_setzero_ps(),
                                  _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect1)),
                                  residuals_mask);
        auto v2 = _mm256_blend_ps(_mm256_setzero_ps(),
                                  _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u const *)pVect2)),
                                  residuals_mask);
        sum = _mm256_mul_ps(v1, v2);
        pVect1 += residual % 8;
        pVect2 += residual % 8;
    }
    if constexpr (residual >= 8 && residual < 16) {
        InnerProductStep(pVect1, pVect2, sum);
    } else if constexpr (residual >= 16 && residual < 24) {
        InnerProductStep(pVect1, pVect2, sum);
        InnerProductStep(pVect1, pVect2, sum);
    } else if constexpr (residual >= 24) {
        InnerProductStep(pVect1, pVect2, sum);
        InnerProductStep(pVect1, pVect2, sum);
        InnerProductStep(pVect1, pVect2, sum);
    }

    // We dealt with the residual part. We are left with some multiple of 32 16-bit floats.
    // In every iteration we process 4 chunk of 128bit (32 FP16)
    do {
        InnerProductStep(pVect1, pVect2, sum);
        InnerProductStep(pVect1, pVect2, sum);
        InnerProductStep(pVect1, pVect2, sum);
        InnerProductStep(pVect1, pVect2, sum);
    } while (pVect1 < pEnd1);

    return 1.0f - my_mm256_reduce_add_ps(sum);
}
