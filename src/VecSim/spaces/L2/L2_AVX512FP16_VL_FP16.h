/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include <cstdint>
#include "VecSim/spaces/space_includes.h"
#include "VecSim/types/float16.h"
#include <cstring>

using float16 = vecsim_types::float16;

// The arithmetic here is fp32, not fp16. `FP16_L2Sqr` widens each stored half with FP16_to_FP32 and
// accumulates into a float, so accumulating in half precision would make this tier disagree with
// the same function computed anywhere else. It is also unsafe: 65504 is the largest finite fp16
// value, so 32 elements of 200.0, all ordinary fp16 values, drive a half precision accumulator
// past it and the result becomes infinity. An fp32 accumulator cannot overflow for any fp16 input.
// This mirrors L2_AVX512F_FP16.h step for step; after widening there is nothing
// half-precision-specific left to do differently.
static inline void L2SqrStep(float16 *&pVect1, float16 *&pVect2, __m512 &sum) {
    // Convert 16 half-floats into floats and store them in 512 bits register.
    auto v1 = _mm512_cvtph_ps(_mm256_lddqu_si256((__m256i *)pVect1));
    auto v2 = _mm512_cvtph_ps(_mm256_lddqu_si256((__m256i *)pVect2));

    // sum = (v1 - v2)^2 + sum
    auto c = _mm512_sub_ps(v1, v2);
    sum = _mm512_fmadd_ps(c, c, sum);
    pVect1 += 16;
    pVect2 += 16;
}

template <unsigned short residual> // 0..31
float FP16_L2SqrSIMD32_AVX512FP16_VL(const void *pVect1v, const void *pVect2v, size_t dimension) {
    auto *pVect1 = (float16 *)pVect1v;
    auto *pVect2 = (float16 *)pVect2v;

    const float16 *pEnd1 = pVect1 + dimension;

    // Two accumulators break the FMA dependency chain, letting more FMAs be in flight at once.
    auto sum0 = _mm512_setzero_ps();
    auto sum1 = _mm512_setzero_ps();

    if constexpr (residual % 16) {
        // Deal with remainder first. The full-width load of 16 16-bit floats is safe because
        // `dim` is at least 16, so the vector spans at least 16 elements.
        __mmask16 constexpr residuals_mask = (1 << (residual % 16)) - 1;
        auto v1 = _mm512_maskz_mov_ps(residuals_mask,
                                      _mm512_cvtph_ps(_mm256_lddqu_si256((__m256i *)pVect1)));
        auto v2 = _mm512_maskz_mov_ps(residuals_mask,
                                      _mm512_cvtph_ps(_mm256_lddqu_si256((__m256i *)pVect2)));
        auto c = _mm512_sub_ps(v1, v2);
        sum0 = _mm512_mul_ps(c, c);
        pVect1 += residual % 16;
        pVect2 += residual % 16;
    }
    // Handle the remaining full 16-element block of the residual (compile-time resolved).
    if constexpr (residual >= 16) {
        L2SqrStep(pVect1, pVect2, sum1);
    }

    // We dealt with the residual part. We are left with some multiple of 32 16-bit floats.
    while (pVect1 < pEnd1) {
        L2SqrStep(pVect1, pVect2, sum0);
        L2SqrStep(pVect1, pVect2, sum1);
    }

    return _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
}
