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

// See the note in L2_AVX512FP16_VL_FP16.h: the products accumulate in fp32, matching
// `FP16_InnerProduct` and mirroring IP_AVX512F_FP16.h, because a half precision accumulator both
// loses precision on every add and overflows to infinity past 65504, the largest finite fp16 value.
static void InnerProductStep(float16 *&pVect1, float16 *&pVect2, __m512 &sum) {
    // Convert 16 half-floats into floats and store them in 512 bits register.
    auto v1 = _mm512_cvtph_ps(_mm256_lddqu_si256((__m256i *)pVect1));
    auto v2 = _mm512_cvtph_ps(_mm256_lddqu_si256((__m256i *)pVect2));

    sum = _mm512_fmadd_ps(v1, v2, sum);
    pVect1 += 16;
    pVect2 += 16;
}

template <unsigned short residual> // 0..31
float FP16_InnerProductSIMD32_AVX512FP16_VL(const void *pVect1v, const void *pVect2v,
                                            size_t dimension) {
    auto *pVect1 = (float16 *)pVect1v;
    auto *pVect2 = (float16 *)pVect2v;

    const float16 *pEnd1 = pVect1 + dimension;

    // Two accumulators break the FMA dependency chain, letting more FMAs be in flight at once.
    auto sum0 = _mm512_setzero_ps();
    auto sum1 = _mm512_setzero_ps();

    if constexpr (residual % 16) {
        __mmask16 constexpr residuals_mask = (1 << (residual % 16)) - 1;
        auto v1 = _mm512_maskz_mov_ps(residuals_mask,
                                      _mm512_cvtph_ps(_mm256_lddqu_si256((__m256i *)pVect1)));
        auto v2 = _mm512_maskz_mov_ps(residuals_mask,
                                      _mm512_cvtph_ps(_mm256_lddqu_si256((__m256i *)pVect2)));
        sum0 = _mm512_mul_ps(v1, v2);
        pVect1 += residual % 16;
        pVect2 += residual % 16;
    }
    if constexpr (residual >= 16) {
        InnerProductStep(pVect1, pVect2, sum1);
    }

    // We dealt with the residual part. We are left with some multiple of 32 16-bit floats.
    while (pVect1 < pEnd1) {
        InnerProductStep(pVect1, pVect2, sum0);
        InnerProductStep(pVect1, pVect2, sum1);
    }

    return 1.0f - _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
}
