/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include <immintrin.h>

/**
 * SQ8-to-SQ8 L2 squared distance using AVX512 VNNI.
 * Computes L2 squared distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses algebraic optimization to leverage integer VNNI instructions:
 *
 * L2² = Σ((q1[i]*δ1 + min1) - (q2[i]*δ2 + min2))²
 *
 * Let c = min1 - min2, then:
 * L2² = Σ(q1[i]*δ1 - q2[i]*δ2 + c)²
 *     = δ1²*Σq1² + δ2²*Σq2² - 2*δ1*δ2*Σ(q1*q2) + 2*c*δ1*Σq1 - 2*c*δ2*Σq2 + dim*c²
 *
 * The vector's sum (Σq) and sum of squares (Σq²) are precomputed and stored in the vector data.
 *
 * This allows using VNNI's _mm512_dpwssd_epi32 for efficient integer dot product computation,
 * then applying scalar corrections at the end using the precomputed values.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)] [sum of squares (float)]
 */

// Process 64 uint8 elements using VNNI with multiple accumulators for ILP
static inline void SQ8_SQ8_L2SqrStep64(const uint8_t *pVec1, const uint8_t *pVec2,
                                       __m512i &dot_acc0, __m512i &dot_acc1) {
    // Load 64 bytes from each vector
    __m512i v1_full = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(pVec1));
    __m512i v2_full = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(pVec2));

    // Extract lower and upper 256-bit halves
    __m256i v1_lo = _mm512_castsi512_si256(v1_full);
    __m256i v1_hi = _mm512_extracti64x4_epi64(v1_full, 1);
    __m256i v2_lo = _mm512_castsi512_si256(v2_full);
    __m256i v2_hi = _mm512_extracti64x4_epi64(v2_full, 1);

    // Convert to int16 (zero-extend)
    __m512i v1_lo_16 = _mm512_cvtepu8_epi16(v1_lo);
    __m512i v1_hi_16 = _mm512_cvtepu8_epi16(v1_hi);
    __m512i v2_lo_16 = _mm512_cvtepu8_epi16(v2_lo);
    __m512i v2_hi_16 = _mm512_cvtepu8_epi16(v2_hi);

    // Compute dot products using VNNI: q1*q2
    dot_acc0 = _mm512_dpwssd_epi32(dot_acc0, v1_lo_16, v2_lo_16);
    dot_acc1 = _mm512_dpwssd_epi32(dot_acc1, v1_hi_16, v2_hi_16);
}

// Process 32 uint8 elements using VNNI
static inline void SQ8_SQ8_L2SqrStep32(const uint8_t *pVec1, const uint8_t *pVec2,
                                       __m512i &dot_acc) {
    // Load 32 bytes from each vector
    __m256i v1_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pVec1));
    __m256i v2_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pVec2));

    // Convert to int16 (zero-extend)
    __m512i v1_16 = _mm512_cvtepu8_epi16(v1_256);
    __m512i v2_16 = _mm512_cvtepu8_epi16(v2_256);

    // Compute dot product: q1*q2
    dot_acc = _mm512_dpwssd_epi32(dot_acc, v1_16, v2_16);
}

// Common implementation for L2 squared distance between two SQ8 vectors
template <unsigned char residual> // 0..63
float SQ8_SQ8_L2SqrSIMD64_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                              size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    const uint8_t *pEnd1 = pVec1 + dimension;

    // Get dequantization parameters and precomputed values from the end of pVec1
    // Layout: [uint8_t values (dim)] [min_val] [delta] [sum] [sum_of_squares]
    const float *params1 = reinterpret_cast<const float *>(pVec1 + dimension);
    const float min1 = params1[0];
    const float delta1 = params1[1];
    const float sum_v1 = params1[2];
    const float sum_sqr1 = params1[3];

    // Get dequantization parameters and precomputed values from the end of pVec2
    const float *params2 = reinterpret_cast<const float *>(pVec2 + dimension);
    const float min2 = params2[0];
    const float delta2 = params2[1];
    const float sum_v2 = params2[2];
    const float sum_sqr2 = params2[3];

    // Multiple accumulators for instruction-level parallelism (only need dot product now)
    __m512i dot_acc0 = _mm512_setzero_si512();  // Σ(q1*q2)
    __m512i dot_acc1 = _mm512_setzero_si512();

    // Handle residual first (0..63 elements)
    if constexpr (residual > 0) {
        if constexpr (residual < 32) {
            // Handle less than 32 elements with mask
            constexpr __mmask32 mask = (1LU << residual) - 1;
            __m256i v1_256 = _mm256_maskz_loadu_epi8(mask, pVec1);
            __m256i v2_256 = _mm256_maskz_loadu_epi8(mask, pVec2);

            // Convert to int16
            __m512i v1_16 = _mm512_cvtepu8_epi16(v1_256);
            __m512i v2_16 = _mm512_cvtepu8_epi16(v2_256);

            // Compute dot product only
            dot_acc0 = _mm512_dpwssd_epi32(dot_acc0, v1_16, v2_16);
        } else if constexpr (residual == 32) {
            // Exactly 32 elements
            SQ8_SQ8_L2SqrStep32(pVec1, pVec2, dot_acc0);
        } else {
            // 33-63 elements: use masked 64-byte load
            constexpr __mmask64 mask = (1LLU << residual) - 1;
            __m512i v1_full = _mm512_maskz_loadu_epi8(mask, pVec1);
            __m512i v2_full = _mm512_maskz_loadu_epi8(mask, pVec2);

            // Extract halves
            __m256i v1_lo = _mm512_castsi512_si256(v1_full);
            __m256i v1_hi = _mm512_extracti64x4_epi64(v1_full, 1);
            __m256i v2_lo = _mm512_castsi512_si256(v2_full);
            __m256i v2_hi = _mm512_extracti64x4_epi64(v2_full, 1);

            // Convert to int16
            __m512i v1_lo_16 = _mm512_cvtepu8_epi16(v1_lo);
            __m512i v1_hi_16 = _mm512_cvtepu8_epi16(v1_hi);
            __m512i v2_lo_16 = _mm512_cvtepu8_epi16(v2_lo);
            __m512i v2_hi_16 = _mm512_cvtepu8_epi16(v2_hi);

            // Compute dot products only
            dot_acc0 = _mm512_dpwssd_epi32(dot_acc0, v1_lo_16, v2_lo_16);
            dot_acc1 = _mm512_dpwssd_epi32(dot_acc1, v1_hi_16, v2_hi_16);
        }
        pVec1 += residual;
        pVec2 += residual;
    }

    // Process full 64-byte chunks
    while (pVec1 < pEnd1) {
        SQ8_SQ8_L2SqrStep64(pVec1, pVec2, dot_acc0, dot_acc1);
        pVec1 += 64;
        pVec2 += 64;
    }

    // Combine accumulators and reduce - only dot product needed
    __m512i dot_total = _mm512_add_epi32(dot_acc0, dot_acc1);
    int64_t dot_product = _mm512_reduce_add_epi32(dot_total);

    // Apply the algebraic formula:
    // L2² = δ1²*Σq1² + δ2²*Σq2² - 2*δ1*δ2*Σ(q1*q2) + 2*c*δ1*Σq1 - 2*c*δ2*Σq2 + dim*c²
    // where c = min1 - min2
    // Use double precision for intermediate calculations to minimize floating-point errors
    double c = static_cast<double>(min1) - static_cast<double>(min2);
    double d1 = static_cast<double>(delta1);
    double d2 = static_cast<double>(delta2);
    double delta1_sq = d1 * d1;
    double delta2_sq = d2 * d2;

    double result = delta1_sq * static_cast<double>(sum_sqr1) +
                    delta2_sq * static_cast<double>(sum_sqr2) -
                    2.0 * d1 * d2 * static_cast<double>(dot_product) +
                    2.0 * c * d1 * static_cast<double>(sum_v1) -
                    2.0 * c * d2 * static_cast<double>(sum_v2) +
                    static_cast<double>(dimension) * c * c;

    return static_cast<float>(result);
}
