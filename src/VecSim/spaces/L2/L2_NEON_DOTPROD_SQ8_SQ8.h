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
#include <arm_neon.h>

/**
 * SQ8-to-SQ8 L2 squared distance functions for NEON with DOTPROD extension.
 * Computes L2 squared distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses algebraic optimization with INTEGER arithmetic throughout:
 *
 * L2² = Σ((q1[i]*δ1 + min1) - (q2[i]*δ2 + min2))²
 *
 * Let c = min1 - min2, then:
 * L2² = δ1²*Σq1² + δ2²*Σq2² - 2*δ1*δ2*Σ(q1*q2) + 2*c*δ1*Σq1 - 2*c*δ2*Σq2 + dim*c²
 *
 * All sums are computed using integer arithmetic, converted to float only at the end.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)]
 */

// Helper function: computes dot product, squares, and element sums using integer arithmetic
__attribute__((always_inline)) static inline void
SQ8_SQ8_L2SqrStep_NEON_DOTPROD(const uint8_t *&pVec1, const uint8_t *&pVec2, uint32x4_t &dot_sum,
                               uint32x4_t &sqr1_sum, uint32x4_t &sqr2_sum, uint32x4_t &sum1,
                               uint32x4_t &sum2) {
    // Ones vector for computing element sums via dot product
    static const uint8x16_t ones = vdupq_n_u8(1);

    // Load 16 uint8 elements
    uint8x16_t v1 = vld1q_u8(pVec1);
    uint8x16_t v2 = vld1q_u8(pVec2);

    // Compute dot product: q1*q2
    dot_sum = vdotq_u32(dot_sum, v1, v2);

    // Compute sum of squares: q1*q1, q2*q2
    sqr1_sum = vdotq_u32(sqr1_sum, v1, v1);
    sqr2_sum = vdotq_u32(sqr2_sum, v2, v2);

    // Compute element sums
    sum1 = vdotq_u32(sum1, v1, ones);
    sum2 = vdotq_u32(sum2, v2, ones);

    pVec1 += 16;
    pVec2 += 16;
}

// Common implementation for L2 squared distance between two SQ8 vectors
template <unsigned char residual> // 0..63
float SQ8_SQ8_L2SqrSIMD64_NEON_DOTPROD(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);

    // Get dequantization parameters
    const float min1 = *reinterpret_cast<const float *>(pVec1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVec1 + dimension + sizeof(float));
    const float min2 = *reinterpret_cast<const float *>(pVec2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVec2 + dimension + sizeof(float));

    // Integer accumulators (dual for ILP)
    uint32x4_t dot_sum0 = vdupq_n_u32(0);
    uint32x4_t dot_sum1 = vdupq_n_u32(0);
    uint32x4_t sqr1_sum0 = vdupq_n_u32(0);
    uint32x4_t sqr1_sum1 = vdupq_n_u32(0);
    uint32x4_t sqr2_sum0 = vdupq_n_u32(0);
    uint32x4_t sqr2_sum1 = vdupq_n_u32(0);
    uint32x4_t sum1_0 = vdupq_n_u32(0);
    uint32x4_t sum1_1 = vdupq_n_u32(0);
    uint32x4_t sum2_0 = vdupq_n_u32(0);
    uint32x4_t sum2_1 = vdupq_n_u32(0);

    // Handle residual elements first (0-15 elements)
    constexpr size_t final_residual = residual % 16;
    if constexpr (final_residual > 0) {
        static const uint8x16_t ones = vdupq_n_u8(1);
        constexpr uint8x16_t mask = {
            0xFF,
            (final_residual >= 2) ? 0xFF : 0,
            (final_residual >= 3) ? 0xFF : 0,
            (final_residual >= 4) ? 0xFF : 0,
            (final_residual >= 5) ? 0xFF : 0,
            (final_residual >= 6) ? 0xFF : 0,
            (final_residual >= 7) ? 0xFF : 0,
            (final_residual >= 8) ? 0xFF : 0,
            (final_residual >= 9) ? 0xFF : 0,
            (final_residual >= 10) ? 0xFF : 0,
            (final_residual >= 11) ? 0xFF : 0,
            (final_residual >= 12) ? 0xFF : 0,
            (final_residual >= 13) ? 0xFF : 0,
            (final_residual >= 14) ? 0xFF : 0,
            (final_residual >= 15) ? 0xFF : 0,
            0,
        };

        uint8x16_t v1 = vld1q_u8(pVec1);
        uint8x16_t v2 = vld1q_u8(pVec2);
        uint8x16_t zeros = vdupq_n_u8(0);
        v1 = vbslq_u8(mask, v1, zeros);
        v2 = vbslq_u8(mask, v2, zeros);

        dot_sum1 = vdotq_u32(dot_sum1, v1, v2);
        sqr1_sum1 = vdotq_u32(sqr1_sum1, v1, v1);
        sqr2_sum1 = vdotq_u32(sqr2_sum1, v2, v2);
        sum1_1 = vdotq_u32(sum1_1, v1, ones);
        sum2_1 = vdotq_u32(sum2_1, v2, ones);

        pVec1 += final_residual;
        pVec2 += final_residual;
    }

    // Process 64 elements at a time
    const size_t num_of_chunks = dimension / 64;
    for (size_t i = 0; i < num_of_chunks; i++) {
        SQ8_SQ8_L2SqrStep_NEON_DOTPROD(pVec1, pVec2, dot_sum0, sqr1_sum0, sqr2_sum0, sum1_0, sum2_0);
        SQ8_SQ8_L2SqrStep_NEON_DOTPROD(pVec1, pVec2, dot_sum1, sqr1_sum1, sqr2_sum1, sum1_1, sum2_1);
        SQ8_SQ8_L2SqrStep_NEON_DOTPROD(pVec1, pVec2, dot_sum0, sqr1_sum0, sqr2_sum0, sum1_0, sum2_0);
        SQ8_SQ8_L2SqrStep_NEON_DOTPROD(pVec1, pVec2, dot_sum1, sqr1_sum1, sqr2_sum1, sum1_1, sum2_1);
    }

    // Handle remaining 16-element chunks
    constexpr size_t residual_chunks = residual / 16;
    if constexpr (residual_chunks >= 1) {
        SQ8_SQ8_L2SqrStep_NEON_DOTPROD(pVec1, pVec2, dot_sum0, sqr1_sum0, sqr2_sum0, sum1_0, sum2_0);
    }
    if constexpr (residual_chunks >= 2) {
        SQ8_SQ8_L2SqrStep_NEON_DOTPROD(pVec1, pVec2, dot_sum1, sqr1_sum1, sqr2_sum1, sum1_1, sum2_1);
    }
    if constexpr (residual_chunks >= 3) {
        SQ8_SQ8_L2SqrStep_NEON_DOTPROD(pVec1, pVec2, dot_sum0, sqr1_sum0, sqr2_sum0, sum1_0, sum2_0);
    }

    // Combine and reduce to scalar integers
    uint32_t dot_product = vaddvq_u32(vaddq_u32(dot_sum0, dot_sum1));
    uint32_t sum_sqr1 = vaddvq_u32(vaddq_u32(sqr1_sum0, sqr1_sum1));
    uint32_t sum_sqr2 = vaddvq_u32(vaddq_u32(sqr2_sum0, sqr2_sum1));
    uint32_t v1_sum = vaddvq_u32(vaddq_u32(sum1_0, sum1_1));
    uint32_t v2_sum = vaddvq_u32(vaddq_u32(sum2_0, sum2_1));

    // Apply the algebraic formula:
    // L2² = δ1²*Σq1² + δ2²*Σq2² - 2*δ1*δ2*Σ(q1*q2) + 2*c*δ1*Σq1 - 2*c*δ2*Σq2 + dim*c²
    float c = min1 - min2;
    return delta1 * delta1 * static_cast<float>(sum_sqr1) +
           delta2 * delta2 * static_cast<float>(sum_sqr2) -
           2.0f * delta1 * delta2 * static_cast<float>(dot_product) +
           2.0f * c * delta1 * static_cast<float>(v1_sum) -
           2.0f * c * delta2 * static_cast<float>(v2_sum) +
           static_cast<float>(dimension) * c * c;
}

