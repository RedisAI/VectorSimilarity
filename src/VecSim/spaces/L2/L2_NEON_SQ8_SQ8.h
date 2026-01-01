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
 * SQ8-to-SQ8 L2 squared distance functions for NEON.
 * Computes L2 squared distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses algebraic optimization:
 *
 * L2² = Σ((q1[i]*δ1 + min1) - (q2[i]*δ2 + min2))²
 *
 * Let c = min1 - min2, then:
 * L2² = δ1²*Σq1² + δ2²*Σq2² - 2*δ1*δ2*Σ(q1*q2) + 2*c*δ1*Σq1 - 2*c*δ2*Σq2 + dim*c²
 *
 * The vector's sum (Σq) and sum of squares (Σq²) are precomputed and stored in the vector data.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)] [sum_of_squares (float)]
 */

// Helper function to process 4 elements - only computes dot product now
static inline void SQ8_SQ8_L2SqrStep_NEON(const uint8_t *&pVec1, const uint8_t *&pVec2,
                                          float32x4_t &dot_sum) {
    // Load 4 uint8 elements from pVec1 and convert to float
    uint8x8_t v1_u8 = vld1_u8(pVec1);
    uint32x4_t v1_u32 = vmovl_u16(vget_low_u16(vmovl_u8(v1_u8)));
    float32x4_t v1_f = vcvtq_f32_u32(v1_u32);

    // Load 4 uint8 elements from pVec2 and convert to float
    uint8x8_t v2_u8 = vld1_u8(pVec2);
    uint32x4_t v2_u32 = vmovl_u16(vget_low_u16(vmovl_u8(v2_u8)));
    float32x4_t v2_f = vcvtq_f32_u32(v2_u32);

    // Accumulate dot product: q1*q2
    dot_sum = vmlaq_f32(dot_sum, v1_f, v2_f);

    pVec1 += 4;
    pVec2 += 4;
}

// Common implementation for L2 squared distance between two SQ8 vectors
template <unsigned char residual> // 0..15
float SQ8_SQ8_L2SqrSIMD16_NEON(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);

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

    const size_t num_of_chunks = dimension / 16;

    // Multiple accumulators for ILP (only need dot product now)
    float32x4_t dot_sum0 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum1 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum2 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum3 = vdupq_n_f32(0.0f);

    // Process 16 elements at a time
    for (size_t i = 0; i < num_of_chunks; i++) {
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum0);
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum1);
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum2);
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum3);
    }

    // Handle residual complete 4-element blocks
    if constexpr (residual >= 4) {
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum0);
    }
    if constexpr (residual >= 8) {
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum1);
    }
    if constexpr (residual >= 12) {
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum2);
    }

    // Combine accumulators
    float32x4_t dot_total = vaddq_f32(vaddq_f32(dot_sum0, dot_sum1), vaddq_f32(dot_sum2, dot_sum3));

    // Handle remaining 1-3 elements
    float dot_scalar = 0.0f;
    constexpr size_t leftover = residual % 4;
    for (size_t i = 0; i < leftover; i++) {
        float v1 = static_cast<float>(pVec1[i]);
        float v2 = static_cast<float>(pVec2[i]);
        dot_scalar += v1 * v2;
    }

    // Horizontal sum for dot product
    float32x2_t dot_h = vadd_f32(vget_low_f32(dot_total), vget_high_f32(dot_total));
    float dot_product = vget_lane_f32(vpadd_f32(dot_h, dot_h), 0) + dot_scalar;

    // Apply the algebraic formula:
    // L2² = δ1²*Σq1² + δ2²*Σq2² - 2*δ1*δ2*Σ(q1*q2) + 2*c*δ1*Σq1 - 2*c*δ2*Σq2 + dim*c²
    float c = min1 - min2;
    return delta1 * delta1 * sum_sqr1 + delta2 * delta2 * sum_sqr2 -
           2.0f * delta1 * delta2 * dot_product + 2.0f * c * delta1 * sum_v1 -
           2.0f * c * delta2 * sum_v2 + static_cast<float>(dimension) * c * c;
}

