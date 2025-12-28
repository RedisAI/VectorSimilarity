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
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)]
 */

// Helper function to process 4 elements
static inline void SQ8_SQ8_L2SqrStep_NEON(const uint8_t *&pVec1, const uint8_t *&pVec2,
                                          float32x4_t &dot_sum, float32x4_t &sqr1_sum,
                                          float32x4_t &sqr2_sum, float32x4_t &sum1,
                                          float32x4_t &sum2) {
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

    // Accumulate sum of squares: q1*q1, q2*q2
    sqr1_sum = vmlaq_f32(sqr1_sum, v1_f, v1_f);
    sqr2_sum = vmlaq_f32(sqr2_sum, v2_f, v2_f);

    // Accumulate element sums
    sum1 = vaddq_f32(sum1, v1_f);
    sum2 = vaddq_f32(sum2, v2_f);

    pVec1 += 4;
    pVec2 += 4;
}

// Common implementation for L2 squared distance between two SQ8 vectors
template <unsigned char residual> // 0..15
float SQ8_SQ8_L2SqrSIMD16_NEON(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);

    // Get dequantization parameters from the end of pVec1
    const float min1 = *reinterpret_cast<const float *>(pVec1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVec1 + dimension + sizeof(float));

    // Get dequantization parameters from the end of pVec2
    const float min2 = *reinterpret_cast<const float *>(pVec2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVec2 + dimension + sizeof(float));

    const size_t num_of_chunks = dimension / 16;

    // Multiple accumulators for ILP
    float32x4_t dot_sum0 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum1 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum2 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum3 = vdupq_n_f32(0.0f);
    float32x4_t sqr1_sum0 = vdupq_n_f32(0.0f);
    float32x4_t sqr1_sum1 = vdupq_n_f32(0.0f);
    float32x4_t sqr1_sum2 = vdupq_n_f32(0.0f);
    float32x4_t sqr1_sum3 = vdupq_n_f32(0.0f);
    float32x4_t sqr2_sum0 = vdupq_n_f32(0.0f);
    float32x4_t sqr2_sum1 = vdupq_n_f32(0.0f);
    float32x4_t sqr2_sum2 = vdupq_n_f32(0.0f);
    float32x4_t sqr2_sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum1_0 = vdupq_n_f32(0.0f);
    float32x4_t sum1_1 = vdupq_n_f32(0.0f);
    float32x4_t sum1_2 = vdupq_n_f32(0.0f);
    float32x4_t sum1_3 = vdupq_n_f32(0.0f);
    float32x4_t sum2_0 = vdupq_n_f32(0.0f);
    float32x4_t sum2_1 = vdupq_n_f32(0.0f);
    float32x4_t sum2_2 = vdupq_n_f32(0.0f);
    float32x4_t sum2_3 = vdupq_n_f32(0.0f);

    // Process 16 elements at a time
    for (size_t i = 0; i < num_of_chunks; i++) {
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum0, sqr1_sum0, sqr2_sum0, sum1_0, sum2_0);
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum1, sqr1_sum1, sqr2_sum1, sum1_1, sum2_1);
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum2, sqr1_sum2, sqr2_sum2, sum1_2, sum2_2);
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum3, sqr1_sum3, sqr2_sum3, sum1_3, sum2_3);
    }

    // Handle residual complete 4-element blocks
    if constexpr (residual >= 4) {
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum0, sqr1_sum0, sqr2_sum0, sum1_0, sum2_0);
    }
    if constexpr (residual >= 8) {
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum1, sqr1_sum1, sqr2_sum1, sum1_1, sum2_1);
    }
    if constexpr (residual >= 12) {
        SQ8_SQ8_L2SqrStep_NEON(pVec1, pVec2, dot_sum2, sqr1_sum2, sqr2_sum2, sum1_2, sum2_2);
    }

    // Combine accumulators
    float32x4_t dot_total = vaddq_f32(vaddq_f32(dot_sum0, dot_sum1), vaddq_f32(dot_sum2, dot_sum3));
    float32x4_t sqr1_total =
        vaddq_f32(vaddq_f32(sqr1_sum0, sqr1_sum1), vaddq_f32(sqr1_sum2, sqr1_sum3));
    float32x4_t sqr2_total =
        vaddq_f32(vaddq_f32(sqr2_sum0, sqr2_sum1), vaddq_f32(sqr2_sum2, sqr2_sum3));
    float32x4_t sum1_total = vaddq_f32(vaddq_f32(sum1_0, sum1_1), vaddq_f32(sum1_2, sum1_3));
    float32x4_t sum2_total = vaddq_f32(vaddq_f32(sum2_0, sum2_1), vaddq_f32(sum2_2, sum2_3));

    // Handle remaining 1-3 elements
    float dot_scalar = 0.0f, sqr1_scalar = 0.0f, sqr2_scalar = 0.0f;
    float sum1_scalar = 0.0f, sum2_scalar = 0.0f;
    constexpr size_t leftover = residual % 4;
    for (size_t i = 0; i < leftover; i++) {
        float v1 = static_cast<float>(pVec1[i]);
        float v2 = static_cast<float>(pVec2[i]);
        dot_scalar += v1 * v2;
        sqr1_scalar += v1 * v1;
        sqr2_scalar += v2 * v2;
        sum1_scalar += v1;
        sum2_scalar += v2;
    }

    // Horizontal sums
    float32x2_t dot_h = vadd_f32(vget_low_f32(dot_total), vget_high_f32(dot_total));
    float dot_product = vget_lane_f32(vpadd_f32(dot_h, dot_h), 0) + dot_scalar;

    float32x2_t sqr1_h = vadd_f32(vget_low_f32(sqr1_total), vget_high_f32(sqr1_total));
    float sum_sqr1 = vget_lane_f32(vpadd_f32(sqr1_h, sqr1_h), 0) + sqr1_scalar;

    float32x2_t sqr2_h = vadd_f32(vget_low_f32(sqr2_total), vget_high_f32(sqr2_total));
    float sum_sqr2 = vget_lane_f32(vpadd_f32(sqr2_h, sqr2_h), 0) + sqr2_scalar;

    float32x2_t s1_h = vadd_f32(vget_low_f32(sum1_total), vget_high_f32(sum1_total));
    float v1_sum = vget_lane_f32(vpadd_f32(s1_h, s1_h), 0) + sum1_scalar;

    float32x2_t s2_h = vadd_f32(vget_low_f32(sum2_total), vget_high_f32(sum2_total));
    float v2_sum = vget_lane_f32(vpadd_f32(s2_h, s2_h), 0) + sum2_scalar;

    // Apply the algebraic formula:
    // L2² = δ1²*Σq1² + δ2²*Σq2² - 2*δ1*δ2*Σ(q1*q2) + 2*c*δ1*Σq1 - 2*c*δ2*Σq2 + dim*c²
    float c = min1 - min2;
    return delta1 * delta1 * sum_sqr1 + delta2 * delta2 * sum_sqr2 -
           2.0f * delta1 * delta2 * dot_product + 2.0f * c * delta1 * v1_sum -
           2.0f * c * delta2 * v2_sum + static_cast<float>(dimension) * c * c;
}

