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

/*
 * Optimized asymmetric SQ8 inner product using algebraic identity:
 *
 *   IP(x, y) = Σ(x_i * y_i)
 *            ≈ Σ((min + delta * q_i) * y_i)
 *            = min * Σy_i + delta * Σ(q_i * y_i)
 *            = min * y_sum + delta * quantized_dot_product
 *
 * where y_sum = Σy_i is precomputed and stored in the query blob.
 * This avoids dequantization in the hot loop - we only compute Σ(q_i * y_i).
 */

// Helper: compute Σ(q_i * y_i) for 4 elements (no dequantization)
static inline void InnerProductStepSQ8(const float *&pVect1, const uint8_t *&pVect2,
                                       float32x4_t &sum) {
    // Load 4 float elements from query
    float32x4_t v1 = vld1q_f32(pVect1);
    pVect1 += 4;

    // Load 4 uint8 elements and convert to float
    uint8x8_t v2_u8 = vld1_u8(pVect2);
    pVect2 += 4;

    uint32x4_t v2_u32 = vmovl_u16(vget_low_u16(vmovl_u8(v2_u8)));
    float32x4_t v2_f = vcvtq_f32_u32(v2_u32);

    // Accumulate q_i * y_i (no dequantization!)
    sum = vmlaq_f32(sum, v2_f, v1);
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_NEON_IMP(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // Multiple accumulators for ILP
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    const size_t num_of_chunks = dimension / 16;

    // Process 16 elements at a time in the main loop
    for (size_t i = 0; i < num_of_chunks; i++) {
        InnerProductStepSQ8(pVect1, pVect2, sum0);
        InnerProductStepSQ8(pVect1, pVect2, sum1);
        InnerProductStepSQ8(pVect1, pVect2, sum2);
        InnerProductStepSQ8(pVect1, pVect2, sum3);
    }

    // Handle remaining complete 4-element blocks within residual
    if constexpr (residual >= 4) {
        InnerProductStepSQ8(pVect1, pVect2, sum0);
    }
    if constexpr (residual >= 8) {
        InnerProductStepSQ8(pVect1, pVect2, sum1);
    }
    if constexpr (residual >= 12) {
        InnerProductStepSQ8(pVect1, pVect2, sum2);
    }

    // Handle final residual elements (0-3 elements)
    constexpr size_t final_residual = residual % 4;
    if constexpr (final_residual > 0) {
        float32x4_t v1 = vdupq_n_f32(0.0f);
        float32x4_t v2_f = vdupq_n_f32(0.0f);

        if constexpr (final_residual >= 1) {
            v1 = vld1q_lane_f32(pVect1, v1, 0);
            float q0 = static_cast<float>(pVect2[0]);
            v2_f = vld1q_lane_f32(&q0, v2_f, 0);
        }
        if constexpr (final_residual >= 2) {
            v1 = vld1q_lane_f32(pVect1 + 1, v1, 1);
            float q1 = static_cast<float>(pVect2[1]);
            v2_f = vld1q_lane_f32(&q1, v2_f, 1);
        }
        if constexpr (final_residual >= 3) {
            v1 = vld1q_lane_f32(pVect1 + 2, v1, 2);
            float q2 = static_cast<float>(pVect2[2]);
            v2_f = vld1q_lane_f32(&q2, v2_f, 2);
        }

        // Compute q_i * y_i (no dequantization)
        sum3 = vmlaq_f32(sum3, v2_f, v1);
    }

    // Combine all four sum accumulators
    float32x4_t sum_combined = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));

    // Horizontal sum to get Σ(q_i * y_i)
    float32x2_t sum_halves = vadd_f32(vget_low_f32(sum_combined), vget_high_f32(sum_combined));
    float32x2_t summed = vpadd_f32(sum_halves, sum_halves);
    float quantized_dot = vget_lane_f32(summed, 0);

    // Get quantization parameters from stored vector (after quantized data)
    const uint8_t *pVect2Base = static_cast<const uint8_t *>(pVect2v);
    const float min_val = *reinterpret_cast<const float *>(pVect2Base + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2Base + dimension + sizeof(float));

    // Get precomputed y_sum from query blob (stored after the dim floats)
    const float *pVect1Base = static_cast<const float *>(pVect1v);
    const float y_sum = pVect1Base[dimension];

    // Apply the algebraic formula: IP = min * y_sum + delta * Σ(q_i * y_i)
    return min_val * y_sum + delta * quantized_dot;
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - SQ8_InnerProductSIMD16_NEON_IMP<residual>(pVect1v, pVect2v, dimension);
}

template <unsigned char residual> // 0..15
float SQ8_CosineSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // Cosine distance = 1 - IP (vectors are pre-normalized)
    return SQ8_InnerProductSIMD16_NEON<residual>(pVect1v, pVect2v, dimension);
}
