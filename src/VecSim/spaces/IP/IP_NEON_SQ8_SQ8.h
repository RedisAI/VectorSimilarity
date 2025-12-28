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
 * SQ8-to-SQ8 distance functions for NEON.
 * These functions compute distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses algebraic optimization to reduce operations per element:
 *
 * IP = Σ (v1[i]*δ1 + min1) * (v2[i]*δ2 + min2)
 *    = δ1*δ2 * Σ(v1[i]*v2[i]) + δ1*min2 * Σv1[i] + δ2*min1 * Σv2[i] + dim*min1*min2
 *
 * This saves 2 FMAs per 4-element step by deferring dequantization to scalar math at the end.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [inv_norm (float)]
 */

// Helper function with algebraic optimization
static inline void SQ8_SQ8_InnerProductStep_NEON(const uint8_t *&pVec1, const uint8_t *&pVec2,
                                                 float32x4_t &dot_sum, float32x4_t &sum1,
                                                 float32x4_t &sum2) {
    // Load 4 uint8 elements from pVec1 and convert to float
    uint8x8_t v1_u8 = vld1_u8(pVec1);
    uint32x4_t v1_u32 = vmovl_u16(vget_low_u16(vmovl_u8(v1_u8)));
    float32x4_t v1_f = vcvtq_f32_u32(v1_u32);

    // Load 4 uint8 elements from pVec2 and convert to float
    uint8x8_t v2_u8 = vld1_u8(pVec2);
    uint32x4_t v2_u32 = vmovl_u16(vget_low_u16(vmovl_u8(v2_u8)));
    float32x4_t v2_f = vcvtq_f32_u32(v2_u32);

    // Accumulate dot product: dot_sum += v1 * v2 (no dequantization)
    dot_sum = vmlaq_f32(dot_sum, v1_f, v2_f);

    // Accumulate element sums
    sum1 = vaddq_f32(sum1, v1_f);
    sum2 = vaddq_f32(sum2, v2_f);

    // Advance pointers
    pVec1 += 4;
    pVec2 += 4;
}

// Common implementation for inner product between two SQ8 vectors
template <unsigned char residual> // 0..15
float SQ8_SQ8_InnerProductSIMD16_NEON_IMP(const void *pVec1v, const void *pVec2v,
                                          size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);

    // Get dequantization parameters from the end of pVec1
    const float min1 = *reinterpret_cast<const float *>(pVec1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVec1 + dimension + sizeof(float));

    // Get dequantization parameters from the end of pVec2
    const float min2 = *reinterpret_cast<const float *>(pVec2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVec2 + dimension + sizeof(float));

    // Multiple accumulators for instruction-level parallelism
    // dot_sum: accumulates v1[i] * v2[i]
    // sum1: accumulates v1[i]
    // sum2: accumulates v2[i]
    float32x4_t dot_sum0 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum1 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum2 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum1_0 = vdupq_n_f32(0.0f);
    float32x4_t sum1_1 = vdupq_n_f32(0.0f);
    float32x4_t sum1_2 = vdupq_n_f32(0.0f);
    float32x4_t sum1_3 = vdupq_n_f32(0.0f);
    float32x4_t sum2_0 = vdupq_n_f32(0.0f);
    float32x4_t sum2_1 = vdupq_n_f32(0.0f);
    float32x4_t sum2_2 = vdupq_n_f32(0.0f);
    float32x4_t sum2_3 = vdupq_n_f32(0.0f);

    const size_t num_of_chunks = dimension / 16;

    // Process 16 elements at a time in the main loop
    for (size_t i = 0; i < num_of_chunks; i++) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum0, sum1_0, sum2_0);
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum1, sum1_1, sum2_1);
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum2, sum1_2, sum2_2);
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum3, sum1_3, sum2_3);
    }

    // Handle remaining complete 4-element blocks within residual
    if constexpr (residual >= 4) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum0, sum1_0, sum2_0);
    }
    if constexpr (residual >= 8) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum1, sum1_1, sum2_1);
    }
    if constexpr (residual >= 12) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum2, sum1_2, sum2_2);
    }

    // Handle final residual elements (0-3 elements) with scalar math
    constexpr size_t final_residual = residual % 4;
    if constexpr (final_residual > 0) {
        float32x4_t v1_f = vdupq_n_f32(0.0f);
        float32x4_t v2_f = vdupq_n_f32(0.0f);

        if constexpr (final_residual >= 1) {
            float val1_0 = static_cast<float>(pVec1[0]);
            float val2_0 = static_cast<float>(pVec2[0]);
            v1_f = vld1q_lane_f32(&val1_0, v1_f, 0);
            v2_f = vld1q_lane_f32(&val2_0, v2_f, 0);
        }
        if constexpr (final_residual >= 2) {
            float val1_1 = static_cast<float>(pVec1[1]);
            float val2_1 = static_cast<float>(pVec2[1]);
            v1_f = vld1q_lane_f32(&val1_1, v1_f, 1);
            v2_f = vld1q_lane_f32(&val2_1, v2_f, 1);
        }
        if constexpr (final_residual >= 3) {
            float val1_2 = static_cast<float>(pVec1[2]);
            float val2_2 = static_cast<float>(pVec2[2]);
            v1_f = vld1q_lane_f32(&val1_2, v1_f, 2);
            v2_f = vld1q_lane_f32(&val2_2, v2_f, 2);
        }

        dot_sum3 = vmlaq_f32(dot_sum3, v1_f, v2_f);
        sum1_3 = vaddq_f32(sum1_3, v1_f);
        sum2_3 = vaddq_f32(sum2_3, v2_f);
    }

    // Combine accumulators
    float32x4_t dot_total = vaddq_f32(vaddq_f32(dot_sum0, dot_sum1), vaddq_f32(dot_sum2, dot_sum3));
    float32x4_t sum1_total = vaddq_f32(vaddq_f32(sum1_0, sum1_1), vaddq_f32(sum1_2, sum1_3));
    float32x4_t sum2_total = vaddq_f32(vaddq_f32(sum2_0, sum2_1), vaddq_f32(sum2_2, sum2_3));

    // Horizontal sum for dot product
    float32x2_t dot_halves = vadd_f32(vget_low_f32(dot_total), vget_high_f32(dot_total));
    float32x2_t dot_summed = vpadd_f32(dot_halves, dot_halves);
    float dot_product = vget_lane_f32(dot_summed, 0);

    // Horizontal sum for v1 sum
    float32x2_t sum1_halves = vadd_f32(vget_low_f32(sum1_total), vget_high_f32(sum1_total));
    float32x2_t sum1_summed = vpadd_f32(sum1_halves, sum1_halves);
    float v1_sum = vget_lane_f32(sum1_summed, 0);

    // Horizontal sum for v2 sum
    float32x2_t sum2_halves = vadd_f32(vget_low_f32(sum2_total), vget_high_f32(sum2_total));
    float32x2_t sum2_summed = vpadd_f32(sum2_halves, sum2_halves);
    float v2_sum = vget_lane_f32(sum2_summed, 0);

    // Apply algebraic formula:
    // IP = δ1*δ2 * Σ(v1*v2) + δ1*min2 * Σv1 + δ2*min1 * Σv2 + dim*min1*min2
    return delta1 * delta2 * dot_product + delta1 * min2 * v1_sum + delta2 * min1 * v2_sum +
           static_cast<float>(dimension) * min1 * min2;
}

// SQ8-to-SQ8 Inner Product distance function
// Returns 1 - inner_product (distance form)
template <unsigned char residual> // 0..15
float SQ8_SQ8_InnerProductSIMD16_NEON(const void *pVec1v, const void *pVec2v, size_t dimension) {
    return 1.0f - SQ8_SQ8_InnerProductSIMD16_NEON_IMP<residual>(pVec1v, pVec2v, dimension);
}

// SQ8-to-SQ8 Cosine distance function
// Assumes both vectors are normalized.
// Returns 1 - inner_product
template <unsigned char residual> // 0..15
float SQ8_SQ8_CosineSIMD16_NEON(const void *pVec1v, const void *pVec2v, size_t dimension) {
    return 1.0f - SQ8_SQ8_InnerProductSIMD16_NEON_IMP<residual>(pVec1v, pVec2v, dimension);
}
