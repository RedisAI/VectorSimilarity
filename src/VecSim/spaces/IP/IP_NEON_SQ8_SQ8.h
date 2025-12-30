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
 * SQ8-to-SQ8 distance functions using ARM NEON with precomputed sum and norm.
 * These functions compute distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses precomputed sum and norm stored in the vector data,
 * eliminating the need to compute them during distance calculation.
 *
 * Uses algebraic optimization:
 *
 * With sum = Σv[i] (sum of original float values), the formula is:
 * IP = min1*sum2 + min2*sum1 - dim*min1*min2 + δ1*δ2 * Σ(q1[i]*q2[i])
 *
 * Since sum is precomputed, we only need to compute the dot product Σ(q1[i]*q2[i]).
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)] [norm
 * (float)]
 */

// Helper function with dot product only (no sum computation needed)
static inline void SQ8_SQ8_InnerProductStep_NEON(const uint8_t *&pVec1, const uint8_t *&pVec2,
                                                 float32x4_t &dot_sum) {
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

    // Advance pointers
    pVec1 += 4;
    pVec2 += 4;
}

// Common implementation for inner product between two SQ8 vectors with precomputed sum/norm
template <unsigned char residual> // 0..15
float SQ8_SQ8_InnerProductSIMD16_NEON_IMP(const void *pVec1v, const void *pVec2v,
                                          size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);

    // Get dequantization parameters and precomputed values from the end of pVec1
    // Layout: [data (dim)] [min (float)] [delta (float)] [sum (float)] [norm (float)]
    const float *params1 = reinterpret_cast<const float *>(pVec1 + dimension);
    const float min1 = params1[0];
    const float delta1 = params1[1];
    const float sum1 = params1[2]; // Precomputed sum of original float elements

    // Get dequantization parameters and precomputed values from the end of pVec2
    const float *params2 = reinterpret_cast<const float *>(pVec2 + dimension);
    const float min2 = params2[0];
    const float delta2 = params2[1];
    const float sum2 = params2[2]; // Precomputed sum of original float elements

    // Calculate number of 16-element chunks
    size_t num_of_chunks = (dimension - residual) / 16;

    // Multiple accumulators for ILP (dot product only)
    float32x4_t dot_sum0 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum1 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum2 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum3 = vdupq_n_f32(0.0f);

    // Process 16 elements at a time in the main loop
    for (size_t i = 0; i < num_of_chunks; i++) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum0);
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum1);
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum2);
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum3);
    }

    // Handle remaining complete 4-element blocks within residual
    if constexpr (residual >= 4) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum0);
    }
    if constexpr (residual >= 8) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum1);
    }
    if constexpr (residual >= 12) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, dot_sum2);
    }

    // Combine dot product accumulators
    float32x4_t dot_total = vaddq_f32(vaddq_f32(dot_sum0, dot_sum1), vaddq_f32(dot_sum2, dot_sum3));

    // Horizontal sum for dot product
    float32x2_t dot_halves = vadd_f32(vget_low_f32(dot_total), vget_high_f32(dot_total));
    float32x2_t dot_summed = vpadd_f32(dot_halves, dot_halves);
    float dot_product = vget_lane_f32(dot_summed, 0);

    // Handle remaining scalar elements (0-3)
    constexpr unsigned char remaining = residual % 4;
    if constexpr (remaining > 0) {
        for (unsigned char i = 0; i < remaining; i++) {
            dot_product += static_cast<float>(pVec1[i]) * static_cast<float>(pVec2[i]);
        }
    }

    // Apply algebraic formula using precomputed sums:
    // IP = min1*sum2 + min2*sum1 - dim*min1*min2 + δ1*δ2 * Σ(q1*q2)
    return min1 * sum2 + min2 * sum1 - static_cast<float>(dimension) * min1 * min2 +
           delta1 * delta2 * dot_product;
}

// SQ8-to-SQ8 Inner Product distance function
// Returns 1 - inner_product (distance form)
template <unsigned char residual> // 0..15
float SQ8_SQ8_InnerProductSIMD16_NEON(const void *pVec1v, const void *pVec2v, size_t dimension) {
    return 1.0f - SQ8_SQ8_InnerProductSIMD16_NEON_IMP<residual>(pVec1v, pVec2v, dimension);
}

// SQ8-to-SQ8 Cosine distance function
// Returns 1 - inner_product (assumes vectors are pre-normalized)
template <unsigned char residual> // 0..15
float SQ8_SQ8_CosineSIMD16_NEON(const void *pVec1v, const void *pVec2v, size_t dimension) {
    return 1.0f - SQ8_SQ8_InnerProductSIMD16_NEON_IMP<residual>(pVec1v, pVec2v, dimension);
}
