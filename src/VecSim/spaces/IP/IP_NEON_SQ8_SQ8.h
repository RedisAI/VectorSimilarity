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
 * where BOTH vectors are uint8 quantized and dequantization is applied to both
 * during computation.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [inv_norm (float)]
 * Dequantization formula: dequantized_value = quantized_value * delta + min_val
 */

// Helper function to perform inner product step for 4 elements with dual dequantization
static inline void SQ8_SQ8_InnerProductStep_NEON(const uint8_t *&pVec1, const uint8_t *&pVec2,
                                                  float32x4_t &sum, const float32x4_t &min_val_vec1,
                                                  const float32x4_t &delta_vec1,
                                                  const float32x4_t &min_val_vec2,
                                                  const float32x4_t &delta_vec2) {
    // Load 4 uint8 elements from pVec1 and convert to float
    uint8x8_t v1_u8 = vld1_u8(pVec1);
    uint32x4_t v1_u32 = vmovl_u16(vget_low_u16(vmovl_u8(v1_u8)));
    float32x4_t v1_f = vcvtq_f32_u32(v1_u32);

    // Dequantize v1: (val * delta1) + min_val1
    float32x4_t v1_dequant = vmlaq_f32(min_val_vec1, v1_f, delta_vec1);

    // Load 4 uint8 elements from pVec2 and convert to float
    uint8x8_t v2_u8 = vld1_u8(pVec2);
    uint32x4_t v2_u32 = vmovl_u16(vget_low_u16(vmovl_u8(v2_u8)));
    float32x4_t v2_f = vcvtq_f32_u32(v2_u32);

    // Dequantize v2: (val * delta2) + min_val2
    float32x4_t v2_dequant = vmlaq_f32(min_val_vec2, v2_f, delta_vec2);

    // Compute dot product and add to sum
    sum = vmlaq_f32(sum, v1_dequant, v2_dequant);

    // Advance pointers
    pVec1 += 4;
    pVec2 += 4;
}

// Common implementation for inner product between two SQ8 vectors
template <unsigned char residual> // 0..15
float SQ8_SQ8_InnerProductSIMD16_NEON_IMP(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);

    // Get dequantization parameters from the end of pVec1
    const float min_val1 = *reinterpret_cast<const float *>(pVec1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVec1 + dimension + sizeof(float));

    // Get dequantization parameters from the end of pVec2
    const float min_val2 = *reinterpret_cast<const float *>(pVec2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVec2 + dimension + sizeof(float));

    // Create broadcast vectors for SIMD operations
    float32x4_t min_val_vec1 = vdupq_n_f32(min_val1);
    float32x4_t delta_vec1 = vdupq_n_f32(delta1);
    float32x4_t min_val_vec2 = vdupq_n_f32(min_val2);
    float32x4_t delta_vec2 = vdupq_n_f32(delta2);

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    const size_t num_of_chunks = dimension / 16;

    // Process 16 elements at a time in the main loop
    for (size_t i = 0; i < num_of_chunks; i++) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, sum0, min_val_vec1, delta_vec1, min_val_vec2,
                                      delta_vec2);
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, sum1, min_val_vec1, delta_vec1, min_val_vec2,
                                      delta_vec2);
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, sum2, min_val_vec1, delta_vec1, min_val_vec2,
                                      delta_vec2);
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, sum3, min_val_vec1, delta_vec1, min_val_vec2,
                                      delta_vec2);
    }

    // Handle remaining complete 4-element blocks within residual
    if constexpr (residual >= 4) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, sum0, min_val_vec1, delta_vec1, min_val_vec2,
                                      delta_vec2);
    }
    if constexpr (residual >= 8) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, sum1, min_val_vec1, delta_vec1, min_val_vec2,
                                      delta_vec2);
    }
    if constexpr (residual >= 12) {
        SQ8_SQ8_InnerProductStep_NEON(pVec1, pVec2, sum2, min_val_vec1, delta_vec1, min_val_vec2,
                                      delta_vec2);
    }

    // Handle final residual elements (0-3 elements)
    constexpr size_t final_residual = residual % 4;
    if constexpr (final_residual > 0) {
        float32x4_t v1_dequant = vdupq_n_f32(0.0f);
        float32x4_t v2_dequant = vdupq_n_f32(0.0f);

        if constexpr (final_residual >= 1) {
            float dequant1_0 = pVec1[0] * delta1 + min_val1;
            float dequant2_0 = pVec2[0] * delta2 + min_val2;
            v1_dequant = vld1q_lane_f32(&dequant1_0, v1_dequant, 0);
            v2_dequant = vld1q_lane_f32(&dequant2_0, v2_dequant, 0);
        }
        if constexpr (final_residual >= 2) {
            float dequant1_1 = pVec1[1] * delta1 + min_val1;
            float dequant2_1 = pVec2[1] * delta2 + min_val2;
            v1_dequant = vld1q_lane_f32(&dequant1_1, v1_dequant, 1);
            v2_dequant = vld1q_lane_f32(&dequant2_1, v2_dequant, 1);
        }
        if constexpr (final_residual >= 3) {
            float dequant1_2 = pVec1[2] * delta1 + min_val1;
            float dequant2_2 = pVec2[2] * delta2 + min_val2;
            v1_dequant = vld1q_lane_f32(&dequant1_2, v1_dequant, 2);
            v2_dequant = vld1q_lane_f32(&dequant2_2, v2_dequant, 2);
        }

        sum3 = vmlaq_f32(sum3, v1_dequant, v2_dequant);
    }

    // Combine all four sum accumulators
    float32x4_t sum_combined = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));

    // Horizontal sum of the 4 elements in the combined NEON register
    float32x2_t sum_halves = vadd_f32(vget_low_f32(sum_combined), vget_high_f32(sum_combined));
    float32x2_t summed = vpadd_f32(sum_halves, sum_halves);
    return vget_lane_f32(summed, 0);
}

// SQ8-to-SQ8 Inner Product distance function
// Assumes both vectors are normalized.
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

