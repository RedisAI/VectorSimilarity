/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

/**
 * SQ8 distance functions (float32 query vs uint8 stored) for NEON.
 *
 * Uses algebraic optimization to reduce operations per element:
 *
 * IP = Σ query[i] * (val[i] * δ + min)
 *    = δ * Σ(query[i] * val[i]) + min * Σ(query[i])
 *
 * This saves 1 FMA per 4-element step by deferring dequantization to scalar math at the end.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)]
 */

// Helper function with algebraic optimization
static inline void InnerProductStep(const float *&pVect1, const uint8_t *&pVect2,
                                    float32x4_t &dot_sum, float32x4_t &query_sum) {
    // Load 4 float elements from query
    float32x4_t v1 = vld1q_f32(pVect1);
    pVect1 += 4;

    // Load 4 uint8 elements and convert to float
    uint8x8_t v2_u8 = vld1_u8(pVect2);
    pVect2 += 4;
    uint32x4_t v2_u32 = vmovl_u16(vget_low_u16(vmovl_u8(v2_u8)));
    float32x4_t v2_f = vcvtq_f32_u32(v2_u32);

    // Accumulate query * val (without dequantization)
    dot_sum = vmlaq_f32(dot_sum, v1, v2_f);

    // Accumulate query sum
    query_sum = vaddq_f32(query_sum, v1);
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_NEON_IMP(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // Get dequantization parameters from the end of quantized vector
    const float min_val = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));

    // Multiple accumulators for instruction-level parallelism
    // dot_sum: accumulates query[i] * val[i]
    // query_sum: accumulates query[i]
    float32x4_t dot_sum0 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum1 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum2 = vdupq_n_f32(0.0f);
    float32x4_t dot_sum3 = vdupq_n_f32(0.0f);
    float32x4_t query_sum0 = vdupq_n_f32(0.0f);
    float32x4_t query_sum1 = vdupq_n_f32(0.0f);
    float32x4_t query_sum2 = vdupq_n_f32(0.0f);
    float32x4_t query_sum3 = vdupq_n_f32(0.0f);

    const size_t num_of_chunks = dimension / 16;

    // Process 16 elements at a time in the main loop
    for (size_t i = 0; i < num_of_chunks; i++) {
        InnerProductStep(pVect1, pVect2, dot_sum0, query_sum0);
        InnerProductStep(pVect1, pVect2, dot_sum1, query_sum1);
        InnerProductStep(pVect1, pVect2, dot_sum2, query_sum2);
        InnerProductStep(pVect1, pVect2, dot_sum3, query_sum3);
    }

    // Handle remaining complete 4-element blocks within residual
    if constexpr (residual >= 4) {
        InnerProductStep(pVect1, pVect2, dot_sum0, query_sum0);
    }
    if constexpr (residual >= 8) {
        InnerProductStep(pVect1, pVect2, dot_sum1, query_sum1);
    }
    if constexpr (residual >= 12) {
        InnerProductStep(pVect1, pVect2, dot_sum2, query_sum2);
    }

    // Handle final residual elements (0-3 elements) with scalar math
    constexpr size_t final_residual = residual % 4;
    if constexpr (final_residual > 0) {
        float32x4_t v1 = vdupq_n_f32(0.0f);
        float32x4_t v2_f = vdupq_n_f32(0.0f);

        if constexpr (final_residual >= 1) {
            v1 = vld1q_lane_f32(pVect1, v1, 0);
            float val0 = static_cast<float>(pVect2[0]);
            v2_f = vld1q_lane_f32(&val0, v2_f, 0);
        }
        if constexpr (final_residual >= 2) {
            v1 = vld1q_lane_f32(pVect1 + 1, v1, 1);
            float val1 = static_cast<float>(pVect2[1]);
            v2_f = vld1q_lane_f32(&val1, v2_f, 1);
        }
        if constexpr (final_residual >= 3) {
            v1 = vld1q_lane_f32(pVect1 + 2, v1, 2);
            float val2 = static_cast<float>(pVect2[2]);
            v2_f = vld1q_lane_f32(&val2, v2_f, 2);
        }

        dot_sum3 = vmlaq_f32(dot_sum3, v1, v2_f);
        query_sum3 = vaddq_f32(query_sum3, v1);
    }

    // Combine accumulators
    float32x4_t dot_total = vaddq_f32(vaddq_f32(dot_sum0, dot_sum1), vaddq_f32(dot_sum2, dot_sum3));
    float32x4_t query_total =
        vaddq_f32(vaddq_f32(query_sum0, query_sum1), vaddq_f32(query_sum2, query_sum3));

    // Horizontal sum
    float32x2_t dot_halves = vadd_f32(vget_low_f32(dot_total), vget_high_f32(dot_total));
    float32x2_t dot_summed = vpadd_f32(dot_halves, dot_halves);
    float dot_product = vget_lane_f32(dot_summed, 0);

    float32x2_t query_halves = vadd_f32(vget_low_f32(query_total), vget_high_f32(query_total));
    float32x2_t query_summed = vpadd_f32(query_halves, query_halves);
    float query_sum = vget_lane_f32(query_summed, 0);

    // Apply algebraic formula: IP = δ * Σ(query*val) + min * Σ(query)
    return delta * dot_product + min_val * query_sum;
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - SQ8_InnerProductSIMD16_NEON_IMP<residual>(pVect1v, pVect2v, dimension);
}

template <unsigned char residual> // 0..15
float SQ8_CosineSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // Assume vectors are normalized.
    return 1.0f - SQ8_InnerProductSIMD16_NEON_IMP<residual>(pVect1v, pVect2v, dimension);
}
