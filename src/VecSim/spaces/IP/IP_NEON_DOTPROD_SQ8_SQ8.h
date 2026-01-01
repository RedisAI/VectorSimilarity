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
 * SQ8-to-SQ8 distance functions using ARM NEON DOTPROD with precomputed sum.
 * These functions compute distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses precomputed sum stored in the vector data,
 * eliminating the need to compute them during distance calculation.
 *
 * Uses algebraic optimization with DOTPROD instruction:
 *
 * With sum = Σv[i] (sum of original float values), the formula is:
 * IP = min1*sum2 + min2*sum1 + δ1*δ2 * Σ(q1[i]*q2[i]) - dim*min1*min2
 *
 * Since sum is precomputed, we only need to compute the dot product Σ(q1[i]*q2[i]).
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)]
 */

// Helper function: computes dot product using DOTPROD instruction (no sum computation needed)
__attribute__((always_inline)) static inline void
SQ8_SQ8_InnerProductStep_NEON_DOTPROD(const uint8_t *&pVec1, const uint8_t *&pVec2,
                                      uint32x4_t &dot_sum) {
    // Load 16 uint8 elements
    uint8x16_t v1 = vld1q_u8(pVec1);
    uint8x16_t v2 = vld1q_u8(pVec2);

    // Compute dot product using DOTPROD instruction: dot_sum += v1 . v2
    dot_sum = vdotq_u32(dot_sum, v1, v2);

    pVec1 += 16;
    pVec2 += 16;
}

// Common implementation for inner product between two SQ8 vectors with precomputed sum
template <unsigned char residual> // 0..63
float SQ8_SQ8_InnerProductSIMD64_NEON_DOTPROD_IMP(const void *pVec1v, const void *pVec2v,
                                                  size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);

    // Get dequantization parameters and precomputed values from the end of pVec1
    // Layout: [data (dim)] [min (float)] [delta (float)] [sum (float)]
    const float *params1 = reinterpret_cast<const float *>(pVec1 + dimension);
    const float min1 = params1[0];
    const float delta1 = params1[1];
    const float sum1 = params1[2]; // Precomputed sum of original float elements

    // Get dequantization parameters and precomputed values from the end of pVec2
    const float *params2 = reinterpret_cast<const float *>(pVec2 + dimension);
    const float min2 = params2[0];
    const float delta2 = params2[1];
    const float sum2 = params2[2]; // Precomputed sum of original float elements

    // Calculate number of 64-element chunks
    size_t num_of_chunks = (dimension - residual) / 64;

    // Multiple accumulators for ILP (dot product only)
    uint32x4_t dot_sum0 = vdupq_n_u32(0);
    uint32x4_t dot_sum1 = vdupq_n_u32(0);
    uint32x4_t dot_sum2 = vdupq_n_u32(0);
    uint32x4_t dot_sum3 = vdupq_n_u32(0);

    // Process 64 elements at a time (4 x 16) in the main loop
    for (size_t i = 0; i < num_of_chunks; i++) {
        SQ8_SQ8_InnerProductStep_NEON_DOTPROD(pVec1, pVec2, dot_sum0);
        SQ8_SQ8_InnerProductStep_NEON_DOTPROD(pVec1, pVec2, dot_sum1);
        SQ8_SQ8_InnerProductStep_NEON_DOTPROD(pVec1, pVec2, dot_sum2);
        SQ8_SQ8_InnerProductStep_NEON_DOTPROD(pVec1, pVec2, dot_sum3);
    }

    // Handle remaining complete 16-element blocks within residual
    if constexpr (residual >= 16) {
        SQ8_SQ8_InnerProductStep_NEON_DOTPROD(pVec1, pVec2, dot_sum0);
    }
    if constexpr (residual >= 32) {
        SQ8_SQ8_InnerProductStep_NEON_DOTPROD(pVec1, pVec2, dot_sum1);
    }
    if constexpr (residual >= 48) {
        SQ8_SQ8_InnerProductStep_NEON_DOTPROD(pVec1, pVec2, dot_sum2);
    }

    // Combine accumulators
    uint32x4_t dot_total = vaddq_u32(vaddq_u32(dot_sum0, dot_sum1), vaddq_u32(dot_sum2, dot_sum3));

    // Horizontal sum for dot product
    uint32_t dot_product = vaddvq_u32(dot_total);

    // Handle remaining scalar elements (0-15)
    constexpr unsigned char remaining = residual % 16;
    if constexpr (remaining > 0) {
        for (unsigned char i = 0; i < remaining; i++) {
            dot_product += static_cast<uint32_t>(pVec1[i]) * static_cast<uint32_t>(pVec2[i]);
        }
    }

    // Apply algebraic formula using precomputed sums:
    // IP = min1*sum2 + min2*sum1 + δ1*δ2 * Σ(q1*q2) - dim*min1*min2
    return min1 * sum2 + min2 * sum1 + delta1 * delta2 * static_cast<float>(dot_product) -
           static_cast<float>(dimension) * min1 * min2;
}

// SQ8-to-SQ8 Inner Product distance function
// Returns 1 - inner_product (distance form)
template <unsigned char residual> // 0..63
float SQ8_SQ8_InnerProductSIMD64_NEON_DOTPROD(const void *pVec1v, const void *pVec2v,
                                              size_t dimension) {
    return 1.0f - SQ8_SQ8_InnerProductSIMD64_NEON_DOTPROD_IMP<residual>(pVec1v, pVec2v, dimension);
}

// SQ8-to-SQ8 Cosine distance function
// Returns 1 - inner_product (assumes vectors are pre-normalized)
template <unsigned char residual> // 0..63
float SQ8_SQ8_CosineSIMD64_NEON_DOTPROD(const void *pVec1v, const void *pVec2v, size_t dimension) {
    return SQ8_SQ8_InnerProductSIMD64_NEON_DOTPROD<residual>(pVec1v, pVec2v, dimension);
}
