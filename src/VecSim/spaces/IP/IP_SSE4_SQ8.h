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
                                       __m128 &sum) {
    // Load 4 float elements from query
    __m128 v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;

    // Load 4 uint8 elements and convert to float
    __m128i v2_i = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*reinterpret_cast<const int32_t *>(pVect2)));
    pVect2 += 4;

    __m128 v2_f = _mm_cvtepi32_ps(v2_i);

    // Accumulate q_i * y_i (no dequantization!)
    // SSE doesn't have FMA, so use mul + add
    sum = _mm_add_ps(sum, _mm_mul_ps(v2_f, v1));
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_SSE4_IMP(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);
    const float *pEnd1 = pVect1 + dimension;

    // Initialize sum accumulator for Σ(q_i * y_i)
    __m128 sum = _mm_setzero_ps();

    // Process residual elements first (1-3 elements)
    if constexpr (residual % 4) {
        __m128 v1;
        __m128 v2_f;

        if constexpr (residual % 4 == 3) {
            v1 = _mm_set_ps(0.0f, pVect1[2], pVect1[1], pVect1[0]);
            v2_f = _mm_set_ps(0.0f, static_cast<float>(pVect2[2]),
                              static_cast<float>(pVect2[1]), static_cast<float>(pVect2[0]));
        } else if constexpr (residual % 4 == 2) {
            v1 = _mm_set_ps(0.0f, 0.0f, pVect1[1], pVect1[0]);
            v2_f = _mm_set_ps(0.0f, 0.0f, static_cast<float>(pVect2[1]),
                              static_cast<float>(pVect2[0]));
        } else if constexpr (residual % 4 == 1) {
            v1 = _mm_set_ps(0.0f, 0.0f, 0.0f, pVect1[0]);
            v2_f = _mm_set_ps(0.0f, 0.0f, 0.0f, static_cast<float>(pVect2[0]));
        }

        pVect1 += residual % 4;
        pVect2 += residual % 4;

        // Compute q_i * y_i (no dequantization)
        sum = _mm_mul_ps(v1, v2_f);
    }

    // Handle remaining residual in chunks of 4 (for residual 4-15)
    if constexpr (residual >= 4) {
        InnerProductStepSQ8(pVect1, pVect2, sum);
    }
    if constexpr (residual >= 8) {
        InnerProductStepSQ8(pVect1, pVect2, sum);
    }
    if constexpr (residual >= 12) {
        InnerProductStepSQ8(pVect1, pVect2, sum);
    }

    // Process remaining full chunks of 4 elements
    // Using do-while since dim > 16 guarantees at least one iteration
    do {
        InnerProductStepSQ8(pVect1, pVect2, sum);
    } while (pVect1 < pEnd1);

    // Horizontal sum to get Σ(q_i * y_i)
    float PORTABLE_ALIGN16 TmpRes[4];
    _mm_store_ps(TmpRes, sum);
    float quantized_dot = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

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
float SQ8_InnerProductSIMD16_SSE4(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - SQ8_InnerProductSIMD16_SSE4_IMP<residual>(pVect1v, pVect2v, dimension);
}

template <unsigned char residual> // 0..15
float SQ8_CosineSIMD16_SSE4(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // Cosine distance = 1 - IP (vectors are pre-normalized)
    return SQ8_InnerProductSIMD16_SSE4<residual>(pVect1v, pVect2v, dimension);
}
