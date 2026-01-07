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
#include "VecSim/spaces/AVX_utils.h"
#include "VecSim/types/sq8.h"
using sq8 = vecsim_types::sq8;

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
 *
 * This version uses FMA instructions for better performance.
 */

// Helper: compute Σ(q_i * y_i) for 8 elements using FMA (no dequantization)
static inline void InnerProductStepSQ8_FMA(const float *&pVect1, const uint8_t *&pVect2,
                                           __m256 &sum256) {
    // Load 8 float elements from query
    __m256 v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;

    // Load 8 uint8 elements and convert to float
    __m128i v2_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pVect2));
    pVect2 += 8;

    __m256i v2_256 = _mm256_cvtepu8_epi32(v2_128);
    __m256 v2_f = _mm256_cvtepi32_ps(v2_256);

    // Accumulate q_i * y_i using FMA (no dequantization!)
    sum256 = _mm256_fmadd_ps(v2_f, v1, sum256);
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductImp_FMA(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);
    const float *pEnd1 = pVect1 + dimension;

    // Initialize sum accumulator for Σ(q_i * y_i)
    __m256 sum256 = _mm256_setzero_ps();

    // Handle residual elements first (0-7 elements)
    if constexpr (residual % 8) {
        __mmask8 constexpr mask = (1 << (residual % 8)) - 1;
        __m256 v1 = my_mm256_maskz_loadu_ps<mask>(pVect1);
        pVect1 += residual % 8;

        // Load uint8 elements and convert to float
        __m128i v2_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pVect2));
        pVect2 += residual % 8;

        __m256i v2_256 = _mm256_cvtepu8_epi32(v2_128);
        __m256 v2_f = _mm256_cvtepi32_ps(v2_256);

        // Compute q_i * y_i (no dequantization)
        sum256 = _mm256_mul_ps(v1, v2_f);
    }

    // If the residual is >=8, have another step of 8 floats
    if constexpr (residual >= 8) {
        InnerProductStepSQ8_FMA(pVect1, pVect2, sum256);
    }

    // Process remaining full chunks of 16 elements (2x8)
    // Using do-while since dim > 16 guarantees at least one iteration
    do {
        InnerProductStepSQ8_FMA(pVect1, pVect2, sum256);
        InnerProductStepSQ8_FMA(pVect1, pVect2, sum256);
    } while (pVect1 < pEnd1);

    // Reduce to get Σ(q_i * y_i)
    float quantized_dot = my_mm256_reduce_add_ps(sum256);

    // Get quantization parameters from stored vector (after quantized data)
    const uint8_t *pVect2Base = static_cast<const uint8_t *>(pVect2v);
    const float *params2 = reinterpret_cast<const float *>(pVect2Base + dimension);
    const float min_val = params2[sq8::MIN_VAL];
    const float delta = params2[sq8::DELTA];

    // Get precomputed y_sum from query blob (stored after the dim floats)
    const float y_sum = static_cast<const float *>(pVect1v)[dimension];

    // Apply the algebraic formula: IP = min * y_sum + delta * Σ(q_i * y_i)
    return min_val * y_sum + delta * quantized_dot;
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_AVX2_FMA(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - SQ8_InnerProductImp_FMA<residual>(pVect1v, pVect2v, dimension);
}

template <unsigned char residual> // 0..15
float SQ8_CosineSIMD16_AVX2_FMA(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // Cosine distance = 1 - IP (vectors are pre-normalized)
    return SQ8_InnerProductSIMD16_AVX2_FMA<residual>(pVect1v, pVect2v, dimension);
}
