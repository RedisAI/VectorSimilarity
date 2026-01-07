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
#include "VecSim/types/sq8.h"
using sq8 = vecsim_types::sq8;
#include <immintrin.h>

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

// Helper: compute Σ(q_i * y_i) for 16 elements
static inline void SQ8_InnerProductStep(const float *&pVec1, const uint8_t *&pVec2, __m512 &sum) {
    // Load 16 float elements from query (pVec1)
    __m512 v1 = _mm512_loadu_ps(pVec1);

    // Load 16 uint8 elements from quantized vector and convert to float
    __m128i v2_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec2));
    __m512i v2_512 = _mm512_cvtepu8_epi32(v2_128);
    __m512 v2_f = _mm512_cvtepi32_ps(v2_512);

    // Accumulate q_i * y_i (no dequantization!)
    sum = _mm512_fmadd_ps(v2_f, v1, sum);

    pVec1 += 16;
    pVec2 += 16;
}

// Common implementation for both inner product and cosine similarity
template <unsigned char residual> // 0..15
float SQ8_InnerProductImp_AVX512(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const float *pVec1 = static_cast<const float *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    const float *pEnd1 = pVec1 + dimension;

    // Initialize sum accumulator for Σ(q_i * y_i)
    __m512 sum = _mm512_setzero_ps();

    // Handle residual elements first (0 to 15)
    if constexpr (residual > 0) {
        __mmask16 mask = (1U << residual) - 1;

        // Load masked float elements from query
        __m512 v1 = _mm512_maskz_loadu_ps(mask, pVec1);

        // Load uint8 elements (safe to load 16 bytes due to padding)
        __m128i v2_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec2));
        __m512i v2_512 = _mm512_cvtepu8_epi32(v2_128);
        __m512 v2_f = _mm512_cvtepi32_ps(v2_512);

        // Compute q_i * y_i with mask (no dequantization)
        sum = _mm512_maskz_mul_ps(mask, v2_f, v1);

        pVec1 += residual;
        pVec2 += residual;
    }

    // Process full chunks of 16 elements
    // Using do-while since dim > 16 guarantees at least one iteration
    do {
        SQ8_InnerProductStep(pVec1, pVec2, sum);
    } while (pVec1 < pEnd1);

    // Reduce to get Σ(q_i * y_i)
    float quantized_dot = _mm512_reduce_add_ps(sum);

    // Get quantization parameters from stored vector (after quantized data)
    // Use the original base pointer since pVec2 has been advanced
    const uint8_t *pVec2Base = static_cast<const uint8_t *>(pVec2v);
    const float *params2 = reinterpret_cast<const float *>(pVec2Base + dimension);
    const float min_val = params2[sq8::MIN_VAL];
    const float delta = params2[sq8::DELTA];

    // Get precomputed y_sum from query blob (stored after the dim floats)
    // Use the original base pointer since pVec1 has been advanced
    const float y_sum = static_cast<const float *>(pVec1v)[dimension];

    // Apply the algebraic formula: IP = min * y_sum + delta * Σ(q_i * y_i)
    return min_val * y_sum + delta * quantized_dot;
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                                size_t dimension) {
    // The inner product similarity is 1 - ip
    return 1.0f - SQ8_InnerProductImp_AVX512<residual>(pVec1v, pVec2v, dimension);
}

template <unsigned char residual> // 0..15
float SQ8_CosineSIMD16_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                          size_t dimension) {
    // Cosine distance = 1 - IP (vectors are pre-normalized)
    return SQ8_InnerProductSIMD16_AVX512F_BW_VL_VNNI<residual>(pVec1v, pVec2v, dimension);
}
