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
#include <immintrin.h>
#include <iostream>

static inline void SQ8_InnerProductStep(const float *&pVec1, const uint8_t *&pVec2, __m512 &sum,
                                        const __m512 &min_val_vec, const __m512 &delta_vec) {
    // Load 16 float elements from pVec1
    __m512 v1 = _mm512_loadu_ps(pVec1);

    // Load 16 uint8 elements from pVec2 and convert to __m512i
    __m128i v2_128 = _mm_loadu_si128((__m128i *)pVec2);
    __m512i v2_512 = _mm512_cvtepu8_epi32(v2_128);

    // Convert uint8 to float
    __m512 v2_f = _mm512_cvtepi32_ps(v2_512);

    // Dequantize: (val * delta) + min_val
    __m512 dequantized = _mm512_fmadd_ps(v2_f, delta_vec, min_val_vec);

    // Compute dot product and add to sum
    sum = _mm512_fmadd_ps(v1, dequantized, sum);

    // Advance pointers
    pVec1 += 16;
    pVec2 += 16;
}

// Common implementation for both inner product and cosine similarity
template <unsigned char residual> // 0..15
float SQ8_InnerProductImp_AVX512(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const float *pVec1 = static_cast<const float *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    const float *pEnd1 = pVec1 + dimension;

    // Get dequantization parameters from the end of pVec2
    const float min_val = *reinterpret_cast<const float *>(pVec2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVec2 + dimension + sizeof(float));

    // Create broadcast vectors for SIMD operations
    __m512 min_val_vec = _mm512_set1_ps(min_val);
    __m512 delta_vec = _mm512_set1_ps(delta);

    // Initialize sum accumulator
    __m512 sum = _mm512_setzero_ps();

    // Deal with remainder first
    if constexpr (residual > 0) {
        // Handle less than 16 elements
        __mmask16 mask = (1U << residual) - 1;

        // Load masked float elements
        __m512 v1 = _mm512_maskz_loadu_ps(mask, pVec1);

        // Load full uint8 elements - we know that the first 16 elements are safe to load
        __m128i v2_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec2));
        __m512i v2_512 = _mm512_cvtepu8_epi32(v2_128);
        __m512 v2_f = _mm512_cvtepi32_ps(v2_512);

        // Dequantize
        __m512 dequantized = _mm512_fmadd_ps(v2_f, delta_vec, min_val_vec);

        // Compute dot product
        __m512 product = _mm512_mul_ps(v1, dequantized);

        // Apply mask to product and add to sum
        sum = _mm512_fmadd_ps(sum, sum, product);

        pVec1 += residual;
        pVec2 += residual;
    }

    // Process remaining full chunks of 16 elements
    do {
        SQ8_InnerProductStep(pVec1, pVec2, sum, min_val_vec, delta_vec);
    } while (pVec1 < pEnd1);

    // Return the raw inner product result
    return _mm512_reduce_add_ps(sum);
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                                size_t dimension) {
    // Calculate inner product using common implementation
    float ip = SQ8_InnerProductImp_AVX512<residual>(pVec1v, pVec2v, dimension);

    // The inner product similarity is 1 - ip
    return 1.0f - ip;
}

template <unsigned char residual> // 0..15
float SQ8_CosineSIMD16_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                          size_t dimension) {
    // Get the inverse norm factor stored after min_val and delta
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    const float inv_norm = *reinterpret_cast<const float *>(pVec2 + dimension + 2 * sizeof(float));

    // Calculate inner product using common implementation with normalization
    float ip = SQ8_InnerProductImp_AVX512<residual>(pVec1v, pVec2v, dimension);

    // The cosine similarity is 1 - ip
    return 1.0f - ip * inv_norm;
}
