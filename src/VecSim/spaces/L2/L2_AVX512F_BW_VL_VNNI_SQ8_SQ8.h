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

/**
 * SQ8-to-SQ8 L2 squared distance functions.
 * These functions compute L2 squared distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized and dequantization is applied to both
 * during computation.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [inv_norm (float)]
 * Dequantization formula: dequantized_value = quantized_value * delta + min_val
 */

// Helper function to perform L2 squared step for 16 elements with dual dequantization
static inline void SQ8_SQ8_L2SqrStep(const uint8_t *&pVec1, const uint8_t *&pVec2, __m512 &sum,
                                     const __m512 &min_val_vec1, const __m512 &delta_vec1,
                                     const __m512 &min_val_vec2, const __m512 &delta_vec2) {
    // Load 16 uint8 elements from pVec1 and convert to float
    __m128i v1_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec1));
    __m512i v1_512 = _mm512_cvtepu8_epi32(v1_128);
    __m512 v1_f = _mm512_cvtepi32_ps(v1_512);

    // Dequantize v1: (val * delta1) + min_val1
    __m512 v1_dequant = _mm512_fmadd_ps(v1_f, delta_vec1, min_val_vec1);

    // Load 16 uint8 elements from pVec2 and convert to float
    __m128i v2_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec2));
    __m512i v2_512 = _mm512_cvtepu8_epi32(v2_128);
    __m512 v2_f = _mm512_cvtepi32_ps(v2_512);

    // Dequantize v2: (val * delta2) + min_val2
    __m512 v2_dequant = _mm512_fmadd_ps(v2_f, delta_vec2, min_val_vec2);

    // Compute difference
    __m512 diff = _mm512_sub_ps(v1_dequant, v2_dequant);

    // Square difference and add to sum: sum += diff * diff
    sum = _mm512_fmadd_ps(diff, diff, sum);

    // Advance pointers
    pVec1 += 16;
    pVec2 += 16;
}

// SQ8-to-SQ8 L2 squared distance function
template <unsigned char residual> // 0..15
float SQ8_SQ8_L2SqrSIMD16_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                              size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    const uint8_t *pEnd1 = pVec1 + dimension;

    // Get dequantization parameters from the end of pVec1
    const float min_val1 = *reinterpret_cast<const float *>(pVec1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVec1 + dimension + sizeof(float));

    // Get dequantization parameters from the end of pVec2
    const float min_val2 = *reinterpret_cast<const float *>(pVec2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVec2 + dimension + sizeof(float));

    // Create broadcast vectors for SIMD operations
    __m512 min_val_vec1 = _mm512_set1_ps(min_val1);
    __m512 delta_vec1 = _mm512_set1_ps(delta1);
    __m512 min_val_vec2 = _mm512_set1_ps(min_val2);
    __m512 delta_vec2 = _mm512_set1_ps(delta2);

    // Initialize sum accumulator
    __m512 sum = _mm512_setzero_ps();

    // Deal with remainder first
    if constexpr (residual > 0) {
        // Handle less than 16 elements
        __mmask16 mask = (1U << residual) - 1;

        // Load and convert v1 elements
        __m128i v1_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec1));
        __m512i v1_512 = _mm512_cvtepu8_epi32(v1_128);
        __m512 v1_f = _mm512_cvtepi32_ps(v1_512);

        // Dequantize v1
        __m512 v1_dequant = _mm512_fmadd_ps(v1_f, delta_vec1, min_val_vec1);

        // Load and convert v2 elements
        __m128i v2_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec2));
        __m512i v2_512 = _mm512_cvtepu8_epi32(v2_128);
        __m512 v2_f = _mm512_cvtepi32_ps(v2_512);

        // Dequantize v2
        __m512 v2_dequant = _mm512_fmadd_ps(v2_f, delta_vec2, min_val_vec2);

        // Compute difference
        __m512 diff = _mm512_sub_ps(v1_dequant, v2_dequant);

        // Square difference with mask
        __m512 squared = _mm512_mul_ps(diff, diff);
        sum = _mm512_maskz_mov_ps(mask, squared);

        pVec1 += residual;
        pVec2 += residual;
    }

    // Process remaining full chunks of 16 elements
    while (pVec1 < pEnd1) {
        SQ8_SQ8_L2SqrStep(pVec1, pVec2, sum, min_val_vec1, delta_vec1, min_val_vec2, delta_vec2);
    }

    // Horizontal sum and return
    return _mm512_reduce_add_ps(sum);
}

