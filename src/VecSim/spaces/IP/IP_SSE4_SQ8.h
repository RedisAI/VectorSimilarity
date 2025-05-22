/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"
#include <iostream>
#include <string.h>

static inline void InnerProductStep(const float *&pVect1, const uint8_t *&pVect2, __m128 &sum_prod1,
                                    __m128 &sum_prod2, const __m128 &min_val_vec,
                                    const __m128 &delta_vec) {
    // Load first 4 elements
    __m128 v1a = _mm_loadu_ps(pVect1);
    __m128i v2a_i = _mm_cvtepu8_epi32(_mm_loadu_si32(pVect2));

    // Load next 4 elements
    __m128 v1b = _mm_loadu_ps(pVect1 + 4);
    __m128i v2b_i = _mm_cvtepu8_epi32(_mm_loadu_si32(pVect2 + 4));

    pVect1 += 8;
    pVect2 += 8;

    // Process both sets
    __m128 v2a_dequant = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v2a_i), delta_vec), min_val_vec);
    __m128 v2b_dequant = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v2b_i), delta_vec), min_val_vec);

    sum_prod1 = _mm_add_ps(sum_prod1, _mm_mul_ps(v1a, v2a_dequant));
    sum_prod2 = _mm_add_ps(sum_prod2, _mm_mul_ps(v1b, v2b_dequant));
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_SSE4_IMP(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    const uint8_t *quantized = static_cast<const uint8_t *>(pVect2v);

    // Get dequantization parameters from the end of quantized vector
    float min = *(float *)(quantized + dimension);
    float delta = *(float *)(quantized + dimension + sizeof(float));

    // Create broadcast vectors for SIMD operations
    __m128 min_val_vec = _mm_set1_ps(min);
    __m128 delta_vec = _mm_set1_ps(delta);

    const float *pEnd1 = pVect1 + dimension;

    // Initialize two sum accumulators
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();

    // Process residual elements if needed
    if constexpr (residual) {
        // Handle residual elements (1-3)
        if constexpr (residual % 4) {
            __m128 v1;
            __m128 v2_dequant = _mm_setzero_ps();

            if constexpr (residual % 4 == 3) {
                // Load 3 floats and set the last one to 0
                v1 = _mm_load_ss(pVect1);                     // load 1 float, set the rest to 0
                v1 = _mm_loadh_pi(v1, (__m64 *)(pVect1 + 1)); // load 2 more floats into high part

                // Dequantize first value
                float dequant0 = quantized[0] * delta + min;
                v2_dequant = _mm_load_ss(&dequant0);

                // Dequantize next two values
                float dequant_high[2] = {quantized[1] * delta + min, quantized[2] * delta + min};
                v2_dequant = _mm_loadh_pi(v2_dequant, (__m64 *)dequant_high);

            } else if constexpr (residual % 4 == 2) {
                // Load 2 floats and set the last two to 0
                v1 = _mm_loadh_pi(_mm_setzero_ps(), (__m64 *)pVect1);

                // Dequantize two values
                float dequant_high[2] = {quantized[0] * delta + min, quantized[1] * delta + min};
                v2_dequant = _mm_loadh_pi(_mm_setzero_ps(), (__m64 *)dequant_high);

            } else if constexpr (residual % 4 == 1) {
                // Load 1 float and set the last three to 0
                v1 = _mm_load_ss(pVect1);

                // Dequantize one value
                float dequant0 = quantized[0] * delta + min;
                v2_dequant = _mm_load_ss(&dequant0);
            }

            pVect1 += residual % 4;
            quantized += residual % 4;
            sum1 = _mm_mul_ps(v1, v2_dequant); // Use sum1 for residual
        }
    }

    // Process 8 elements at a time
    while (pVect1 < pEnd1) {
        InnerProductStep(pVect1, quantized, sum1, sum2, min_val_vec, delta_vec);
    }

    // Combine the two sums
    __m128 sum = _mm_add_ps(sum1, sum2);

    // TmpRes must be 16 bytes aligned.
    float PORTABLE_ALIGN16 TmpRes[4];
    _mm_store_ps(TmpRes, sum);
    float result = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return result;
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_SSE4(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - SQ8_InnerProductSIMD16_SSE4_IMP<residual>(pVect1v, pVect2v, dimension);
}

template <unsigned char residual> // 0..15
float SQ8_CosineSIMD16_SSE4(const void *pVect1v, const void *pVect2v, size_t dimension) {

    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);
    // Get quantization parameters
    const float inv_norm = *reinterpret_cast<const float *>(pVect2 + dimension + 2 * sizeof(float));

    // Compute inner product with dequantization using the common function
    // We need to cast away const for the inner product function, but it doesn't modify the vectors
    const float res = SQ8_InnerProductSIMD16_SSE4_IMP<residual>(pVect1v, pVect2v, dimension);

    // For cosine, we need to account for the vector norms
    // The inv_norm parameter is stored after min_val and delta in the quantized vector
    return 1.0f - res * inv_norm;
}
