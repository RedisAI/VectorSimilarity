/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/
#include "VecSim/spaces/space_includes.h"

// Helper function to perform L2 squared distance calculation for a chunk of 16 elements
static inline void
SQ8_L2SqrStep(const float *&pVect1, const uint8_t *&pVect2, __m512 &sum,
              const __m512 &min_val_vec, const __m512 &delta_vec, const __m512 &inv_norm_vec) {
    // Load 16 float elements from pVect1
    __m512 v1 = _mm512_loadu_ps(pVect1);

    // Load 16 uint8 elements from pVect2 and convert to __m512i
    __m128i v2_128 = _mm_loadu_si128((__m128i*)pVect2);
    __m512i v2_512 = _mm512_cvtepu8_epi32(v2_128);

    // Convert uint8 to float
    __m512 v2_f = _mm512_cvtepi32_ps(v2_512);

    // Dequantize: (val * delta + min_val) * inv_norm
    __m512 dequantized = _mm512_fmadd_ps(v2_f, delta_vec, min_val_vec);
    dequantized = _mm512_mul_ps(dequantized, inv_norm_vec);

    // Compute difference
    __m512 diff = _mm512_sub_ps(v1, dequantized);

    // Square difference and add to sum
    sum = _mm512_fmadd_ps(diff, diff, sum);

    // Advance pointers
    pVect1 += 16;
    pVect2 += 16;
}

template <unsigned char residual> // 0..15
float SQ8_L2SqrSIMD16_AVX512F_BW_VL_VNNI(const void *pVect1v, const void *pVect2v,
                                          size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);
    const float *pEnd1 = pVect1 + dimension;

    // Get dequantization parameters from the end of pVect2
    const float min_val = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));
    const float inv_norm = *reinterpret_cast<const float *>(pVect2 + dimension + 2 * sizeof(float));

    // Create broadcast vectors for SIMD operations
    __m512 min_val_vec = _mm512_set1_ps(min_val);
    __m512 delta_vec = _mm512_set1_ps(delta);
    __m512 inv_norm_vec = _mm512_set1_ps(inv_norm);

    // Initialize sum accumulator
    __m512 sum = _mm512_setzero_ps();
    
    // Handle residual elements (0 to 15)
    if constexpr (residual > 0) {
        // Create mask for residual elements
        __mmask16 mask = (1U << residual) - 1;

        // Load masked float elements from pVect1
        __m512 v1 = _mm512_maskz_loadu_ps(mask, pVect1);

        // Load masked uint8 elements from pVect2
        __m128i v2_128 = _mm_maskz_loadu_epi8(mask, reinterpret_cast<const __m128i*>(pVect2));
        __m512i v2_512 = _mm512_cvtepu8_epi32(v2_128);
        __m512 v2_f = _mm512_cvtepi32_ps(v2_512);

        // Dequantize: (val * delta + min_val) * inv_norm
        __m512 dequantized = _mm512_fmadd_ps(v2_f, delta_vec, min_val_vec);
        dequantized = _mm512_mul_ps(dequantized, inv_norm_vec);

        // Compute difference
        __m512 diff = _mm512_sub_ps(v1, dequantized);

        // Square difference and add to sum (with mask)
        __m512 squared = _mm512_mul_ps(diff, diff);
        sum = _mm512_mask_add_ps(sum, mask, sum, squared);

        // Advance pointers
        pVect1 += residual;
        pVect2 += residual;
    }

    // Process remaining full chunks of 16 elements
    do  {
        SQ8_L2SqrStep(pVect1, pVect2, sum, min_val_vec, delta_vec, inv_norm_vec);
    }while (pVect1 < pEnd1);

    // Horizontal sum
    float result = _mm512_reduce_add_ps(sum);
    
    return result;
}
