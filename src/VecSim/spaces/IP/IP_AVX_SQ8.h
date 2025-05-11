/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"

static inline void InnerProductStepSQ8(const float *&pVect1, const uint8_t *&pVect2, __m256 &sum256,
                                      const __m256 &min_val_vec, const __m256 &delta_vec) {
    // Load 8 float elements from pVect1
    __m256 v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    
    // Load 8 uint8 elements from pVect2, convert to int32, then to float
    __m128i v2_128 = _mm_loadl_epi64((__m128i*)pVect2);
    pVect2 += 8;
    
    // Zero-extend uint8 to int32
    __m256i v2_256 = _mm256_cvtepu8_epi32(v2_128);
    
    // Convert int32 to float
    __m256 v2_f = _mm256_cvtepi32_ps(v2_256);
    
    // Dequantize: (val * delta) + min_val
    __m256 v2_dequant = _mm256_add_ps(_mm256_mul_ps(v2_f, delta_vec), min_val_vec);
    
    // Compute dot product and add to sum
    sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2_dequant));
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    // pVect2 is a quantized uint8_t vector
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);
    const float *pEnd1 = pVect1 + dimension;
    
    // Get dequantization parameters from the end of quantized vector
    const float min_val = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));
    // Create broadcast vectors for SIMD operations
    __m256 min_val_vec = _mm256_set1_ps(min_val);
    __m256 delta_vec = _mm256_set1_ps(delta);

    __m256 sum256 = _mm256_setzero_ps();

    // Deal with 1-7 floats with mask loading, if needed. `dim` is >16, so we have at least one
    // 16-float block, so mask loading is guaranteed to be safe.
    if constexpr (residual % 8) {
        __mmask8 constexpr mask = (1 << (residual % 8)) - 1;
        __m256 v1 = my_mm256_maskz_loadu_ps<mask>(pVect1);
        pVect1 += residual % 8;
        
        // Load quantized values and dequantize
        __m128i v2_128 = _mm_loadl_epi64((__m128i*)pVect2);
        pVect2 += residual % 8;
        
        // Zero-extend uint8 to int32
        __m256i v2_256 = _mm256_cvtepu8_epi32(v2_128);
        
        // Convert int32 to float
        __m256 v2_f = _mm256_cvtepi32_ps(v2_256);
        
        // Dequantize: (val * delta) + min_val
        __m256 v2_dequant = _mm256_add_ps(_mm256_mul_ps(v2_f, delta_vec), min_val_vec);
        
        // Compute dot product with masking
        sum256 = _mm256_mul_ps(v1, v2_dequant);
    }

    // If the reminder is >=8, have another step of 8 floats
    if constexpr (residual >= 8) {
        InnerProductStepSQ8(pVect1, pVect2, sum256, min_val_vec, delta_vec);
    }

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    // In each iteration we calculate 16 floats = 512 bits.
    do {
        InnerProductStepSQ8(pVect1, pVect2, sum256, min_val_vec, delta_vec);
        InnerProductStepSQ8(pVect1, pVect2, sum256, min_val_vec, delta_vec);
    } while (pVect1 < pEnd1);

    return my_mm256_reduce_add_ps(sum256);
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_AVX(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - SQ8_InnerProductImp<residual>(pVect1v, pVect2v, dimension);
}

template <unsigned char residual> // 0..15
float SQ8_CosineSIMD16_AVX(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // Get dequantization parameters from the end of quantized vector
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);
    const float inv_norm = *reinterpret_cast<const float *>(pVect2 + dimension + 2 * sizeof(float));
    
    // Calculate inner product using common implementation with normalization
    float ip = SQ8_InnerProductImp<residual>(pVect1v, pVect2v, dimension);
    
    // For cosine, we need to account for the vector norms
    // The inv_norm parameter is stored after min_val and delta in the quantized vector
    return 1.0f - ip * inv_norm;
}
