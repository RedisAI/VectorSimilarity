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

static inline void InnerProductStepSQ8(float *&pVect1, uint8_t *&pVect2, __m256 &sum256,
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
float SQ8_InnerProductSIMD16_AVX(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    uint8_t *quantized = (uint8_t *)pVect2v;

    // Get dequantization parameters from the end of quantized vector
    float min = *(float *)(quantized + dimension);
    float delta = *(float *)(quantized + dimension + sizeof(float));
    
    // Create broadcast vectors for SIMD operations
    __m256 min_val_vec = _mm256_set1_ps(min);
    __m256 delta_vec = _mm256_set1_ps(delta);

    const float *pEnd1 = pVect1 + dimension;

    __m256 sum256 = _mm256_setzero_ps();

    // Deal with 1-7 floats with mask loading, if needed
    if constexpr (residual % 8) {
        __mmask8 constexpr mask = (1 << (residual % 8)) - 1;
        
        // Load masked float elements
        __m256 v1 = my_mm256_maskz_loadu_ps<mask>(pVect1);
        pVect1 += residual % 8;
        
        // Load masked uint8 elements
        __m128i v2_128;
        if constexpr (residual % 8 <= 4) {
            // Load 4 or fewer bytes directly using unaligned loads and shifts
            uint32_t temp = 0;
            // Direct byte-by-byte loading to avoid memcpy
            switch (residual % 8) {
                case 4: temp |= (uint32_t)quantized[3] << 24;
                case 3: temp |= (uint32_t)quantized[2] << 16;
                case 2: temp |= (uint32_t)quantized[1] << 8;
                case 1: temp |= quantized[0];
            }
            v2_128 = _mm_cvtsi32_si128(temp);
        } else {
            // Load 5-7 bytes directly using unaligned loads and shifts
            uint64_t temp = 0;
            // Direct byte-by-byte loading to avoid memcpy
            switch (residual % 8) {
                case 7: temp |= (uint64_t)quantized[6] << 48;
                case 6: temp |= (uint64_t)quantized[5] << 40;
                case 5: temp |= (uint64_t)quantized[4] << 32;
                case 4: temp |= (uint64_t)quantized[3] << 24;
                case 3: temp |= (uint64_t)quantized[2] << 16;
                case 2: temp |= (uint64_t)quantized[1] << 8;
                case 1: temp |= quantized[0];
            }
            v2_128 = _mm_cvtsi64_si128(temp);
        }
        quantized += residual % 8;
        
        // Zero-extend uint8 to int32
        __m256i v2_256 = _mm256_cvtepu8_epi32(v2_128);
        
        // Convert int32 to float
        __m256 v2_f = _mm256_cvtepi32_ps(v2_256);
        
        // Dequantize: (val * delta) + min
        __m256 v2_dequant = _mm256_add_ps(_mm256_mul_ps(v2_f, delta_vec), min_val_vec);
        
        // Compute dot product with masking
        sum256 = _mm256_mul_ps(v1, v2_dequant);
    }

    // If the reminder is >=8, have another step of 8 floats
    if constexpr (residual >= 8) {
        InnerProductStepSQ8(pVect1, quantized, sum256, min_val_vec, delta_vec);
    }

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    // In each iteration we calculate 16 floats = 512 bits.
    do {
        InnerProductStepSQ8(pVect1, quantized, sum256, min_val_vec, delta_vec);
        InnerProductStepSQ8(pVect1, quantized, sum256, min_val_vec, delta_vec);
    } while (pVect1 < pEnd1);

    return 1.0f - my_mm256_reduce_add_ps(sum256);
}
