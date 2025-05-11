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
#include <iostream>

static inline void L2SqrStep(float *&pVect1, uint8_t *&pVect2, __m256 &sum, 
                            const __m256 &min_val_vec, const __m256 &delta_vec) {
    // Load 8 float elements from pVect1
    __m256 v1 = _mm256_loadu_ps(pVect1);
    
    // Load 8 uint8 elements from pVect2
    __m128i v2_128 = _mm_loadl_epi64((__m128i*)pVect2);
    
    // Zero-extend uint8 to int32
    __m256i v2_256 = _mm256_cvtepu8_epi32(v2_128);
    
    // Convert int32 to float
    __m256 v2_f = _mm256_cvtepi32_ps(v2_256);
    
    // Dequantize: (val * delta) + min_val
    __m256 v2_dequant = _mm256_add_ps(_mm256_mul_ps(v2_f, delta_vec), min_val_vec);
    
    // Compute difference
    __m256 diff = _mm256_sub_ps(v1, v2_dequant);
    
    // Square difference and add to sum
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    
    // Advance pointers
    pVect1 += 8;
    pVect2 += 8;
}

template <unsigned char residual> // 0..15
float SQ8_L2SqrSIMD16_AVX(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    uint8_t *pVect2 = (uint8_t *)pVect2v;
    float *pVect1_debug = (float *)pVect1v;
    uint8_t *pVect2_debug = (uint8_t *)pVect2v; 
    // Get dequantization parameters from the end of quantized vector
    const float min_val = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));
    // Create broadcast vectors for SIMD operations
    __m256 min_val_vec = _mm256_set1_ps(min_val);
    __m256 delta_vec = _mm256_set1_ps(delta);

    const float *pEnd1 = pVect1 + dimension;

    __m256 sum = _mm256_setzero_ps();

    // Deal with 1-7 floats with mask loading, if needed
    if constexpr (residual % 8) {
        __mmask8 constexpr mask = (1 << (residual % 8)) - 1;
        __m256 v1 = my_mm256_maskz_loadu_ps<mask>(pVect1);
        pVect1 += residual % 8;
        
        uint8_t temp_buf[8] = {0};
        // Manually copy elements
        for (size_t i = 0; i < residual % 8; i++) {
            temp_buf[i] = pVect2[i];
        }
        // Load from buffer
        __m128i v2_128 = _mm_loadl_epi64((__m128i*)temp_buf);
        pVect2 += residual % 8;
        
        // Zero-extend uint8 to int32
        __m256i v2_256 = _mm256_cvtepu8_epi32(v2_128);
        
        // Convert int32 to float
        __m256 v2_f = _mm256_cvtepi32_ps(v2_256);
        
        // Dequantize: (val * delta) + min_val
        __m256 v2_dequant = _mm256_add_ps(_mm256_mul_ps(v2_f, delta_vec), min_val_vec);
        // print debug information
        // std::cout << "v2_dequant before: ";
        // for (size_t i = 0; i <  8; i++) {
        //     std::cout <<  v2_dequant[i] << " ";
        // }
        // std::cout << std::endl;
        
        v2_dequant = _mm256_blend_ps(_mm256_setzero_ps(), v2_dequant, mask);
        // std::cout << "v2_dequant after: ";
        // for (size_t i = 0; i <  8; i++) {
        //     std::cout <<  v2_dequant[i] << " ";
        // }
        // std::cout << std::endl;

        __m256 diff = _mm256_sub_ps(v1, v2_dequant);


        sum = _mm256_mul_ps(diff, diff);
        // print sum
    }

    // If the reminder is >= 8, have another step of 8 floats
    if constexpr (residual >= 8) {
        L2SqrStep(pVect1, pVect2, sum, min_val_vec, delta_vec);
    }
    float naive_sum = 0;
    for (size_t i = 0; i < residual; i++) {
        auto dequantized_V2 = (pVect2_debug[i] * delta + min_val);
        float t = pVect1_debug[i] - dequantized_V2;
        naive_sum += t * t;
    }
    
    // We dealt with the residual part. We are left with some multiple of 16 floats.
    // In each iteration we calculate 16 floats = 512 bits.
    do {
        L2SqrStep(pVect1, pVect2, sum, min_val_vec, delta_vec);
        L2SqrStep(pVect1, pVect2, sum, min_val_vec, delta_vec);
    } while (pVect1 < pEnd1);

    return my_mm256_reduce_add_ps(sum);
}
