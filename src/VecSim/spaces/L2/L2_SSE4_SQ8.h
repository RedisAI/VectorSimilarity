/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"
#include <string.h>

static inline void L2SqrStep(const float *&pVect1, const uint8_t *&pVect2, __m128 &sum,
                             const __m128 &min_val_vec, const __m128 &delta_vec) {
    // Load 4 float elements from pVect1
    __m128 v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;

    // Load 4 uint8 elements from pVect2, convert to int32, then to float
    __m128i v2_i = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t *)pVect2));
    pVect2 += 4;

    // Convert int32 to float
    __m128 v2_f = _mm_cvtepi32_ps(v2_i);

    // Dequantize: (val * delta) + min_val
    __m128 v2_dequant = _mm_add_ps(_mm_mul_ps(v2_f, delta_vec), min_val_vec);

    // Compute difference
    __m128 diff = _mm_sub_ps(v1, v2_dequant);

    // Square difference and add to sum
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
}

template <unsigned char residual> // 0..15
float SQ8_L2SqrSIMD16_SSE4(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    const uint8_t *quantized = static_cast<const uint8_t *>(pVect2v);

    // Get dequantization parameters from the end of quantized vector
    const float min_val = *reinterpret_cast<const float *>(quantized + dimension);
    const float delta = *reinterpret_cast<const float *>(quantized + dimension + sizeof(float));

    // Create broadcast vectors for SIMD operations
    __m128 min_val_vec = _mm_set1_ps(min_val);
    __m128 delta_vec = _mm_set1_ps(delta);

    const float *pEnd1 = pVect1 + dimension;

    __m128 sum = _mm_setzero_ps();

    // Process residual elements if needed
    if constexpr (residual) {
        // Handle residual elements (1-3)
        if constexpr (residual % 4) {
            __m128 v1;
            __m128 v2_dequant;

            if constexpr (residual % 4 == 3) {
                // Load 3 floats and set the last one to 0
                v1 = _mm_set_ps(0.0f, pVect1[2], pVect1[1], pVect1[0]);

                // Dequantize and set 3 values
                v2_dequant = _mm_set_ps(0.0f, quantized[2] * delta + min_val,
                                                  quantized[1] * delta + min_val, quantized[0] * delta + min_val);

            } else if constexpr (residual % 4 == 2) {
                // Set 2 floats and the last two to 0
                v1 = _mm_set_ps(0.0f, 0.0f, pVect1[1], pVect1[0]);

                // Dequantize and set 2 valuesAdd commentMore actions
                v2_dequant = _mm_set_ps(0.0f, 0.0f,
                                       quantized[1] * delta + min_val,
                                       quantized[0] * delta + min_val);

            } else if constexpr (residual % 4 == 1) {
                // Set 1 float and the last three to 0Add commentMore actions
                v1 = _mm_set_ps(0.0f, 0.0f, 0.0f, pVect1[0]);

                // Dequantize and set 1 value
                v2_dequant = _mm_set_ps(0.0f, 0.0f, 0.0f, quantized[0] * delta + min_val);
            }

            pVect1 += residual % 4;
            quantized += residual % 4;

            // Compute difference
            __m128 diff = _mm_sub_ps(v1, v2_dequant);

            // Square difference and initialize sum
            sum = _mm_mul_ps(diff, diff);
        }

        // Process remaining blocks of 4 elements based on residual
        if constexpr (residual >= 12)
            L2SqrStep(pVect1, quantized, sum, min_val_vec, delta_vec);
        if constexpr (residual >= 8)
            L2SqrStep(pVect1, quantized, sum, min_val_vec, delta_vec);
        if constexpr (residual >= 4)
            L2SqrStep(pVect1, quantized, sum, min_val_vec, delta_vec);
    }

    // Process 16 elements at a time (4 elements per step, 4 steps)
    while (pVect1 < pEnd1) {
        L2SqrStep(pVect1, quantized, sum, min_val_vec, delta_vec);
        L2SqrStep(pVect1, quantized, sum, min_val_vec, delta_vec);
        L2SqrStep(pVect1, quantized, sum, min_val_vec, delta_vec);
        L2SqrStep(pVect1, quantized, sum, min_val_vec, delta_vec);
    }

    // TmpRes must be 16 bytes aligned
    float PORTABLE_ALIGN16 TmpRes[4];
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
