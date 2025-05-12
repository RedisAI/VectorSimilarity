/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/
#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

static inline void L2SqrStep(float *&pVect1, uint8_t *&pVect2, float32x4_t &sum,
                            const float32x4_t &min_val_vec, const float32x4_t &delta_vec) {
    // Load 4 float elements from pVect1
    float32x4_t v1 = vld1q_f32(pVect1);
    pVect1 += 4;

    // Load 4 uint8 elements from pVect2
    uint8x8_t v2_u8 = vld1_u8(pVect2);
    pVect2 += 4;

    // Convert uint8 to uint32
    uint32x4_t v2_u32 = vmovl_u16(vget_low_u16(vmovl_u8(v2_u8)));

    // Convert uint32 to float32
    float32x4_t v2_f = vcvtq_f32_u32(v2_u32);

    // Dequantize: (val * delta) + min_val
    float32x4_t v2_dequant = vmlaq_f32(min_val_vec, v2_f, delta_vec);

    // Compute difference
    float32x4_t diff = vsubq_f32(v1, v2_dequant);

    // Square difference and add to sum
    sum = vmlaq_f32(sum, diff, diff);
}

template <unsigned char residual> // 0..15
float SQ8_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    uint8_t *pVect2 = (uint8_t *)pVect2v;

    // Get dequantization parameters from the end of quantized vector
    const float min_val = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));

    // Create broadcast vectors for SIMD operations
    float32x4_t min_val_vec = vdupq_n_f32(min_val);
    float32x4_t delta_vec = vdupq_n_f32(delta);

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    const size_t num_of_chunks = dimension / 16;

    // Process 16 elements at a time in the main loop
    for (size_t i = 0; i < num_of_chunks; i++) {
        L2SqrStep(pVect1, pVect2, sum0, min_val_vec, delta_vec);
        L2SqrStep(pVect1, pVect2, sum1, min_val_vec, delta_vec);
        L2SqrStep(pVect1, pVect2, sum2, min_val_vec, delta_vec);
        L2SqrStep(pVect1, pVect2, sum3, min_val_vec, delta_vec);
    }

    // Handle remaining complete 4-float blocks within residual
    if constexpr (residual >= 4) {
        L2SqrStep(pVect1, pVect2, sum0, min_val_vec, delta_vec);
    }
    if constexpr (residual >= 8) {
        L2SqrStep(pVect1, pVect2, sum1, min_val_vec, delta_vec);
    }
    if constexpr (residual >= 12) {
        L2SqrStep(pVect1, pVect2, sum2, min_val_vec, delta_vec);
    }

    // Handle final residual elements (0-3 elements)
    constexpr size_t final_residual = residual % 4;
    if constexpr (final_residual > 0) {
        float32x4_t v1 = vdupq_n_f32(0.0f);
        float32x4_t v2_dequant = vdupq_n_f32(0.0f);

        if constexpr (final_residual >= 1) {
            v1 = vld1q_lane_f32(pVect1, v1, 0);
            float dequant0 = pVect2[0] * delta + min_val;
            v2_dequant = vld1q_lane_f32(&dequant0, v2_dequant, 0);
        }
        if constexpr (final_residual >= 2) {
            v1 = vld1q_lane_f32(pVect1 + 1, v1, 1);
            float dequant1 = pVect2[1] * delta + min_val;
            v2_dequant = vld1q_lane_f32(&dequant1, v2_dequant, 1);
        }
        if constexpr (final_residual >= 3) {
            v1 = vld1q_lane_f32(pVect1 + 2, v1, 2);
            float dequant2 = pVect2[2] * delta + min_val;
            v2_dequant = vld1q_lane_f32(&dequant2, v2_dequant, 2);
        }

        float32x4_t diff = vsubq_f32(v1, v2_dequant);
        sum3 = vmlaq_f32(sum3, diff, diff);
    }

    // Combine all four sum accumulators
    float32x4_t sum_combined = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));

    // Horizontal sum of the 4 elements in the combined NEON register
    float32x2_t sum_halves = vadd_f32(vget_low_f32(sum_combined), vget_high_f32(sum_combined));
    float32x2_t summed = vpadd_f32(sum_halves, sum_halves);
    float sum = vget_lane_f32(summed, 0);

    return sum;
}
