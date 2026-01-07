/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include <arm_neon.h>

// Load 8 FP16 elements, convert to FP32, compute diff^2, and accumulate in FP32.
// Uses two FP32 accumulators (low and high) for the 8 FP16 elements.
inline void FP16_L2Sqr_Step(const float16_t *&vec1, const float16_t *&vec2,
                            float32x4_t &acc_lo, float32x4_t &acc_hi) {
    // Load 8 half-precision elements
    float16x8_t v1_f16 = vld1q_f16(vec1);
    float16x8_t v2_f16 = vld1q_f16(vec2);
    vec1 += 8;
    vec2 += 8;

    // Convert low 4 FP16 elements to FP32 and compute
    float32x4_t v1_lo = vcvt_f32_f16(vget_low_f16(v1_f16));
    float32x4_t v2_lo = vcvt_f32_f16(vget_low_f16(v2_f16));
    float32x4_t diff_lo = vsubq_f32(v1_lo, v2_lo);
    acc_lo = vfmaq_f32(acc_lo, diff_lo, diff_lo);

    // Convert high 4 FP16 elements to FP32 and compute
    float32x4_t v1_hi = vcvt_f32_f16(vget_high_f16(v1_f16));
    float32x4_t v2_hi = vcvt_f32_f16(vget_high_f16(v2_f16));
    float32x4_t diff_hi = vsubq_f32(v1_hi, v2_hi);
    acc_hi = vfmaq_f32(acc_hi, diff_hi, diff_hi);
}

template <unsigned char residual> // 0..31
float FP16_L2Sqr_NEON_HP(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *vec1 = static_cast<const float16_t *>(pVect1v);
    const auto *vec2 = static_cast<const float16_t *>(pVect2v);
    const auto *const v1End = vec1 + dimension;

    // Accumulate in FP32 for precision - use pairs for low/high halves
    float32x4_t acc1_lo = vdupq_n_f32(0.0f);
    float32x4_t acc1_hi = vdupq_n_f32(0.0f);
    float32x4_t acc2_lo = vdupq_n_f32(0.0f);
    float32x4_t acc2_hi = vdupq_n_f32(0.0f);

    // First, handle the partial chunk residual
    if constexpr (residual % 8) {
        auto constexpr chunk_residual = residual % 8;

        // Load partial vectors
        float16x8_t v1_f16 = vld1q_f16(vec1);
        float16x8_t v2_f16 = vld1q_f16(vec2);

        // Create zero vector for masking
        float16x8_t zero_f16 = vdupq_n_f16(0.0f);

        // Apply mask to both vectors
        constexpr uint16x8_t mask = {
            0xFFFF,
            (chunk_residual >= 2) ? 0xFFFF : 0,
            (chunk_residual >= 3) ? 0xFFFF : 0,
            (chunk_residual >= 4) ? 0xFFFF : 0,
            (chunk_residual >= 5) ? 0xFFFF : 0,
            (chunk_residual >= 6) ? 0xFFFF : 0,
            (chunk_residual >= 7) ? 0xFFFF : 0,
            0,
        };
        float16x8_t masked_v1 = vbslq_f16(mask, v1_f16, zero_f16);
        float16x8_t masked_v2 = vbslq_f16(mask, v2_f16, zero_f16);

        // Convert to FP32 and compute
        float32x4_t v1_lo = vcvt_f32_f16(vget_low_f16(masked_v1));
        float32x4_t v2_lo = vcvt_f32_f16(vget_low_f16(masked_v2));
        float32x4_t diff_lo = vsubq_f32(v1_lo, v2_lo);
        acc1_lo = vfmaq_f32(acc1_lo, diff_lo, diff_lo);

        float32x4_t v1_hi = vcvt_f32_f16(vget_high_f16(masked_v1));
        float32x4_t v2_hi = vcvt_f32_f16(vget_high_f16(masked_v2));
        float32x4_t diff_hi = vsubq_f32(v1_hi, v2_hi);
        acc1_hi = vfmaq_f32(acc1_hi, diff_hi, diff_hi);

        // Advance pointers
        vec1 += chunk_residual;
        vec2 += chunk_residual;
    }

    // Handle (residual - (residual % 8)) in chunks of 8 float16
    if constexpr (residual >= 8)
        FP16_L2Sqr_Step(vec1, vec2, acc2_lo, acc2_hi);
    if constexpr (residual >= 16)
        FP16_L2Sqr_Step(vec1, vec2, acc1_lo, acc1_hi);
    if constexpr (residual >= 24)
        FP16_L2Sqr_Step(vec1, vec2, acc2_lo, acc2_hi);

    // Process the rest of the vectors (the full chunks part)
    while (vec1 < v1End) {
        FP16_L2Sqr_Step(vec1, vec2, acc1_lo, acc1_hi);
        FP16_L2Sqr_Step(vec1, vec2, acc2_lo, acc2_hi);
        FP16_L2Sqr_Step(vec1, vec2, acc1_lo, acc1_hi);
        FP16_L2Sqr_Step(vec1, vec2, acc2_lo, acc2_hi);
    }

    // Combine all accumulators
    acc1_lo = vaddq_f32(acc1_lo, acc2_lo);
    acc1_hi = vaddq_f32(acc1_hi, acc2_hi);
    acc1_lo = vaddq_f32(acc1_lo, acc1_hi);

    // Horizontal sum
    float32x2_t sum_2 = vadd_f32(vget_low_f32(acc1_lo), vget_high_f32(acc1_lo));
    sum_2 = vpadd_f32(sum_2, sum_2);

    return vget_lane_f32(sum_2, 0);
}
