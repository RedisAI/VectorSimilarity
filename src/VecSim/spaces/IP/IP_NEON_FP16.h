/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include <arm_neon.h>

// The products accumulate in fp32, not fp16. `FP16_InnerProduct` widens each stored half with
// FP16_to_FP32 and accumulates into a float, so a half precision accumulator would make this tier
// disagree with the same function computed anywhere else. It is also unsafe: 65504 is the largest
// finite fp16 value, so 32 elements of 200.0, all ordinary fp16 values, drive a half precision
// accumulator past it and the result becomes infinity. An fp32 accumulator cannot overflow for
// any fp16 input.
inline void InnerProduct_Step(const float16_t *&vec1, const float16_t *&vec2, float32x4_t &acc_lo,
                              float32x4_t &acc_hi) {
    // Load half-precision vectors
    float16x8_t v1 = vld1q_f16(vec1);
    float16x8_t v2 = vld1q_f16(vec2);
    vec1 += 8;
    vec2 += 8;

    // Widen both halves to single precision, then multiply and accumulate in fp32.
    acc_lo = vfmaq_f32(acc_lo, vcvt_f32_f16(vget_low_f16(v1)), vcvt_f32_f16(vget_low_f16(v2)));
    acc_hi = vfmaq_f32(acc_hi, vcvt_high_f32_f16(v1), vcvt_high_f32_f16(v2));
}

template <unsigned char residual> // 0..31
float FP16_InnerProduct_NEON_HP(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *vec1 = static_cast<const float16_t *>(pVect1v);
    const auto *vec2 = static_cast<const float16_t *>(pVect2v);
    const auto *const v1End = vec1 + dimension;
    // Four accumulator pairs, keeping the original four-way unrolling with fp32 accumulators.
    float32x4_t acc1_lo = vdupq_n_f32(0.0f), acc1_hi = vdupq_n_f32(0.0f);
    float32x4_t acc2_lo = vdupq_n_f32(0.0f), acc2_hi = vdupq_n_f32(0.0f);
    float32x4_t acc3_lo = vdupq_n_f32(0.0f), acc3_hi = vdupq_n_f32(0.0f);
    float32x4_t acc4_lo = vdupq_n_f32(0.0f), acc4_hi = vdupq_n_f32(0.0f);

    // First, handle the partial chunk residual
    if constexpr (residual % 8) {
        auto constexpr chunk_residual = residual % 8;
        // TODO: spacial cases for some residuals and benchmark if its better
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

        // Load partial vectors
        float16x8_t v1 = vld1q_f16(vec1);
        float16x8_t v2 = vld1q_f16(vec2);

        // Apply mask to both vectors, zeroing the lanes past the residual.
        const float16x8_t zero_h = vdupq_n_f16(0.0f);
        float16x8_t masked_v1 = vbslq_f16(mask, v1, zero_h);
        float16x8_t masked_v2 = vbslq_f16(mask, v2, zero_h);

        // Multiply and accumulate in fp32; the masked lanes contribute zero.
        acc1_lo = vfmaq_f32(acc1_lo, vcvt_f32_f16(vget_low_f16(masked_v1)),
                            vcvt_f32_f16(vget_low_f16(masked_v2)));
        acc1_hi = vfmaq_f32(acc1_hi, vcvt_high_f32_f16(masked_v1), vcvt_high_f32_f16(masked_v2));

        // Advance pointers
        vec1 += chunk_residual;
        vec2 += chunk_residual;
    }

    // Handle (residual - (residual % 8)) in chunks of 8 float16
    if constexpr (residual >= 8)
        InnerProduct_Step(vec1, vec2, acc2_lo, acc2_hi);
    if constexpr (residual >= 16)
        InnerProduct_Step(vec1, vec2, acc3_lo, acc3_hi);
    if constexpr (residual >= 24)
        InnerProduct_Step(vec1, vec2, acc4_lo, acc4_hi);

    // Process the rest of the vectors (the full chunks part)
    while (vec1 < v1End) {
        // TODO: use `vld1q_f16_x4` for quad-loading?
        InnerProduct_Step(vec1, vec2, acc1_lo, acc1_hi);
        InnerProduct_Step(vec1, vec2, acc2_lo, acc2_hi);
        InnerProduct_Step(vec1, vec2, acc3_lo, acc3_hi);
        InnerProduct_Step(vec1, vec2, acc4_lo, acc4_hi);
    }

    // Accumulate accumulators, all in fp32.
    float32x4_t sum_f32 = vaddq_f32(vaddq_f32(acc1_lo, acc1_hi), vaddq_f32(acc2_lo, acc2_hi));
    sum_f32 = vaddq_f32(sum_f32, vaddq_f32(acc3_lo, acc3_hi));
    sum_f32 = vaddq_f32(sum_f32, vaddq_f32(acc4_lo, acc4_hi));

    // Extract result
    return 1.0f - vaddvq_f32(sum_f32);
}
