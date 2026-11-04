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
#include "VecSim/spaces/IP/IP_NEON_SQ4_FP32.h" // SQ4_NEON_WidenIndex
#include "VecSim/spaces/IP/IP_NEON_SQ4_SQ4.h"  // SQ4TailNibbleMask
#include "VecSim/types/sq4.h"
#include "VecSim/utils/alignment.h"
#include <arm_neon.h>

namespace {

/*
 * Symmetric SQ4-SQ4 L2 squared distance, reconstructed per element as the scalar SQ4_SQ4_L2Sqr:
 *
 *   ||x - y||^2 = Σ((min_x - min_y) + (delta_x * qx_i - delta_y * qy_i))^2
 *
 * Subtracting the offsets first keeps rounding at the scale of the residual, not of the norms,
 * when the data has a large common offset (the ||x||^2 + ||y||^2 - 2 * IP identity loses HNSW
 * recall there). Low nibbles (even coordinates) and high nibbles (odd coordinates) are processed in
 * separate lanes; both operands share the packing and L2 is a plain sum, so no reordering is
 * needed.
 */

// Accumulate diff^2 for eight coordinates held as codes in u8 lanes, into two accumulators. Only
// the first `kept` lanes count: a tail masks the others' diff to zero.
template <unsigned kept = 8>
static inline void L2StepSQ4_SQ4_NEON(float32x4_t &sum_a, float32x4_t &sum_b, uint8x8_t qx,
                                      uint8x8_t qy, float32x4_t min_diff, float32x4_t delta_x,
                                      float32x4_t delta_y) {
    const uint16x8_t wx = vmovl_u8(qx), wy = vmovl_u8(qy);
    const float32x4_t xa = vcvtq_f32_u32(vmovl_u16(vget_low_u16(wx)));
    const float32x4_t xb = vcvtq_f32_u32(vmovl_u16(vget_high_u16(wx)));
    const float32x4_t ya = vcvtq_f32_u32(vmovl_u16(vget_low_u16(wy)));
    const float32x4_t yb = vcvtq_f32_u32(vmovl_u16(vget_high_u16(wy)));
    float32x4_t da = vaddq_f32(min_diff, vfmsq_f32(vmulq_f32(delta_x, xa), delta_y, ya));
    float32x4_t db = vaddq_f32(min_diff, vfmsq_f32(vmulq_f32(delta_x, xb), delta_y, yb));
    if constexpr (kept < 8) {
        const uint32x4_t lane = {0, 1, 2, 3};
        const uint32x4_t keep_a = vcltq_u32(lane, vdupq_n_u32(kept));
        const uint32x4_t keep_b = vcltq_u32(vaddq_u32(lane, vdupq_n_u32(4)), vdupq_n_u32(kept));
        da = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(da), keep_a));
        db = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(db), keep_b));
    }
    sum_a = vfmaq_f32(sum_a, da, da);
    sum_b = vfmaq_f32(sum_b, db, db);
}

// diff^2 for four coordinates held as codes in u32 lanes.
static inline float32x4_t L2StepSQ4_SQ4_NEON_U32(float32x4_t sum, uint32x4_t qx, uint32x4_t qy,
                                                 float32x4_t min_diff, float32x4_t delta_x,
                                                 float32x4_t delta_y) {
    const float32x4_t d = vaddq_f32(
        min_diff, vfmsq_f32(vmulq_f32(delta_x, vcvtq_f32_u32(qx)), delta_y, vcvtq_f32_u32(qy)));
    return vfmaq_f32(sum, d, d);
}

template <unsigned char residual> // 0..15
float SQ4_SQ4_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    using vecsim_types::sq4;
    const auto *x = static_cast<const uint8_t *>(pVect1v);
    const auto *y = static_cast<const uint8_t *>(pVect2v);
    const auto *params_x = x + sq4::data_bytes_count(dimension);
    const auto *params_y = y + sq4::data_bytes_count(dimension);
    const float min_x = load_unaligned<float>(params_x + sq4::MIN_VAL * sizeof(float));
    const float delta_x = load_unaligned<float>(params_x + sq4::DELTA * sizeof(float));
    const float min_y = load_unaligned<float>(params_y + sq4::MIN_VAL * sizeof(float));
    const float delta_y = load_unaligned<float>(params_y + sq4::DELTA * sizeof(float));
    const float32x4_t min_diff = vdupq_n_f32(min_x - min_y);
    const float32x4_t delta_x_vec = vdupq_n_f32(delta_x);
    const float32x4_t delta_y_vec = vdupq_n_f32(delta_y);
    const uint8x8_t nibble = vdup_n_u8(15);
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    const size_t complete = dimension - residual;
    size_t i = 0;
    // One 8-byte load per operand covers 16 coordinates.
    const uint8x16_t widen0 = SQ4_NEON_WidenIndex(0), widen1 = SQ4_NEON_WidenIndex(4);
    const uint32x4_t low_mask = vdupq_n_u32(15);
    for (; i < complete; i += 16) {
        // One 8-byte load per operand covers 16 coordinates. TBL puts one packed byte in each u32
        // lane; the low nibbles (even coordinates) and high nibbles (odd ones) of both operands
        // pair up without reordering.
        const uint8x16_t xb = vcombine_u8(vld1_u8(x + i / 2), vdup_n_u8(0));
        const uint8x16_t yb = vcombine_u8(vld1_u8(y + i / 2), vdup_n_u8(0));
        const uint32x4_t x0 = vreinterpretq_u32_u8(vqtbl1q_u8(xb, widen0));
        const uint32x4_t x1 = vreinterpretq_u32_u8(vqtbl1q_u8(xb, widen1));
        const uint32x4_t y0 = vreinterpretq_u32_u8(vqtbl1q_u8(yb, widen0));
        const uint32x4_t y1 = vreinterpretq_u32_u8(vqtbl1q_u8(yb, widen1));
        sum0 = L2StepSQ4_SQ4_NEON_U32(sum0, vandq_u32(x0, low_mask), vandq_u32(y0, low_mask),
                                      min_diff, delta_x_vec, delta_y_vec);
        sum1 = L2StepSQ4_SQ4_NEON_U32(sum1, vandq_u32(x1, low_mask), vandq_u32(y1, low_mask),
                                      min_diff, delta_x_vec, delta_y_vec);
        sum2 = L2StepSQ4_SQ4_NEON_U32(sum2, vshrq_n_u32(x0, 4), vshrq_n_u32(y0, 4), min_diff,
                                      delta_x_vec, delta_y_vec);
        sum3 = L2StepSQ4_SQ4_NEON_U32(sum3, vshrq_n_u32(x1, 4), vshrq_n_u32(y1, 4), min_diff,
                                      delta_x_vec, delta_y_vec);
    }
    if constexpr (residual > 0) {
        // One 8-byte load per operand from an even coordinate. The bytes past the last code are the
        // metadata that follows the codes in every stored blob (at least 12 bytes, so the load
        // stays inside it). The lane masks zero their diff, and an odd dim's unused high nibble's.
        const uint8x8_t xb = vld1_u8(x + i / 2), yb = vld1_u8(y + i / 2);
        L2StepSQ4_SQ4_NEON<(residual + 1) / 2>(sum0, sum1, vand_u8(xb, nibble), vand_u8(yb, nibble),
                                               min_diff, delta_x_vec, delta_y_vec);
        if constexpr (residual >= 2) {
            L2StepSQ4_SQ4_NEON<residual / 2>(sum2, sum3, vshr_n_u8(xb, 4), vshr_n_u8(yb, 4),
                                             min_diff, delta_x_vec, delta_y_vec);
        }
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
}

} // namespace
