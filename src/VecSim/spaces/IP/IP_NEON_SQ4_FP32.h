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
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/types/sq4.h"
#include "VecSim/utils/alignment.h"
#include <arm_neon.h>
#include <cmath>

namespace {

/*
 * Asymmetric SQ4 inner product with the same identity as SQ8:
 *
 *   IP(x, y) ≈ Σ((min + delta * q_i) * y_i) = min * y_sum + delta * Σ(q_i * y_i)
 *
 * Storage packs coordinate i into byte i / 2, the low nibble for even i and the high nibble for
 * odd i, followed by unaligned FP32 metadata. The query blob carries y_sum after its dim floats.
 */

// Load exactly two packed bytes, yielding four coordinates in order without touching metadata.
static inline float32x4_t SQ4_NEON_Unpack4(const uint8_t *data) {
    const uint8x8_t packed = vcreate_u8(load_unaligned<uint16_t>(data));
    const uint8x8_t ordered = vzip_u8(vand_u8(packed, vdup_n_u8(15)), vshr_n_u8(packed, 4)).val[0];
    return vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(ordered))));
}

template <unsigned char residual> // 0..15
float SQ4_FP32_InnerProductSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    using vecsim_types::sq4;
    const auto *x = static_cast<const uint8_t *>(pVect1v);
    const auto *y = static_cast<const float *>(pVect2v);
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    const size_t complete = dimension - residual;
    size_t i = 0;
    for (; i < complete; i += 16) {
        // One eight-byte load covers exactly sixteen coordinates. Zipping the low and high
        // nibbles before widening keeps each four-lane vector in coordinate order.
        const uint8x8_t packed = vld1_u8(x + i / 2);
        const uint8x8x2_t ordered = vzip_u8(vand_u8(packed, vdup_n_u8(15)), vshr_n_u8(packed, 4));
        const uint16x8_t first = vmovl_u8(ordered.val[0]);
        const uint16x8_t second = vmovl_u8(ordered.val[1]);
        sum0 = vmlaq_f32(sum0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(first))), vld1q_f32(y + i));
        sum1 =
            vmlaq_f32(sum1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(first))), vld1q_f32(y + i + 4));
        sum2 =
            vmlaq_f32(sum2, vcvtq_f32_u32(vmovl_u16(vget_low_u16(second))), vld1q_f32(y + i + 8));
        sum3 =
            vmlaq_f32(sum3, vcvtq_f32_u32(vmovl_u16(vget_high_u16(second))), vld1q_f32(y + i + 12));
    }
    // Packed tail: groups of four read two whole bytes, the last 1..3 coordinates read nibbles,
    // so no load passes the code bytes (an odd dim's unused high nibble is never read).
    if constexpr (residual >= 4) {
        sum0 = vmlaq_f32(sum0, SQ4_NEON_Unpack4(x + i / 2), vld1q_f32(y + i));
        i += 4;
    }
    if constexpr (residual >= 8) {
        sum1 = vmlaq_f32(sum1, SQ4_NEON_Unpack4(x + i / 2), vld1q_f32(y + i));
        i += 4;
    }
    if constexpr (residual >= 12) {
        sum2 = vmlaq_f32(sum2, SQ4_NEON_Unpack4(x + i / 2), vld1q_f32(y + i));
        i += 4;
    }
    if constexpr (residual % 4 > 0) {
        float32x4_t codes = vdupq_n_f32(0.0f);
        float32x4_t query = vdupq_n_f32(0.0f);
        const float q0 = sq4::load_code(x, i);
        codes = vld1q_lane_f32(&q0, codes, 0);
        query = vld1q_lane_f32(y + i, query, 0);
        if constexpr (residual % 4 >= 2) {
            const float q1 = sq4::load_code(x, i + 1);
            codes = vld1q_lane_f32(&q1, codes, 1);
            query = vld1q_lane_f32(y + i + 1, query, 1);
        }
        if constexpr (residual % 4 >= 3) {
            const float q2 = sq4::load_code(x, i + 2);
            codes = vld1q_lane_f32(&q2, codes, 2);
            query = vld1q_lane_f32(y + i + 2, query, 2);
        }
        sum3 = vmlaq_f32(sum3, codes, query);
    }
    const float dot = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
    const auto *params = x + sq4::data_bytes_count(dimension);
    const float min_val = load_unaligned<float>(params + sq4::MIN_VAL * sizeof(float));
    const float delta = load_unaligned<float>(params + sq4::DELTA * sizeof(float));
    const float distance = 1.0f - (min_val * y[dimension + sq4::SUM_QUERY] + delta * dot);
    // The unscaled code dot can overflow even when the reconstructed products stay finite.
    if (!std::isfinite(distance)) {
        return SQ4_FP32_InnerProductReconstructed(pVect1v, pVect2v, dimension);
    }
    return distance;
}

} // namespace
