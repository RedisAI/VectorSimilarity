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
#include "VecSim/spaces/IP/IP_NEON_SQ4_FP32.h" // SQ4_NEON_Unpack4
#include "VecSim/types/sq4.h"
#include "VecSim/utils/alignment.h"
#include <arm_neon.h>

namespace {

/*
 * Asymmetric SQ4-FP32 L2 squared distance via direct residual accumulation, as SQ8:
 *
 *   ||x - y||² = Σ(delta * q_i + (min - y_i))²
 *
 * Keeping min - y first (one fused multiply-add per coordinate) rounds at the scale of the
 * residual, not of min, when the data has a large common offset. Storage packs coordinate i into
 * byte i / 2, the low nibble for even i and the high nibble for odd i, followed by unaligned FP32
 * metadata.
 */

// Accumulate Σ(diff_i²) for four coordinates.
static inline float32x4_t L2StepSQ4_FP32_NEON(float32x4_t sum, float32x4_t codes, float32x4_t query,
                                              float32x4_t min_val_vec, float32x4_t delta_vec) {
    const float32x4_t diff = vfmaq_f32(vsubq_f32(min_val_vec, query), delta_vec, codes);
    return vfmaq_f32(sum, diff, diff);
}

template <unsigned char residual> // 0..15
float SQ4_FP32_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    using vecsim_types::sq4;
    const auto *x = static_cast<const uint8_t *>(pVect1v);
    const auto *y = static_cast<const float *>(pVect2v);
    const auto *params = x + sq4::data_bytes_count(dimension);
    const float min_val = load_unaligned<float>(params + sq4::MIN_VAL * sizeof(float));
    const float delta = load_unaligned<float>(params + sq4::DELTA * sizeof(float));
    const float32x4_t min_val_vec = vdupq_n_f32(min_val);
    const float32x4_t delta_vec = vdupq_n_f32(delta);
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
        sum0 = L2StepSQ4_FP32_NEON(sum0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(first))),
                                   vld1q_f32(y + i), min_val_vec, delta_vec);
        sum1 = L2StepSQ4_FP32_NEON(sum1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(first))),
                                   vld1q_f32(y + i + 4), min_val_vec, delta_vec);
        sum2 = L2StepSQ4_FP32_NEON(sum2, vcvtq_f32_u32(vmovl_u16(vget_low_u16(second))),
                                   vld1q_f32(y + i + 8), min_val_vec, delta_vec);
        sum3 = L2StepSQ4_FP32_NEON(sum3, vcvtq_f32_u32(vmovl_u16(vget_high_u16(second))),
                                   vld1q_f32(y + i + 12), min_val_vec, delta_vec);
    }
    // Packed tail: groups of four read two whole bytes, the last 1..3 coordinates read nibbles,
    // so no load passes the code bytes (an odd dim's unused high nibble is never read).
    if constexpr (residual >= 4) {
        sum0 = L2StepSQ4_FP32_NEON(sum0, SQ4_NEON_Unpack4(x + i / 2), vld1q_f32(y + i), min_val_vec,
                                   delta_vec);
        i += 4;
    }
    if constexpr (residual >= 8) {
        sum1 = L2StepSQ4_FP32_NEON(sum1, SQ4_NEON_Unpack4(x + i / 2), vld1q_f32(y + i), min_val_vec,
                                   delta_vec);
        i += 4;
    }
    if constexpr (residual >= 12) {
        sum2 = L2StepSQ4_FP32_NEON(sum2, SQ4_NEON_Unpack4(x + i / 2), vld1q_f32(y + i), min_val_vec,
                                   delta_vec);
        i += 4;
    }
    if constexpr (residual % 4 > 0) {
        // Padding lanes get q = 0 and y = min, so their diff is exactly 0.
        float32x4_t codes = vdupq_n_f32(0.0f);
        float32x4_t query = min_val_vec;
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
        sum3 = L2StepSQ4_FP32_NEON(sum3, codes, query, min_val_vec, delta_vec);
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
}

} // namespace
