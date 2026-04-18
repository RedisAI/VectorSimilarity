/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "TQ_POLAR_NEON.h"

#include <arm_neon.h>

namespace spaces {

namespace {

inline int FinishPackedResidualSignDot(const uint8_t *lhs, const uint8_t *rhs, size_t projections,
                                       size_t processed_bytes, uint32_t diff_count_total) {
    const size_t full_bytes = projections / 8;
    const size_t tail_bits = projections % 8;
    int sign_dot =
        static_cast<int>(processed_bytes * 8) - 2 * static_cast<int>(diff_count_total);

    for (size_t idx = processed_bytes; idx < full_bytes; ++idx) {
        const int diff_count =
            __builtin_popcount(static_cast<unsigned int>(lhs[idx] ^ rhs[idx]));
        sign_dot += 8 - (2 * diff_count);
    }

    if (tail_bits != 0) {
        const uint8_t valid_mask = static_cast<uint8_t>((uint16_t{1} << tail_bits) - 1u);
        const uint8_t diff_bits = static_cast<uint8_t>((lhs[full_bytes] ^ rhs[full_bytes]) &
                                                       valid_mask);
        const int diff_count = __builtin_popcount(static_cast<unsigned int>(diff_bits));
        sign_dot += static_cast<int>(tail_bits) - (2 * diff_count);
    }

    return sign_dot;
}

template <unsigned char residual>
int TQ_PackedResidualSignDotSIMD16_NEON(const uint8_t *lhs, const uint8_t *rhs,
                                        size_t projections) {
    const size_t full_bytes = projections / 8;
    const size_t simd_end = full_bytes - residual;
    uint32_t diff_count_total = 0;

    size_t idx = 0;
    for (; idx < simd_end; idx += 16) {
        const uint8x16_t diff = veorq_u8(vld1q_u8(lhs + idx), vld1q_u8(rhs + idx));
        const uint8x16_t bit_counts = vcntq_u8(diff);
        const uint16x8_t partial16 = vpaddlq_u8(bit_counts);
        const uint32x4_t partial32 = vpaddlq_u16(partial16);
        diff_count_total += vaddvq_u32(partial32);
    }

    return FinishPackedResidualSignDot(lhs, rhs, projections, idx, diff_count_total);
}

template <unsigned char residual>
float TQ_SymmetricPolarSIMD8_NEON(const float *lhs_radii, const uint8_t *lhs_angles,
                                  const float *rhs_radii, const uint8_t *rhs_angles,
                                  const float *delta_cos_lut, uint8_t angle_delta_mask,
                                  size_t pairs) {
    const size_t simd_end = pairs - residual;
    const uint8x8_t mask_vec = vdup_n_u8(angle_delta_mask);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    alignas(16) uint8_t deltas[8];
    alignas(16) float delta_cos_values[8];

    size_t idx = 0;
    for (; idx < simd_end; idx += 8) {
        const uint8x8_t lhs_vec = vld1_u8(lhs_angles + idx);
        const uint8x8_t rhs_vec = vld1_u8(rhs_angles + idx);
        const uint8x8_t delta_vec = vand_u8(vsub_u8(lhs_vec, rhs_vec), mask_vec);
        vst1_u8(deltas, delta_vec);
        for (size_t lane = 0; lane < 8; ++lane) {
            delta_cos_values[lane] = delta_cos_lut[deltas[lane]];
        }

        const float32x4_t lhs_radii_0 = vld1q_f32(lhs_radii + idx);
        const float32x4_t rhs_radii_0 = vld1q_f32(rhs_radii + idx);
        const float32x4_t delta_cos_0 = vld1q_f32(delta_cos_values);
        acc0 = vmlaq_f32(acc0, vmulq_f32(lhs_radii_0, rhs_radii_0), delta_cos_0);

        const float32x4_t lhs_radii_1 = vld1q_f32(lhs_radii + idx + 4);
        const float32x4_t rhs_radii_1 = vld1q_f32(rhs_radii + idx + 4);
        const float32x4_t delta_cos_1 = vld1q_f32(delta_cos_values + 4);
        acc1 = vmlaq_f32(acc1, vmulq_f32(lhs_radii_1, rhs_radii_1), delta_cos_1);
    }

    float sum = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    for (; idx < pairs; ++idx) {
        const size_t delta =
            (static_cast<size_t>(lhs_angles[idx]) - static_cast<size_t>(rhs_angles[idx])) &
            angle_delta_mask;
        sum += lhs_radii[idx] * rhs_radii[idx] * delta_cos_lut[delta];
    }

    return sum;
}

} // namespace

float TQ_SymmetricPolarEstimate_NEON(const float *lhs_radii, const void *lhs_angles,
                                     const float *rhs_radii, const void *rhs_angles, size_t pairs,
                                     size_t angle_delta_mask, const float *delta_cos_lut,
                                     bool nibble_angle_codes, bool compact_angle_codes) {
    if (!compact_angle_codes || nibble_angle_codes) {
        return TQ_SymmetricPolarEstimate(lhs_radii, lhs_angles, rhs_radii, rhs_angles, pairs,
                                         angle_delta_mask, delta_cos_lut, nibble_angle_codes,
                                         compact_angle_codes);
    }

    const auto *lhs_angle_bytes = static_cast<const uint8_t *>(lhs_angles);
    const auto *rhs_angle_bytes = static_cast<const uint8_t *>(rhs_angles);
    const uint8x8_t mask_vec = vdup_n_u8(static_cast<uint8_t>(angle_delta_mask));
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    alignas(16) uint8_t deltas[8];
    alignas(16) float delta_cos_values[8];
    size_t idx = 0;

    for (; idx + 8 <= pairs; idx += 8) {
        const uint8x8_t lhs_vec = vld1_u8(lhs_angle_bytes + idx);
        const uint8x8_t rhs_vec = vld1_u8(rhs_angle_bytes + idx);
        const uint8x8_t delta_vec = vand_u8(vsub_u8(lhs_vec, rhs_vec), mask_vec);
        vst1_u8(deltas, delta_vec);

        for (size_t lane = 0; lane < 8; ++lane) {
            delta_cos_values[lane] = delta_cos_lut[deltas[lane]];
        }

        const float32x4_t lhs_radii_0 = vld1q_f32(lhs_radii + idx);
        const float32x4_t rhs_radii_0 = vld1q_f32(rhs_radii + idx);
        const float32x4_t delta_cos_0 = vld1q_f32(delta_cos_values);
        acc0 = vmlaq_f32(acc0, vmulq_f32(lhs_radii_0, rhs_radii_0), delta_cos_0);

        const float32x4_t lhs_radii_1 = vld1q_f32(lhs_radii + idx + 4);
        const float32x4_t rhs_radii_1 = vld1q_f32(rhs_radii + idx + 4);
        const float32x4_t delta_cos_1 = vld1q_f32(delta_cos_values + 4);
        acc1 = vmlaq_f32(acc1, vmulq_f32(lhs_radii_1, rhs_radii_1), delta_cos_1);
    }

    float polar_estimate = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    if (idx < pairs) {
        polar_estimate += TQ_SymmetricPolarEstimate(lhs_radii + idx, lhs_angle_bytes + idx,
                                                    rhs_radii + idx, rhs_angle_bytes + idx,
                                                    pairs - idx, angle_delta_mask, delta_cos_lut,
                                                    false, true);
    }

    return polar_estimate;
}

int TQ_PackedSignDot_NEON(const uint8_t *lhs, const uint8_t *rhs, size_t projections) {
    const size_t full_bytes = projections / 8;
    const size_t tail_bits = projections % 8;
    int sign_dot = 0;

    size_t idx = 0;
    uint32_t diff_count_total = 0;
    for (; idx + 16 <= full_bytes; idx += 16) {
        const uint8x16_t diff = veorq_u8(vld1q_u8(lhs + idx), vld1q_u8(rhs + idx));
        const uint8x16_t bit_counts = vcntq_u8(diff);
        const uint16x8_t partial16 = vpaddlq_u8(bit_counts);
        const uint32x4_t partial32 = vpaddlq_u16(partial16);
        diff_count_total += vaddvq_u32(partial32);
    }
    sign_dot += static_cast<int>(idx * 8) - 2 * static_cast<int>(diff_count_total);

    if (idx < full_bytes) {
        sign_dot += TQ_PackedSignDot(lhs + idx, rhs + idx, (full_bytes - idx) * 8);
    }
    if (tail_bits != 0) {
        sign_dot += TQ_PackedSignDot(lhs + full_bytes, rhs + full_bytes, tail_bits);
    }

    return sign_dot;
}

tq_symmetric_polar_estimate_func_t Choose_TQ_SymmetricPolarEstimate_implementation_NEON(void) {
    return TQ_SymmetricPolarEstimate_NEON;
}

tq_packed_sign_dot_func_t Choose_TQ_PackedSignDot_implementation_NEON(void) {
    return TQ_PackedSignDot_NEON;
}


#include "implementation_chooser.h"

tq_packed_residual_sign_dot_func_t
Choose_TQ_PackedResidualSignDot_implementation_NEON(size_t projections) {
    const size_t full_bytes = projections / 8;
    tq_packed_residual_sign_dot_func_t ret_func;
    CHOOSE_IMPLEMENTATION(ret_func, full_bytes, 16, TQ_PackedResidualSignDotSIMD16_NEON);
    return ret_func;
}

tq_symmetric_polar_func_t Choose_TQ_SymmetricPolar_implementation_NEON(size_t pairs) {
    tq_symmetric_polar_func_t ret_func;
    CHOOSE_IMPLEMENTATION(ret_func, pairs, 8, TQ_SymmetricPolarSIMD8_NEON);
    return ret_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
