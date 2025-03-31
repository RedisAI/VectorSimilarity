/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

static inline void L2SquareStep(int8_t *&pVect1, int8_t *&pVect2, int32x4_t &sum) {
    // Load 16 int8 elements (16 bytes) into NEON registers
    int8x16_t v1 = vld1q_s8(pVect1);
    int8x16_t v2 = vld1q_s8(pVect2);

    // Split into low and high halves
    int8x8_t v1_low = vget_low_s8(v1);
    int8x8_t v1_high = vget_high_s8(v1);
    int8x8_t v2_low = vget_low_s8(v2);
    int8x8_t v2_high = vget_high_s8(v2);

    // Widen to 16-bit before subtraction
    int16x8_t v1_low_wide = vmovl_s8(v1_low);
    int16x8_t v1_high_wide = vmovl_s8(v1_high);
    int16x8_t v2_low_wide = vmovl_s8(v2_low);
    int16x8_t v2_high_wide = vmovl_s8(v2_high);

    // Calculate difference on widened values
    int16x8_t diff_low = vsubq_s16(v1_low_wide, v2_low_wide);
    int16x8_t diff_high = vsubq_s16(v1_high_wide, v2_high_wide);

    // Further widen differences to 32-bit for safer squaring
    int32x4_t diff_low_0 = vmovl_s16(vget_low_s16(diff_low));
    int32x4_t diff_low_1 = vmovl_s16(vget_high_s16(diff_low));
    int32x4_t diff_high_0 = vmovl_s16(vget_low_s16(diff_high));
    int32x4_t diff_high_1 = vmovl_s16(vget_high_s16(diff_high));

    // Square differences in 32-bit
    int32x4_t square_low_0 = vmulq_s32(diff_low_0, diff_low_0);
    int32x4_t square_low_1 = vmulq_s32(diff_low_1, diff_low_1);
    int32x4_t square_high_0 = vmulq_s32(diff_high_0, diff_high_0);
    int32x4_t square_high_1 = vmulq_s32(diff_high_1, diff_high_1);

    // Accumulate into 32-bit sum
    sum = vaddq_s32(sum, square_low_0);
    sum = vaddq_s32(sum, square_low_1);
    sum = vaddq_s32(sum, square_high_0);
    sum = vaddq_s32(sum, square_high_1);

    pVect1 += 16;
    pVect2 += 16;
}

static inline void L2SquareStep_32(int8_t *&pVect1, int8_t *&pVect2, int32x4_t &sum0,
                                   int32x4_t &sum1) {
    // Load 32 int8 elements (32 bytes) at once
    int8x16x2_t v1 = vld1q_s8_x2(pVect1);
    int8x16x2_t v2 = vld1q_s8_x2(pVect2);

    // Process first chunk
    int8x8_t v1_low_0 = vget_low_s8(v1.val[0]);
    int8x8_t v1_high_0 = vget_high_s8(v1.val[0]);
    int8x8_t v2_low_0 = vget_low_s8(v2.val[0]);
    int8x8_t v2_high_0 = vget_high_s8(v2.val[0]);

    int16x8_t v1_low_wide_0 = vmovl_s8(v1_low_0);
    int16x8_t v1_high_wide_0 = vmovl_s8(v1_high_0);
    int16x8_t v2_low_wide_0 = vmovl_s8(v2_low_0);
    int16x8_t v2_high_wide_0 = vmovl_s8(v2_high_0);

    int16x8_t diff_low_0 = vsubq_s16(v1_low_wide_0, v2_low_wide_0);
    int16x8_t diff_high_0 = vsubq_s16(v1_high_wide_0, v2_high_wide_0);

    int32x4_t diff_low_00 = vmovl_s16(vget_low_s16(diff_low_0));
    int32x4_t diff_low_01 = vmovl_s16(vget_high_s16(diff_low_0));
    int32x4_t diff_high_00 = vmovl_s16(vget_low_s16(diff_high_0));
    int32x4_t diff_high_01 = vmovl_s16(vget_high_s16(diff_high_0));

    int32x4_t square_low_00 = vmulq_s32(diff_low_00, diff_low_00);
    int32x4_t square_low_01 = vmulq_s32(diff_low_01, diff_low_01);
    int32x4_t square_high_00 = vmulq_s32(diff_high_00, diff_high_00);
    int32x4_t square_high_01 = vmulq_s32(diff_high_01, diff_high_01);

    sum0 = vaddq_s32(sum0, square_low_00);
    sum0 = vaddq_s32(sum0, square_low_01);
    sum0 = vaddq_s32(sum0, square_high_00);
    sum0 = vaddq_s32(sum0, square_high_01);

    // Process second chunk
    int8x8_t v1_low_1 = vget_low_s8(v1.val[1]);
    int8x8_t v1_high_1 = vget_high_s8(v1.val[1]);
    int8x8_t v2_low_1 = vget_low_s8(v2.val[1]);
    int8x8_t v2_high_1 = vget_high_s8(v2.val[1]);

    int16x8_t v1_low_wide_1 = vmovl_s8(v1_low_1);
    int16x8_t v1_high_wide_1 = vmovl_s8(v1_high_1);
    int16x8_t v2_low_wide_1 = vmovl_s8(v2_low_1);
    int16x8_t v2_high_wide_1 = vmovl_s8(v2_high_1);

    int16x8_t diff_low_1 = vsubq_s16(v1_low_wide_1, v2_low_wide_1);
    int16x8_t diff_high_1 = vsubq_s16(v1_high_wide_1, v2_high_wide_1);

    int32x4_t diff_low_10 = vmovl_s16(vget_low_s16(diff_low_1));
    int32x4_t diff_low_11 = vmovl_s16(vget_high_s16(diff_low_1));
    int32x4_t diff_high_10 = vmovl_s16(vget_low_s16(diff_high_1));
    int32x4_t diff_high_11 = vmovl_s16(vget_high_s16(diff_high_1));

    int32x4_t square_low_10 = vmulq_s32(diff_low_10, diff_low_10);
    int32x4_t square_low_11 = vmulq_s32(diff_low_11, diff_low_11);
    int32x4_t square_high_10 = vmulq_s32(diff_high_10, diff_high_10);
    int32x4_t square_high_11 = vmulq_s32(diff_high_11, diff_high_11);

    sum1 = vaddq_s32(sum1, square_low_10);
    sum1 = vaddq_s32(sum1, square_low_11);
    sum1 = vaddq_s32(sum1, square_high_10);
    sum1 = vaddq_s32(sum1, square_high_11);

    pVect1 += 32;
    pVect2 += 32;
}

template <unsigned char residual> // 0..63
float INT8_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;

    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);

    // Process 64 elements at a time in the main loop
    size_t num_of_chunks = dimension / 64;

    for (size_t i = 0; i < num_of_chunks; i++) {
        L2SquareStep_32(pVect1, pVect2, sum0, sum1);
        L2SquareStep_32(pVect1, pVect2, sum2, sum3);
    }

    constexpr size_t num_of_32_chunks = residual / 32;

    if constexpr (num_of_32_chunks) {
        L2SquareStep_32(pVect1, pVect2, sum0, sum1);
    }

    constexpr size_t residual_chunks = (residual % 32) / 16;
    if constexpr (residual_chunks >= 1) {
        L2SquareStep(pVect1, pVect2, sum2);
    }
    if constexpr (residual_chunks >= 2) {
        L2SquareStep(pVect1, pVect2, sum3);
    }

    // Then handle the final 0-15 elements
    constexpr size_t final_residual = (residual % 32) % 16;
    if constexpr (final_residual > 0) {
        // Create temporary arrays with padding
        uint8x16_t indices =
            vcombine_u8(vcreate_u8(0x0706050403020100ULL), vcreate_u8(0x0F0E0D0C0B0A0908ULL));

        // Create threshold vector with all elements = final_residual
        uint8x16_t threshold = vdupq_n_u8(final_residual);

        // Create mask: indices < final_residual ? 0xFF : 0x00
        uint8x16_t mask = vcltq_u8(indices, threshold);

        // Load data directly from input vectors
        int8x16_t v1 = vld1q_s8(pVect1);
        int8x16_t v2 = vld1q_s8(pVect2);

        // Apply mask to zero out irrelevant elements
        v1 = vandq_s8(v1, vreinterpretq_s8_u8(mask));
        v2 = vandq_s8(v2, vreinterpretq_s8_u8(mask));
        // Split into low and high halves
        int8x8_t v1_low = vget_low_s8(v1);
        int8x8_t v1_high = vget_high_s8(v1);
        int8x8_t v2_low = vget_low_s8(v2);
        int8x8_t v2_high = vget_high_s8(v2);

        // Widen to 16-bit before subtraction
        int16x8_t v1_low_wide = vmovl_s8(v1_low);
        int16x8_t v1_high_wide = vmovl_s8(v1_high);
        int16x8_t v2_low_wide = vmovl_s8(v2_low);
        int16x8_t v2_high_wide = vmovl_s8(v2_high);

        // Calculate difference on widened values
        int16x8_t diff_low = vsubq_s16(v1_low_wide, v2_low_wide);
        int16x8_t diff_high = vsubq_s16(v1_high_wide, v2_high_wide);

        // Further widen differences to 32-bit for safer squaring
        int32x4_t diff_low_0 = vmovl_s16(vget_low_s16(diff_low));
        int32x4_t diff_low_1 = vmovl_s16(vget_high_s16(diff_low));
        int32x4_t diff_high_0 = vmovl_s16(vget_low_s16(diff_high));
        int32x4_t diff_high_1 = vmovl_s16(vget_high_s16(diff_high));

        // Square differences in 32-bit
        int32x4_t square_low_0 = vmulq_s32(diff_low_0, diff_low_0);
        int32x4_t square_low_1 = vmulq_s32(diff_low_1, diff_low_1);
        int32x4_t square_high_0 = vmulq_s32(diff_high_0, diff_high_0);
        int32x4_t square_high_1 = vmulq_s32(diff_high_1, diff_high_1);

        // Accumulate into 32-bit sum
        sum0 = vaddq_s32(sum0, square_low_0);
        sum1 = vaddq_s32(sum1, square_low_1);

        if constexpr (final_residual > 8) {
            sum2 = vaddq_s32(sum2, square_high_0);
            sum3 = vaddq_s32(sum3, square_high_1);
        }
    }

    // Horizontal sum of the 4 elements in the sum register to get final result
    int32x4_t total_sum = vaddq_s32(sum0, sum1);

    total_sum = vaddq_s32(total_sum, sum2);
    total_sum = vaddq_s32(total_sum, sum3);

    // Horizontal sum of the 4 elements in the combined sum register
    int32_t result = vaddvq_s32(total_sum);

    // Return the L2 squared distance as a float
    return static_cast<float>(result);
}
