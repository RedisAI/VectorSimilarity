/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>
#include <iostream>

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

template <unsigned char residual> // 0..63
float INT8_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *start = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;

    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);
    // Process 64 elements at a time in the main loop
    size_t num_of_chunks = dimension / 64;

    for (size_t i = 0; i < num_of_chunks; i++) {
        L2SquareStep(pVect1, pVect2, sum0);
        L2SquareStep(pVect1, pVect2, sum1);
        L2SquareStep(pVect1, pVect2, sum2);
        L2SquareStep(pVect1, pVect2, sum3);
    }

    // Handle residual elements (0-63)
    // First, process full chunks of 16 elements
    constexpr size_t residual_chunks = residual / 16;

    if constexpr (residual_chunks > 0) {

        if constexpr (residual_chunks >= 1) {
            L2SquareStep(pVect1, pVect2, sum0);
        }
        if constexpr (residual_chunks >= 2) {
            L2SquareStep(pVect1, pVect2, sum1);
        }
        if constexpr (residual_chunks >= 3) {
            L2SquareStep(pVect1, pVect2, sum2);
        }
    }

    // Then handle the final 0-15 elements
    constexpr size_t final_residual = residual % 16;
    if constexpr (final_residual > 0) {
        // Create temporary arrays with padding
    uint8x16_t indices = vcombine_u8(
        vcreate_u8(0x0706050403020100ULL), 
        vcreate_u8(0x0F0E0D0C0B0A0908ULL)
    );
    
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
        sum1 = vaddq_s32(sum1, square_low_0);
        sum1 = vaddq_s32(sum1, square_low_1);

        if constexpr (final_residual > 8) {
            sum1 = vaddq_s32(sum1, square_high_0);
            sum1 = vaddq_s32(sum1, square_high_1);
        }
    }

    int32x4_t paired_sum = vpaddq_s32(vpaddq_s32(sum0, sum1), vpaddq_s32(sum2, sum3));
    int32_t result = vgetq_lane_s32(paired_sum, 0);

    // Return the L2 squared distance as a float
    return static_cast<float>(result);
}
