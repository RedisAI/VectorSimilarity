/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

static inline void L2SquareStep(uint8_t *&pVect1, uint8_t *&pVect2, int32x4_t &sum) {
    // Load 16 uint8 elements (16 bytes) into NEON registers
    uint8x16_t v1 = vld1q_u8(pVect1);
    uint8x16_t v2 = vld1q_u8(pVect2);

    // Split into low and high halves
    uint8x8_t v1_low = vget_low_u8(v1);
    uint8x8_t v1_high = vget_high_u8(v1);
    uint8x8_t v2_low = vget_low_u8(v2);
    uint8x8_t v2_high = vget_high_u8(v2);

    // Widen to 16-bit before subtraction
    uint16x8_t v1_low_wide = vmovl_u8(v1_low);
    uint16x8_t v1_high_wide = vmovl_u8(v1_high);
    uint16x8_t v2_low_wide = vmovl_u8(v2_low);
    uint16x8_t v2_high_wide = vmovl_u8(v2_high);

    // Calculate difference on widened values
    uint16x8_t diff_low = vsubq_u16(v1_low_wide, v2_low_wide);
    uint16x8_t diff_high = vsubq_u16(v1_high_wide, v2_high_wide);

    // Further widen differences to 32-bit for safer squaring
    uint32x4_t diff_low_0 = vmovl_u16(vget_low_u16(diff_low));
    uint32x4_t diff_low_1 = vmovl_u16(vget_high_u16(diff_low));
    uint32x4_t diff_high_0 = vmovl_u16(vget_low_u16(diff_high));
    uint32x4_t diff_high_1 = vmovl_u16(vget_high_u16(diff_high));

    // Square differences in 32-bit
    int32x4_t square_low_0 = vreinterpretq_s32_u32(vmulq_u32(diff_low_0, diff_low_0));
    int32x4_t square_low_1 = vreinterpretq_s32_u32(vmulq_u32(diff_low_1, diff_low_1));
    int32x4_t square_high_0 = vreinterpretq_s32_u32(vmulq_u32(diff_high_0, diff_high_0));
    int32x4_t square_high_1 = vreinterpretq_s32_u32(vmulq_u32(diff_high_1, diff_high_1));

    // Accumulate into 32-bit sum
    sum = vaddq_s32(sum, square_low_0);
    sum = vaddq_s32(sum, square_low_1);
    sum = vaddq_s32(sum, square_high_0);
    sum = vaddq_s32(sum, square_high_1);

    pVect1 += 16;
    pVect2 += 16;
}

template <unsigned char residual> // 0..63
float UINT8_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    uint8_t *pVect1 = (uint8_t *)pVect1v;
    uint8_t *pVect2 = (uint8_t *)pVect2v;

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
        uint8x16_t v1 = vld1q_u8(pVect1);
        uint8x16_t v2 = vld1q_u8(pVect2);

        // Apply mask to zero out irrelevant elements
        v1 = vandq_u8(v1, mask);
        v2 = vandq_u8(v2, mask);
        // Split into low and high halves
        uint8x8_t v1_low = vget_low_u8(v1);
        uint8x8_t v1_high = vget_high_u8(v1);
        uint8x8_t v2_low = vget_low_u8(v2);
        uint8x8_t v2_high = vget_high_u8(v2);

        // Widen to 16-bit before subtraction
        uint16x8_t v1_low_wide = vmovl_u8(v1_low);
        uint16x8_t v1_high_wide = vmovl_u8(v1_high);
        uint16x8_t v2_low_wide = vmovl_u8(v2_low);
        uint16x8_t v2_high_wide = vmovl_u8(v2_high);
        uint16x8_t max_low = vmaxq_u16(v1_low_wide, v2_low_wide);
        uint16x8_t min_low = vminq_u16(v1_low_wide, v2_low_wide);
        uint16x8_t abs_diff_low = vsubq_u16(max_low, min_low);
        
        uint16x8_t max_high = vmaxq_u16(v1_high_wide, v2_high_wide);
        uint16x8_t min_high = vminq_u16(v1_high_wide, v2_high_wide);
        uint16x8_t abs_diff_high = vsubq_u16(max_high, min_high);
        
        // Convert to signed for further processing
        int16x8_t diff_low = vreinterpretq_s16_u16(abs_diff_low);
        int16x8_t diff_high = vreinterpretq_s16_u16(abs_diff_high);

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
