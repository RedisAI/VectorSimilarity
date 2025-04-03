/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

__attribute__((always_inline)) static inline void
L2SquareOp(const uint8x16_t &v1_first, const uint8x16_t &v2_first, int32x4_t &sum) {
    // Split into low and high 8-bit halves
    uint8x8_t v1_low = vget_low_u8(v1_first);
    uint8x8_t v1_high = vget_high_u8(v1_first);
    uint8x8_t v2_low = vget_low_u8(v2_first);
    uint8x8_t v2_high = vget_high_u8(v2_first);

    // Compute absolute differences and widen to 16-bit in one step
    uint16x8_t diff_low_u = vabdl_u8(v1_low, v2_low);
    uint16x8_t diff_high_u = vabdl_u8(v1_high, v2_high);

    // Reinterpret as signed for compatibility with the rest of the code
    int16x8_t diff_low = vreinterpretq_s16_u16(diff_low_u);
    int16x8_t diff_high = vreinterpretq_s16_u16(diff_high_u);

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

    // Accumulate all results into sum
    sum = vaddq_s32(sum, square_low_0);
    sum = vaddq_s32(sum, square_low_1);
    sum = vaddq_s32(sum, square_high_0);
    sum = vaddq_s32(sum, square_high_1);
}

__attribute__((always_inline)) static inline void L2SquareStep16(uint8_t *&pVect1, uint8_t *&pVect2,
                                                                 int32x4_t &sum) {
    // Load 16 uint8 elements into NEON registers
    uint8x16_t v1 = vld1q_u8(pVect1);
    uint8x16_t v2 = vld1q_u8(pVect2);

    L2SquareOp(v1, v2, sum);

    pVect1 += 16;
    pVect2 += 16;
}

__attribute__((always_inline)) static inline void L2SquareStep32(uint8_t *&pVect1, uint8_t *&pVect2,
                                                                 int32x4_t &sum1, int32x4_t &sum2) {
    uint8x16x2_t v1_pair = vld1q_u8_x2(pVect1);
    uint8x16x2_t v2_pair = vld1q_u8_x2(pVect2);

    // Reference the individual vectors
    uint8x16_t v1_first = v1_pair.val[0];
    uint8x16_t v1_second = v1_pair.val[1];
    uint8x16_t v2_first = v2_pair.val[0];
    uint8x16_t v2_second = v2_pair.val[1];

    L2SquareOp(v1_first, v2_first, sum1);
    L2SquareOp(v1_second, v2_second, sum2);

    pVect1 += 32;
    pVect2 += 32;
}

template <unsigned char residual> // 0..63
float UINT8_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    uint8_t *pVect1 = (uint8_t *)pVect1v;
    uint8_t *pVect2 = (uint8_t *)pVect2v;

    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);

    constexpr size_t final_residual = residual % 16;
    if constexpr (final_residual > 0) {
        constexpr uint8x16_t mask = {0xFF,
                                     (final_residual >= 2) ? 0xFF : 0,
                                     (final_residual >= 3) ? 0xFF : 0,
                                     (final_residual >= 4) ? 0xFF : 0,
                                     (final_residual >= 5) ? 0xFF : 0,
                                     (final_residual >= 6) ? 0xFF : 0,
                                     (final_residual >= 7) ? 0xFF : 0,
                                     (final_residual >= 8) ? 0xFF : 0,
                                     (final_residual >= 9) ? 0xFF : 0,
                                     (final_residual >= 10) ? 0xFF : 0,
                                     (final_residual >= 11) ? 0xFF : 0,
                                     (final_residual >= 12) ? 0xFF : 0,
                                     (final_residual >= 13) ? 0xFF : 0,
                                     (final_residual >= 14) ? 0xFF : 0,
                                     (final_residual >= 15) ? 0xFF : 0,
                                     0};

        // Load data directly from input vectors
        uint8x16_t v1 = vld1q_u8(pVect1);
        uint8x16_t v2 = vld1q_u8(pVect2);

        // Apply mask to zero out irrelevant elements
        v1 = vandq_u8(v1, mask);
        v2 = vandq_u8(v2, mask);
        L2SquareOp(v1, v2, sum1);
        pVect1 += final_residual;
        pVect2 += final_residual;
    }

    // Process 64 elements at a time in the main loop
    size_t num_of_chunks = dimension / 64;

    for (size_t i = 0; i < num_of_chunks; i++) {
        L2SquareStep32(pVect1, pVect2, sum0, sum2);
        L2SquareStep32(pVect1, pVect2, sum1, sum3);
    }

    constexpr size_t num_of_32_chunks = residual / 32;

    if constexpr (num_of_32_chunks) {
        L2SquareStep32(pVect1, pVect2, sum0, sum1);
    }

    // Handle residual elements (0-63)
    // First, process full chunks of 16 elements
    constexpr size_t residual_chunks = (residual % 32) / 16;
    if constexpr (residual_chunks >= 1) {
        L2SquareStep16(pVect1, pVect2, sum0);
    }
    if constexpr (residual_chunks >= 2) {
        L2SquareStep16(pVect1, pVect2, sum1);
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
