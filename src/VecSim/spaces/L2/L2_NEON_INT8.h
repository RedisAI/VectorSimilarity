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

    // Square differences
    int16x8_t square_low = vmulq_s16(diff_low, diff_low);
    int16x8_t square_high = vmulq_s16(diff_high, diff_high);

    // Accumulate into 32-bit sum
    sum = vpadalq_s16(sum, square_low);
    sum = vpadalq_s16(sum, square_high);

    pVect1 += 16;
    pVect2 += 16;
}

template <unsigned char residual> // 0..63
float INT8_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *start = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;

    int32x4_t sum = vdupq_n_s32(0);
    // print tht vector
    for (size_t i = 0; i < dimension; i++) {
        std::cout << (int)pVect1[i] << " ";
    }
    std::cout << "\n";
    for (size_t i = 0; i < dimension; i++) {
        std::cout << (int)pVect2[i] << " ";
    }
    std::cout << "\n";
    // Process 64 elements at a time in the main loop
    size_t num_of_chunks = dimension / 64;

    for (size_t i = 0; i < num_of_chunks; i++) {
        L2SquareStep(pVect1, pVect2, sum);
        L2SquareStep(pVect1, pVect2, sum);
        L2SquareStep(pVect1, pVect2, sum);
        L2SquareStep(pVect1, pVect2, sum);
    }
    std::cout << "AFTER loop first " << pVect1 - start << "\n";
    // Handle residual elements (0-63)
    // First, process full chunks of 16 elements
    constexpr size_t residual_chunks = residual / 16;

    if constexpr (residual_chunks > 0) {
        for (size_t i = 0; i < residual_chunks; i++) {
            L2SquareStep(pVect1, pVect2, sum);
            std::cout << "sum: " << vaddvq_s32(sum) << "\n";
        }
    }
    std::cout << "AFTER second first " << pVect1 - start << "\n";

    // Then handle the final 0-15 elements
    constexpr size_t final_residual = residual % 16;
    if constexpr (final_residual > 0) {
        // Create temporary arrays with padding
        int8_t temp1[16] = {0};
        int8_t temp2[16] = {0};

        for (size_t i = 0; i < final_residual; i++) {
            temp1[i] = pVect1[i];
            temp2[i] = pVect2[i];
        }

        // Process with NEON SIMD
        int8x16_t v1 = vld1q_s8(temp1);
        int8x16_t v2 = vld1q_s8(temp2);

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

        // Square differences
        int16x8_t square_low = vmulq_s16(diff_low, diff_low);
        int16x8_t square_high = vmulq_s16(diff_high, diff_high);

        // Accumulate into 32-bit sum
        sum = vpadalq_s16(sum, square_low);
        if constexpr (final_residual > 8) {
            sum = vpadalq_s16(sum, square_high);
        }
    }
    std::cout << "AFTER res first " << pVect1 - start << "\n";

    std::cout << "num_of_chunks: " << (int)num_of_chunks << "\n";
    std::cout << "residual_chunks: " << (int)residual_chunks << "\n";
    std::cout << "final_residual: " << (int)final_residual << "\n";

    // Horizontal sum of the 4 elements in the sum register to get final result
    int32_t result = vaddvq_s32(sum);

    // Return the L2 squared distance as a float
    return static_cast<float>(result);
}
