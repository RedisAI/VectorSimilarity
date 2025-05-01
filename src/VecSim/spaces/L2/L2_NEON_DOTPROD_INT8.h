/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/
#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

__attribute__((always_inline)) static inline void L2SquareOp(const int8x16_t &v1,
                                                             const int8x16_t &v2, uint32x4_t &sum) {
    // Explicitly reinterpret the int8 vectors as uint8 for vabdq_u8

    // Compute absolute differences (results in uint8x16_t)
    int8x16_t diff = vabdq_s8(v1, v2);

    // Reinterpret back to int8x16_t for vdotq_s32
    uint8x16_t diff_s8 = vreinterpretq_u8_s8(diff);

    // Use dot product to square and accumulate (diff·diff)
    sum = vdotq_u32(sum, diff_s8, diff_s8);
}

__attribute__((always_inline)) static inline void L2SquareStep16(int8_t *&pVect1, int8_t *&pVect2,
                                                                 uint32x4_t &sum) {
    // Load 16 int8 elements (16 bytes) into NEON registers
    int8x16_t v1 = vld1q_s8(pVect1);
    int8x16_t v2 = vld1q_s8(pVect2);

    L2SquareOp(v1, v2, sum);

    pVect1 += 16;
    pVect2 += 16;
}

static inline void L2SquareStep32(int8_t *&pVect1, int8_t *&pVect2, uint32x4_t &sum0,
                                  uint32x4_t &sum1) {
    // Load 32 int8 elements (32 bytes) at once
    int8x16x2_t v1 = vld1q_s8_x2(pVect1);
    int8x16x2_t v2 = vld1q_s8_x2(pVect2);

    auto v1_0 = v1.val[0];
    auto v2_0 = v2.val[0];
    L2SquareOp(v1_0, v2_0, sum0);

    auto v1_1 = v1.val[1];
    auto v2_1 = v2.val[1];
    L2SquareOp(v1_1, v2_1, sum1);

    pVect1 += 32;
    pVect2 += 32;
}

template <unsigned char residual> // 0..63
float INT8_L2SqrSIMD16_NEON_DOTPROD(const void *pVect1v, const void *pVect2v, size_t dimension) {
    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;

    uint32x4_t sum0 = vdupq_n_u32(0);
    uint32x4_t sum1 = vdupq_n_u32(0);
    uint32x4_t sum2 = vdupq_n_u32(0);
    uint32x4_t sum3 = vdupq_n_u32(0);

    constexpr size_t final_residual = residual % 16;
    if constexpr (final_residual > 0) {
        // Define a compile-time constant mask based on final_residual
        constexpr uint8x16_t mask = {
            0xFF,
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
            0,
        };

        // Load data directly from input vectors
        int8x16_t v1 = vld1q_s8(pVect1);
        int8x16_t v2 = vld1q_s8(pVect2);

        // Zero vector for replacement
        int8x16_t zeros = vdupq_n_s8(0);

        // Apply bit select to zero out irrelevant elements
        v1 = vbslq_s8(mask, v1, zeros);
        v2 = vbslq_s8(mask, v2, zeros);
        L2SquareOp(v1, v2, sum0);
        pVect1 += final_residual;
        pVect2 += final_residual;
    }

    // Process 64 elements at a time in the main loop
    size_t num_of_chunks = dimension / 64;

    for (size_t i = 0; i < num_of_chunks; i++) {
        L2SquareStep32(pVect1, pVect2, sum0, sum1);
        L2SquareStep32(pVect1, pVect2, sum2, sum3);
    }

    constexpr size_t num_of_32_chunks = residual / 32;
    if constexpr (num_of_32_chunks) {
        L2SquareStep32(pVect1, pVect2, sum0, sum1);
    }

    constexpr size_t residual_chunks = (residual % 32) / 16;
    if constexpr (residual_chunks > 0) {
        L2SquareStep16(pVect1, pVect2, sum2);
    }

    // Horizontal sum of the 4 elements in the sum register to get final result
    uint32x4_t total_sum = vaddq_u32(sum0, sum1);

    total_sum = vaddq_u32(total_sum, sum2);
    total_sum = vaddq_u32(total_sum, sum3);

    // Horizontal sum of the 4 elements in the combined sum register
    uint32_t result = vaddvq_u32(total_sum);

    // Return the L2 squared distance as a float
    return static_cast<float>(result);
}
