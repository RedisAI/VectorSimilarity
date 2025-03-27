/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

static inline void L2SquareStep(float *&pVect1, float *&pVect2, float32x4_t &sum) {
    float32x4_t v1 = vld1q_f32(pVect1);
    float32x4_t v2 = vld1q_f32(pVect2);

    float32x4_t diff = vsubq_f32(v1, v2);

    sum = vmlaq_f32(sum, diff, diff);

    pVect1 += 4;
    pVect2 += 4;
}

template <unsigned char residual> // 0..15
float FP32_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    float32x4_t sum_squares = vdupq_n_f32(0.0f);

    const size_t main_blocks = (dimension - residual) / 16;

    for (size_t i = 0; i < main_blocks; i++) {
        L2SquareStep(pVect1, pVect2, sum_squares);
        L2SquareStep(pVect1, pVect2, sum_squares);
        L2SquareStep(pVect1, pVect2, sum_squares);
        L2SquareStep(pVect1, pVect2, sum_squares);
    }

    // Handle remaining complete 4-float blocks within residual
    constexpr size_t remaining_quads = residual / 4;
    if constexpr (remaining_quads > 0) {
        for (size_t i = 0; i < remaining_quads; i++) {
            L2SquareStep(pVect1, pVect2, sum_squares);
        }
    }

    // Handle final residual elements (0-3 elements)
    constexpr size_t final_residual = residual % 4;
    if constexpr (final_residual > 0) {
        float32x4_t v1 = vdupq_n_f32(0.0f);
        float32x4_t v2 = vdupq_n_f32(0.0f);

        // loads the elements to the corresponding lane on the vector
        if constexpr (final_residual >= 1) {
            v1 = vld1q_lane_f32(pVect1, v1, 0);
            v2 = vld1q_lane_f32(pVect2, v2, 0);
        }
        if constexpr (final_residual >= 2) {
            v1 = vld1q_lane_f32(pVect1 + 1, v1, 1);
            v2 = vld1q_lane_f32(pVect2 + 1, v2, 1);
        }
        if constexpr (final_residual >= 3) {
            v1 = vld1q_lane_f32(pVect1 + 2, v1, 2);
            v2 = vld1q_lane_f32(pVect2 + 2, v2, 2);
        }

        // Calculate difference and square
        float32x4_t diff = vsubq_f32(v1, v2);
        sum_squares = vmlaq_f32(sum_squares, diff, diff);
    }

    // Horizontal sum of the 4 elements in the NEON register
    float32x2_t sum_halves = vadd_f32(vget_low_f32(sum_squares), vget_high_f32(sum_squares));
    float32x2_t summed = vpadd_f32(sum_halves, sum_halves);

    return vget_lane_f32(summed, 0);
}
