/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

static inline void L2SquareStep(float *&pVect1, float *&pVect2, float32x4_t &sum) {
    float32x4_t v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    float32x4_t v2 = vld1q_f32(pVect2);
    pVect2 += 4;

    // Calculate difference between vectors
    float32x4_t diff = vsubq_f32(v1, v2);

    // Square and accumulate
    sum = vmlaq_f32(sum, diff, diff);
}

template <unsigned char residual> // 0..15
float FP32_L2Sqr_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    float32x4_t sum_squares = vdupq_n_f32(0.0f);

    // Deal with %4 remainder first
    if constexpr (residual % 4) {
        float32x4_t v1 = vdupq_n_f32(0.0f);
        float32x4_t v2 = vdupq_n_f32(0.0f);

        if constexpr (residual % 4 == 3) {
            // Load 3 floats
            v1 = vld1q_lane_f32(pVect1, v1, 0);
            v2 = vld1q_lane_f32(pVect2, v2, 0);
            v1 = vld1q_lane_f32(pVect1 + 1, v1, 1);
            v2 = vld1q_lane_f32(pVect2 + 1, v2, 1);
            v1 = vld1q_lane_f32(pVect1 + 2, v1, 2);
            v2 = vld1q_lane_f32(pVect2 + 2, v2, 2);
        } else if constexpr (residual % 4 == 2) {
            // Load 2 floats
            v1 = vld1q_lane_f32(pVect1, v1, 0);
            v2 = vld1q_lane_f32(pVect2, v2, 0);
            v1 = vld1q_lane_f32(pVect1 + 1, v1, 1);
            v2 = vld1q_lane_f32(pVect2 + 1, v2, 1);
        } else if constexpr (residual % 4 == 1) {
            // Load 1 float
            v1 = vld1q_lane_f32(pVect1, v1, 0);
            v2 = vld1q_lane_f32(pVect2, v2, 0);
        }
        pVect1 += residual % 4;
        pVect2 += residual % 4;

        // Calculate difference and square
        float32x4_t diff = vsubq_f32(v1, v2);
        sum_squares = vmlaq_f32(sum_squares, diff, diff);
    }

    // Have another 1, 2 or 3 4-float steps according to residual
    if constexpr (residual >= 12)
        L2SquareStep(pVect1, pVect2, sum_squares);
    if constexpr (residual >= 8)
        L2SquareStep(pVect1, pVect2, sum_squares);
    if constexpr (residual >= 4)
        L2SquareStep(pVect1, pVect2, sum_squares);

    // Process remaining 16-float blocks (4 vectors at a time)
    while (pVect1 < pEnd1) {
        L2SquareStep(pVect1, pVect2, sum_squares);
        L2SquareStep(pVect1, pVect2, sum_squares);
        L2SquareStep(pVect1, pVect2, sum_squares);
        L2SquareStep(pVect1, pVect2, sum_squares);
    }

    // Horizontal sum of the 4 elements in the NEON register
    float32x2_t sum_halves = vadd_f32(vget_low_f32(sum_squares), vget_high_f32(sum_squares));
    float32x2_t summed = vpadd_f32(sum_halves, sum_halves); // contains duplicate values

    return vget_lane_f32(summed, 0);
}