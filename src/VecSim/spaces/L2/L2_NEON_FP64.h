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

inline void L2SquareStep(double *&pVect1, double *&pVect2, float64x2_t &sum) {
    float64x2_t v1 = vld1q_f64(pVect1);
    float64x2_t v2 = vld1q_f64(pVect2);

    // Calculate difference between vectors
    float64x2_t diff = vsubq_f64(v1, v2);

    // Square and accumulate
    sum = vmlaq_f64(sum, diff, diff);

    pVect1 += 2;
    pVect2 += 2;
}

template <unsigned char residual> // 0..7
double FP64_L2SqrSIMD8_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    float64x2_t sum0 = vdupq_n_f64(0.0);
    float64x2_t sum1 = vdupq_n_f64(0.0);
    float64x2_t sum2 = vdupq_n_f64(0.0);
    float64x2_t sum3 = vdupq_n_f64(0.0);
    // These are compile-time constants derived from the template parameter

    // Calculate how many full 8-element blocks to process
    const size_t num_of_chunks = dimension / 8;

    for (size_t i = 0; i < num_of_chunks; i++) {
        L2SquareStep(pVect1, pVect2, sum0);
        L2SquareStep(pVect1, pVect2, sum1);
        L2SquareStep(pVect1, pVect2, sum2);
        L2SquareStep(pVect1, pVect2, sum3);
    }

    // Handle remaining complete 2-float blocks within residual
    constexpr size_t remaining_chunks = residual / 2;
    // Unrolled loop for the 2-float blocks
    if constexpr (remaining_chunks >= 1) {
        L2SquareStep(pVect1, pVect2, sum0);
    }
    if constexpr (remaining_chunks >= 2) {
        L2SquareStep(pVect1, pVect2, sum1);
    }
    if constexpr (remaining_chunks >= 3) {
        L2SquareStep(pVect1, pVect2, sum2);
    }

    // Handle final residual element
    constexpr size_t final_residual = residual % 2; // Final element
    if constexpr (final_residual > 0) {
        float64x2_t v1 = vdupq_n_f64(0.0);
        float64x2_t v2 = vdupq_n_f64(0.0);
        v1 = vld1q_lane_f64(pVect1, v1, 0);
        v2 = vld1q_lane_f64(pVect2, v2, 0);

        // Calculate difference and square
        float64x2_t diff = vsubq_f64(v1, v2);
        sum3 = vmlaq_f64(sum3, diff, diff);
    }

    float64x2_t sum_combined = vaddq_f64(vaddq_f64(sum0, sum1), vaddq_f64(sum2, sum3));

    // Horizontal sum of the 4 elements in the NEON register
    float64x1_t sum = vadd_f64(vget_low_f64(sum_combined), vget_high_f64(sum_combined));
    return vget_lane_f64(sum, 0);
}
