/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

static inline void L2SquareStep(double *&pVect1, double *&pVect2, float64x4_t &sum) {
    float64x4_t v1 = vld1q_f64(pVect1);
    float64x4_t v2 = vld1q_f64(pVect2);

    // Calculate difference between vectors
    float64x4_t diff = vsubq_f64(v1, v2);

    // Square and accumulate
    sum = vmlaq_f64(sum, diff, diff);

    pVect1 += 4;
    pVect2 += 4;
}

template <unsigned char residual> // 0..15
double FP64_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    float64x4_t sum_squares = vdupq_n_f64(0.0f);

    // These are compile-time constants derived from the template parameter
    constexpr size_t remaining_quads = residual / 4; // Complete 4-element vectors in residual
    constexpr size_t final_residual = residual % 4;  // Final 0-3 elements

    // Calculate how many full 16-element blocks to process
    const size_t main_blocks = (dimension - residual) / 16;

    // Process all complete 16-float blocks (4 vectors at a time)
    for (size_t i = 0; i < main_blocks; i++) {
        // Prefetch next data block (64 bytes ahead = 16 floats)
        __builtin_prefetch(pVect1 + 64, 0, 0);
        __builtin_prefetch(pVect2 + 64, 0, 0);

        // Process 4 NEON vectors (16 floats) per iteration
        L2SquareStep(pVect1, pVect2, sum_squares);
        L2SquareStep(pVect1, pVect2, sum_squares);
        L2SquareStep(pVect1, pVect2, sum_squares);
        L2SquareStep(pVect1, pVect2, sum_squares);
    }

    // Handle remaining complete 4-float blocks within residual
    if constexpr (remaining_quads > 0) {
        for (size_t i = 0; i < remaining_quads; i++) {
            L2SquareStep(pVect1, pVect2, sum_squares);
        }
    }

    // Handle final residual elements (0-3 elements)
    if constexpr (final_residual > 0) {
        float64x4_t v1 = vdupq_n_f64(0.0f);
        float64x4_t v2 = vdupq_n_f64(0.0f);

        if constexpr (final_residual >= 1) {
            v1 = vld1q_lane_f64(pVect1, v1, 0);
            v2 = vld1q_lane_f64(pVect2, v2, 0);
        }
        if constexpr (final_residual >= 2) {
            v1 = vld1q_lane_f64(pVect1 + 1, v1, 1);
            v2 = vld1q_lane_f64(pVect2 + 1, v2, 1);
        }
        if constexpr (final_residual >= 3) {
            v1 = vld1q_lane_f64(pVect1 + 2, v1, 2);
            v2 = vld1q_lane_f64(pVect2 + 2, v2, 2);
        }

        // Calculate difference and square
        float64x4_t diff = vsubq_f64(v1, v2);
        sum_squares = vmlaq_f64(sum_squares, diff, diff);
    }

    // Horizontal sum of the 4 elements in the NEON register
    float64x2_t sum_halves = vadd_f64(vget_low_f64(sum_squares), vget_high_f64(sum_squares));
    float64x2_t summed = vpadd_f64(sum_halves, sum_halves);

    return vget_lane_f64(summed, 0);
}