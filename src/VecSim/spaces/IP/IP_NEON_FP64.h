/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

static inline void InnerProductStep(double *&pVect1, double *&pVect2, float64x2_t &sum) {
    float64x2_t v1 = vld1q_f64(pVect1);
    float64x2_t v2 = vld1q_f64(pVect2);
    sum = vmlaq_f64(sum, v1, v2);
    pVect1 += 2;
    pVect2 += 2;
}

template <unsigned char residual> // 0..7
double FP64_InnerProductSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    float64x2_t sum_prod = vdupq_n_f64(0.0f);

    // These are compile-time constants derived from the template parameter

    // Calculate how many full 8-element blocks to process (each block = 4 NEON vectors)
    // This ensures we process dimension-residual elements in the main loop
    const size_t main_blocks = (dimension - residual) / 8;

    // Process all complete 8-float blocks (2 vectors at a time)
    for (size_t i = 0; i < main_blocks; i++) {
        // Prefetch next data block (64 bytes ahead = 8 floats)
        __builtin_prefetch(pVect1 + 64, 0, 0);
        __builtin_prefetch(pVect2 + 64, 0, 0);

        // Process 2 NEON vectors (4 floats) per iteration
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);

    }

    // Handle remaining complete 4-float blocks within residual
    // This code generates specialized paths at compile time based on residual
    constexpr size_t remaining_duos = residual / 2; // Complete 4-element vectors in residual
    if constexpr (remaining_duos > 0) {
        for (size_t i = 0; i < remaining_duos; i++) {
            InnerProductStep(pVect1, pVect2, sum_prod);
        }
    }

    // Handle final residual elements (0-1 elements)
    // This entire block is eliminated at compile time if final_residual is 0
    constexpr size_t final_residual = residual % 2; // Final 0-1 elements
    if constexpr (final_residual == 1) {
        float64x2_t v1 = vdupq_n_f64(0.0f);
        float64x2_t v2 = vdupq_n_f64(0.0f);
        v1 = vld1q_lane_f64(pVect1, v1, 0);
        v2 = vld1q_lane_f64(pVect2, v2, 0);

        sum_prod = vmlaq_f64(sum_prod, v1, v2);
    }

    // Horizontal sum of the 4 elements in the NEON register
    float64x1_t sum = vpaddq_f64(sum_prod, sum_prod);

    return 1.0f - sum;
}