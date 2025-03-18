/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

static inline void InnerProductStep(float *&pVect1, float *&pVect2, float32x4_t &sum) {
    float32x4_t v1 = vld1q_f32(pVect1);
    float32x4_t v2 = vld1q_f32(pVect2);
    
    sum = vmlaq_f32(sum, v1, v2);
    
    pVect1 += 4;
    pVect2 += 4;
}

template <unsigned char residual> // 0..16
float FP32_InnerProductSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    float32x4_t sum = vdupq_n_f32(0.0f);

    if constexpr (residual > 0) {
        float32x4_t v1 = vdupq_n_f32(0.0f);
        float32x4_t v2 = vdupq_n_f32(0.0f);
        
        if constexpr (residual >= 1) {
            v1 = vsetq_lane_f32(pVect1[0], v1, 0);
            v2 = vsetq_lane_f32(pVect2[0], v2, 0);
        }
        if constexpr (residual >= 2) {
            v1 = vsetq_lane_f32(pVect1[1], v1, 1);
            v2 = vsetq_lane_f32(pVect2[1], v2, 1);
        }
        if constexpr (residual >= 3) {
            v1 = vsetq_lane_f32(pVect1[2], v1, 2);
            v2 = vsetq_lane_f32(pVect2[2], v2, 2);
        }
        
        // Multiply and accumulate into sum
        sum = vmlaq_f32(sum, v1, v2);
        
        pVect1 += residual;
        pVect2 += residual;
    }

    // We dealt with the residual part. We are left with some multiple of 4 floats.
    while (pVect1 < pEnd1) {
        InnerProductStep(pVect1, pVect2, sum);
    }

    // Horizontal sum of the 4 elements in the NEON register
    float32x2_t sum_halves = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    return vget_lane_f32(vpadd_f32(sum_halves, sum_halves), 0);
}
