/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

// SVE2 offers specialized dot product instructions for better efficiency
static inline void InnerProductStep_SVE2(float *&pVect1, float *&pVect2, svfloat32_t &sum,
                                         svbool_t pg) {
    // Get vector length
    size_t vl = svcntw();

    // Load vectors from memory
    svfloat32_t v1 = svld1_f32(pg, pVect1);
    svfloat32_t v2 = svld1_f32(pg, pVect2);

// Use SVE2 dot product instruction if available
#ifdef __ARM_FEATURE_SVE2
    // Process 4 elements at a time using SVE2's dot product
    // SVE2 dotproduct accumulates into sum
    sum = svdot_f32(sum, v1, v2);
#else
    // Fallback to standard SVE multiply-add if SVE2 not available
    sum = svmla_f32_x(pg, sum, v1, v2);
#endif

    // Advance pointers
    pVect1 += vl;
    pVect2 += vl;
}

template <unsigned char residual> // 0..63 (assuming max SVE vector length of 2048 bits = 64 floats)
float FP32_InnerProductSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    // Get the scalable vector length
    size_t vl = svcntw();
    const float *pEnd1 = pVect1 + dimension;

    // Create predicate for full vectors
    svbool_t pg = svptrue_b32();

    // Initialize sum to zero
    svfloat32_t sum = svdup_n_f32(0.0f);

    // Handle residual elements first
    if constexpr (residual > 0) {
        // Create a predicate for the residual elements
        svbool_t residual_pred = svwhilelt_b32(0, residual);

        // Load and process the residual elements
        svfloat32_t v1 = svld1_f32(residual_pred, pVect1);
        pVect1 += residual;
        svfloat32_t v2 = svld1_f32(residual_pred, pVect2);
        pVect2 += residual;

// Initialize sum with the residual product
#ifdef __ARM_FEATURE_SVE2
        // For SVE2, we initialize sum first, then use dot product
        sum = svdot_f32(sum, v1, v2);
#else
        // Fallback to multiply for non-SVE2
        sum = svmul_f32_x(residual_pred, v1, v2);
#endif
    }

    // Process the remaining full vectors
    while (pVect1 < pEnd1) {
        // Process as many elements as fit in an SVE vector
        svbool_t active_pred = svwhilelt_b32(0, pEnd1 - pVect1);
        if (svcntp_b32(pg, active_pred) < vl) {
            // This is the last partial vector
            svfloat32_t v1 = svld1_f32(active_pred, pVect1);
            svfloat32_t v2 = svld1_f32(active_pred, pVect2);

#ifdef __ARM_FEATURE_SVE2
            sum = svdot_f32(sum, v1, v2);
#else
            sum = svmla_f32_x(active_pred, sum, v1, v2);
#endif

            break;
        } else {
            // This is a full vector
            InnerProductStep_SVE2(pVect1, pVect2, sum, pg);
        }
    }

    // Horizontal sum of all elements
    float result = svaddv_f32(pg, sum);

    return result;
}
