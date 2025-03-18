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
    // Load vectors from memory
    svfloat32_t v1 = svld1_f32(pg, pVect1);
    svfloat32_t v2 = svld1_f32(pg, pVect2);

    // Use fmla instead of dot for float32 (svdot is for int8/int32)
    // This does element-wise multiply and add to accumulator
    sum = svmla_f32_x(pg, sum, v1, v2);

    // Advance pointers by the vector length
    size_t vl = svcntw();
    pVect1 += vl;
    pVect2 += vl;
}

template <unsigned char residual> // 0..63 (assuming max SVE vector length of 2048 bits = 64 floats)
float FP32_InnerProductSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    // Get the scalable vector length
    size_t vl = svcntw();

    // Create predicate for full vectors
    svbool_t pg_all = svptrue_b32();

    // Create predicate for residual elements
    svbool_t pg_residual = svwhilelt_b32_u32(0, residual);

    // Initialize sum to zero
    svfloat32_t sum = svdup_n_f32(0.0f);

    // Handle residual elements first (if any)
    if constexpr (residual > 0) {
        svfloat32_t v1 = svld1_f32(pg_residual, pVect1);
        svfloat32_t v2 = svld1_f32(pg_residual, pVect2);
        sum = svmla_f32_x(pg_residual, sum, v1, v2);
        pVect1 += residual;
        pVect2 += residual;
    }

    // Process main loop with full vectors - calculate iterations based on dimension
    size_t main_iterations = (dimension - residual) / vl;
    for (size_t i = 0; i < main_iterations; i++) {
        InnerProductStep_SVE2(pVect1, pVect2, sum, pg_all);
    }

    // Handle any remaining elements not covered by residual or full vectors
    size_t remaining = (dimension - residual) % vl;
    if (remaining > 0) {
        svbool_t pg_remain = svwhilelt_b32_u32(0, remaining);
        svfloat32_t v1 = svld1_f32(pg_remain, pVect1);
        svfloat32_t v2 = svld1_f32(pg_remain, pVect2);
        sum = svmla_f32_x(pg_remain, sum, v1, v2);
    }

    // Horizontal sum of all elements
    float final_sum = svaddv_f32(pg_all, sum);

    // Return 1.0f - sum to match the convention from SSE/SVE implementations
    return 1.0f - final_sum;
}