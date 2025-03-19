/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

static inline void L2SquareStep_SVE(float *&pVect1, float *&pVect2, svfloat32_t &sum, svbool_t pg) {
    svfloat32_t v1 = svld1_f32(pg, pVect1);
    svfloat32_t v2 = svld1_f32(pg, pVect2);

    // Calculate difference between vectors
    svfloat32_t diff = svsub_f32_x(pg, v1, v2);

    // Square and accumulate: sum += diff * diff
    sum = svmla_f32_x(pg, sum, diff, diff);

    // Advance pointers by vector length
    size_t vl = svcntw();
    pVect1 += vl;
    pVect2 += vl;
}

template <unsigned char residual> // 0..63 (assuming max SVE vector length of 2048 bits = 64 floats)
float FP32_L2SqrSIMD64_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    // Get SVE vector length at runtime
    size_t vl = svcntw();

    // Create predicates for full vectors and residual elements
    svbool_t pg_all = svptrue_b32();
    svbool_t pg_residual = svwhilelt_b32_u32(0, residual);

    // Initialize accumulator
    svfloat32_t sum_squares = svdup_n_f32(0.0f);

    // Handle residual elements first (if any)
    if constexpr (residual > 0) {
        svfloat32_t v1 = svld1_f32(pg_residual, pVect1);
        svfloat32_t v2 = svld1_f32(pg_residual, pVect2);
        svfloat32_t diff = svsub_f32_x(pg_residual, v1, v2);
        sum_squares = svmla_f32_x(pg_residual, sum_squares, diff, diff);
        pVect1 += residual;
        pVect2 += residual;
    }

    // Process main loop with full vectors
    size_t main_iterations = (dimension - residual) / vl;
    for (size_t i = 0; i < main_iterations; i++) {
        L2SquareStep_SVE(pVect1, pVect2, sum_squares, pg_all);
    }

    // Handle any remaining elements not covered by residual or main loop
    size_t remaining = (dimension - residual) % vl;
    if (remaining > 0) {
        svbool_t pg_remain = svwhilelt_b32_u32(0, remaining);
        svfloat32_t v1 = svld1_f32(pg_remain, pVect1);
        svfloat32_t v2 = svld1_f32(pg_remain, pVect2);
        svfloat32_t diff = svsub_f32_x(pg_remain, v1, v2);
        sum_squares = svmla_f32_x(pg_remain, sum_squares, diff, diff);
    }

    // Horizontal sum of all elements in the SVE register
    float result = svaddv_f32(pg_all, sum_squares);

    return result;
}