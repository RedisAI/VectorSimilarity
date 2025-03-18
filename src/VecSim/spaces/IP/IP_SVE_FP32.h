/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

#include <arm_sve.h>

static inline void InnerProductStep(float *&pVect1, float *&pVect2, svfloat32_t &sum, svbool_t pg) {
    svfloat32_t v1 = svld1_f32(pg, pVect1);
    svfloat32_t v2 = svld1_f32(pg, pVect2);
    
    // Increment pointers by the number of active elements in predicate
    uint64_t vlen = svcntw();
    pVect1 += vlen;
    pVect2 += vlen;
    
    // Multiply-accumulate
    sum = svmla_f32_x(pg, sum, v1, v2);
}

template <unsigned char residual> // 0..63 (assuming max SVE vector length of 2048 bits = 64 floats)
float FP32_InnerProductSIMD64_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    // Get vector length at runtime
    const uint64_t vl = svcntw();
    
    // Create predicates for full vectors and residual elements
    svbool_t pg_all = svptrue_b32();
    svbool_t pg_residual = svwhilelt_b32_u32(0, residual);
    
    // Initialize accumulator
    svfloat32_t sum = svdup_n_f32(0.0f);
    
    // Handle residual elements first (if any)
    if constexpr (residual > 0) {
        svfloat32_t v1 = svld1_f32(pg_residual, pVect1);
        svfloat32_t v2 = svld1_f32(pg_residual, pVect2);
        sum = svmla_f32_x(pg_residual, sum, v1, v2);
        pVect1 += residual;
        pVect2 += residual;
    }
    
    // Process main loop with full vectors
    size_t main_iterations = dimension / vl;
    for (size_t i = 0; i < main_iterations; i++) {
        InnerProductStep(pVect1, pVect2, sum, pg_all);
    }
    
    // Horizontal sum of all elements in the SVE register
    float final_sum = svaddv_f32(pg_all, sum);
    
    return 1.0f - final_sum; // Match SSE implementation return value
}