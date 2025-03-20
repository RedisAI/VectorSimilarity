/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

#include <arm_sve.h>

static void InnerProductStep(float *&pVect1, float *&pVect2, svfloat32_t &sum) {
    // Load vectors
    svfloat32_t v1 = svld1_f32(svptrue_b32(), pVect1);
    svfloat32_t v2 = svld1_f32(svptrue_b32(), pVect2);

    // Multiply-accumulate
    sum = svmla_f32_z(svptrue_b32(), sum, v1, v2);

    // Advance pointers
    pVect1 += svcntw();
    pVect2 += svcntw();
}

template <unsigned char residual>
float FP32_InnerProductSIMD64_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    const float *pVect2 = static_cast<const float *>(pVect2v);

    uint64_t vl = svcntw();

    // Multiple accumulators to increase instruction-level parallelism
    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    // Process vectors in chunks, with unrolling for better pipelining
    size_t i = 0;
    for (; i + 4 * vl <= dimension; i += 4 * vl) {
        // Prefetch future data (critical for high dimensions)
        svprfw(svptrue_b32(), pVect1 + i + 16 * vl, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), pVect2 + i + 16 * vl, SV_PLDL1KEEP);

        float *vec1_0 = const_cast<float *>(pVect1 + i);
        float *vec2_0 = const_cast<float *>(pVect2 + i);
        InnerProductStep(vec1_0, vec2_0, sum0);
        float *vec1_1 = const_cast<float *>(pVect1 + i + vl);
        float *vec2_1 = const_cast<float *>(pVect2 + i + vl);
        InnerProductStep(vec1_1, vec2_1, sum1);
        float *vec1_2 = const_cast<float *>(pVect1 + i + 2 * vl);
        float *vec2_2 = const_cast<float *>(pVect2 + i + 2 * vl);
        InnerProductStep(vec1_2, vec2_2, sum2);
        float *vec1_3 = const_cast<float *>(pVect1 + i + 3 * vl);
        float *vec2_3 = const_cast<float *>(pVect2 + i + 3 * vl);

        InnerProductStep(vec1_3, vec2_3, sum3);
    }

    // Handle remaining elements (less than 4*vl)
    for (; i < dimension; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dimension);
        svfloat32_t v1 = svld1_f32(pg, pVect1 + i);
        svfloat32_t v2 = svld1_f32(pg, pVect2 + i);
        sum0 = svmla_f32_m(pg, sum0, v1, v2);
    }

    // Combine the partial sums
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum1);
    sum2 = svadd_f32_z(svptrue_b32(), sum2, sum3);
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum2);

    // Horizontal sum
    float result = svaddv_f32(svptrue_b32(), sum0);
    return 1.0f - result;
}