/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

static void L2SquareStep_SVE2(float *&pVect1, float *&pVect2, svfloat32_t &sum) {
    // Load vectors
    svfloat32_t v1 = svld1_f32(svptrue_b32(), pVect1);
    svfloat32_t v2 = svld1_f32(svptrue_b32(), pVect2);

    // Calculate difference between vectors (using SVE2's fused operations)
    svfloat32_t diff = svsub_f32_z(svptrue_b32(), v1, v2);

    // Square the difference and accumulate: sum += diff * diff
    sum = svmla_f32_z(svptrue_b32(), sum, diff, diff);

    // Advance pointers by the vector length
    pVect1 += svcntw();
    pVect2 += svcntw();
}

template <unsigned char residual>
float FP32_L2SqrSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    const float *pVect2 = static_cast<const float *>(pVect2v);

    // Get the number of 32-bit elements per vector at runtime
    uint64_t vl = svcntw();

    // Multiple accumulators to increase instruction-level parallelism
    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    // Process vectors in chunks, with unrolling for better pipelining
    size_t i = 0;
    for (; i + 4 * vl <= dimension; i += 4 * vl) {
        // Prefetch future data
        svprfw(svptrue_b32(), pVect1 + i + 16 * vl, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), pVect2 + i + 16 * vl, SV_PLDL1KEEP);

        // Process 4 chunks with separate accumulators and properly separated pointers
        float *vec1_0 = const_cast<float *>(pVect1 + i);
        float *vec2_0 = const_cast<float *>(pVect2 + i);
        float *vec1_1 = const_cast<float *>(pVect1 + i + vl);
        float *vec2_1 = const_cast<float *>(pVect2 + i + vl);
        float *vec1_2 = const_cast<float *>(pVect1 + i + 2 * vl);
        float *vec2_2 = const_cast<float *>(pVect2 + i + 2 * vl);
        float *vec1_3 = const_cast<float *>(pVect1 + i + 3 * vl);
        float *vec2_3 = const_cast<float *>(pVect2 + i + 3 * vl);

        L2SquareStep_SVE2(vec1_0, vec2_0, sum0);
        L2SquareStep_SVE2(vec1_1, vec2_1, sum1);
        L2SquareStep_SVE2(vec1_2, vec2_2, sum2);
        L2SquareStep_SVE2(vec1_3, vec2_3, sum3);
    }

    // Handle remaining elements (less than 4*vl)
    for (; i < dimension; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dimension);

        // Load vectors with predication
        svfloat32_t v1 = svld1_f32(pg, pVect1 + i);
        svfloat32_t v2 = svld1_f32(pg, pVect2 + i);

        // Calculate difference with predication
        svfloat32_t diff = svsub_f32_m(pg, v1, v2);

        // Square the difference and accumulate with predication
        sum0 = svmla_f32_m(pg, sum0, diff, diff);
    }

    // Combine the partial sums
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum1);
    sum2 = svadd_f32_z(svptrue_b32(), sum2, sum3);
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum2);

    // Horizontal sum and convert to similarity score
    float result = svaddv_f32(svptrue_b32(), sum0);
    return result; // Convert distance to similarity score
}