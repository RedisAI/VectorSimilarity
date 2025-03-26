/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

static inline void L2SquareStep(float *&pVect1, float *&pVect2, size_t &offset, svfloat32_t &sum) {
    // Load vectors
    svfloat32_t v1 = svld1_f32(svptrue_b32(), pVect1 + offset);
    svfloat32_t v2 = svld1_f32(svptrue_b32(), pVect2 + offset);

    // Calculate difference between vectors
    svfloat32_t diff = svsub_f32_x(svptrue_b32(), v1, v2);

    // Square the difference and accumulate: sum += diff * diff
    sum = svmla_f32_z(svptrue_b32(), sum, diff, diff);

    // Advance pointers by the vector length
    offset += svcntw();
}

template <bool partial_chunk, unsigned char additional_steps>
float FP32_L2SqrSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t offset = 0;

    // Get the number of 32-bit elements per vector at runtime
    uint64_t vl = svcntw();

    // Multiple accumulators to increase instruction-level parallelism
    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    // Process vectors in chunks, with unrolling for better pipelining
    auto chunk_size = 4 * vl;
    size_t number_of_chunks = dimension / chunk_size;
    for (size_t i = 0; i < number_of_chunks; ++i) {
        // Process 4 chunks with separate accumulators
        L2SquareStep(pVect1, pVect2, offset, sum0);
        L2SquareStep(pVect1, pVect2, offset, sum1);
        L2SquareStep(pVect1, pVect2, offset, sum2);
        L2SquareStep(pVect1, pVect2, offset, sum3);
    }

    if constexpr (additional_steps > 0) {
        if constexpr (additional_steps >= 1) {
            L2SquareStep(pVect1, pVect2, offset, sum0);
        }
        if constexpr (additional_steps >= 2) {
            L2SquareStep(pVect1, pVect2, offset, sum1);
        }
        if constexpr (additional_steps >= 3) {
            L2SquareStep(pVect1, pVect2, offset, sum2);
        }
    }

    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b32(offset, dimension);

        // Load vectors with predication
        svfloat32_t v1 = svld1_f32(pg, pVect1 + offset);
        svfloat32_t v2 = svld1_f32(pg, pVect2 + offset);

        svfloat32_t diff = svsub_f32_m(pg, v1, v2);

        sum3 = svmla_f32_m(pg, sum3, diff, diff);
    }

    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum1);
    sum2 = svadd_f32_z(svptrue_b32(), sum2, sum3);
    svfloat32_t sum_all = svadd_f32_z(svptrue_b32(), sum0, sum2);
    float result = svaddv_f32(svptrue_b32(), sum_all);
    return result;
}
