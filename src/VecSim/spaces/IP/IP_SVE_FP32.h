/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

#include <arm_sve.h>

static inline void InnerProductStep(float *&pVect1, float *&pVect2, size_t &offset,
                                    svfloat32_t &sum) {
    svfloat32_t v1 = svld1_f32(svptrue_b32(), pVect1 + offset);
    svfloat32_t v2 = svld1_f32(svptrue_b32(), pVect2 + offset);

    sum = svmla_f32_x(svptrue_b32(), sum, v1, v2);

    offset += svcntw();
}

template <bool partial_chunk, unsigned char additional_steps>
float FP32_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t offset = 0;

    uint64_t sve_word_count = svcntw();

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    auto chunk_size = 4 * sve_word_count;
    const size_t number_of_chunks = dimension / chunk_size;
    for (size_t i = 0; i < number_of_chunks; i++) {
        InnerProductStep(pVect1, pVect2, offset, sum0);
        InnerProductStep(pVect1, pVect2, offset, sum1);
        InnerProductStep(pVect1, pVect2, offset, sum2);
        InnerProductStep(pVect1, pVect2, offset, sum3);
    }

    // Process remaining complete SVE vectors that didn't fit into the main loop
    // These are full vector operations (0-3 elements)
    if constexpr (additional_steps > 0) {
        if constexpr (additional_steps >= 1) {
            InnerProductStep(pVect1, pVect2, offset, sum0);
        }
        if constexpr (additional_steps >= 2) {
            InnerProductStep(pVect1, pVect2, offset, sum1);
        }
        if constexpr (additional_steps >= 3) {
            InnerProductStep(pVect1, pVect2, offset, sum3);
        }
    }

    // Process final tail elements that don't form a complete vector
    // This section handles the case when dimension is not evenly divisible by SVE vector length
    if constexpr (partial_chunk) {
        // Create a predicate mask where each lane is active only for the remaining elements
        svbool_t pg = svwhilelt_b32(offset, dimension);

        // Load vectors with predication
        svfloat32_t v1 = svld1_f32(pg, pVect1 + offset);
        svfloat32_t v2 = svld1_f32(pg, pVect2 + offset);
        sum3 = svmla_f32_m(pg, sum3, v1, v2);
    }

    sum0 = svadd_f32_x(svptrue_b32(), sum0, sum1);
    sum2 = svadd_f32_x(svptrue_b32(), sum2, sum3);
    // Perform vector addition in parallel
    svfloat32_t sum_all = svadd_f32_x(svptrue_b32(), sum0, sum2);
    // Single horizontal reduction at the end
    float result = svaddv_f32(svptrue_b32(), sum_all);
    return 1.0f - result;
}
