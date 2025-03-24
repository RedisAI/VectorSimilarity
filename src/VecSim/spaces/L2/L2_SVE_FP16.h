/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

// #include "VecSim/spaces/space_includes.h"

#include <arm_sve.h>

template <bool has_residual>
float FP16_L2Sqr_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const float16_t *>(pVect1v);
    const auto *pVect2 = static_cast<const float16_t *>(pVect2v);
    const size_t chunk = svcnth(); // number of 16-bit elements in a register
    svbool_t all16 = svptrue_b16();
    svbool_t all32 = svptrue_b32();
    svfloat32_t acc = svdup_f32(0.0f);
    size_t i = 0;

    // Process all full vectors
    const size_t full_iterations = dimension / chunk;
    for (size_t iter = 0; iter < full_iterations; iter++) {
        // Load half-precision vectors.
        svfloat16_t v1 = svld1_f16(all16, pVect1 + i);
        svfloat16_t v2 = svld1_f16(all16, pVect2 + i);
        // Compute difference in half precision.
        svfloat16_t diff = svsub_f16_x(all16, v1, v2);
        // Square the differences.
        svfloat16_t diffSq = svmul_f16_x(all16, diff, diff);
        // Convert to single-precision for accumulation.
        svfloat32_t diffSqF32_even = svcvt_f32_f16_x(all32, diffSq);
        acc = svadd_f32_x(all32, acc, diffSqF32_even);
        svfloat32_t diffSqF32_odd = svcvtlt_f32_f16_x(all32, diffSq);
        acc = svadd_f32_x(all32, acc, diffSqF32_odd);

        // Move to next chunk
        i += chunk;
    }

    // Handle the tail with the residual predicate
    if constexpr (has_residual) {
        svbool_t pg = svwhilelt_b16(i, dimension);

        // Load half-precision vectors.
        svfloat16_t v1 = svld1_f16(pg, pVect1 + i);
        svfloat16_t v2 = svld1_f16(pg, pVect2 + i);
        // Compute difference in half precision.
        svfloat16_t diff = svsub_f16_x(pg, v1, v2);
        // Square the differences.
        // Use `z` suffix to zero-out the irrelevant part before casting
        svfloat16_t diffSq = svmul_f16_z(pg, diff, diff);
        // Convert to single-precision for accumulation.
        svfloat32_t diffSqF32_1 = svcvt_f32_f16(all32, diffSq);
        acc = svadd_f32_x(all32, acc, diffSqF32_1);
        svfloat32_t diffSqF32_2 = svcvtlt_f32_f16(all32, diffSq);
        acc = svadd_f32_x(all32, acc, diffSqF32_2);
    }

    // Reduce the accumulated sum.
    float result = svaddv_f32(all32, acc);
    return result;
}

template <bool has_residual>
float FP16_L2Sqr_SVE_direct(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const float16_t *>(pVect1v);
    const auto *pVect2 = static_cast<const float16_t *>(pVect2v);
    const size_t chunk = svcnth(); // number of 16-bit elements in a register
    svbool_t all = svptrue_b16();
    svfloat16_t acc = svdup_f16(0.0f);
    size_t i = 0;

    // Process all full vectors
    const size_t full_iterations = dimension / chunk;
    for (size_t iter = 0; iter < full_iterations; iter++) {
        // Load half-precision vectors.
        svfloat16_t v1 = svld1_f16(all, pVect1 + i);
        svfloat16_t v2 = svld1_f16(all, pVect2 + i);
        // Compute difference in half precision.
        svfloat16_t diff = svsub_f16_x(all, v1, v2);
        // Square the differences and accumulate
        acc = svmla_f16_x(all, acc, diff, diff);

        // Move to next chunk
        i += chunk;
    }

    // Handle the tail with the residual predicate
    if constexpr (has_residual) {
        svbool_t pg = svwhilelt_b16(i, dimension);

        // Load half-precision vectors.
        svfloat16_t v1 = svld1_f16(pg, pVect1 + i);
        svfloat16_t v2 = svld1_f16(pg, pVect2 + i);
        // Compute difference in half precision.
        svfloat16_t diff = svsub_f16_x(pg, v1, v2);
        // Square the differences.
        // Use `m` suffix to keep the inactive elements as they are in `acc`
        acc = svmla_f16_m(pg, acc, diff, diff);
    }

    // Reduce the accumulated sum.
    float result = svaddv_f16(all, acc);
    return result;
}
