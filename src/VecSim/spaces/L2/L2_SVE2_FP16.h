/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

// #include "VecSim/spaces/space_includes.h"

#include <arm_sve.h>

template <bool has_residual>
float FP16_L2Sqr_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const float16_t *>(pVect1v);
    const auto *pVect2 = static_cast<const float16_t *>(pVect2v);
    const size_t chunk = svcnth(); // number of 16-bit elements in a register
    svbool_t all = svptrue_b16();
    svfloat32_t acc = svdup_f32(0.0f);
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
        acc = svmlalt_f32(acc, diff, diff);
        acc = svmlalb_f32(acc, diff, diff);

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
        // Use `z` suffix to zero out the inactive part
        svfloat16_t diff = svsub_f16_z(pg, v1, v2);
        // Square the differences.
        acc = svmlalt_f32(acc, diff, diff);
        acc = svmlalb_f32(acc, diff, diff);
    }

    // Reduce the accumulated sum.
    float result = svaddv_f32(svptrue_b32(), acc);
    return result;
}
