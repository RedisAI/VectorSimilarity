/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include <arm_sve.h>

// Compute L2 squared for one FP16 vector pair, accumulating in FP32 for precision.
// svcvt_f32_f16_x converts even-indexed FP16 elements to FP32
// For odd-indexed elements, we shift the vector by 1 element first, then use svcvt_f32_f16_x
inline void FP16_L2Sqr_Op(svfloat32_t &acc, svfloat16_t v1_f16, svfloat16_t v2_f16) {
    svbool_t pg32 = svptrue_b32();

    // Convert even-indexed FP16 elements to FP32 and compute
    svfloat32_t v1_lo = svcvt_f32_f16_x(pg32, v1_f16);
    svfloat32_t v2_lo = svcvt_f32_f16_x(pg32, v2_f16);
    svfloat32_t diff_lo = svsub_f32_x(pg32, v1_lo, v2_lo);
    acc = svmla_f32_x(pg32, acc, diff_lo, diff_lo);

    // Shift FP16 vectors by 1 element to move odd elements to even positions
    // svext extracts elements from concatenation of two vectors at given offset
    svfloat16_t v1_shifted = svext_f16(v1_f16, v1_f16, 1);
    svfloat16_t v2_shifted = svext_f16(v2_f16, v2_f16, 1);

    // Now convert (previously odd-indexed) elements to FP32 and compute
    svfloat32_t v1_hi = svcvt_f32_f16_x(pg32, v1_shifted);
    svfloat32_t v2_hi = svcvt_f32_f16_x(pg32, v2_shifted);
    svfloat32_t diff_hi = svsub_f32_x(pg32, v1_hi, v2_hi);
    acc = svmla_f32_x(pg32, acc, diff_hi, diff_hi);
}

inline void FP16_L2Sqr_Step(const float16_t *vec1, const float16_t *vec2, svfloat32_t &acc,
                            size_t &offset, const size_t chunk) {
    svbool_t pg16 = svptrue_b16();

    // Load FP16 vectors
    svfloat16_t v1_f16 = svld1_f16(pg16, vec1 + offset);
    svfloat16_t v2_f16 = svld1_f16(pg16, vec2 + offset);

    // Compute L2 squared with FP32 accumulation
    FP16_L2Sqr_Op(acc, v1_f16, v2_f16);

    offset += chunk;
}

template <bool partial_chunk, unsigned char additional_steps> // [t/f, 0..3]
float FP16_L2Sqr_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *vec1 = static_cast<const float16_t *>(pVect1v);
    const auto *vec2 = static_cast<const float16_t *>(pVect2v);
    const size_t chunk = svcnth(); // number of 16-bit elements in a register

    // Accumulate in FP32 for precision
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);
    svfloat32_t acc4 = svdup_f32(0.0f);
    size_t offset = 0;

    // Process all full vectors (4 chunks at a time for loop unrolling)
    const size_t full_iterations = dimension / chunk / 4;
    for (size_t iter = 0; iter < full_iterations; iter++) {
        FP16_L2Sqr_Step(vec1, vec2, acc1, offset, chunk);
        FP16_L2Sqr_Step(vec1, vec2, acc2, offset, chunk);
        FP16_L2Sqr_Step(vec1, vec2, acc3, offset, chunk);
        FP16_L2Sqr_Step(vec1, vec2, acc4, offset, chunk);
    }

    // Perform between 0 and 3 additional steps, according to `additional_steps` value
    if constexpr (additional_steps >= 1)
        FP16_L2Sqr_Step(vec1, vec2, acc1, offset, chunk);
    if constexpr (additional_steps >= 2)
        FP16_L2Sqr_Step(vec1, vec2, acc2, offset, chunk);
    if constexpr (additional_steps >= 3)
        FP16_L2Sqr_Step(vec1, vec2, acc3, offset, chunk);

    // Handle partial chunk, if needed
    if constexpr (partial_chunk) {
        svbool_t pg16 = svwhilelt_b16_u64(offset, dimension);

        // Load FP16 vectors with predicate (inactive elements are zeros)
        svfloat16_t v1_f16 = svld1_f16(pg16, vec1 + offset);
        svfloat16_t v2_f16 = svld1_f16(pg16, vec2 + offset);

        // Compute L2 squared with FP32 accumulation
        // Zero elements contribute 0 to the sum, so no special handling needed
        FP16_L2Sqr_Op(acc4, v1_f16, v2_f16);
    }

    // Accumulate all accumulators in FP32
    svbool_t pg32 = svptrue_b32();
    acc1 = svadd_f32_x(pg32, acc1, acc3);
    acc2 = svadd_f32_x(pg32, acc2, acc4);
    acc1 = svadd_f32_x(pg32, acc1, acc2);

    // Reduce the accumulated sum in FP32
    float result = svaddv_f32(pg32, acc1);
    return result;
}
