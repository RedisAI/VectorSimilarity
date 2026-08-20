/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include <arm_sve.h>

// Widen a half-precision vector into two single-precision vectors. `svcvt_f32_f16` reads each
// input from the bottom 16 bits of a 32-bit lane, and ZIP1/ZIP2 interleave the lower and upper
// halves of their operands, so zipping against zero leaves `lo` holding the vector's lower
// contiguous half (elements 0 .. N/2-1) and `hi` the upper half, one value per 32-bit lane.
// Splitting this way changes the summation order relative to a scalar loop, and fp32 addition still
// rounds, so the result can differ in the last bits. That is true of every accumulator split in
// these kernels, which is why the randomized tests compare against a tolerance.
inline void widen_f16_to_f32(svfloat16_t v, svfloat32_t &lo, svfloat32_t &hi) {
    const svfloat16_t zero = svdup_f16(0.0f);
    lo = svcvt_f32_f16_x(svptrue_b32(), svzip1_f16(v, zero));
    hi = svcvt_f32_f16_x(svptrue_b32(), svzip2_f16(v, zero));
}

// The products accumulate in fp32, not fp16. `FP16_InnerProduct` widens each stored half with
// FP16_to_FP32 and accumulates into a float, so a half precision accumulator would make this tier
// disagree with the same function computed anywhere else. It is also unsafe: 65504 is the largest
// finite fp16 value, so 32 elements of 200.0, all ordinary fp16 values, drive a half precision
// accumulator past it and the result becomes infinity. An fp32 accumulator cannot overflow for
// any fp16 input.
inline void InnerProduct_Step(const float16_t *vec1, const float16_t *vec2, svfloat32_t &acc_lo,
                              svfloat32_t &acc_hi, size_t &offset, const size_t chunk) {
    svbool_t all = svptrue_b16();
    svbool_t all32 = svptrue_b32();

    // Load half-precision vectors.
    svfloat16_t v1 = svld1_f16(all, vec1 + offset);
    svfloat16_t v2 = svld1_f16(all, vec2 + offset);

    svfloat32_t a_lo, a_hi, b_lo, b_hi;
    widen_f16_to_f32(v1, a_lo, a_hi);
    widen_f16_to_f32(v2, b_lo, b_hi);

    // Compute multiplications and add to the accumulators, in single precision.
    acc_lo = svmla_f32_x(all32, acc_lo, a_lo, b_lo);
    acc_hi = svmla_f32_x(all32, acc_hi, a_hi, b_hi);

    // Move to next chunk
    offset += chunk;
}

template <bool partial_chunk, unsigned char additional_steps> // [t/f, 0..3]
float FP16_InnerProduct_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *vec1 = static_cast<const float16_t *>(pVect1v);
    const auto *vec2 = static_cast<const float16_t *>(pVect2v);
    const size_t chunk = svcnth(); // number of 16-bit elements in a register
    // Four accumulator pairs, keeping the original four-way unrolling with fp32 accumulators.
    svfloat32_t acc1_lo = svdup_f32(0.0f), acc1_hi = svdup_f32(0.0f);
    svfloat32_t acc2_lo = svdup_f32(0.0f), acc2_hi = svdup_f32(0.0f);
    svfloat32_t acc3_lo = svdup_f32(0.0f), acc3_hi = svdup_f32(0.0f);
    svfloat32_t acc4_lo = svdup_f32(0.0f), acc4_hi = svdup_f32(0.0f);
    size_t offset = 0;

    // Process all full vectors
    const size_t full_iterations = dimension / chunk / 4;
    for (size_t iter = 0; iter < full_iterations; iter++) {
        InnerProduct_Step(vec1, vec2, acc1_lo, acc1_hi, offset, chunk);
        InnerProduct_Step(vec1, vec2, acc2_lo, acc2_hi, offset, chunk);
        InnerProduct_Step(vec1, vec2, acc3_lo, acc3_hi, offset, chunk);
        InnerProduct_Step(vec1, vec2, acc4_lo, acc4_hi, offset, chunk);
    }

    // Perform between 0 and 3 additional steps, according to `additional_steps` value
    if constexpr (additional_steps >= 1)
        InnerProduct_Step(vec1, vec2, acc1_lo, acc1_hi, offset, chunk);
    if constexpr (additional_steps >= 2)
        InnerProduct_Step(vec1, vec2, acc2_lo, acc2_hi, offset, chunk);
    if constexpr (additional_steps >= 3)
        InnerProduct_Step(vec1, vec2, acc3_lo, acc3_hi, offset, chunk);

    // Handle the tail with the residual predicate
    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b16_u64(offset, dimension);

        // The predicated load zeroes the inactive lanes, so the arithmetic below can run
        // unpredicated: those lanes contribute a zero product.
        svfloat16_t v1 = svld1_f16(pg, vec1 + offset);
        svfloat16_t v2 = svld1_f16(pg, vec2 + offset);

        svfloat32_t a_lo, a_hi, b_lo, b_hi;
        widen_f16_to_f32(v1, a_lo, a_hi);
        widen_f16_to_f32(v2, b_lo, b_hi);

        svbool_t all32 = svptrue_b32();
        acc4_lo = svmla_f32_x(all32, acc4_lo, a_lo, b_lo);
        acc4_hi = svmla_f32_x(all32, acc4_hi, a_hi, b_hi);
    }

    // Accumulate accumulators, all in fp32.
    svbool_t all32 = svptrue_b32();
    svfloat32_t sum = svadd_f32_x(all32, svadd_f32_x(all32, acc1_lo, acc1_hi),
                                  svadd_f32_x(all32, acc2_lo, acc2_hi));
    sum = svadd_f32_x(all32, sum, svadd_f32_x(all32, acc3_lo, acc3_hi));
    sum = svadd_f32_x(all32, sum, svadd_f32_x(all32, acc4_lo, acc4_hi));

    // Reduce the accumulated sum.
    return 1.0f - svaddv_f32(all32, sum);
}
