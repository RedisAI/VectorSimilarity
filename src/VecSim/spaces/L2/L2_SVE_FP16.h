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
// widen_f16_to_f32 lives here: SVE.cpp compiles both headers, so it must be defined once.
#include "VecSim/spaces/IP/IP_SVE_FP16.h"

// The squared differences accumulate in fp32, not fp16. `FP16_L2Sqr` widens each stored half with
// FP16_to_FP32, subtracts in float and accumulates into a float, so a half precision accumulator
// would make this tier disagree with the same function computed anywhere else. It is also unsafe:
// 65504 is the largest finite fp16 value, so 32 elements of 200.0, all ordinary fp16 values, drive
// a half precision accumulator past it and the result becomes infinity. An fp32 accumulator cannot
// overflow for any fp16 input.
inline void L2Sqr_Step(const float16_t *vec1, const float16_t *vec2, svfloat32_t &acc_lo,
                       svfloat32_t &acc_hi, size_t &offset, const size_t chunk) {
    svbool_t all = svptrue_b16();
    svbool_t all32 = svptrue_b32();

    svfloat16_t v1 = svld1_f16(all, vec1 + offset);
    svfloat16_t v2 = svld1_f16(all, vec2 + offset);

    svfloat32_t a_lo, a_hi, b_lo, b_hi;
    widen_f16_to_f32(v1, a_lo, a_hi);
    widen_f16_to_f32(v2, b_lo, b_hi);

    // Subtract and accumulate in single precision.
    svfloat32_t d_lo = svsub_f32_x(all32, a_lo, b_lo);
    svfloat32_t d_hi = svsub_f32_x(all32, a_hi, b_hi);
    acc_lo = svmla_f32_x(all32, acc_lo, d_lo, d_lo);
    acc_hi = svmla_f32_x(all32, acc_hi, d_hi, d_hi);
    offset += chunk;
}

template <bool partial_chunk, unsigned char additional_steps> // [t/f, 0..3]
float FP16_L2Sqr_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
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
        L2Sqr_Step(vec1, vec2, acc1_lo, acc1_hi, offset, chunk);
        L2Sqr_Step(vec1, vec2, acc2_lo, acc2_hi, offset, chunk);
        L2Sqr_Step(vec1, vec2, acc3_lo, acc3_hi, offset, chunk);
        L2Sqr_Step(vec1, vec2, acc4_lo, acc4_hi, offset, chunk);
    }

    // Perform between 0 and 3 additional steps, according to `additional_steps` value
    if constexpr (additional_steps >= 1)
        L2Sqr_Step(vec1, vec2, acc1_lo, acc1_hi, offset, chunk);
    if constexpr (additional_steps >= 2)
        L2Sqr_Step(vec1, vec2, acc2_lo, acc2_hi, offset, chunk);
    if constexpr (additional_steps >= 3)
        L2Sqr_Step(vec1, vec2, acc3_lo, acc3_hi, offset, chunk);

    // Handle partial chunk, if needed
    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b16_u64(offset, dimension);

        // Load half-precision vectors. The predicated load zeroes the inactive lanes, so the
        // arithmetic below can run unpredicated: those lanes contribute a squared zero.
        svfloat16_t v1 = svld1_f16(pg, vec1 + offset);
        svfloat16_t v2 = svld1_f16(pg, vec2 + offset);

        svfloat32_t a_lo, a_hi, b_lo, b_hi;
        widen_f16_to_f32(v1, a_lo, a_hi);
        widen_f16_to_f32(v2, b_lo, b_hi);

        svbool_t all32 = svptrue_b32();
        svfloat32_t d_lo = svsub_f32_x(all32, a_lo, b_lo);
        svfloat32_t d_hi = svsub_f32_x(all32, a_hi, b_hi);
        acc4_lo = svmla_f32_x(all32, acc4_lo, d_lo, d_lo);
        acc4_hi = svmla_f32_x(all32, acc4_hi, d_hi, d_hi);
    }

    // Accumulate accumulators, all in fp32.
    svbool_t all32 = svptrue_b32();
    svfloat32_t sum = svadd_f32_x(all32, svadd_f32_x(all32, acc1_lo, acc1_hi),
                                  svadd_f32_x(all32, acc2_lo, acc2_hi));
    sum = svadd_f32_x(all32, sum, svadd_f32_x(all32, acc3_lo, acc3_hi));
    sum = svadd_f32_x(all32, sum, svadd_f32_x(all32, acc4_lo, acc4_hi));

    // Reduce the accumulated sum.
    return svaddv_f32(all32, sum);
}
