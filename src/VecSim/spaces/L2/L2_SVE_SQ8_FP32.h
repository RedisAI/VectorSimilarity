/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include "VecSim/types/sq8.h"
#include <arm_sve.h>

// SVE.cpp and SVE2.cpp both compile this header, under different -march flags. The
// anonymous namespace keeps each tier's bodies to itself; without it they are weak
// symbols that both objects define and link order picks the -march. Only the Choose_*
// entry points stay external. Dependencies above must stay outside the namespace.
namespace {

using sq8 = vecsim_types::sq8;

/*
 * Asymmetric SQ8-FP32 L2 squared distance computed via direct residual accumulation:
 *
 *   ||x - y||² = Σ(dequant(x_i) - y_i)²
 *   where dequant(x_i) = min_val + delta * q_i
 *
 * This avoids the algebraic-identity/cancellation approach, which catastrophically cancels in
 * FP32 when x and y share a large common offset relative to their spread.
 *
 * The subtract is fused into the multiply-add (diff = min_minus_y + delta*q) via svmla_f32_x,
 * which matters for performance.
 */

// Helper: compute Σ(diff_i²) for one SVE vector width, where diff_i = dequant(x_i) - y_i.
// pVect1 = SQ8 storage (quantized values), pVect2 = FP32 query.
// min_val_vec/delta_vec are broadcast scalars from the stored vector's metadata.
static inline void L2StepSQ8_FP32_SVE(const uint8_t *pVect1, const float *pVect2, size_t &offset,
                                      svfloat32_t &sum, const size_t chunk, svfloat32_t min_val_vec,
                                      svfloat32_t delta_vec) {
    svbool_t pg = svptrue_b32();

    // Load uint8 elements and zero-extend to uint32
    svuint32_t v1_u32 = svld1ub_u32(pg, pVect1 + offset);

    // Convert uint32 to float32
    svfloat32_t v1_f = svcvt_f32_u32_x(pg, v1_u32);

    // Load float elements from query
    svfloat32_t v2 = svld1_f32(pg, pVect2 + offset);

    // min - y computed once per lane, then fuse the dequantize-and-subtract:
    // diff = min_minus_y + delta*q, via a single fused multiply-add.
    svfloat32_t min_minus_y = svsub_f32_x(pg, min_val_vec, v2);
    svfloat32_t diff = svmla_f32_x(pg, min_minus_y, delta_vec, v1_f);

    sum = svmla_f32_x(pg, sum, diff, diff);

    offset += chunk;
}

// pVect1v = SQ8 storage, pVect2v = FP32 query
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP32_L2SqrSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const uint8_t *pVect1 = static_cast<const uint8_t *>(pVect1v); // SQ8 storage
    const float *pVect2 = static_cast<const float *>(pVect2v);     // FP32 query
    size_t offset = 0;

    svbool_t pg = svptrue_b32();

    // Get the number of 32-bit elements per vector at runtime
    uint64_t chunk = svcntw();

    // Get quantization parameters from stored vector (after quantized data)
    const auto *params1 = pVect1 + dimension;
    const float min_val_scalar = load_unaligned<float>(params1 + sq8::MIN_VAL * sizeof(float));
    const float delta_scalar = load_unaligned<float>(params1 + sq8::DELTA * sizeof(float));
    const svfloat32_t min_val_vec = svdup_f32(min_val_scalar);
    const svfloat32_t delta_vec = svdup_f32(delta_scalar);

    // Multiple accumulators for ILP
    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    // Full-width groups first, predicated tail last, and every bound compared rather than
    // divided.
    //
    // `chunk` is a runtime value (svcntw), so `dimension % chunk` and `/ chunk_size` compile to
    // real `udiv` instructions -- ~12-20 cycles each and not pipelined, on a function called
    // thousands of times per query. They are also redundant: CHOOSE_SVE_IMPLEMENTATION already
    // divides once, at chooser time, and hands the results down as `partial_chunk` and
    // `additional_steps`. Doing the full vectors first additionally keeps every unpredicated
    // load at a multiple of the vector length; a leading partial chunk would push all of them
    // to a non-VL-multiple offset.
    //
    // This is why the shape here differs from the prefix-first sibling SQ8 SVE kernels. Given
    // dimension = k*chunk + r (r = dimension % chunk, so r > 0 exactly when partial_chunk),
    // the loop below runs floor(k/4) times, leaving (k % 4) == additional_steps full vectors
    // plus r tail elements.
    //
    // Measured on Graviton4 (Neoverse-V2, 128-bit VL), median of 9, cv <= 0.33%, against the
    // prefix-first shape: SVE2 is faster everywhere (dim 1024 160 -> 141ns, dim 513 81.9 ->
    // 79.4ns), and SVE is faster where dimension is a multiple of the vector length (dim 1024
    // 164 -> 153ns) but ~8% slower at dims that leave a tail (dim 513 78.3 -> 84.6ns). Kept
    // because SVE2 is the tier the chooser prefers when both are present, and because real
    // embedding dims (128/256/384/512/768/1024/1536) all divide evenly at 128- and 256-bit VL,
    // landing on the faster path. Feeding the tail through its own accumulator instead of
    // svmla_f32_m was also tried and measured slightly worse (dim 513 85.7ns).
    //
    // Note `dimension & (chunk - 1)` is NOT a valid substitute for the modulo: SVE permits any
    // vector length that is a multiple of 128 bits, not only powers of two.
    const size_t chunk_size = 4 * chunk;
    while (offset + chunk_size <= dimension) {
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum0, chunk, min_val_vec, delta_vec);
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum1, chunk, min_val_vec, delta_vec);
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum2, chunk, min_val_vec, delta_vec);
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum3, chunk, min_val_vec, delta_vec);
    }

    // Handle remaining full-width steps (0-3), resolved at compile time.
    if constexpr (additional_steps > 0) {
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum0, chunk, min_val_vec, delta_vec);
    }
    if constexpr (additional_steps > 1) {
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum1, chunk, min_val_vec, delta_vec);
    }
    if constexpr (additional_steps > 2) {
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum2, chunk, min_val_vec, delta_vec);
    }

    // Predicated tail for the final partial vector. svwhilelt derives the predicate from the
    // current offset against `dimension`, so no residual arithmetic is needed, and inactive
    // lanes of the `_z` (zeroing) forms contribute 0 to the accumulator.
    if constexpr (partial_chunk) {
        svbool_t pg_tail =
            svwhilelt_b32(static_cast<uint32_t>(offset), static_cast<uint32_t>(dimension));

        svuint32_t v1_u32 = svld1ub_u32(pg_tail, pVect1 + offset);
        svfloat32_t v1_f = svcvt_f32_u32_z(pg_tail, v1_u32);
        svfloat32_t v2 = svld1_f32(pg_tail, pVect2 + offset);

        svfloat32_t min_minus_y = svsub_f32_z(pg_tail, min_val_vec, v2);
        svfloat32_t diff = svmla_f32_z(pg_tail, min_minus_y, delta_vec, v1_f);

        // Merging (_m), not zeroing (_z): sum3 already holds full-vector partial sums from the
        // main loop, and _z would zero every lane outside pg_tail, silently dropping them. _z is
        // only safe on a freshly zeroed accumulator, which is why the pre-restructure kernel
        // could use it here -- back then this block ran first, against an all-zero sum0.
        // The temporaries above stay _z because they are fresh values, not accumulators.
        sum3 = svmla_f32_m(pg_tail, sum3, diff, diff);
    }

    // Combine the accumulators
    svfloat32_t sum = svadd_f32_z(pg, sum0, sum1);
    sum = svadd_f32_z(pg, sum, sum2);
    sum = svadd_f32_z(pg, sum, sum3);

    // Horizontal sum to get Σ(diff_i²)
    return svaddv_f32(pg, sum);
}
} // namespace
