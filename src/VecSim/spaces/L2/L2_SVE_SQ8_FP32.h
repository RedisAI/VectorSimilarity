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

    // Handle partial chunk if needed
    if constexpr (partial_chunk) {
        size_t remaining = dimension % chunk;
        if (remaining > 0) {
            // Create predicate for the remaining elements
            svbool_t pg_partial =
                svwhilelt_b32(static_cast<uint32_t>(0), static_cast<uint32_t>(remaining));

            // Load uint8 elements and zero-extend to uint32
            svuint32_t v1_u32 = svld1ub_u32(pg_partial, pVect1 + offset);

            // Convert uint32 to float32
            svfloat32_t v1_f = svcvt_f32_u32_z(pg_partial, v1_u32);

            // Load float elements from query with predicate. `+ offset` is 0 here (this block
            // runs before any full-width step), but spell it out to match the load above and
            // so the two stay correct if the block order ever changes.
            svfloat32_t v2 = svld1_f32(pg_partial, pVect2 + offset);

            // min - y, then dequantize-and-subtract. Inactive lanes of a `_z` (zeroing)
            // predicated op become zero, which is exactly what we want when accumulating.
            svfloat32_t min_minus_y = svsub_f32_z(pg_partial, min_val_vec, v2);
            svfloat32_t diff = svmla_f32_z(pg_partial, min_minus_y, delta_vec, v1_f);
            sum0 = svmla_f32_z(pg_partial, sum0, diff, diff);

            offset += remaining;
        }
    }

    // Process 4 chunks at a time in the main loop
    auto chunk_size = 4 * chunk;
    const size_t number_of_chunks =
        (dimension - (partial_chunk ? dimension % chunk : 0)) / chunk_size;

    for (size_t i = 0; i < number_of_chunks; i++) {
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum0, chunk, min_val_vec, delta_vec);
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum1, chunk, min_val_vec, delta_vec);
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum2, chunk, min_val_vec, delta_vec);
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum3, chunk, min_val_vec, delta_vec);
    }

    // Handle remaining steps (0-3)
    if constexpr (additional_steps > 0) {
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum0, chunk, min_val_vec, delta_vec);
    }
    if constexpr (additional_steps > 1) {
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum1, chunk, min_val_vec, delta_vec);
    }
    if constexpr (additional_steps > 2) {
        L2StepSQ8_FP32_SVE(pVect1, pVect2, offset, sum2, chunk, min_val_vec, delta_vec);
    }

    // Combine the accumulators
    svfloat32_t sum = svadd_f32_z(pg, sum0, sum1);
    sum = svadd_f32_z(pg, sum, sum2);
    sum = svadd_f32_z(pg, sum, sum3);

    // Horizontal sum to get Σ(diff_i²)
    return svaddv_f32(pg, sum);
}
} // namespace
