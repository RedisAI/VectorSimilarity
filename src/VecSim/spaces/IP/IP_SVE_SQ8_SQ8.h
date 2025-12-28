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
#include <arm_sve.h>

/**
 * SQ8-to-SQ8 distance functions for SVE.
 * These functions compute distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses algebraic optimization to reduce operations per element:
 *
 * IP = Σ (v1[i]*δ1 + min1) * (v2[i]*δ2 + min2)
 *    = δ1*δ2 * Σ(v1[i]*v2[i]) + δ1*min2 * Σv1[i] + δ2*min1 * Σv2[i] + dim*min1*min2
 *
 * This saves 2 FMAs per chunk by deferring dequantization to scalar math at the end.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [inv_norm (float)]
 */

// Helper function to perform inner product step with algebraic optimization
static inline void SQ8_SQ8_InnerProductStep_SVE(const uint8_t *pVec1, const uint8_t *pVec2,
                                                size_t offset, svfloat32_t &dot_sum,
                                                svfloat32_t &sum1, svfloat32_t &sum2,
                                                const size_t chunk) {
    svbool_t pg = svptrue_b32();

    // Load uint8 elements from pVec1 and convert to float
    svuint32_t v1_u32 = svld1ub_u32(pg, pVec1 + offset);
    svfloat32_t v1_f = svcvt_f32_u32_x(pg, v1_u32);

    // Load uint8 elements from pVec2 and convert to float
    svuint32_t v2_u32 = svld1ub_u32(pg, pVec2 + offset);
    svfloat32_t v2_f = svcvt_f32_u32_x(pg, v2_u32);

    // Accumulate dot product: dot_sum += v1 * v2 (no dequantization)
    dot_sum = svmla_f32_x(pg, dot_sum, v1_f, v2_f);

    // Accumulate element sums
    sum1 = svadd_f32_x(pg, sum1, v1_f);
    sum2 = svadd_f32_x(pg, sum2, v2_f);
}

// Common implementation for inner product between two SQ8 vectors
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_SQ8_InnerProductSIMD_SVE_IMP(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    size_t offset = 0;

    // Get dequantization parameters from the end of pVec1
    const float min1 = *reinterpret_cast<const float *>(pVec1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVec1 + dimension + sizeof(float));

    // Get dequantization parameters from the end of pVec2
    const float min2 = *reinterpret_cast<const float *>(pVec2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVec2 + dimension + sizeof(float));

    svbool_t pg = svptrue_b32();

    // Get the number of 32-bit elements per vector at runtime
    uint64_t chunk = svcntw();

    // Multiple accumulators for instruction-level parallelism
    // dot_sum: accumulates v1[i] * v2[i]
    // sum1: accumulates v1[i]
    // sum2: accumulates v2[i]
    svfloat32_t dot_sum0 = svdup_f32(0.0f);
    svfloat32_t dot_sum1 = svdup_f32(0.0f);
    svfloat32_t dot_sum2 = svdup_f32(0.0f);
    svfloat32_t dot_sum3 = svdup_f32(0.0f);
    svfloat32_t sum1_0 = svdup_f32(0.0f);
    svfloat32_t sum1_1 = svdup_f32(0.0f);
    svfloat32_t sum1_2 = svdup_f32(0.0f);
    svfloat32_t sum1_3 = svdup_f32(0.0f);
    svfloat32_t sum2_0 = svdup_f32(0.0f);
    svfloat32_t sum2_1 = svdup_f32(0.0f);
    svfloat32_t sum2_2 = svdup_f32(0.0f);
    svfloat32_t sum2_3 = svdup_f32(0.0f);

    // Handle partial chunk if needed
    if constexpr (partial_chunk) {
        size_t remaining = dimension % chunk;
        if (remaining > 0) {
            // Create predicate for the remaining elements
            svbool_t pg_partial =
                svwhilelt_b32(static_cast<uint32_t>(0), static_cast<uint32_t>(remaining));

            // Load and convert v1 elements
            svuint32_t v1_u32 = svld1ub_u32(pg_partial, pVec1 + offset);
            svfloat32_t v1_f = svcvt_f32_u32_z(pg_partial, v1_u32);

            // Load and convert v2 elements
            svuint32_t v2_u32 = svld1ub_u32(pg_partial, pVec2 + offset);
            svfloat32_t v2_f = svcvt_f32_u32_z(pg_partial, v2_u32);

            // Accumulate dot product (no dequantization)
            dot_sum0 = svmla_f32_z(pg_partial, dot_sum0, v1_f, v2_f);

            // Accumulate element sums
            sum1_0 = svadd_f32_z(pg_partial, sum1_0, v1_f);
            sum2_0 = svadd_f32_z(pg_partial, sum2_0, v2_f);

            // Move past the partial chunk
            offset += remaining;
        }
    }

    // Process 4 chunks at a time in the main loop
    auto chunk_size = 4 * chunk;
    const size_t number_of_chunks =
        (dimension - (partial_chunk ? dimension % chunk : 0)) / chunk_size;

    for (size_t i = 0; i < number_of_chunks; i++) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum0, sum1_0, sum2_0, chunk);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset + chunk, dot_sum1, sum1_1, sum2_1, chunk);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset + 2 * chunk, dot_sum2, sum1_2, sum2_2,
                                     chunk);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset + 3 * chunk, dot_sum3, sum1_3, sum2_3,
                                     chunk);
        offset += chunk_size;
    }

    // Handle remaining steps (0-3)
    if constexpr (additional_steps > 0) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum0, sum1_0, sum2_0, chunk);
        offset += chunk;
    }
    if constexpr (additional_steps > 1) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum1, sum1_1, sum2_1, chunk);
        offset += chunk;
    }
    if constexpr (additional_steps > 2) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum2, sum1_2, sum2_2, chunk);
    }

    // Combine the accumulators
    svfloat32_t dot_total = svadd_f32_x(pg, svadd_f32_x(pg, dot_sum0, dot_sum1),
                                        svadd_f32_x(pg, dot_sum2, dot_sum3));
    svfloat32_t sum1_total = svadd_f32_x(pg, svadd_f32_x(pg, sum1_0, sum1_1),
                                         svadd_f32_x(pg, sum1_2, sum1_3));
    svfloat32_t sum2_total = svadd_f32_x(pg, svadd_f32_x(pg, sum2_0, sum2_1),
                                         svadd_f32_x(pg, sum2_2, sum2_3));

    // Horizontal sum of all elements
    float dot_product = svaddv_f32(pg, dot_total);
    float v1_sum = svaddv_f32(pg, sum1_total);
    float v2_sum = svaddv_f32(pg, sum2_total);

    // Apply algebraic formula:
    // IP = δ1*δ2 * Σ(v1*v2) + δ1*min2 * Σv1 + δ2*min1 * Σv2 + dim*min1*min2
    return delta1 * delta2 * dot_product + delta1 * min2 * v1_sum + delta2 * min1 * v2_sum +
           static_cast<float>(dimension) * min1 * min2;
}

// SQ8-to-SQ8 Inner Product distance function
// Returns 1 - inner_product (distance form)
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_SQ8_InnerProductSIMD_SVE(const void *pVec1v, const void *pVec2v, size_t dimension) {
    return 1.0f - SQ8_SQ8_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(pVec1v, pVec2v,
                                                                                    dimension);
}

// SQ8-to-SQ8 Cosine distance function
// Returns 1 - inner_product (assumes vectors are pre-normalized)
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_SQ8_CosineSIMD_SVE(const void *pVec1v, const void *pVec2v, size_t dimension) {
    return 1.0f - SQ8_SQ8_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(pVec1v, pVec2v,
                                                                                    dimension);
}
