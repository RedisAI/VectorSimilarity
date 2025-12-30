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
 * SQ8-to-SQ8 distance functions using ARM SVE with precomputed sum and norm.
 * These functions compute distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses precomputed sum and norm stored in the vector data,
 * eliminating the need to compute them during distance calculation.
 *
 * Uses algebraic optimization with SVE dot product instruction:
 *
 * With sum = Σv[i] (sum of original float values), the formula is:
 * IP = min1*sum2 + min2*sum1 - dim*min1*min2 + δ1*δ2 * Σ(q1[i]*q2[i])
 *
 * Since sum is precomputed, we only need to compute the dot product Σ(q1[i]*q2[i]).
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)] [norm
 * (float)]
 */

// Helper function to perform inner product step using integer dot product (no sum computation)
static inline void SQ8_SQ8_InnerProductStep_SVE(const uint8_t *pVec1, const uint8_t *pVec2,
                                                size_t &offset, svuint32_t &dot_sum,
                                                const size_t chunk) {
    svbool_t pg = svptrue_b8();

    // Load uint8 vectors
    svuint8_t v1_u8 = svld1_u8(pg, pVec1 + offset);
    svuint8_t v2_u8 = svld1_u8(pg, pVec2 + offset);

    // Compute dot product using integer svdot instruction
    dot_sum = svdot_u32(dot_sum, v1_u8, v2_u8);

    offset += chunk;
}

// Common implementation for inner product between two SQ8 vectors with precomputed sum/norm
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_SQ8_InnerProductSIMD_SVE_IMP(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    size_t offset = 0;

    // Get dequantization parameters and precomputed values from the end of pVec1
    // Layout: [data (dim)] [min (float)] [delta (float)] [sum (float)] [norm (float)]
    const float *params1 = reinterpret_cast<const float *>(pVec1 + dimension);
    const float min1 = params1[0];
    const float delta1 = params1[1];
    const float sum1 = params1[2]; // Precomputed sum of original float elements

    // Get dequantization parameters and precomputed values from the end of pVec2
    const float *params2 = reinterpret_cast<const float *>(pVec2 + dimension);
    const float min2 = params2[0];
    const float delta2 = params2[1];
    const float sum2 = params2[2]; // Precomputed sum of original float elements

    // Get the vector length for uint8 elements
    const size_t vl = svcntb();

    // Calculate number of complete 4-chunk groups
    size_t number_of_chunks = dimension / (vl * 4);

    // Multiple accumulators for ILP (dot product only)
    svuint32_t dot_sum0 = svdup_u32(0);
    svuint32_t dot_sum1 = svdup_u32(0);
    svuint32_t dot_sum2 = svdup_u32(0);
    svuint32_t dot_sum3 = svdup_u32(0);

    for (size_t i = 0; i < number_of_chunks; i++) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum0, vl);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum1, vl);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum2, vl);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum3, vl);
    }

    // Handle remaining steps (0-3 complete chunks)
    if constexpr (additional_steps >= 1) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum0, vl);
    }
    if constexpr (additional_steps >= 2) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum1, vl);
    }
    if constexpr (additional_steps >= 3) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum2, vl);
    }

    // Handle partial chunk if needed
    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b8(offset, dimension);
        svuint8_t v1_u8 = svld1_u8(pg, pVec1 + offset);
        svuint8_t v2_u8 = svld1_u8(pg, pVec2 + offset);
        dot_sum3 = svdot_u32(dot_sum3, v1_u8, v2_u8);
    }

    // Combine all accumulators
    svuint32_t dot_total = svadd_u32_x(svptrue_b32(), dot_sum0, dot_sum1);
    dot_total = svadd_u32_x(svptrue_b32(), dot_total, dot_sum2);
    dot_total = svadd_u32_x(svptrue_b32(), dot_total, dot_sum3);

    // Horizontal sum to scalar integer
    svbool_t pg32 = svptrue_b32();
    uint32_t dot_product = svaddv_u32(pg32, dot_total);

    // Apply algebraic formula with float conversion only at the end:
    // IP = min1*sum2 + min2*sum1 - dim*min1*min2 + δ1*δ2 * Σ(q1*q2)
    return min1 * sum2 + min2 * sum1 - static_cast<float>(dimension) * min1 * min2 +
           delta1 * delta2 * static_cast<float>(dot_product);
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
