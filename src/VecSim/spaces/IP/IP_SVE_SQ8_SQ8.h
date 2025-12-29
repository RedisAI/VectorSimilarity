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
 * Uses algebraic optimization with INTEGER arithmetic throughout:
 *
 * IP = Σ (v1[i]*δ1 + min1) * (v2[i]*δ2 + min2)
 *    = δ1*δ2 * Σ(v1[i]*v2[i]) + δ1*min2 * Σv1[i] + δ2*min1 * Σv2[i] + dim*min1*min2
 *
 * All sums are computed using integer dot product instructions, converted to float only at the end.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)]]
 */

// Helper function to perform inner product step using integer dot product
static inline void SQ8_SQ8_InnerProductStep_SVE(const uint8_t *pVec1, const uint8_t *pVec2,
                                                size_t &offset, svuint32_t &dot_sum,
                                                svuint32_t &sum1, svuint32_t &sum2,
                                                const size_t chunk) {
    svbool_t pg = svptrue_b8();

    // Load uint8 vectors
    svuint8_t v1_u8 = svld1_u8(pg, pVec1 + offset);
    svuint8_t v2_u8 = svld1_u8(pg, pVec2 + offset);

    // Compute dot product using integer svdot instruction
    dot_sum = svdot_u32(dot_sum, v1_u8, v2_u8);

    // Compute element sums using dot product with ones vector
    svuint8_t ones = svdup_u8(1);
    sum1 = svdot_u32(sum1, v1_u8, ones);
    sum2 = svdot_u32(sum2, v2_u8, ones);

    offset += chunk;
}

// Common implementation for inner product between two SQ8 vectors
// Uses integer arithmetic throughout for maximum performance
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

    // Get the number of 8-bit elements per vector at runtime
    const size_t vl = svcntb();
    const size_t chunk_size = 4 * vl;

    // Integer accumulators for dot product and element sums
    svuint32_t dot_sum0 = svdup_u32(0);
    svuint32_t dot_sum1 = svdup_u32(0);
    svuint32_t dot_sum2 = svdup_u32(0);
    svuint32_t dot_sum3 = svdup_u32(0);
    svuint32_t sum1_0 = svdup_u32(0);
    svuint32_t sum1_1 = svdup_u32(0);
    svuint32_t sum1_2 = svdup_u32(0);
    svuint32_t sum1_3 = svdup_u32(0);
    svuint32_t sum2_0 = svdup_u32(0);
    svuint32_t sum2_1 = svdup_u32(0);
    svuint32_t sum2_2 = svdup_u32(0);
    svuint32_t sum2_3 = svdup_u32(0);

    // Process 4 chunks at a time in the main loop
    const size_t number_of_chunks = dimension / chunk_size;

    for (size_t i = 0; i < number_of_chunks; i++) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum0, sum1_0, sum2_0, vl);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum1, sum1_1, sum2_1, vl);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum2, sum1_2, sum2_2, vl);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum3, sum1_3, sum2_3, vl);
    }

    // Handle remaining steps (0-3 complete chunks)
    if constexpr (additional_steps >= 1) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum0, sum1_0, sum2_0, vl);
    }
    if constexpr (additional_steps >= 2) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum1, sum1_1, sum2_1, vl);
    }
    if constexpr (additional_steps >= 3) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, dot_sum2, sum1_2, sum2_2, vl);
    }

    // Handle partial chunk if needed
    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b8_u64(offset, dimension);
        svuint8_t v1_u8 = svld1_u8(pg, pVec1 + offset);
        svuint8_t v2_u8 = svld1_u8(pg, pVec2 + offset);

        // Compute dot product and sums (inactive lanes are already zeroed by svld1)
        dot_sum3 = svdot_u32(dot_sum3, v1_u8, v2_u8);
        svuint8_t ones = svdup_u8(1);
        sum1_3 = svdot_u32(sum1_3, v1_u8, ones);
        sum2_3 = svdot_u32(sum2_3, v2_u8, ones);
    }

    // Combine the integer accumulators
    svbool_t pg32 = svptrue_b32();
    svuint32_t dot_total = svadd_u32_x(pg32, svadd_u32_x(pg32, dot_sum0, dot_sum1),
                                       svadd_u32_x(pg32, dot_sum2, dot_sum3));
    svuint32_t sum1_total =
        svadd_u32_x(pg32, svadd_u32_x(pg32, sum1_0, sum1_1), svadd_u32_x(pg32, sum1_2, sum1_3));
    svuint32_t sum2_total =
        svadd_u32_x(pg32, svadd_u32_x(pg32, sum2_0, sum2_1), svadd_u32_x(pg32, sum2_2, sum2_3));

    // Horizontal sum to scalar integers
    uint32_t dot_product = svaddv_u32(pg32, dot_total);
    uint32_t v1_sum = svaddv_u32(pg32, sum1_total);
    uint32_t v2_sum = svaddv_u32(pg32, sum2_total);

    // Apply algebraic formula with float conversion only at the end:
    // IP = δ1*δ2 * Σ(v1*v2) + δ1*min2 * Σv1 + δ2*min1 * Σv2 + dim*min1*min2
    return delta1 * delta2 * static_cast<float>(dot_product) +
           delta1 * min2 * static_cast<float>(v1_sum) + delta2 * min1 * static_cast<float>(v2_sum) +
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
