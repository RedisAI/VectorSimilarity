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
 * SQ8-to-SQ8 L2 squared distance functions for SVE.
 * Computes L2 squared distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses algebraic optimization with INTEGER arithmetic throughout:
 *
 * L2² = Σ((q1[i]*δ1 + min1) - (q2[i]*δ2 + min2))²
 *
 * Let c = min1 - min2, then:
 * L2² = δ1²*Σq1² + δ2²*Σq2² - 2*δ1*δ2*Σ(q1*q2) + 2*c*δ1*Σq1 - 2*c*δ2*Σq2 + dim*c²
 *
 * The vector's sum (Σq) and sum of squares (Σq²) are precomputed and stored in the vector data.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)] [sum_of_squares (float)]
 */

// Helper function to perform L2 squared step using integer dot product - only computes dot product now
static inline void SQ8_SQ8_L2SqrStep_SVE(const uint8_t *pVec1, const uint8_t *pVec2, size_t &offset,
                                         svuint32_t &dot_sum, const size_t chunk) {
    svbool_t pg = svptrue_b8();

    // Load uint8 vectors
    svuint8_t v1_u8 = svld1_u8(pg, pVec1 + offset);
    svuint8_t v2_u8 = svld1_u8(pg, pVec2 + offset);

    // Compute dot product: q1*q2
    dot_sum = svdot_u32(dot_sum, v1_u8, v2_u8);

    offset += chunk;
}

// Common implementation for L2 squared distance between two SQ8 vectors
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_SQ8_L2SqrSIMD_SVE(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    size_t offset = 0;

    // Get dequantization parameters and precomputed values from the end of pVec1
    // Layout: [uint8_t values (dim)] [min_val] [delta] [sum] [sum_of_squares]
    const float *params1 = reinterpret_cast<const float *>(pVec1 + dimension);
    const float min1 = params1[0];
    const float delta1 = params1[1];
    const float sum_v1 = params1[2];
    const float sum_sqr1 = params1[3];

    // Get dequantization parameters and precomputed values from the end of pVec2
    const float *params2 = reinterpret_cast<const float *>(pVec2 + dimension);
    const float min2 = params2[0];
    const float delta2 = params2[1];
    const float sum_v2 = params2[2];
    const float sum_sqr2 = params2[3];

    // Get the number of 8-bit elements per vector at runtime
    const size_t vl = svcntb();
    const size_t chunk_size = 4 * vl;

    // Integer accumulators (4x for ILP) - only need dot product now
    svuint32_t dot_sum0 = svdup_u32(0), dot_sum1 = svdup_u32(0);
    svuint32_t dot_sum2 = svdup_u32(0), dot_sum3 = svdup_u32(0);

    // Process 4 chunks at a time
    const size_t number_of_chunks = dimension / chunk_size;
    for (size_t i = 0; i < number_of_chunks; i++) {
        SQ8_SQ8_L2SqrStep_SVE(pVec1, pVec2, offset, dot_sum0, vl);
        SQ8_SQ8_L2SqrStep_SVE(pVec1, pVec2, offset, dot_sum1, vl);
        SQ8_SQ8_L2SqrStep_SVE(pVec1, pVec2, offset, dot_sum2, vl);
        SQ8_SQ8_L2SqrStep_SVE(pVec1, pVec2, offset, dot_sum3, vl);
    }

    // Handle remaining steps (0-3 complete chunks)
    if constexpr (additional_steps >= 1) {
        SQ8_SQ8_L2SqrStep_SVE(pVec1, pVec2, offset, dot_sum0, vl);
    }
    if constexpr (additional_steps >= 2) {
        SQ8_SQ8_L2SqrStep_SVE(pVec1, pVec2, offset, dot_sum1, vl);
    }
    if constexpr (additional_steps >= 3) {
        SQ8_SQ8_L2SqrStep_SVE(pVec1, pVec2, offset, dot_sum2, vl);
    }

    // Handle partial chunk if needed
    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b8_u64(offset, dimension);
        svuint8_t v1_u8 = svld1_u8(pg, pVec1 + offset);
        svuint8_t v2_u8 = svld1_u8(pg, pVec2 + offset);

        dot_sum3 = svdot_u32(dot_sum3, v1_u8, v2_u8);
    }

    // Combine accumulators and reduce - only dot product needed
    svbool_t pg32 = svptrue_b32();
    svuint32_t dot_total = svadd_u32_x(pg32, svadd_u32_x(pg32, dot_sum0, dot_sum1),
                                       svadd_u32_x(pg32, dot_sum2, dot_sum3));

    // Horizontal sum to scalar integer
    uint32_t dot_product = svaddv_u32(pg32, dot_total);

    // Apply the algebraic formula:
    // L2² = δ1²*Σq1² + δ2²*Σq2² - 2*δ1*δ2*Σ(q1*q2) + 2*c*δ1*Σq1 - 2*c*δ2*Σq2 + dim*c²
    float c = min1 - min2;
    return delta1 * delta1 * sum_sqr1 + delta2 * delta2 * sum_sqr2 -
           2.0f * delta1 * delta2 * static_cast<float>(dot_product) +
           2.0f * c * delta1 * sum_v1 - 2.0f * c * delta2 * sum_v2 +
           static_cast<float>(dimension) * c * c;
}

