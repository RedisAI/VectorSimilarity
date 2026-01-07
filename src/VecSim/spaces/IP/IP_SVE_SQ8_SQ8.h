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
#include "VecSim/spaces/IP/IP_SVE_UINT8.h"
#include "VecSim/types/sq8.h"
#include <arm_sve.h>

using sq8 = vecsim_types::sq8;

/**
 * SQ8-to-SQ8 distance functions using ARM SVE with precomputed sum.
 * These functions compute distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses precomputed sum stored in the vector data,
 * eliminating the need to compute them during distance calculation.
 *
 * Uses algebraic optimization with SVE dot product instruction:
 *
 * With sum = Σv[i] (sum of original float values), the formula is:
 * IP = min1*sum2 + min2*sum1 + δ1*δ2 * Σ(q1[i]*q2[i]) - dim*min1*min2
 *
 * Since sum is precomputed, we only need to compute the dot product Σ(q1[i]*q2[i]).
 * The dot product is computed using the efficient UINT8_InnerProductImp which uses
 * SVE dot product instruction (svdot_u32) for native uint8 dot product computation.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)]
 */

// Common implementation for inner product between two SQ8 vectors with precomputed sum
// Uses UINT8_InnerProductImp for efficient dot product computation with SVE
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_SQ8_InnerProductSIMD_SVE_IMP(const void *pVec1v, const void *pVec2v, size_t dimension) {
    // Compute raw dot product using efficient UINT8 SVE implementation
    // UINT8_InnerProductImp uses svdot_u32 for native uint8 dot product
    float dot_product =
        UINT8_InnerProductImp<partial_chunk, additional_steps>(pVec1v, pVec2v, dimension);

    // Get dequantization parameters and precomputed values from the end of vectors
    // Layout: [data (dim)] [min (float)] [delta (float)] [sum (float)]
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);

    const float *params1 = reinterpret_cast<const float *>(pVec1 + dimension);
    const float min1 = params1[sq8::MIN_VAL];
    const float delta1 = params1[sq8::DELTA];
    const float sum1 = params1[sq8::SUM]; // Precomputed sum of original float elements

    const float *params2 = reinterpret_cast<const float *>(pVec2 + dimension);
    const float min2 = params2[sq8::MIN_VAL];
    const float delta2 = params2[sq8::DELTA];
    const float sum2 = params2[sq8::SUM]; // Precomputed sum of original float elements

    // Apply algebraic formula with float conversion only at the end:
    // IP = min1*sum2 + min2*sum1 + δ1*δ2 * Σ(q1*q2) - dim*min1*min2
    return min1 * sum2 + min2 * sum1 + delta1 * delta2 * dot_product -
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
    // Assume vectors are normalized.
    return SQ8_SQ8_InnerProductSIMD_SVE<partial_chunk, additional_steps>(pVec1v, pVec2v, dimension);
}
