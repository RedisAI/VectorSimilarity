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
#include <immintrin.h>

/**
 * SQ8-to-SQ8 distance functions using AVX512 VNNI with precomputed sum.
 * These functions compute distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses precomputed sum stored in the vector data,
 * eliminating the need to compute them during distance calculation.
 *
 * Uses algebraic optimization to leverage integer VNNI instructions:
 *
 * With sum = Σv[i] (sum of original float values), the formula is:
 * IP = min1*sum2 + min2*sum1 + δ1*δ2 * Σ(q1[i]*q2[i]) - dim*min1*min2
 *
 * Since sum is precomputed, we only need to compute the dot product Σ(q1[i]*q2[i]).
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)]
 */

// Process 64 uint8 elements using VNNI with multiple accumulators for ILP (dot product only)
static inline void SQ8_SQ8_InnerProductStep64(const uint8_t *pVec1, const uint8_t *pVec2,
                                              __m512i &dot_acc0, __m512i &dot_acc1) {
    // Load 64 bytes from each vector
    __m512i v1_full = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(pVec1));
    __m512i v2_full = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(pVec2));

    // Extract lower and upper 256-bit halves
    __m256i v1_lo = _mm512_castsi512_si256(v1_full);
    __m256i v1_hi = _mm512_extracti64x4_epi64(v1_full, 1);
    __m256i v2_lo = _mm512_castsi512_si256(v2_full);
    __m256i v2_hi = _mm512_extracti64x4_epi64(v2_full, 1);

    // Convert to int16 (zero-extend) and compute dot products using VNNI
    // dpwssd: multiply pairs of int16, sum pairs to int32, accumulate
    dot_acc0 =
        _mm512_dpwssd_epi32(dot_acc0, _mm512_cvtepu8_epi16(v1_lo), _mm512_cvtepu8_epi16(v2_lo));
    dot_acc1 =
        _mm512_dpwssd_epi32(dot_acc1, _mm512_cvtepu8_epi16(v1_hi), _mm512_cvtepu8_epi16(v2_hi));
}

// Process 32 uint8 elements using VNNI (dot product only)
static inline void SQ8_SQ8_InnerProductStep32(const uint8_t *pVec1, const uint8_t *pVec2,
                                              __m512i &dot_acc) {
    // Load 32 bytes from each vector
    __m256i v1_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pVec1));
    __m256i v2_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pVec2));

    // Convert to int16 (zero-extend) and compute dot product using VNNI
    dot_acc =
        _mm512_dpwssd_epi32(dot_acc, _mm512_cvtepu8_epi16(v1_256), _mm512_cvtepu8_epi16(v2_256));
}

// Common implementation for inner product between two SQ8 vectors with precomputed sum
template <unsigned char residual> // 0..63
float SQ8_SQ8_InnerProductImp(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    const uint8_t *pEnd1 = pVec1 + dimension;

    // Get dequantization parameters and precomputed values from the end of pVec1
    // Layout: [data (dim)] [min (float)] [delta (float)] [sum (float)]
    const float *params1 = reinterpret_cast<const float *>(pVec1 + dimension);
    const float min1 = params1[0];
    const float delta1 = params1[1];
    const float sum1 = params1[2]; // Precomputed sum of original float elements

    // Get dequantization parameters and precomputed values from the end of pVec2
    const float *params2 = reinterpret_cast<const float *>(pVec2 + dimension);
    const float min2 = params2[0];
    const float delta2 = params2[1];
    const float sum2 = params2[2]; // Precomputed sum of original float elements

    // Multiple accumulators for instruction-level parallelism (dot product only)
    __m512i dot_acc0 = _mm512_setzero_si512();
    __m512i dot_acc1 = _mm512_setzero_si512();

    // Handle residual first (0..63 elements)
    if constexpr (residual > 0) {
        if constexpr (residual < 32) {
            // Handle less than 32 elements with mask
            constexpr __mmask32 mask = (1LU << residual) - 1;
            __m256i v1_256 = _mm256_maskz_loadu_epi8(mask, pVec1);
            __m256i v2_256 = _mm256_maskz_loadu_epi8(mask, pVec2);

            // Convert to int16 and compute dot product
            dot_acc0 = _mm512_dpwssd_epi32(dot_acc0, _mm512_cvtepu8_epi16(v1_256),
                                           _mm512_cvtepu8_epi16(v2_256));
        } else if constexpr (residual == 32) {
            // Exactly 32 elements
            SQ8_SQ8_InnerProductStep32(pVec1, pVec2, dot_acc0);
        } else {
            // 33-63 elements: use masked 64-byte load
            constexpr __mmask64 mask = (1LLU << residual) - 1;
            __m512i v1_full = _mm512_maskz_loadu_epi8(mask, pVec1);
            __m512i v2_full = _mm512_maskz_loadu_epi8(mask, pVec2);

            // Extract halves and compute dot products
            __m256i v1_lo = _mm512_castsi512_si256(v1_full);
            __m256i v1_hi = _mm512_extracti64x4_epi64(v1_full, 1);
            __m256i v2_lo = _mm512_castsi512_si256(v2_full);
            __m256i v2_hi = _mm512_extracti64x4_epi64(v2_full, 1);

            dot_acc0 = _mm512_dpwssd_epi32(dot_acc0, _mm512_cvtepu8_epi16(v1_lo),
                                           _mm512_cvtepu8_epi16(v2_lo));
            dot_acc1 = _mm512_dpwssd_epi32(dot_acc1, _mm512_cvtepu8_epi16(v1_hi),
                                           _mm512_cvtepu8_epi16(v2_hi));
        }
        pVec1 += residual;
        pVec2 += residual;
    }

    // Process full 64-byte chunks
    while (pVec1 < pEnd1) {
        SQ8_SQ8_InnerProductStep64(pVec1, pVec2, dot_acc0, dot_acc1);
        pVec1 += 64;
        pVec2 += 64;
    }

    // Combine dot product accumulators and reduce
    __m512i dot_total = _mm512_add_epi32(dot_acc0, dot_acc1);
    int64_t dot_product = _mm512_reduce_add_epi32(dot_total);

    // Apply the algebraic formula using precomputed sums:
    // IP = min1*sum2 + min2*sum1 + δ1*δ2 * Σ(q1[i]*q2[i]) - dim*min1*min2
    float result = min1 * sum2 + min2 * sum1 + delta1 * delta2 * static_cast<float>(dot_product) -
                   static_cast<float>(dimension) * min1 * min2;

    return result;
}

// SQ8-to-SQ8 Inner Product distance function
// Returns 1 - inner_product (distance form)
template <unsigned char residual> // 0..63
float SQ8_SQ8_InnerProductSIMD64_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                                    size_t dimension) {
    return 1.0f - SQ8_SQ8_InnerProductImp<residual>(pVec1v, pVec2v, dimension);
}

// SQ8-to-SQ8 Cosine distance function
// Returns 1 - (inner_product)
template <unsigned char residual> // 0..63
float SQ8_SQ8_CosineSIMD64_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                              size_t dimension) {
    // Assume vectors are normalized.
    return SQ8_SQ8_InnerProductSIMD64_AVX512F_BW_VL_VNNI<residual>(pVec1v, pVec2v, dimension);
}
