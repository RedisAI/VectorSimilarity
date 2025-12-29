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
 * SQ8-to-SQ8 distance functions using AVX512 VNNI.
 * These functions compute distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized.
 *
 * Uses algebraic optimization to leverage integer VNNI instructions:
 *
 * IP = Σ (v1[i]*δ1 + min1) * (v2[i]*δ2 + min2)
 *    = δ1*δ2 * Σ(v1[i]*v2[i]) + δ1*min2 * Σv1[i] + δ2*min1 * Σv2[i] + dim*min1*min2
 * TODO: Can store the vector's norm and sum of elements in the vector data, and use it here.
 *
 * This allows using VNNI's _mm512_dpwssd_epi32 for efficient integer dot product,
 * then applying scalar corrections at the end.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)]]
 */

// Process 64 uint8 elements using VNNI with multiple accumulators for ILP
static inline void SQ8_SQ8_InnerProductStep64(const uint8_t *pVec1, const uint8_t *pVec2,
                                              __m512i &dot_acc0, __m512i &dot_acc1,
                                              __m512i &sum1_acc, __m512i &sum2_acc) {
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

    // Sum of elements using SAD with zero (sums bytes in groups of 8 -> 8x 64-bit results)
    __m512i zero = _mm512_setzero_si512();
    sum1_acc = _mm512_add_epi64(sum1_acc, _mm512_sad_epu8(v1_full, zero));
    sum2_acc = _mm512_add_epi64(sum2_acc, _mm512_sad_epu8(v2_full, zero));
}

// Process 32 uint8 elements using VNNI
static inline void SQ8_SQ8_InnerProductStep32(const uint8_t *pVec1, const uint8_t *pVec2,
                                              __m512i &dot_acc, __m512i &sum1_acc,
                                              __m512i &sum2_acc) {
    // Load 32 bytes from each vector
    __m256i v1_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pVec1));
    __m256i v2_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pVec2));

    // Convert to int16 (zero-extend) and compute dot product using VNNI
    dot_acc =
        _mm512_dpwssd_epi32(dot_acc, _mm512_cvtepu8_epi16(v1_256), _mm512_cvtepu8_epi16(v2_256));

    // Sum of elements - extend to 512-bit and use SAD
    // Use zextsi256_si512 to properly zero the upper half
    __m512i v1_full = _mm512_zextsi256_si512(v1_256);
    __m512i v2_full = _mm512_zextsi256_si512(v2_256);
    __m512i zero = _mm512_setzero_si512();
    sum1_acc = _mm512_add_epi64(sum1_acc, _mm512_sad_epu8(v1_full, zero));
    sum2_acc = _mm512_add_epi64(sum2_acc, _mm512_sad_epu8(v2_full, zero));
}

// Common implementation for inner product between two SQ8 vectors
template <unsigned char residual> // 0..63
float SQ8_SQ8_InnerProductImp(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    const uint8_t *pEnd1 = pVec1 + dimension;

    // Get dequantization parameters from the end of pVec1
    const float min1 = *reinterpret_cast<const float *>(pVec1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVec1 + dimension + sizeof(float));

    // Get dequantization parameters from the end of pVec2
    const float min2 = *reinterpret_cast<const float *>(pVec2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVec2 + dimension + sizeof(float));

    // Multiple accumulators for instruction-level parallelism
    __m512i dot_acc0 = _mm512_setzero_si512();
    __m512i dot_acc1 = _mm512_setzero_si512();
    __m512i sum1_acc = _mm512_setzero_si512(); // Sum of v1 elements
    __m512i sum2_acc = _mm512_setzero_si512(); // Sum of v2 elements

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

            // Sum using SAD (masked load already zeroed unused bytes)
            __m512i v1_full = _mm512_zextsi256_si512(v1_256);
            __m512i v2_full = _mm512_zextsi256_si512(v2_256);
            __m512i zero = _mm512_setzero_si512();
            sum1_acc = _mm512_sad_epu8(v1_full, zero);
            sum2_acc = _mm512_sad_epu8(v2_full, zero);
        } else if constexpr (residual == 32) {
            // Exactly 32 elements
            SQ8_SQ8_InnerProductStep32(pVec1, pVec2, dot_acc0, sum1_acc, sum2_acc);
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

            // Sum using SAD (masked load already zeroed unused bytes)
            __m512i zero = _mm512_setzero_si512();
            sum1_acc = _mm512_sad_epu8(v1_full, zero);
            sum2_acc = _mm512_sad_epu8(v2_full, zero);
        }
        pVec1 += residual;
        pVec2 += residual;
    }

    // Process full 64-byte chunks
    while (pVec1 < pEnd1) {
        SQ8_SQ8_InnerProductStep64(pVec1, pVec2, dot_acc0, dot_acc1, sum1_acc, sum2_acc);
        pVec1 += 64;
        pVec2 += 64;
    }

    // Combine dot product accumulators and reduce
    __m512i dot_total = _mm512_add_epi32(dot_acc0, dot_acc1);
    int64_t dot_product = _mm512_reduce_add_epi32(dot_total);

    // Reduce sum accumulators (SAD produces 8 x 64-bit sums)
    int64_t sum_v1 = _mm512_reduce_add_epi64(sum1_acc);
    int64_t sum_v2 = _mm512_reduce_add_epi64(sum2_acc);

    // Apply the algebraic formula:
    // IP = δ1*δ2 * Σ(v1[i]*v2[i]) + δ1*min2 * Σv1[i] + δ2*min1 * Σv2[i] + dim*min1*min2
    float result = delta1 * delta2 * static_cast<float>(dot_product) +
                   delta1 * min2 * static_cast<float>(sum_v1) +
                   delta2 * min1 * static_cast<float>(sum_v2) +
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
    return 1.0f - SQ8_SQ8_InnerProductImp<residual>(pVec1v, pVec2v, dimension);
}
