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
 * SQ8 distance functions (float32 query vs uint8 stored) using AVX512.
 *
 * Uses algebraic optimization to reduce operations per element:
 *
 * IP = Σ query[i] * (val[i] * δ + min)
 *    = δ * Σ(query[i] * val[i]) + min * Σ(query[i])
 *
 * This saves one FMA per 16 elements by separating:
 * - dot_sum: accumulates query[i] * val[i]
 * - query_sum: accumulates query[i]
 * Then combines at the end: result = δ * dot_sum + min * query_sum
 *
 * Also uses multiple accumulators for better instruction-level parallelism.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)]
 */

// Process 16 elements with algebraic optimization
static inline void SQ8_InnerProductStep(const float *pVec1, const uint8_t *pVec2, __m512 &dot_sum,
                                        __m512 &query_sum) {
    // Load 16 float elements from query
    __m512 v1 = _mm512_loadu_ps(pVec1);

    // Load 16 uint8 elements and convert to float
    __m128i v2_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec2));
    __m512i v2_512 = _mm512_cvtepu8_epi32(v2_128);
    __m512 v2_f = _mm512_cvtepi32_ps(v2_512);

    // Accumulate query * val (without dequantization)
    dot_sum = _mm512_fmadd_ps(v1, v2_f, dot_sum);

    // Accumulate query sum
    query_sum = _mm512_add_ps(query_sum, v1);
}

// Common implementation for both inner product and cosine similarity
template <unsigned char residual> // 0..15
float SQ8_InnerProductImp_AVX512(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const float *pVec1 = static_cast<const float *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);

    // Get dequantization parameters from the end of pVec2
    const float min_val = *reinterpret_cast<const float *>(pVec2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVec2 + dimension + sizeof(float));

    // Multiple accumulators for instruction-level parallelism
    __m512 dot_sum0 = _mm512_setzero_ps();
    __m512 dot_sum1 = _mm512_setzero_ps();
    __m512 dot_sum2 = _mm512_setzero_ps();
    __m512 dot_sum3 = _mm512_setzero_ps();
    __m512 query_sum0 = _mm512_setzero_ps();
    __m512 query_sum1 = _mm512_setzero_ps();
    __m512 query_sum2 = _mm512_setzero_ps();
    __m512 query_sum3 = _mm512_setzero_ps();

    size_t offset = 0;

    // Deal with remainder first
    if constexpr (residual > 0) {
        // Handle less than 16 elements
        __mmask16 mask = (1U << residual) - 1;

        // Load masked float elements from query
        __m512 v1 = _mm512_maskz_loadu_ps(mask, pVec1);

        // Load uint8 elements and convert to float
        __m128i v2_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec2));
        __m512i v2_512 = _mm512_cvtepu8_epi32(v2_128);
        __m512 v2_f = _mm512_cvtepi32_ps(v2_512);

        // Masked accumulation (mask already zeroed unused elements in v1)
        dot_sum0 = _mm512_mul_ps(v1, v2_f);
        query_sum0 = v1;

        offset = residual;
    }

    // Calculate number of full 64-element chunks (4 x 16)
    size_t num_chunks = (dimension - residual) / 64;

    // Process 4 chunks at a time for maximum ILP
    for (size_t i = 0; i < num_chunks; i++) {
        SQ8_InnerProductStep(pVec1 + offset, pVec2 + offset, dot_sum0, query_sum0);
        SQ8_InnerProductStep(pVec1 + offset + 16, pVec2 + offset + 16, dot_sum1, query_sum1);
        SQ8_InnerProductStep(pVec1 + offset + 32, pVec2 + offset + 32, dot_sum2, query_sum2);
        SQ8_InnerProductStep(pVec1 + offset + 48, pVec2 + offset + 48, dot_sum3, query_sum3);
        offset += 64;
    }

    // Handle remaining 16-element chunks (0-3 remaining)
    size_t remaining = (dimension - residual) % 64;
    if (remaining >= 16) {
        SQ8_InnerProductStep(pVec1 + offset, pVec2 + offset, dot_sum0, query_sum0);
        offset += 16;
        remaining -= 16;
    }
    if (remaining >= 16) {
        SQ8_InnerProductStep(pVec1 + offset, pVec2 + offset, dot_sum1, query_sum1);
        offset += 16;
        remaining -= 16;
    }
    if (remaining >= 16) {
        SQ8_InnerProductStep(pVec1 + offset, pVec2 + offset, dot_sum2, query_sum2);
    }

    // Combine accumulators
    __m512 dot_total =
        _mm512_add_ps(_mm512_add_ps(dot_sum0, dot_sum1), _mm512_add_ps(dot_sum2, dot_sum3));
    __m512 query_total =
        _mm512_add_ps(_mm512_add_ps(query_sum0, query_sum1), _mm512_add_ps(query_sum2, query_sum3));

    // Reduce to scalar
    float dot_product = _mm512_reduce_add_ps(dot_total);
    float query_sum = _mm512_reduce_add_ps(query_total);

    // Apply algebraic formula: IP = δ * Σ(query*val) + min * Σ(query)
    return delta * dot_product + min_val * query_sum;
}

template <unsigned char residual> // 0..15
float SQ8_InnerProductSIMD16_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                                size_t dimension) {
    // The inner product similarity is 1 - ip
    return 1.0f -SQ8_InnerProductImp_AVX512<residual>(pVec1v, pVec2v, dimension);;
}

template <unsigned char residual> // 0..15
float SQ8_CosineSIMD16_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                          size_t dimension) {
    // Assume vectors are normalized.
    return SQ8_InnerProductSIMD16_AVX512F_BW_VL_VNNI<residual>(pVec1v, pVec2v, dimension);
}
