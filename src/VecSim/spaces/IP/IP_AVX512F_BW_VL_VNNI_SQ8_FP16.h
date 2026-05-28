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
#include "VecSim/types/float16.h"
#include "VecSim/utils/alignment.h"
#include <immintrin.h>

using sq8 = vecsim_types::sq8;
using float16 = vecsim_types::float16;

// Helper: load 16 SQ8 + 16 FP16 lanes, widen both to FP32, fused-multiply-add into sum.
static inline void SQ8_FP16_InnerProductStep_AVX512(const uint8_t *&pVec1, const float16 *&pVec2,
                                                    __m512 &sum) {
    // 16 uint8 -> 16 fp32
    __m128i v1_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec1));
    __m512i v1_512 = _mm512_cvtepu8_epi32(v1_128);
    __m512 v1_f = _mm512_cvtepi32_ps(v1_512);

    // 16 fp16 -> 16 fp32. _mm512_cvtph_ps is part of AVX512F.
    __m256i v2_16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pVec2));
    __m512 v2_f = _mm512_cvtph_ps(v2_16);

    sum = _mm512_fmadd_ps(v1_f, v2_f, sum);

    pVec1 += 16;
    pVec2 += 16;
}

// Raw inner product Σ((min + delta * q_i) * y_i). Used by both InnerProduct/Cosine wrappers
// and by the L2 kernel.
// Precondition: dim >= 16. Caller is the dispatcher in IP_space.cpp / L2_space.cpp, which gates
// this. The residual block reads 16 SQ8 bytes and 32 FP16 bytes unconditionally; shorter blobs
// would under-read.
template <unsigned char residual> // 0..15
float SQ8_FP16_InnerProductImp_AVX512(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v); // SQ8 storage
    const float16 *pVec2 = static_cast<const float16 *>(pVec2v); // FP16 query
    const uint8_t *pEnd1 = pVec1 + dimension;

    // Four independent accumulators break the FMA dependency chain so the inner loop can
    // saturate both FMA ports on Sapphire Rapids / Zen 4.
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    if constexpr (residual > 0) {
        __mmask16 mask = (1U << residual) - 1;

        __m128i v1_128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVec1));
        __m512i v1_512 = _mm512_cvtepu8_epi32(v1_128);
        __m512 v1_f = _mm512_cvtepi32_ps(v1_512);

        // Safe to read the full 32-byte FP16 chunk: dim >= 16 and the FP16 metadata follows
        // the lanes, so the load stays within the query blob.
        __m256i v2_16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pVec2));
        __m512 v2_f = _mm512_cvtph_ps(v2_16);

        // Mask out unused lanes by folding the mask into the multiply.
        sum0 = _mm512_maskz_mul_ps(mask, v1_f, v2_f);

        pVec1 += residual;
        pVec2 += residual;
    }

    // Main unrolled loop: 4 chunks of 16 lanes per iteration, one chunk per accumulator.
    // Residual leaves `dim - residual` lanes remaining (a multiple of 16), so the
    // pointer comparison stays exact.
    while (pVec1 + 64 <= pEnd1) {
        SQ8_FP16_InnerProductStep_AVX512(pVec1, pVec2, sum0);
        SQ8_FP16_InnerProductStep_AVX512(pVec1, pVec2, sum1);
        SQ8_FP16_InnerProductStep_AVX512(pVec1, pVec2, sum2);
        SQ8_FP16_InnerProductStep_AVX512(pVec1, pVec2, sum3);
    }

    // Reduce the four accumulators into one.
    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));

    // Tail: at most three remaining 16-lane chunks.
    while (pVec1 < pEnd1) {
        SQ8_FP16_InnerProductStep_AVX512(pVec1, pVec2, sum);
    }

    float quantized_dot = _mm512_reduce_add_ps(sum);

    // SQ8 metadata starts at byte offset `dimension`; for odd `dimension` it is not
    // 4-byte aligned, so use load_unaligned. Mirrors the scalar SQ8_FP16_Impl pattern.
    const uint8_t *pVec1Base = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *params_bytes = pVec1Base + dimension;
    const float min_val = load_unaligned<float>(params_bytes + sq8::MIN_VAL * sizeof(float));
    const float delta = load_unaligned<float>(params_bytes + sq8::DELTA * sizeof(float));

    // FP16 query metadata sits at byte offset 2*dimension; for odd `dimension` it is
    // 2-byte aligned only.
    const float16 *pVec2Base = static_cast<const float16 *>(pVec2v);
    const auto *query_meta_bytes = reinterpret_cast<const uint8_t *>(pVec2Base + dimension);
    const float y_sum = load_unaligned<float>(query_meta_bytes + sq8::SUM_QUERY * sizeof(float));

    return min_val * y_sum + delta * quantized_dot;
}

template <unsigned char residual> // 0..15
float SQ8_FP16_InnerProductSIMD16_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                                     size_t dimension) {
    return 1.0f - SQ8_FP16_InnerProductImp_AVX512<residual>(pVec1v, pVec2v, dimension);
}

template <unsigned char residual> // 0..15
float SQ8_FP16_CosineSIMD16_AVX512F_BW_VL_VNNI(const void *pVec1v, const void *pVec2v,
                                               size_t dimension) {
    // Cosine distance = 1 - IP for pre-normalised vectors. Aliases InnerProduct, matching the
    // SQ8_FP32 pattern.
    return SQ8_FP16_InnerProductSIMD16_AVX512F_BW_VL_VNNI<residual>(pVec1v, pVec2v, dimension);
}
