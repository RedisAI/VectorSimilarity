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
#include "VecSim/spaces/quantized_distance_helpers.h"
#include <immintrin.h>

namespace {

/*
 * SQ8 storage vs an A8 query (vecsim_types::sq8_a8_query). The stored uint8 codes and the int8
 * query codes are exactly the unsigned-by-signed operands of vpdpbusd, which adds four products
 * into each int32 lane without saturation. The rest of the distance is the shared double
 * epilogue, so this kernel returns the scalar reference's float for the same code dot.
 *
 * Four independent accumulators take 256 bytes per iteration, up to three remaining 64-byte
 * chunks go to their own accumulators, and one masked load of each operand reads the last
 * `residual` bytes, so nothing past the codes is read.
 *
 * Exactness: |a_i * p_i| <= 255 * 128, and every lane and every partial sum of the reduction adds
 * at most dim products, so all of them are exact in int32 while 32,640 * dim fits INT32_MAX
 * (spaces::SQ8_A8_MAX_EXACT_SIMD_DIM).
 */

// A distinct zero for each accumulator. When the accumulators start from one shared zero register,
// GCC 13 copies each tied vpdpbusd accumulator twice per loop iteration; the empty asm makes every
// initial zero its own value, which removes the copies. It changes no arithmetic.
static inline __m512i SQ8_A8_ZeroAccumulator_AVX512F_BW_VL_VNNI() {
    __m512i zero = _mm512_setzero_si512();
    __asm__ volatile("" : "+v"(zero));
    return zero;
}

template <unsigned char residual> // 0..63
static inline int32_t SQ8_A8_CodeDot_AVX512F_BW_VL_VNNI(const uint8_t *codes,
                                                        const int8_t *query_codes, size_t dim) {
    __m512i acc0 = SQ8_A8_ZeroAccumulator_AVX512F_BW_VL_VNNI();
    __m512i acc1 = SQ8_A8_ZeroAccumulator_AVX512F_BW_VL_VNNI();
    __m512i acc2 = SQ8_A8_ZeroAccumulator_AVX512F_BW_VL_VNNI();
    __m512i acc3 = SQ8_A8_ZeroAccumulator_AVX512F_BW_VL_VNNI();
    const size_t complete = dim - residual;
    size_t i = 0;
    for (; i + 256 <= complete; i += 256) {
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512(codes + i),
                                   _mm512_loadu_si512(query_codes + i));
        acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512(codes + i + 64),
                                   _mm512_loadu_si512(query_codes + i + 64));
        acc2 = _mm512_dpbusd_epi32(acc2, _mm512_loadu_si512(codes + i + 128),
                                   _mm512_loadu_si512(query_codes + i + 128));
        acc3 = _mm512_dpbusd_epi32(acc3, _mm512_loadu_si512(codes + i + 192),
                                   _mm512_loadu_si512(query_codes + i + 192));
    }
    if (i < complete) {
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512(codes + i),
                                   _mm512_loadu_si512(query_codes + i));
        i += 64;
        if (i < complete) {
            acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512(codes + i),
                                       _mm512_loadu_si512(query_codes + i));
            i += 64;
            if (i < complete) {
                acc2 = _mm512_dpbusd_epi32(acc2, _mm512_loadu_si512(codes + i),
                                           _mm512_loadu_si512(query_codes + i));
                i += 64;
            }
        }
    }
    __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));
    // The tail goes into the summed accumulator: accumulating it into one of the loop's
    // accumulators makes GCC 13 copy that accumulator twice per loop iteration.
    if constexpr (residual > 0) {
        constexpr __mmask64 mask = (uint64_t(1) << residual) - 1;
        acc = _mm512_dpbusd_epi32(acc, _mm512_maskz_loadu_epi8(mask, codes + i),
                                  _mm512_maskz_loadu_epi8(mask, query_codes + i));
    }
    return _mm512_reduce_add_epi32(acc);
}

// Returns the distance 1 - IP; Cosine on normalized vectors uses the same kernel.
template <unsigned char residual> // 0..63
float SQ8_A8_InnerProductSIMD64_AVX512F_BW_VL_VNNI(const void *pVect1v, const void *pVect2v,
                                                   size_t dimension) {
    const auto *stored = static_cast<const uint8_t *>(pVect1v);
    const auto *query = static_cast<const uint8_t *>(pVect2v);
    const int32_t code_dot = SQ8_A8_CodeDot_AVX512F_BW_VL_VNNI<residual>(
        stored, reinterpret_cast<const int8_t *>(query), dimension);
    return spaces::quantized::SQ8_A8_InnerProduct_FromCodeDot(stored, query, dimension, code_dot);
}

} // namespace
