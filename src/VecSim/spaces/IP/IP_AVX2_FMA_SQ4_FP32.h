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
#include "VecSim/spaces/AVX_utils.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/types/sq4.h"
#include "VecSim/utils/alignment.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>

namespace {

/*
 * Asymmetric SQ4 inner product with the same identity as SQ8:
 *
 *   IP(x, y) ≈ Σ((min + delta * q_i) * y_i) = min * y_sum + delta * Σ(q_i * y_i)
 *
 * Storage packs coordinate i into byte i / 2, the low nibble for even i and the high nibble for
 * odd i, followed by unaligned FP32 metadata. The query blob carries y_sum after its dim floats.
 */

// Interleave the low and high nibbles of the packed bytes, so `first` holds the codes of the low
// eight bytes and `second` those of the high eight bytes, each in coordinate order.
static inline void SQ4_Unpack_AVX2_FMA(__m128i packed, __m128i &first, __m128i &second) {
    const __m128i nibble = _mm_set1_epi8(15);
    const __m128i low = _mm_and_si128(packed, nibble);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble);
    first = _mm_unpacklo_epi8(low, high);
    second = _mm_unpackhi_epi8(low, high);
}

// Sixteen byte codes in coordinate order to two float vectors.
static inline void SQ4_ToFloats_AVX2_FMA(__m128i codes, __m256 &first, __m256 &second) {
    first = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes));
    second = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_unpackhi_epi64(codes, codes)));
}

// Exactly `bytes` (1..8) packed bytes in the low half of a vector, the rest zero.
template <unsigned bytes>
static inline __m128i SQ4_LoadBytes_AVX2_FMA(const uint8_t *data) {
    static_assert(bytes >= 1 && bytes <= 8);
    uint64_t bits = 0;
    std::memcpy(&bits, data, bytes);
    return _mm_cvtsi64_si128(static_cast<long long>(bits));
}

// All ones in lanes 0..lanes-1, zero above: a maskload mask that reads exactly `lanes` floats.
template <unsigned lanes> // 1..8
static inline __m256i SQ4_LaneMask_AVX2_FMA() {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(lanes), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}

// Accumulate Σ(q_i * y_i) for 32 coordinates: one 16-byte load of packed codes.
static inline void SQ4_FP32_IPStep32_AVX2_FMA(const uint8_t *x, const float *y, __m256 &sum0,
                                              __m256 &sum1, __m256 &sum2, __m256 &sum3) {
    __m128i ordered0, ordered1;
    SQ4_Unpack_AVX2_FMA(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x)), ordered0, ordered1);
    __m256 codes0, codes1, codes2, codes3;
    SQ4_ToFloats_AVX2_FMA(ordered0, codes0, codes1);
    SQ4_ToFloats_AVX2_FMA(ordered1, codes2, codes3);
    sum0 = _mm256_fmadd_ps(codes0, _mm256_loadu_ps(y), sum0);
    sum1 = _mm256_fmadd_ps(codes1, _mm256_loadu_ps(y + 8), sum1);
    sum2 = _mm256_fmadd_ps(codes2, _mm256_loadu_ps(y + 16), sum2);
    sum3 = _mm256_fmadd_ps(codes3, _mm256_loadu_ps(y + 24), sum3);
}

template <unsigned char residual> // 0..31
float SQ4_FP32_InnerProductSIMD32_AVX2_FMA(const void *pVect1v, const void *pVect2v,
                                           size_t dimension) {
    using vecsim_types::sq4;
    const auto *x = static_cast<const uint8_t *>(pVect1v);
    const auto *y = static_cast<const float *>(pVect2v);
    // Eight accumulators break the FMA dependency chain: two 32-coordinate steps per iteration.
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps(), sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps(), sum5 = _mm256_setzero_ps();
    __m256 sum6 = _mm256_setzero_ps(), sum7 = _mm256_setzero_ps();
    const size_t complete = dimension - residual;
    size_t i = 0;
    for (; i + 64 <= complete; i += 64) {
        SQ4_FP32_IPStep32_AVX2_FMA(x + i / 2, y + i, sum0, sum1, sum2, sum3);
        SQ4_FP32_IPStep32_AVX2_FMA(x + i / 2 + 16, y + i + 32, sum4, sum5, sum6, sum7);
    }
    if (i < complete) {
        SQ4_FP32_IPStep32_AVX2_FMA(x + i / 2, y + i, sum0, sum1, sum2, sum3);
    }
    sum0 = _mm256_add_ps(sum0, sum4);
    sum1 = _mm256_add_ps(sum1, sum5);
    sum2 = _mm256_add_ps(sum2, sum6);
    sum3 = _mm256_add_ps(sum3, sum7);
    i = complete;
    if constexpr (residual >= 16) {
        // One 8-byte load covers exactly 16 coordinates.
        __m128i ordered, unused;
        SQ4_Unpack_AVX2_FMA(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(x + i / 2)), ordered,
                            unused);
        __m256 codes0, codes1;
        SQ4_ToFloats_AVX2_FMA(ordered, codes0, codes1);
        sum0 = _mm256_fmadd_ps(codes0, _mm256_loadu_ps(y + i), sum0);
        sum1 = _mm256_fmadd_ps(codes1, _mm256_loadu_ps(y + i + 8), sum1);
        i += 16;
    }
    constexpr unsigned rest = residual % 16;
    if constexpr (rest > 0) {
        // Exactly the tail's code bytes and, through maskload (which does not touch masked-out
        // lanes), exactly its query floats. Masked-out query lanes are zero, so the padding nibble
        // of an odd dim adds nothing.
        __m128i ordered, unused;
        SQ4_Unpack_AVX2_FMA(SQ4_LoadBytes_AVX2_FMA<(rest + 1) / 2>(x + i / 2), ordered, unused);
        __m256 codes0, codes1;
        SQ4_ToFloats_AVX2_FMA(ordered, codes0, codes1);
        constexpr unsigned first = rest < 8 ? rest : 8;
        sum2 = _mm256_fmadd_ps(codes0, _mm256_maskload_ps(y + i, SQ4_LaneMask_AVX2_FMA<first>()),
                               sum2);
        if constexpr (rest > 8) {
            sum3 = _mm256_fmadd_ps(
                codes1, _mm256_maskload_ps(y + i + 8, SQ4_LaneMask_AVX2_FMA<rest - 8>()), sum3);
        }
    }
    const float dot = my_mm256_reduce_add_ps_tree(
        _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)));
    const auto *params = x + sq4::data_bytes_count(dimension);
    const float min_val = load_unaligned<float>(params + sq4::MIN_VAL * sizeof(float));
    const float delta = load_unaligned<float>(params + sq4::DELTA * sizeof(float));
    const float distance = 1.0f - (min_val * y[dimension + sq4::SUM_QUERY] + delta * dot);
    // The unscaled code dot can overflow even when the reconstructed products stay finite.
    if (!std::isfinite(distance)) {
        return SQ4_FP32_InnerProductReconstructed(pVect1v, pVect2v, dimension);
    }
    return distance;
}

} // namespace
