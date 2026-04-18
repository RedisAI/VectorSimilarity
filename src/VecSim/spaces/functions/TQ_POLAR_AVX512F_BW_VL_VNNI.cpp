/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "TQ_POLAR_AVX512F_BW_VL_VNNI.h"

#include <immintrin.h>

namespace spaces {

namespace {

inline int FinishPackedResidualSignDot(const uint8_t *lhs, const uint8_t *rhs, size_t projections,
                                       size_t processed_bytes, uint64_t diff_count_total) {
    const size_t full_bytes = projections / 8;
    const size_t tail_bits = projections % 8;
    int sign_dot =
        static_cast<int>(processed_bytes * 8) - 2 * static_cast<int>(diff_count_total);

    for (size_t idx = processed_bytes; idx < full_bytes; ++idx) {
        const int diff_count =
            __builtin_popcount(static_cast<unsigned int>(lhs[idx] ^ rhs[idx]));
        sign_dot += 8 - (2 * diff_count);
    }

    if (tail_bits != 0) {
        const uint8_t valid_mask = static_cast<uint8_t>((uint16_t{1} << tail_bits) - 1u);
        const uint8_t diff_bits = static_cast<uint8_t>((lhs[full_bytes] ^ rhs[full_bytes]) &
                                                       valid_mask);
        const int diff_count = __builtin_popcount(static_cast<unsigned int>(diff_bits));
        sign_dot += static_cast<int>(tail_bits) - (2 * diff_count);
    }

    return sign_dot;
}

inline __m512i PopcountBytes(__m512i bytes) {
    const __m512i low_mask = _mm512_set1_epi8(0x0F);
    const __m512i lookup =
        _mm512_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2,
                         2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3,
                         2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m512i low = _mm512_and_si512(bytes, low_mask);
    const __m512i high =
        _mm512_and_si512(_mm512_srli_epi16(bytes, 4), low_mask);
    return _mm512_add_epi8(_mm512_shuffle_epi8(lookup, low), _mm512_shuffle_epi8(lookup, high));
}

template <unsigned char residual>
int TQ_PackedResidualSignDotSIMD64_AVX512F_BW_VL_VNNI(const uint8_t *lhs, const uint8_t *rhs,
                                                      size_t projections) {
    const size_t full_bytes = projections / 8;
    const size_t simd_end = full_bytes - residual;
    uint64_t diff_count_total = 0;
    const __m512i zero = _mm512_setzero_si512();

    size_t idx = 0;
    for (; idx < simd_end; idx += 64) {
        const __m512i diff =
            _mm512_xor_si512(_mm512_loadu_si512(reinterpret_cast<const void *>(lhs + idx)),
                             _mm512_loadu_si512(reinterpret_cast<const void *>(rhs + idx)));
        const __m512i bit_counts = PopcountBytes(diff);
        const __m512i partial = _mm512_sad_epu8(bit_counts, zero);
        alignas(64) uint64_t sums[8];
        _mm512_store_si512(reinterpret_cast<void *>(sums), partial);
        for (uint64_t value : sums) {
            diff_count_total += value;
        }
    }

    return FinishPackedResidualSignDot(lhs, rhs, projections, idx, diff_count_total);
}

template <unsigned char residual>
float TQ_SymmetricPolarSIMD16_AVX512F_BW_VL_VNNI(const float *lhs_radii, const uint8_t *lhs_angles,
                                                 const float *rhs_radii,
                                                 const uint8_t *rhs_angles,
                                                 const float *delta_cos_lut,
                                                 uint8_t angle_delta_mask, size_t pairs) {
    const size_t simd_end = pairs - residual;
    const __m128i mask_vec = _mm_set1_epi8(static_cast<char>(angle_delta_mask));
    __m512 acc = _mm512_setzero_ps();

    alignas(16) uint8_t deltas[16];
    alignas(64) float lut_values[16];

    size_t idx = 0;
    for (; idx < simd_end; idx += 16) {
        const __m128i lhs_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhs_angles + idx));
        const __m128i rhs_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs_angles + idx));
        const __m128i delta_vec = _mm_and_si128(_mm_sub_epi8(lhs_vec, rhs_vec), mask_vec);
        _mm_store_si128(reinterpret_cast<__m128i *>(deltas), delta_vec);
        for (size_t lane = 0; lane < 16; ++lane) {
            lut_values[lane] = delta_cos_lut[deltas[lane]];
        }

        const __m512 lhs_vec_f = _mm512_loadu_ps(lhs_radii + idx);
        const __m512 rhs_vec_f = _mm512_loadu_ps(rhs_radii + idx);
        const __m512 lut_vec = _mm512_load_ps(lut_values);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_mul_ps(lhs_vec_f, rhs_vec_f), lut_vec));
    }

    alignas(64) float partials[16];
    _mm512_store_ps(partials, acc);
    float sum = 0.0f;
    for (float partial : partials) {
        sum += partial;
    }

    for (; idx < pairs; ++idx) {
        const size_t delta =
            (static_cast<size_t>(lhs_angles[idx]) - static_cast<size_t>(rhs_angles[idx])) &
            angle_delta_mask;
        sum += lhs_radii[idx] * rhs_radii[idx] * delta_cos_lut[delta];
    }

    return sum;
}

} // namespace

#include "implementation_chooser.h"

tq_packed_residual_sign_dot_func_t
Choose_TQ_PackedResidualSignDot_implementation_AVX512F_BW_VL_VNNI(size_t projections) {
    const size_t full_bytes = projections / 8;
    tq_packed_residual_sign_dot_func_t ret_func;
    CHOOSE_IMPLEMENTATION(ret_func, full_bytes, 64,
                          TQ_PackedResidualSignDotSIMD64_AVX512F_BW_VL_VNNI);
    return ret_func;
}

tq_symmetric_polar_func_t
Choose_TQ_SymmetricPolar_implementation_AVX512F_BW_VL_VNNI(size_t pairs) {
    tq_symmetric_polar_func_t ret_func;
    CHOOSE_IMPLEMENTATION(ret_func, pairs, 16, TQ_SymmetricPolarSIMD16_AVX512F_BW_VL_VNNI);
    return ret_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
