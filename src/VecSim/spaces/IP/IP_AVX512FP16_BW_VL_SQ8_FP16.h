/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cmath>
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/types/sq8.h"
#include "VecSim/types/float16.h"
#include "VecSim/utils/alignment.h"

static inline void SQ8_FP16_InnerProductStep_AVX512FP16(const uint8_t *&codes,
                                                        const vecsim_types::float16 *&query,
                                                        __m512 &low_sum, __m512 &high_sum) {
    const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(codes));
    const __m512h values = _mm512_cvtepu16_ph(_mm512_cvtepu8_epi16(bytes));
    const __m512h query_values = _mm512_loadu_ph(query);
    // Directed rounding can saturate an overflowing product instead of triggering the fallback.
    constexpr int rounding = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    const __m512h product = _mm512_mul_round_ph(values, query_values, rounding);
    // A byte times a half has at most 19 significant bits, so its finite half residual is exact.
    // Recover it before SQ8's scale/offset correction can amplify the product rounding error.
    const __m512h error = _mm512_fmsub_round_ph(values, query_values, product, rounding);
    const __m512i product_bits = _mm512_castph_si512(product);
    const __m512i error_bits = _mm512_castph_si512(error);
    const __m512 low = _mm512_add_ps(_mm512_cvtph_ps(_mm512_castsi512_si256(product_bits)),
                                     _mm512_cvtph_ps(_mm512_castsi512_si256(error_bits)));
    const __m512 high = _mm512_add_ps(_mm512_cvtph_ps(_mm512_extracti64x4_epi64(product_bits, 1)),
                                      _mm512_cvtph_ps(_mm512_extracti64x4_epi64(error_bits, 1)));
    low_sum = _mm512_add_ps(low_sum, low);
    high_sum = _mm512_add_ps(high_sum, high);
    codes += 32;
    query += 32;
}

template <unsigned char residual>
float SQ8_FP16_InnerProductSIMD32_AVX512FP16_BW_VL(const void *storage, const void *query_blob,
                                                   size_t dimension) {
    using sq8 = vecsim_types::sq8;
    using float16 = vecsim_types::float16;
    const auto *codes = static_cast<const uint8_t *>(storage);
    const auto *query = static_cast<const float16 *>(query_blob);
    size_t remaining = dimension - residual;
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    while (remaining >= 64) {
        SQ8_FP16_InnerProductStep_AVX512FP16(codes, query, sum0, sum1);
        SQ8_FP16_InnerProductStep_AVX512FP16(codes, query, sum2, sum3);
        remaining -= 64;
    }
    if (remaining) {
        SQ8_FP16_InnerProductStep_AVX512FP16(codes, query, sum0, sum1);
    }
    float dot =
        _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3)));
    if (!std::isfinite(dot)) {
        // Unnormalized inputs can overflow a half product before SQ8 scaling.
        return spaces::Choose_SQ8_FP16_IP_implementation_AVX512F(dimension)(storage, query_blob,
                                                                            dimension);
    }
    for (size_t i = 0; i < residual; ++i) {
        dot += static_cast<float>(codes[i]) * vecsim_types::FP16_to_FP32(query[i]);
    }

    const auto *storage_meta = static_cast<const uint8_t *>(storage) + dimension;
    const float min_val = load_unaligned<float>(storage_meta + sq8::MIN_VAL * sizeof(float));
    const float delta = load_unaligned<float>(storage_meta + sq8::DELTA * sizeof(float));
    const auto *query_meta =
        reinterpret_cast<const uint8_t *>(static_cast<const float16 *>(query_blob) + dimension);
    const float query_sum = load_unaligned<float>(query_meta + sq8::SUM_QUERY * sizeof(float));
    return 1.0f - (min_val * query_sum + delta * dot);
}
