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

static inline __m512h SQ8_FP16_LoadCodes_AVX512FP16(const uint8_t *codes) {
    const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(codes));
    return _mm512_cvtepu16_ph(_mm512_cvtepu8_epi16(bytes));
}

static inline void SQ8_FP16_InnerProductStep_AVX512FP16(const uint8_t *&codes,
                                                        const vecsim_types::float16 *&query,
                                                        __m512h &sum) {
    const __m512h values = SQ8_FP16_LoadCodes_AVX512FP16(codes);
    const __m512h query_values = _mm512_loadu_ph(query);
    sum = _mm512_fmadd_ph(values, query_values, sum);
    codes += 32;
    query += 32;
}

// dim >= 32 keeps the residual's full loads within the vector payloads.
template <unsigned char residual>
float SQ8_FP16_InnerProductSIMD32_AVX512FP16_BW_VL(const void *storage, const void *query_blob,
                                                   size_t dimension) {
    using sq8 = vecsim_types::sq8;
    using float16 = vecsim_types::float16;
    // Directed rounding can saturate half overflow, hiding it from the finite check below.
    if ((_mm_getcsr() & _MM_ROUND_MASK) != _MM_ROUND_NEAREST)
        return spaces::Choose_SQ8_FP16_IP_implementation_AVX512F(dimension)(storage, query_blob,
                                                                            dimension);
    const auto *codes = static_cast<const uint8_t *>(storage);
    const auto *query = static_cast<const float16 *>(query_blob);
    const auto *end = codes + dimension;
    __m512h sum = _mm512_setzero_ph();
    if constexpr (residual) {
        constexpr __mmask32 mask = (1U << residual) - 1;
        const __m512h values = SQ8_FP16_LoadCodes_AVX512FP16(codes);
        const __m512h query_values = _mm512_loadu_ph(query);
        sum = _mm512_maskz_mul_ph(mask, values, query_values);
        codes += residual;
        query += residual;
    }
    do {
        SQ8_FP16_InnerProductStep_AVX512FP16(codes, query, sum);
    } while (codes < end);
    const _Float16 reduced = _mm512_reduce_add_ph(sum);
    const float dot = static_cast<float>(reduced);
    // Nearest rounding leaves intermediate and reduction overflow nonfinite.
    if (!std::isfinite(dot))
        return spaces::Choose_SQ8_FP16_IP_implementation_AVX512F(dimension)(storage, query_blob,
                                                                            dimension);

    const auto *storage_meta = static_cast<const uint8_t *>(storage) + dimension;
    const float min_val = load_unaligned<float>(storage_meta + sq8::MIN_VAL * sizeof(float));
    const float delta = load_unaligned<float>(storage_meta + sq8::DELTA * sizeof(float));
    const auto *query_meta =
        reinterpret_cast<const uint8_t *>(static_cast<const float16 *>(query_blob) + dimension);
    const float query_sum = load_unaligned<float>(query_meta + sq8::SUM_QUERY * sizeof(float));
    return 1.0f - (min_val * query_sum + delta * dot);
}
