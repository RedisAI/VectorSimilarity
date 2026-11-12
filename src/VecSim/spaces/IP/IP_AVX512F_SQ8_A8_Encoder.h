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
#include "VecSim/spaces/sq8_a8_query_encoder.h"
#include "VecSim/types/float16.h"

#include <type_traits>

/*
 * Vectorized A8 query encoder (vecsim_types::sq8_a8_query), AVX-512F only.
 *
 * Pass 1 finds the float range 16 lanes at a time. Pass 2 converts 16 floats per iteration to two
 * halves of 8 doubles, and each lane performs the scalar encoder's double operations in the same
 * order (v = y - c, v / s + 0.5, floor, max with -128, then min with 127), so every code equals
 * SQ8_A8_EncodeElement's. The min / max operand order matches the scalar comparisons too: a NaN
 * element is skipped by the range unless it is the first one, and its code is -128. The last
 * dim % 16 elements use the scalar step. The sums E and PE are added in a different order than
 * the scalar encoder's, so the constants QS and QK may differ in their last bits.
 *
 * With a mean (mean-centered index), each element is centered in FP32 first, the same float
 * subtraction the scalar encoder does, and <mean, y> is accumulated in double.
 */

namespace spaces {
namespace a8_avx512f {

// 16 input elements widened to FP32 (FP16 through vcvtph2ps, exact like FP16_to_FP32), centered
// when WithMean.
template <typename InputType, bool WithMean>
static inline __m512 LoadValues16(const InputType *query, const float *mean, size_t i,
                                  __m512 &raw) {
    if constexpr (std::is_same_v<InputType, vecsim_types::float16>) {
        raw = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + i)));
    } else {
        raw = _mm512_loadu_ps(query + i);
    }
    if constexpr (WithMean) {
        return _mm512_sub_ps(raw, _mm512_loadu_ps(mean + i));
    } else {
        return raw;
    }
}

static inline __m512d LowHalf(__m512 x) { return _mm512_cvtps_pd(_mm512_castps512_ps256(x)); }
static inline __m512d HighHalf(__m512 x) {
    return _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(x), 1)));
}

struct Sums8 {
    __m512d P = _mm512_setzero_pd();
    __m512d E = _mm512_setzero_pd();
    __m512d PE = _mm512_setzero_pd();
    __m512d mean_ip = _mm512_setzero_pd();
};

// Codes of 8 elements as int32, accumulating P, E and PE. The quotient is a true division: a
// multiply by 1 / s needs a division fallback near integers to keep the codes exact, and is not
// faster on AVX-512 hardware.
static inline __m256i EncodeHalf(__m512d y, __m512d offset, __m512d step, Sums8 &sums) {
    const __m512d v = _mm512_sub_pd(y, offset);
    const __m512d t = _mm512_add_pd(_mm512_div_pd(v, step), _mm512_set1_pd(0.5));
    __m512d code = _mm512_roundscale_pd(t, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    // max(code, -128) returns -128 for NaN, like the scalar clamp.
    code = _mm512_max_pd(code, _mm512_set1_pd(-128.0));
    code = _mm512_min_pd(code, _mm512_set1_pd(127.0));
    const __m512d err = _mm512_sub_pd(v, _mm512_mul_pd(step, code));
    sums.P = _mm512_add_pd(sums.P, code);
    sums.E = _mm512_add_pd(sums.E, err);
    sums.PE = _mm512_fmadd_pd(code, err, sums.PE);
    return _mm512_cvtpd_epi32(code); // exact: code is an integer in [-128, 127]
}

template <typename InputType, bool WithMean>
static void EncodeQuery(const InputType *query, const float *mean, size_t dim, void *blob) {
    assert(dim > 0);
    const auto value = [=](size_t i) -> float {
        const float y = quantized::SQ8_A8_InputValue(query, i);
        if constexpr (WithMean) {
            return y - mean[i];
        } else {
            return y;
        }
    };

    // Pass 1: range. Every lane starts from element 0, as the scalar loop does.
    float min_val = value(0);
    float max_val = min_val;
    size_t i = 0;
    __m512 raw;
    {
        __m512 lo = _mm512_set1_ps(min_val);
        __m512 hi = lo;
        for (; i + 16 <= dim; i += 16) {
            const __m512 x = LoadValues16<InputType, WithMean>(query, mean, i, raw);
            lo = _mm512_min_ps(x, lo); // x < lo ? x : lo
            hi = _mm512_max_ps(x, hi); // x > hi ? x : hi
        }
        min_val = _mm512_reduce_min_ps(lo);
        max_val = _mm512_reduce_max_ps(hi);
    }
    for (; i < dim; i++) {
        const float y = value(i);
        min_val = y < min_val ? y : min_val;
        max_val = y > max_val ? y : max_val;
    }
    const quantized::SQ8_A8_QueryGrid grid = quantized::SQ8_A8_QueryGridFromRange(min_val, max_val);

    // Pass 2: codes and sums.
    const __m512d offset = _mm512_set1_pd(grid.offset);
    const __m512d step = _mm512_set1_pd(grid.step);
    auto *codes = static_cast<int8_t *>(blob);
    Sums8 low_sums, high_sums;
    i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m512 x = LoadValues16<InputType, WithMean>(query, mean, i, raw);
        const __m256i low = EncodeHalf(LowHalf(x), offset, step, low_sums);
        const __m256i high = EncodeHalf(HighHalf(x), offset, step, high_sums);
        const __m512i codes32 = _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(codes + i), _mm512_cvtepi32_epi8(codes32));
        if constexpr (WithMean) {
            const __m512 m = _mm512_loadu_ps(mean + i);
            low_sums.mean_ip = _mm512_fmadd_pd(LowHalf(m), LowHalf(raw), low_sums.mean_ip);
            high_sums.mean_ip = _mm512_fmadd_pd(HighHalf(m), HighHalf(raw), high_sums.mean_ip);
        }
    }
    quantized::SQ8_A8_QuerySums sums;
    sums.P = static_cast<int64_t>(_mm512_reduce_add_pd(_mm512_add_pd(low_sums.P, high_sums.P)));
    sums.E = _mm512_reduce_add_pd(_mm512_add_pd(low_sums.E, high_sums.E));
    sums.PE = _mm512_reduce_add_pd(_mm512_add_pd(low_sums.PE, high_sums.PE));
    [[maybe_unused]] double y_mean_ip =
        _mm512_reduce_add_pd(_mm512_add_pd(low_sums.mean_ip, high_sums.mean_ip));
    for (size_t tail = i; tail < dim; tail++) {
        codes[tail] = quantized::SQ8_A8_EncodeElement(value(tail), grid, sums);
    }
    quantized::SQ8_A8_FinishQuery(grid, sums, dim, blob);

    if constexpr (WithMean) {
        for (; i < dim; i++) {
            y_mean_ip += static_cast<double>(mean[i]) * quantized::SQ8_A8_InputValue(query, i);
        }
        quantized::SQ8_A8_SetQueryMeanIP(y_mean_ip, dim, blob);
    }
}

} // namespace a8_avx512f

template <typename InputType>
void SQ8_A8_EncodeQuery_AVX512F(const InputType *query, const float *mean, size_t dim, void *blob) {
    if (mean) {
        a8_avx512f::EncodeQuery<InputType, true>(query, mean, dim, blob);
    } else {
        a8_avx512f::EncodeQuery<InputType, false>(query, nullptr, dim, blob);
    }
}

} // namespace spaces
