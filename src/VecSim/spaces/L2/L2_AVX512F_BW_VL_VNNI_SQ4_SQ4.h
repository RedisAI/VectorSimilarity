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
// SQ4_Unpack32_AVX512F_BW_VL_VNNI, SQ4TailMasks_AVX512F_BW_VL_VNNI
#include "VecSim/spaces/IP/IP_AVX512F_BW_VL_VNNI_SQ4_FP32.h"
// SQ4_SQ4_InnerProductSIMD64_AVX512F_BW_VL_VNNI_IMP
#include "VecSim/spaces/IP/IP_AVX512F_BW_VL_VNNI_SQ4_SQ4.h"
#include "VecSim/types/sq4.h"
#include "VecSim/utils/alignment.h"
#include <immintrin.h>

namespace {

/*
 * Symmetric SQ4-SQ4 L2 squared distance, reconstructed per element as the scalar SQ4_SQ4_L2Sqr:
 *
 *   ||x - y||^2 = Σ((min_x - min_y) + (delta_x * qx_i - delta_y * qy_i))^2
 *
 * Subtracting the offsets first keeps rounding at the scale of the residual, not of the norms,
 * when the data has a large common offset (the ||x||^2 + ||y||^2 - 2 * IP identity loses HNSW
 * recall there). Both operands are unpacked into coordinate order. Each difference is evaluated
 * with two fused multiply-adds as ((min_x - min_y) + delta_x * qx_i) - delta_y * qy_i, so the
 * offsets are still subtracted first.
 */

static inline __m512 SQ4_SQ4_L2Diff_AVX512F_BW_VL_VNNI(__m512 qx, __m512 qy, __m512 min_diff,
                                                       __m512 delta_x, __m512 delta_y) {
    return _mm512_fnmadd_ps(delta_y, qy, _mm512_fmadd_ps(delta_x, qx, min_diff));
}

template <unsigned char residual> // 0..31
float SQ4_SQ4_L2SqrSIMD32_AVX512F_BW_VL_VNNI(const void *pVect1v, const void *pVect2v,
                                             size_t dimension) {
    using vecsim_types::sq4;
    const auto *x = static_cast<const uint8_t *>(pVect1v);
    const auto *y = static_cast<const uint8_t *>(pVect2v);
    const auto *params_x = x + sq4::data_bytes_count(dimension);
    const auto *params_y = y + sq4::data_bytes_count(dimension);
    const float min_x = load_unaligned<float>(params_x + sq4::MIN_VAL * sizeof(float));
    const float delta_x = load_unaligned<float>(params_x + sq4::DELTA * sizeof(float));
    const float min_y = load_unaligned<float>(params_y + sq4::MIN_VAL * sizeof(float));
    const float delta_y = load_unaligned<float>(params_y + sq4::DELTA * sizeof(float));
    const __m512 min_diff = _mm512_set1_ps(min_x - min_y);
    const __m512 delta_x_vec = _mm512_set1_ps(delta_x);
    const __m512 delta_y_vec = _mm512_set1_ps(delta_y);
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    const size_t complete = dimension - residual;
    for (size_t i = 0; i < complete; i += 32) {
        __m512 x0, x1, y0, y1;
        SQ4_Unpack32_AVX512F_BW_VL_VNNI(
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i / 2)), x0, x1);
        SQ4_Unpack32_AVX512F_BW_VL_VNNI(
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + i / 2)), y0, y1);
        const __m512 d0 =
            SQ4_SQ4_L2Diff_AVX512F_BW_VL_VNNI(x0, y0, min_diff, delta_x_vec, delta_y_vec);
        const __m512 d1 =
            SQ4_SQ4_L2Diff_AVX512F_BW_VL_VNNI(x1, y1, min_diff, delta_x_vec, delta_y_vec);
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
    }
    if constexpr (residual > 0) {
        // Masked loads of exactly the tail's code bytes, so nothing past the codes is read. Lanes
        // past the tail (including an odd dim's padding nibble) keep the running sum: a zero code
        // there would still give diff = min_x - min_y.
        using masks = SQ4TailMasks_AVX512F_BW_VL_VNNI<residual>;
        __m512 x0, x1, y0, y1;
        SQ4_Unpack32_AVX512F_BW_VL_VNNI(_mm_maskz_loadu_epi8(masks::bytes, x + complete / 2), x0,
                                        x1);
        SQ4_Unpack32_AVX512F_BW_VL_VNNI(_mm_maskz_loadu_epi8(masks::bytes, y + complete / 2), y0,
                                        y1);
        const __m512 d0 =
            SQ4_SQ4_L2Diff_AVX512F_BW_VL_VNNI(x0, y0, min_diff, delta_x_vec, delta_y_vec);
        sum0 = _mm512_mask3_fmadd_ps(d0, d0, sum0, masks::first);
        if constexpr (masks::second != 0) {
            const __m512 d1 =
                SQ4_SQ4_L2Diff_AVX512F_BW_VL_VNNI(x1, y1, min_diff, delta_x_vec, delta_y_vec);
            sum1 = _mm512_mask3_fmadd_ps(d1, d1, sum1, masks::second);
        }
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
}

/*
 * Symmetric SQ4-SQ4 L2 squared distance with the same identity as SQ8-SQ8, for centered indexes:
 *
 *   ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * IP(x, y)
 *
 * where ||x||^2 and ||y||^2 are the stored SUM_SQUARES metadata and IP(x, y) comes from the
 * integer code dot of the inner product kernel. It rounds at the scale of the norms, so the index
 * factory uses it only when the stored vectors are centered (a mean is set); plain indexes keep the
 * per-element reconstruction above. Exact-dim bound and dispatch as the inner product kernel.
 */
template <unsigned char residual> // 0..63
float SQ4_SQ4_L2SqrIdentitySIMD64_AVX512F_BW_VL_VNNI(const void *pVect1v, const void *pVect2v,
                                                     size_t dimension) {
    using vecsim_types::sq4;
    const float ip =
        SQ4_SQ4_InnerProductSIMD64_AVX512F_BW_VL_VNNI_IMP<residual>(pVect1v, pVect2v, dimension);
    const auto *x = static_cast<const uint8_t *>(pVect1v);
    const auto *y = static_cast<const uint8_t *>(pVect2v);
    const size_t data_bytes = sq4::data_bytes_count(dimension);
    const float sum_sq_x = load_unaligned<float>(x + data_bytes + sq4::SUM_SQUARES * sizeof(float));
    const float sum_sq_y = load_unaligned<float>(y + data_bytes + sq4::SUM_SQUARES * sizeof(float));
    return sum_sq_x + sum_sq_y - 2.0f * ip;
}

} // namespace
