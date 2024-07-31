/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <cstdint>
#include "VecSim/spaces/space_includes.h"
#include "VecSim/types/float16.h"
#include <cstring>

using float16 = vecsim_types::float16;

static inline void L2SqrStep(float16 *&pVect1, float16 *&pVect2, __m512h &sum) {
    __m512h v1 = _mm512_loadu_ph(pVect1);
    __m512h v2 = _mm512_loadu_ph(pVect2);

    __m512h diff = _mm512_sub_ph(v1, v2);

    sum = _mm512_fmadd_ph(diff, diff, sum);
    pVect1 += 32;
    pVect2 += 32;
}

template <unsigned short residual> // 0..31
float FP16_L2SqrSIMD32_AVX512FP16(const void *pVect1v, const void *pVect2v, size_t dimension) {
    auto *pVect1 = (float16 *)pVect1v;
    auto *pVect2 = (float16 *)pVect2v;

    const float16 *pEnd1 = pVect1 + dimension;

    __m512h sum = _mm512_setzero_ph();

    if constexpr (residual) {
        constexpr __mmask32 mask = (1LU << residual) - 1;
        __m512h v1 = _mm512_loadu_ph(pVect1);
        pVect1 += residual;
        __m512h v2 = _mm512_loadu_ph(pVect2);
        pVect2 += residual;
        __m512h diff = _mm512_maskz_sub_ph(mask, v1, v2);

        sum = _mm512_mul_ph(diff, diff);
    }

    // We dealt with the residual part. We are left with some multiple of 32 16-bit floats.
    do {
        L2SqrStep(pVect1, pVect2, sum);
    } while (pVect1 < pEnd1);

    // Extract the lower 256 bits
    __m256h low = _mm512_castph512_ph256(sum);
    // Extract the upper 256 bits
    __m512i sum_i = _mm512_castph_si512(sum);
    __m256i high_i = _mm512_extracti32x8_epi32(sum_i, 1);
    // Convert to FP16
    __m256h high = _mm256_castsi256_ph(high_i);
    _Float16 res_low = _mm256_reduce_add_ph(low);
    _Float16 res_high = _mm256_reduce_add_ph(high);
    _Float16 res = res_low + res_high;
    return res;
}
