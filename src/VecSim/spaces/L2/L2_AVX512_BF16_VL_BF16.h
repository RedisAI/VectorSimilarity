/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "VecSim/types/bfloat16.h"

using bfloat16 = vecsim_types::bfloat16;

static inline void L2SqrStep(__m512i &bfloat16_chunk1, __m512i &bfloat16_chunk2, __m512 &sum) {
    // L2 = sum((a-b)^2) = a^2 - 2ab + b^2
    // sum += a^2
    sum = _mm512_dpbf16_ps(sum, (__m512bh)bfloat16_chunk1, (__m512bh)bfloat16_chunk1);
    // sum += b^2
    sum = _mm512_dpbf16_ps(sum, (__m512bh)bfloat16_chunk2, (__m512bh)bfloat16_chunk2);
    // ab
    __m512 v1_v2_dp = _mm512_setzero_ps();
    v1_v2_dp = _mm512_dpbf16_ps(v1_v2_dp, (__m512bh)bfloat16_chunk1, (__m512bh)bfloat16_chunk2);
    // sum = sum -ab -ab
    sum = _mm512_sub_ps(sum, v1_v2_dp);
    sum = _mm512_sub_ps(sum, v1_v2_dp);
}

template <unsigned char residual> // 0..31
float BF16_L2SqrSIMD32_AVX512BF16_VL(const void *pVect1v, const void *pVect2v, size_t dimension) {
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    const bfloat16 *pEnd1 = pVect1 + dimension;

    __m512 sum = _mm512_setzero_ps();

    if constexpr (residual) {
        constexpr __mmask32 mask = (1LU << residual) - 1;
        __m512i v1 = _mm512_maskz_loadu_epi16(mask, pVect1);
        pVect1 += residual;
        __m512i v2 = _mm512_maskz_loadu_epi16(mask, pVect2);
        pVect2 += residual;
        L2SqrStep(v1, v2, sum);
    }

    do {
        __m512i v1 = _mm512_loadu_si512((__m512i *)pVect1);
        __m512i v2 = _mm512_loadu_si512((__m512i *)pVect2);
        L2SqrStep(v1, v2, sum);
        pVect1 += 32;
        pVect2 += 32;
    } while (pVect1 < pEnd1);

    return _mm512_reduce_add_ps(sum);
}
