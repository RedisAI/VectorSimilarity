/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "VecSim/utils/types_decl.h"

static inline void InnerProductHalfStep(bfloat16 *&pVect1, bfloat16 *&pVect2, __m512 &sum,
                                        __mmask32 mask) {
    __m512i v1 = _mm512_maskz_expandloadu_epi16(mask, pVect1); // AVX512_VBMI2
    __m512i v2 = _mm512_maskz_expandloadu_epi16(mask, pVect2); // AVX512_VBMI2
    sum = _mm512_add_ps(sum, _mm512_mul_ps((__m512)v1, (__m512)v2));
}

// static inline void InnerProductStep(bfloat16 *&pVect1, bfloat16 *&pVect2, __m512 &sum) {
//     __mmask32 mask = 0xAAAAAAAA;
//     InnerProductHalfStep(pVect1, pVect2, sum, mask);
//     pVect1 += 16;
//     pVect2 += 16;
//     InnerProductHalfStep(pVect1, pVect2, sum, mask);
//     pVect1 += 16;
//     pVect2 += 16;
// }

// Alternative imp, faster according to local BM
static inline void InnerProductStep(bfloat16 *&pVect1, bfloat16 *&pVect2, __m512 &sum) {
    __m512i v1 = _mm512_loadu_si512((__m512i *)pVect1);
    __m512i v2 = _mm512_loadu_si512((__m512i *)pVect2);
    pVect1 += 32;
    pVect2 += 32;
    __m512i zeros = _mm512_setzero_si512();

    // covert 0:3, 8:11, .. 28:31 to float32
    __m512i v1_low = _mm512_unpacklo_epi16(zeros, v1); // AVX512BW
    __m512i v2_low = _mm512_unpacklo_epi16(zeros, v2);
    sum = _mm512_add_ps(sum, _mm512_mul_ps((__m512)v1_low, (__m512)v2_low));

    // covert 4:7, 12:15, .. 24:27 to float32
    __m512i v1_high = _mm512_unpackhi_epi16(zeros, v1);
    __m512i v2_high = _mm512_unpackhi_epi16(zeros, v2);
    sum = _mm512_add_ps(sum, _mm512_mul_ps((__m512)v1_high, (__m512)v2_high));
}

template <unsigned char residual> // 0..31
float BF16_InnerProductSIMD32_AVX512BW_VBMI2(const void *pVect1v, const void *pVect2v,
                                             size_t dimension) {
    // cast to bfloat16 *
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    // define end
    const bfloat16 *pEnd1 = pVect1 + dimension;

    // declare sum
    __m512 sum = _mm512_setzero_ps();

    // handle first residual % 32 elements
    if (residual) {
        const __mmask32 mask = 0xAAAAAAAA; // 01010101...
        // calc first 16
        if (residual >= 16) {
            InnerProductHalfStep(pVect1, pVect2, sum, mask);
            pVect1 += 16;
            pVect2 += 16;
        }
        // each elements is represented by a pair of 01 bits
        // create a mask for the elements we want to process:
        // mask2 = {01 * (residual % 16)}0000...
        const __mmask32 mask2 = mask & ((1 << ((residual % 16) * 2)) - 1);
        InnerProductHalfStep(pVect1, pVect2, sum, mask2);
        pVect1 += residual % 16;
        pVect2 += residual % 16;
    }

    // handle 512 bits (32 bfloat16) in chunks of max SIMD = 512 bits = 32 bfloat16
    do {
        InnerProductStep(pVect1, pVect2, sum);
    } while (pVect1 < pEnd1);

    return 1.0f - _mm512_reduce_add_ps(sum);
}
