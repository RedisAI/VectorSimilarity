/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "VecSim/utils/types_decl.h"

static inline void InnerProductLowHalfStep(__m128i v1, __m128i v2, __m128i zeros,
                                           __m128 &sum_prod) {
    // convert next 0..3 bf16 to 4 floats
    __m128i bf16_low1 = _mm_unpacklo_epi16(zeros, v1); // SSE2
    __m128i bf16_low2 = _mm_unpacklo_epi16(zeros, v2);

    // compute dist
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps((__m128)bf16_low1, (__m128)bf16_low2));
}

static inline void InnerProductHighHalfStep(__m128i v1, __m128i v2, __m128i zeros,
                                            __m128 &sum_prod) {
    // convert next 4..7 bf16 to 4 floats
    __m128i bf16_high1 = _mm_unpackhi_epi16(zeros, v1);
    __m128i bf16_high2 = _mm_unpackhi_epi16(zeros, v2);

    // compute dist
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps((__m128)bf16_high1, (__m128)bf16_high2));
}

static inline void InnerProductStep(bfloat16 *&pVect1, bfloat16 *&pVect2, __m128 &sum_prod) {
    // load 8 bf16 elements
    __m128i v1 = _mm_lddqu_si128((__m128i *)pVect1); // SSE3
    pVect1 += 8;
    __m128i v2 = _mm_lddqu_si128((__m128i *)pVect2);
    pVect2 += 8;

    __m128i zeros = _mm_setzero_si128(); // SSE2

    // Compute dist for 0..3 bf16
    InnerProductLowHalfStep(v1, v2, zeros, sum_prod);

    // Compute dist for 4..7 bf16
    InnerProductHighHalfStep(v1, v2, zeros, sum_prod);
}

template <unsigned char residual> // 0..31
float BF16_InnerProductSIMD32_SSE3(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // cast to bfloat16 *
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    // define end
    const bfloat16 *pEnd1 = pVect1 + dimension;

    // declare sum
    __m128 sum_prod = _mm_setzero_ps();

    // handle first residual % 8 elements (smaller than step chunk size)
    unsigned char skip = 0;
    if (residual % 8) {
        __m128i v1, v2;
        bfloat16 zero = 0;
        // | bf16_0 | bf16_1 | bf16_2 | bf16_3 |
        if (residual % 8 >= 4) {
            InnerProductLowHalfStep(_mm_lddqu_si128((__m128i *)pVect1),
                                    _mm_lddqu_si128((__m128i *)pVect2), _mm_setzero_si128(),
                                    sum_prod);
            pVect1 += 4;
            pVect2 += 4;
        }
        // bf16_0 | bf16_1 | bf16_2 || bf16_4 | bf16_5 | bf16_6
        if (residual % 8 == 3 || residual % 8 == 7) {
            v1 = _mm_setr_epi16(zero, pVect1[0], zero, pVect1[1], zero, pVect1[2], zero,
                                zero); // SSE2
            v2 = _mm_setr_epi16(zero, pVect2[0], zero, pVect2[1], zero, pVect2[2], zero, zero);
            skip = 3;
        }
        // bf16_0 | bf16_1 || bf16_4 | bf16_5
        if (residual % 8 == 2 || residual % 8 == 6) {
            // load 2 bf16 element set the rest to 0
            v1 = _mm_setr_epi16(zero, pVect1[0], zero, pVect1[1], zero, zero, zero, zero); // SSE2
            v2 = _mm_setr_epi16(zero, pVect2[0], zero, pVect2[1], zero, zero, zero, zero);
            skip = 2;
        }
        // bf16_0 || bf16_4
        if (residual % 8 == 1 || residual % 8 == 5) {
            // load only first element
            v1 = _mm_setr_epi16(zero, pVect1[0], zero, zero, zero, zero, zero, zero); // SSE2
            v2 = _mm_setr_epi16(zero, pVect2[0], zero, zero, zero, zero, zero, zero);
            skip = 1;
        }
        if (skip) {
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps((__m128)v1, (__m128)v2));
            pVect1 += skip;
            pVect2 += skip;
        }
    }

    // handle (residual - (residual % 8)) in chunks of 8 bfloat16
    if (residual >= 24)
        InnerProductStep(pVect1, pVect2, sum_prod);
    if (residual >= 16)
        InnerProductStep(pVect1, pVect2, sum_prod);
    if (residual >= 8)
        InnerProductStep(pVect1, pVect2, sum_prod);

    // handle 512 bits (32 bfloat16) in chuncks of max SIMD = 128 bits = 8 bfloat16
    do {
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
    } while (pVect1 < pEnd1);

    // TmpRes must be 16 bytes aligned
    float PORTABLE_ALIGN16 TmpRes[4];
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return 1.0f - sum;
}
