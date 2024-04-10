/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "VecSim/utils/types_decl.h"

static inline void InnerProductLowHalfStep(__m256i v1, __m256i v2, __m256i zeros,
                                           __m256 &sum_prod) {
    // convert next  0:3, 8:11 bf16 to 8 floats
    __m256i bf16_low1 = _mm256_unpacklo_epi16(zeros, v1); // AVX2
    __m256i bf16_low2 = _mm256_unpacklo_epi16(zeros, v2);

    // compute dist
    sum_prod = _mm256_add_ps(sum_prod, _mm256_mul_ps((__m256)bf16_low1, (__m256)bf16_low2));
}

static inline void InnerProductHighHalfStep(__m256i v1, __m256i v2, __m256i zeros,
                                            __m256 &sum_prod) {
    // convert next 4:7, 12:15 bf16 to 8 floats
    __m256i bf16_high1 = _mm256_unpackhi_epi16(zeros, v1);
    __m256i bf16_high2 = _mm256_unpackhi_epi16(zeros, v2);

    // compute dist
    sum_prod = _mm256_add_ps(sum_prod, _mm256_mul_ps((__m256)bf16_high1, (__m256)bf16_high2));
}

static inline void InnerProductStep(bfloat16 *&pVect1, bfloat16 *&pVect2, __m256 &sum_prod) {
    // load 16 bf16 elements
    __m256i v1 = _mm256_lddqu_si256((__m256i *)pVect1); // avx
    pVect1 += 16;
    __m256i v2 = _mm256_lddqu_si256((__m256i *)pVect2);
    pVect2 += 16;

    __m256i zeros = _mm256_setzero_si256(); // avx

    // Compute dist for 0:3, 8:11 bf16
    InnerProductLowHalfStep(v1, v2, zeros, sum_prod);

    // Compute dist for 4:7, 12:15 bf16
    InnerProductHighHalfStep(v1, v2, zeros, sum_prod);
}

template <unsigned char residual> // 0..31
float BF16_InnerProductSIMD32_AVX2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // cast to bfloat16 *
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    // define end
    const bfloat16 *pEnd1 = pVect1 + dimension;

    // declare sum
    __m256 sum_prod = _mm256_setzero_ps();

    // handle first residual % 16 elements
    if (residual % 16) {
        // load all 16 elements to a 256 bit register
        __m256i v1 = _mm256_lddqu_si256((__m256i *)pVect1); // avx
        pVect1 += residual % 16;
        __m256i v2 = _mm256_lddqu_si256((__m256i *)pVect2);
        pVect2 += residual % 16;

        // unpack  0:3, 8:11 bf16 to 8 floats
        __m256i zeros = _mm256_setzero_si256();
        __m256i v1_low = _mm256_unpacklo_epi16(zeros, v1);
        __m256i v2_low = _mm256_unpacklo_epi16(zeros, v2);

        __m256 low_mul = _mm256_mul_ps((__m256)v1_low, (__m256)v2_low);
        if (residual % 16 <= 4) {
            const unsigned char elem_to_calc = residual % 16;
            __mmask8 constexpr mask = (1 << elem_to_calc) - 1;
            low_mul = _mm256_blend_ps(_mm256_setzero_ps(), low_mul, mask);
        } else {
            __m256i v1_high = _mm256_unpackhi_epi16(zeros, v1);
            __m256i v2_high = _mm256_unpackhi_epi16(zeros, v2);
            __m256 high_mul = _mm256_mul_ps((__m256)v1_high, (__m256)v2_high);
            if (4 < residual % 16 && residual % 16 <= 8) {
                // keep only 4 first elemnts of low pack
                __mmask8 mask = (1 << 4) - 1;
                low_mul = _mm256_blend_ps(_mm256_setzero_ps(), low_mul, mask);

                // keep residual % 16 - 4 first elements of high_mul
                const unsigned char elem_to_calc = residual % 16 - 4;
                mask = (1 << elem_to_calc) - 1;
                high_mul = _mm256_blend_ps(_mm256_setzero_ps(), high_mul, mask);
            }
            if (8 < residual % 16 && residual % 16 < 12) {
                // keep residual % 16 - 4 first elements of low_mul
                const unsigned char elem_to_calc = residual % 16 - 4;
                __mmask8 mask = (1 << elem_to_calc) - 1;
                low_mul = _mm256_blend_ps(_mm256_setzero_ps(), low_mul, mask);

                // keep ony 4 first elements of high_mul
                mask = (1 << 4) - 1;
                high_mul = _mm256_blend_ps(_mm256_setzero_ps(), high_mul, mask);
            }
            if (residual % 16 >= 12) {
                // keep residual % 16 - 8 first elements of high
                const unsigned char elem_to_calc = residual % 16 - 8;
                __mmask8 mask = (1 << elem_to_calc) - 1;
                high_mul = _mm256_blend_ps(_mm256_setzero_ps(), high_mul, mask);
            }
            sum_prod = _mm256_add_ps(sum_prod, high_mul);
        }
        sum_prod = _mm256_add_ps(sum_prod, low_mul);
    }

    // do a single step if residual >=16
    if (residual >= 16) {
        InnerProductStep(pVect1, pVect2, sum_prod);
    }

    // handle 512 bits (32 bfloat16) in chuncks of max SIMD = 256 bits = 16 bfloat16
    do {
        InnerProductStep(pVect1, pVect2, sum_prod);
        InnerProductStep(pVect1, pVect2, sum_prod);
    } while (pVect1 < pEnd1);

    // TmpRes must be 16 bytes aligned
    float PORTABLE_ALIGN32 TmpRes[8];
    _mm256_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7];

    return 1.0f - sum;
}
