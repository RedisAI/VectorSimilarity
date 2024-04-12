/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "VecSim/types/bfloat16.h"

static inline void L2SqrLowHalfStep(__m256i v1, __m256i v2, __m256i zeros, __m256 &sum) {
    // convert next  0:3, 8:11 bf16 to 8 floats
    __m256i bf16_low1 = _mm256_unpacklo_epi16(zeros, v1); // AVX2
    __m256i bf16_low2 = _mm256_unpacklo_epi16(zeros, v2);

    // compute dist
    __m256 diff = _mm256_sub_ps((__m256)bf16_low1, (__m256)bf16_low2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
}

static inline void L2SqrHighHalfStep(__m256i v1, __m256i v2, __m256i zeros, __m256 &sum) {
    // convert next 4:7, 12:15 bf16 to 8 floats
    __m256i bf16_high1 = _mm256_unpackhi_epi16(zeros, v1);
    __m256i bf16_high2 = _mm256_unpackhi_epi16(zeros, v2);

    // compute dist
    __m256 diff = _mm256_sub_ps((__m256)bf16_high1, (__m256)bf16_high2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
}

static inline void L2SqrStep(bfloat16 *&pVect1, bfloat16 *&pVect2, __m256 &sum) {
    // load 16 bf16 elements
    __m256i v1 = _mm256_lddqu_si256((__m256i *)pVect1); // avx
    pVect1 += 16;
    __m256i v2 = _mm256_lddqu_si256((__m256i *)pVect2);
    pVect2 += 16;

    __m256i zeros = _mm256_setzero_si256(); // avx

    // Compute dist for 0:3, 8:11 bf16
    L2SqrLowHalfStep(v1, v2, zeros, sum);

    // Compute dist for 4:7, 12:15 bf16
    L2SqrHighHalfStep(v1, v2, zeros, sum);
}

template <unsigned char residual> // 0..31
float BF16_L2SqrSIMD32_AVX2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // cast to bfloat16 *
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    // define end
    const bfloat16 *pEnd1 = pVect1 + dimension;

    // declare sum
    __m256 sum = _mm256_setzero_ps();

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

        __m256 low_diff = _mm256_sub_ps((__m256)v1_low, (__m256)v2_low);
        if (residual % 16 <= 4) {
            unsigned char constexpr elem_to_calc = residual % 16;
            const __mmask8 mask = (1 << elem_to_calc) - 1;
            low_diff = _mm256_blend_ps(_mm256_setzero_ps(), low_diff, mask);
        } else {
            __m256i v1_high = _mm256_unpackhi_epi16(zeros, v1);
            __m256i v2_high = _mm256_unpackhi_epi16(zeros, v2);
            __m256 high_diff = _mm256_sub_ps((__m256)v1_high, (__m256)v2_high);

            if (4 < residual % 16 && residual % 16 <= 8) {
                // keep only 4 first elemnts of low pack
                const __mmask8 mask2 = (1 << 4) - 1;
                low_diff = _mm256_blend_ps(_mm256_setzero_ps(), low_diff, mask2);

                // keep residual % 16 - 4 first elements of high_diff
                unsigned char constexpr elem_to_calc = residual % 16 - 4;
                const __mmask8 mask3 = (1 << elem_to_calc) - 1;
                high_diff = _mm256_blend_ps(_mm256_setzero_ps(), high_diff, mask3);
            }
            if (8 < residual % 16 && residual % 16 < 12) {
                // keep residual % 16 - 4 first elements of low_diff
                unsigned char constexpr elem_to_calc = residual % 16 - 4;
                const __mmask8 mask2 = (1 << elem_to_calc) - 1;
                low_diff = _mm256_blend_ps(_mm256_setzero_ps(), low_diff, mask2);

                // keep ony 4 first elements of high_diff
                const __mmask8 mask3 = (1 << 4) - 1;
                high_diff = _mm256_blend_ps(_mm256_setzero_ps(), high_diff, mask3);
            }
            if (residual % 16 >= 12) {
                // keep residual % 16 - 8 first elements of high
                unsigned char constexpr elem_to_calc = residual % 16 - 8;
                const __mmask8 mask2 = (1 << elem_to_calc) - 1;
                high_diff = _mm256_blend_ps(_mm256_setzero_ps(), high_diff, mask2);
            }
            sum = _mm256_add_ps(sum, _mm256_mul_ps(high_diff, high_diff));
        }
        sum = _mm256_add_ps(sum, _mm256_mul_ps(low_diff, low_diff));
    }

    // do a single step if residual >=16
    if (residual >= 16) {
        L2SqrStep(pVect1, pVect2, sum);
    }

    // handle 512 bits (32 bfloat16) in chuncks of max SIMD = 256 bits = 16 bfloat16
    do {
        L2SqrStep(pVect1, pVect2, sum);
        L2SqrStep(pVect1, pVect2, sum);
    } while (pVect1 < pEnd1);

    // TmpRes must be 16 bytes aligned
    float PORTABLE_ALIGN32 TmpRes[8];
    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
           TmpRes[7];
}
