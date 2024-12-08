/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void InnerProductStep(int8_t *&pVect1, int8_t *&pVect2, __m512i &sum) {
    __m256i temp_a = _mm256_loadu_epi8(pVect1);
    __m512i va = _mm512_cvtepi8_epi16(temp_a);
    pVect1 += 32;

    __m256i temp_b = _mm256_loadu_epi8(pVect2);
    __m512i vb = _mm512_cvtepi8_epi16(temp_b);
    pVect2 += 32;

    // _mm512_dpwssd_epi32(src, a, b)
    // Multiply groups of 2 adjacent pairs of signed 16-bit integers in `a` with corresponding
    // 16-bit integers in `b`, producing 2 intermediate signed 32-bit results. Sum these 2 results
    // with the corresponding 32-bit integer in src, and store the packed 32-bit results in dst.
    sum = _mm512_dpwssd_epi32(sum, va, vb);
}

template <unsigned char residual> // 0..32
static inline int INT8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;

    const int8_t *pEnd1 = pVect1 + dimension;

    __m512i sum = _mm512_setzero_epi32();

    // Deal with remainder first. `dim` is more than 32, so we have at least one 32-int_8 block,
    // so mask loading is guaranteed to be safe
    if constexpr (residual) {
        __mmask32 mask = (1LU << residual) - 1;
        __m256i temp_a = _mm256_maskz_loadu_epi8(mask, pVect1);
        __m512i va = _mm512_cvtepi8_epi16(temp_a);
        pVect1 += residual;

        __m256i temp_b = _mm256_maskz_loadu_epi8(mask, pVect2);
        __m512i vb = _mm512_cvtepi8_epi16(temp_b);
        pVect2 += residual;

        sum = _mm512_dpwssd_epi32(sum, va, vb);
    }

    // We dealt with the residual part. We are left with some multiple of 32-int_8.
    do {
        InnerProductStep(pVect1, pVect2, sum);
    } while (pVect1 < pEnd1);

    return _mm512_reduce_add_epi32(sum);
}

template <unsigned char residual> // 0..32
float INT8_InnerProductSIMD32_AVX512F_BW_VL_VNNI(const void *pVect1v, const void *pVect2v,
                                                 size_t dimension) {

    return 1 - INT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension);
}
template <unsigned char residual> // 0..32
float INT8_CosineSIMD32_AVX512F_BW_VL_VNNI(const void *pVect1v, const void *pVect2v,
                                           size_t dimension) {
    float norm_v1 = *(float *)((int8_t *)pVect1v + dimension);
    float norm_v2 = *(float *)((int8_t *)pVect2v + dimension);
    return 1.0f -
           float(INT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension)) / (norm_v1 * norm_v2);
}
