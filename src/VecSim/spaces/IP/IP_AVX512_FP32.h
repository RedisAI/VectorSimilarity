/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void InnerProductStep(float *&pVect1, float *&pVect2, __m512 &sum512) {
    __m512 v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    __m512 v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
}

template <unsigned char residual> // 0..15
float FP32_InnerProductSIMD16_AVX512(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    __m512 sum512 = _mm512_setzero_ps();

    // Deal with remainder first. `dim` is more than 16, so we have at least one 16-float block,
    // so mask loading is guaranteed to be safe
    if (residual) {
        __mmask16 constexpr mask = (1 << residual) - 1;
        __m512 v1 = _mm512_maskz_loadu_ps(mask, pVect1);
        pVect1 += residual;
        __m512 v2 = _mm512_maskz_loadu_ps(mask, pVect2);
        pVect2 += residual;
        sum512 = _mm512_mul_ps(v1, v2);
    }

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    do {
        InnerProductStep(pVect1, pVect2, sum512);
    } while (pVect1 < pEnd1);

    return 1.0f - _mm512_reduce_add_ps(sum512);
}

#include <cstring>
/* ***************BF16 FUNCTIONS***************  */
template <unsigned char residual> // 0..15
void cast_fp32_to_bf16_imp(float *& vec, float *vec_out, __mmask32 mask) {
    unsigned char promote = residual ? residual : 16;

    __m512i padded_vec = _mm512_maskz_loadu_epi16(mask, vec);
    // Convert back to float32
    __m512 final_data = _mm512_castsi512_ps(padded_vec);
    _mm512_storeu_ps(vec_out, final_data);
    vec += promote;
}

template <unsigned char residual> // 0..15
float BF16_InnerProductSIMD16_AVX512(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // Convert to bf16 and back to float
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    float vec1[dimension] = {0};
    float vec2[dimension] = {0};

    float *res_vec1p = vec1;
    float *res_vec2p = vec2;

    __mmask32 mask = 0xAAAAAAAA;

    if (residual) {
        __mmask32 res_mask = mask & ((1L << (residual*2)) - 1);

        // load only residual number of float
        cast_fp32_to_bf16_imp<residual>(pVect1, res_vec1p, res_mask);
        cast_fp32_to_bf16_imp<residual>(pVect2, res_vec2p, res_mask);
        res_vec1p += residual;
        res_vec2p += residual;
    }

    do {
        cast_fp32_to_bf16_imp<0>(pVect1, res_vec1p, mask);
        cast_fp32_to_bf16_imp<0>(pVect2, res_vec2p, mask);
        res_vec1p += 16;
        res_vec2p += 16;
    } while (pVect1 < pEnd1);

    return FP32_InnerProductSIMD16_AVX512<residual>(vec1, vec2, dimension);
}

/* ***************FP16 FUNCTIONS***************  */

void cast_fp32_to_fp16_imp(__m512 vec, float *vec_out, unsigned char promote) {
    // convert to fp16
    __m256i v_fp16 = _mm512_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT);
    // convert back to fp32
    __m512 v_back_to_fp32 = _mm512_cvtph_ps(v_fp16);
    // copy to the vectors
    float __attribute__((aligned(64))) mem_addr[16];
    //mem_addr needs to be aligned and since we (may) promoted vec_out by residual,
    // it went out of alignment.
    _mm512_store_ps(mem_addr, v_back_to_fp32);

    memcpy(vec_out, mem_addr, sizeof(mem_addr));
}

template <unsigned char residual> // 0..15
void cast_fp32_to_fp16(float *&pVect1, float *&pVect2, float *vec1_out, float *vec2_out) {
    unsigned char promote = residual ? residual : 16;
    // Load original fp32 chunk
    __m512 v1, v2;
    if (residual) {
        // load only residual number of float
        __mmask16 constexpr mask = (1 << residual) - 1;

        v1 = _mm512_maskz_loadu_ps(mask, pVect1);
        v2 = _mm512_maskz_loadu_ps(mask, pVect2);
    } else {
        v1 = _mm512_loadu_ps(pVect1);
        v2 = _mm512_loadu_ps(pVect2);
    }
    pVect1 += promote;
    pVect2 += promote;


    cast_fp32_to_fp16_imp(v1, vec1_out, promote);
    cast_fp32_to_fp16_imp(v2, vec2_out, promote);
}

template <unsigned char residual> // 0..15
float FP16_InnerProductSIMD16_AVX512(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // Convert to bf16 and back to float
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect1 + dimension;

    float vec1[dimension] = {0};
    float vec2[dimension] = {0};

    float *res_vec1p = vec1;
    float *res_vec2p = vec2;
    if (residual) {
        // load only residual number of float
        cast_fp32_to_fp16<residual>(pVect1, pVect2, res_vec1p, res_vec2p);
        res_vec1p += residual;
        res_vec2p += residual;
    }

    do {
        cast_fp32_to_fp16<0>(pVect1, pVect2, res_vec1p, res_vec2p);
        res_vec1p += 16;
        res_vec2p += 16;
    } while (pVect1 < pEnd1);

    return FP32_InnerProductSIMD16_AVX512<residual>(vec1, vec2, dimension);
}
