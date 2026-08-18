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
#include "VecSim/spaces/spaces.h" // spaces::UINT8_CHUNK_ELEMENTS

static inline void InnerProductStep(uint8_t *&pVect1, uint8_t *&pVect2, __m512i &sum) {
    __m512i va = _mm512_loadu_epi8(pVect1); // AVX512BW
    pVect1 += 64;

    __m512i vb = _mm512_loadu_epi8(pVect2); // AVX512BW
    pVect2 += 64;

    __m512i va_lo = _mm512_unpacklo_epi8(va, _mm512_setzero_si512()); // AVX512BW
    __m512i vb_lo = _mm512_unpacklo_epi8(vb, _mm512_setzero_si512());
    sum = _mm512_dpwssd_epi32(sum, va_lo, vb_lo);

    __m512i va_hi = _mm512_unpackhi_epi8(va, _mm512_setzero_si512()); // AVX512BW
    __m512i vb_hi = _mm512_unpackhi_epi8(vb, _mm512_setzero_si512());
    sum = _mm512_dpwssd_epi32(sum, va_hi, vb_hi);

    // _mm512_dpwssd_epi32(src, a, b)
    // Multiply groups of 2 adjacent pairs of signed 16-bit integers in `a` with corresponding
    // 16-bit integers in `b`, producing 2 intermediate signed 32-bit results. Sum these 2 results
    // with the corresponding 32-bit integer in src, and store the packed 32-bit results in dst.
}

// always_inline, not merely inline: the chunked wrapper below calls this twice, and without the
// attribute GCC outlines it once it has several callers, which also costs the plain wrapper its
// inlining. Measured: the plain residual-0 wrapper went from 33 instructions to 9 plus a call.
template <unsigned char residual> // 0..63
__attribute__((always_inline)) static inline uint32_t
UINT8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
    uint8_t *pVect1 = (uint8_t *)pVect1v;
    uint8_t *pVect2 = (uint8_t *)pVect2v;

    const uint8_t *pEnd1 = pVect1 + dimension;

    __m512i sum = _mm512_setzero_epi32();

    // Deal with remainder first.
    if constexpr (residual) {
        if constexpr (residual < 32) {
            constexpr __mmask32 mask = (1LU << residual) - 1;
            __m256i temp_a = _mm256_maskz_loadu_epi8(mask, pVect1);
            __m512i va = _mm512_cvtepu8_epi16(temp_a);

            __m256i temp_b = _mm256_maskz_loadu_epi8(mask, pVect2);
            __m512i vb = _mm512_cvtepu8_epi16(temp_b);

            sum = _mm512_dpwssd_epi32(sum, va, vb);
        } else if constexpr (residual == 32) {
            __m256i temp_a = _mm256_loadu_epi8(pVect1);
            __m512i va = _mm512_cvtepu8_epi16(temp_a);

            __m256i temp_b = _mm256_loadu_epi8(pVect2);
            __m512i vb = _mm512_cvtepu8_epi16(temp_b);

            sum = _mm512_dpwssd_epi32(sum, va, vb);
        } else {
            constexpr __mmask64 mask = (1LU << residual) - 1;
            __m512i va = _mm512_maskz_loadu_epi8(mask, pVect1);
            __m512i vb = _mm512_maskz_loadu_epi8(mask, pVect2);

            __m512i va_lo = _mm512_unpacklo_epi8(va, _mm512_setzero_si512());
            __m512i vb_lo = _mm512_unpacklo_epi8(vb, _mm512_setzero_si512());
            sum = _mm512_dpwssd_epi32(sum, va_lo, vb_lo);

            __m512i va_hi = _mm512_unpackhi_epi8(va, _mm512_setzero_si512());
            __m512i vb_hi = _mm512_unpackhi_epi8(vb, _mm512_setzero_si512());
            sum = _mm512_dpwssd_epi32(sum, va_hi, vb_hi);
        }
        pVect1 += residual;
        pVect2 += residual;

        // We dealt with the residual part.
        // We are left with some multiple of 64-uint_8 (might be 0).
        while (pVect1 < pEnd1) {
            InnerProductStep(pVect1, pVect2, sum);
        }
    } else {
        // We have no residual, we have some non-zero multiple of 64-uint_8.
        do {
            InnerProductStep(pVect1, pVect2, sum);
        } while (pVect1 < pEnd1);
    }

    // Unsigned reduce. The lanes are in range individually, but their total reaches 255*255*dim,
    // which passes INT_MAX from dimension 33,027, so reading the result as a signed int wrapped it.
    // The intrinsic's adds are vector operations, so the bit pattern is already correct modulo
    // 2^32 and this cast simply reads it as unsigned. Exact for up to
    // spaces::UINT8_CHUNK_ELEMENTS elements, which is what the caller guarantees.
    return static_cast<uint32_t>(_mm512_reduce_add_epi32(sum));
}

template <unsigned char residual> // 0..63
float UINT8_InnerProductSIMD64_AVX512F_BW_VL_VNNI(const void *pVect1v, const void *pVect2v,
                                                  size_t dimension) {

    return 1 - static_cast<int64_t>(UINT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension));
}
template <unsigned char residual> // 0..63
float UINT8_CosineSIMD64_AVX512F_BW_VL_VNNI(const void *pVect1v, const void *pVect2v,
                                            size_t dimension) {
    float ip = static_cast<float>(UINT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension));
    const float norm_v1 = load_unaligned<float>(static_cast<const uint8_t *>(pVect1v) + dimension);
    const float norm_v2 = load_unaligned<float>(static_cast<const uint8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}

// One out-of-line copy of the residual-0 kernel, called once per whole chunk. Inlining it into
// every chunked wrapper cost text for nothing: one call per 65,536 elements is unmeasurable, and
// keeping it out of line leaves the first chunk's register allocation alone.
__attribute__((noinline)) static uint32_t
UINT8_InnerProductFullChunk_AVX512F_BW_VL_VNNI(const uint8_t *pVect1, const uint8_t *pVect2,
                                               size_t dimension) {
    return UINT8_InnerProductImp<0>(pVect1, pVect2, dimension);
}

// Chunked variants, selected by the chooser for dimensions past spaces::UINT8_CHUNK_ELEMENTS. Each
// chunk's 32-bit total is exact, and the chunks are folded in 64 bits, so these are exact at any
// dimension. The plain wrappers above are left untouched so their inlining is unaffected, and the
// choice is made once per index rather than per call.
//
// The first chunk absorbs the residual, which leaves the remaining length a whole multiple of 64,
// so every later chunk satisfies the residual-0 kernel's precondition.
template <unsigned char residual> // 0..63
static inline uint64_t UINT8_InnerProductChunkedImp(const void *pVect1v, const void *pVect2v,
                                                    size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);

    constexpr size_t chunk = spaces::UINT8_CHUNK_ELEMENTS;
    // Runtime min rather than the constant alone: with a compile-time trip count GCC split this
    // loop's accumulator and copied it in and out every 64 elements, measured at 8-9.5% on Ice
    // Lake. The min also makes this wrapper correct at any dimension, not only past the chunk size.
    constexpr size_t first_chunk = residual + (chunk - residual) / 64 * 64;
    const size_t first = dimension < first_chunk ? dimension : first_chunk;
    uint64_t total = UINT8_InnerProductImp<residual>(pVect1, pVect2, first);
    pVect1 += first;
    pVect2 += first;
    size_t remaining = dimension - first;

    while (remaining) {
        const size_t step = remaining < chunk ? remaining : chunk;
        total += UINT8_InnerProductFullChunk_AVX512F_BW_VL_VNNI(pVect1, pVect2, step);
        pVect1 += step;
        pVect2 += step;
        remaining -= step;
    }
    return total;
}

template <unsigned char residual> // 0..63
float UINT8_InnerProductSIMD64_AVX512F_BW_VL_VNNI_Chunked(const void *pVect1v, const void *pVect2v,
                                                          size_t dimension) {
    return 1 - static_cast<int64_t>(
                   UINT8_InnerProductChunkedImp<residual>(pVect1v, pVect2v, dimension));
}

template <unsigned char residual> // 0..63
float UINT8_CosineSIMD64_AVX512F_BW_VL_VNNI_Chunked(const void *pVect1v, const void *pVect2v,
                                                    size_t dimension) {
    const float ip =
        static_cast<float>(UINT8_InnerProductChunkedImp<residual>(pVect1v, pVect2v, dimension));
    const float norm_v1 = load_unaligned<float>(static_cast<const uint8_t *>(pVect1v) + dimension);
    const float norm_v2 = load_unaligned<float>(static_cast<const uint8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}
