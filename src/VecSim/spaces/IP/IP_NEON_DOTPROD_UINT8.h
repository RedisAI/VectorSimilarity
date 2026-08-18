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
#include <arm_neon.h>

__attribute__((always_inline)) static inline void InnerProductOp(uint8x16_t &v1, uint8x16_t &v2,
                                                                 uint32x4_t &sum) {
    sum = vdotq_u32(sum, v1, v2);
}

__attribute__((always_inline)) static inline void
InnerProductStep(uint8_t *&pVect1, uint8_t *&pVect2, uint32x4_t &sum) {
    // Load 16 uint8 elements (16 bytes) into NEON registers
    uint8x16_t v1 = vld1q_u8(pVect1);
    uint8x16_t v2 = vld1q_u8(pVect2);
    InnerProductOp(v1, v2, sum);

    pVect1 += 16;
    pVect2 += 16;
}

// Returns the raw integer total, and is static and always_inline; see the NEON header for why each
// of those three matters. The internal linkage is what keeps this body and the NEON one apart
// despite the shared name.
template <unsigned char residual> // 0..63
__attribute__((always_inline)) static inline uint32_t
UINT8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
    uint8_t *pVect1 = (uint8_t *)pVect1v;
    uint8_t *pVect2 = (uint8_t *)pVect2v;

    // Initialize multiple sum accumulators for better parallelism
    uint32x4_t sum0 = vdupq_n_u32(0);
    uint32x4_t sum1 = vdupq_n_u32(0);

    constexpr size_t final_residual = residual % 16;
    if constexpr (final_residual > 0) {
        constexpr uint8x16_t mask = {
            0xFF,
            (final_residual >= 2) ? 0xFF : 0,
            (final_residual >= 3) ? 0xFF : 0,
            (final_residual >= 4) ? 0xFF : 0,
            (final_residual >= 5) ? 0xFF : 0,
            (final_residual >= 6) ? 0xFF : 0,
            (final_residual >= 7) ? 0xFF : 0,
            (final_residual >= 8) ? 0xFF : 0,
            (final_residual >= 9) ? 0xFF : 0,
            (final_residual >= 10) ? 0xFF : 0,
            (final_residual >= 11) ? 0xFF : 0,
            (final_residual >= 12) ? 0xFF : 0,
            (final_residual >= 13) ? 0xFF : 0,
            (final_residual >= 14) ? 0xFF : 0,
            (final_residual >= 15) ? 0xFF : 0,
            0,
        };

        // Load data directly from input vectors
        uint8x16_t v1 = vld1q_u8(pVect1);
        uint8x16_t v2 = vld1q_u8(pVect2);

        // Zero vector for replacement
        uint8x16_t zeros = vdupq_n_u8(0);

        // Apply bit select to zero out irrelevant elements
        v1 = vbslq_u8(mask, v1, zeros);
        v2 = vbslq_u8(mask, v2, zeros);
        InnerProductOp(v1, v2, sum1);
        pVect1 += final_residual;
        pVect2 += final_residual;
    }

    // Process 64 elements at a time in the main loop
    const size_t num_of_chunks = dimension / 64;

    for (size_t i = 0; i < num_of_chunks; i++) {
        InnerProductStep(pVect1, pVect2, sum0);
        InnerProductStep(pVect1, pVect2, sum1);
        InnerProductStep(pVect1, pVect2, sum0);
        InnerProductStep(pVect1, pVect2, sum1);
    }

    constexpr size_t residual_chunks = residual / 16;

    if constexpr (residual_chunks > 0) {
        if constexpr (residual_chunks >= 1) {
            InnerProductStep(pVect1, pVect2, sum0);
        }
        if constexpr (residual_chunks >= 2) {
            InnerProductStep(pVect1, pVect2, sum1);
        }
        if constexpr (residual_chunks >= 3) {
            InnerProductStep(pVect1, pVect2, sum0);
        }
    }

    uint32x4_t total_sum = vaddq_u32(sum0, sum1);

    // ADDV, unsigned. The total reaches 255*255*dim, so the previous int32_t receiving this
    // wrapped negative from dimension 33,027. Exact for up to spaces::UINT8_CHUNK_ELEMENTS
    // elements, which is what the caller guarantees.
    return vaddvq_u32(total_sum);
}

template <unsigned char residual> // 0..63
float UINT8_InnerProductSIMD16_NEON_DOTPROD(const void *pVect1v, const void *pVect2v,
                                            size_t dimension) {
    return 1 - static_cast<int64_t>(UINT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension));
}

template <unsigned char residual> // 0..63
float UINT8_CosineSIMD_NEON_DOTPROD(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float ip = static_cast<float>(UINT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension));
    const float norm_v1 = load_unaligned<float>(static_cast<const uint8_t *>(pVect1v) + dimension);
    const float norm_v2 = load_unaligned<float>(static_cast<const uint8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}

// One out-of-line copy of the residual-0 kernel, called once per whole chunk. Inlining it into
// every chunked wrapper cost text for nothing: one call per 65,536 elements is unmeasurable, and
// keeping it out of line leaves the first chunk's register allocation alone.
__attribute__((noinline)) static uint32_t
UINT8_InnerProductFullChunk_NEON_DOTPROD(const uint8_t *pVect1, const uint8_t *pVect2,
                                         size_t dimension) {
    return UINT8_InnerProductImp<0>(pVect1, pVect2, dimension);
}

// See the NEON header for why each chunk's 32-bit total is exact and why the first chunk absorbs
// the residual.
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
        total += UINT8_InnerProductFullChunk_NEON_DOTPROD(pVect1, pVect2, step);
        pVect1 += step;
        pVect2 += step;
        remaining -= step;
    }
    return total;
}

template <unsigned char residual> // 0..63
float UINT8_InnerProductSIMD16_NEON_DOTPROD_Chunked(const void *pVect1v, const void *pVect2v,
                                                    size_t dimension) {
    return 1 - static_cast<int64_t>(
                   UINT8_InnerProductChunkedImp<residual>(pVect1v, pVect2v, dimension));
}

template <unsigned char residual> // 0..63
float UINT8_CosineSIMD_NEON_DOTPROD_Chunked(const void *pVect1v, const void *pVect2v,
                                            size_t dimension) {
    float ip =
        static_cast<float>(UINT8_InnerProductChunkedImp<residual>(pVect1v, pVect2v, dimension));
    const float norm_v1 = load_unaligned<float>(static_cast<const uint8_t *>(pVect1v) + dimension);
    const float norm_v2 = load_unaligned<float>(static_cast<const uint8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}
