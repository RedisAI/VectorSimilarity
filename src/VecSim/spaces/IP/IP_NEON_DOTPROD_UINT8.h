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
#include "VecSim/spaces/uint8_chunking.h"
#include <arm_neon.h>

// uint8 inner product: Imp returns the raw integer total and the wrappers convert it. The chooser
// picks plain up to spaces::UINT8_CHUNK_ELEMENTS and chunked above it, once per index; spaces.h
// carries the chunk-size argument.
//
// Imp is static because IP_NEON_UINT8.h defines the same name with a different body and
// aarch64 gcc 12.3 outlines it, so shared linkage lets a NEON call site execute udot and fault
// where asimddp is absent. always_inline keeps the plain wrapper's codegen unchanged.
// The chunked wrapper's first chunk absorbs the residual and its length is a runtime min against
// the dimension, because a compile-time trip count cost 8-9.5% in accumulator copies; later chunks
// share one out-of-line copy of the kernel.
// The inner product subtracts in integer and converts once, signed because the total is not.

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

    // ADDV, unsigned, and exact for up to spaces::UINT8_CHUNK_ELEMENTS elements.
    return vaddvq_u32(total_sum);
}

template <unsigned char residual> // 0..63
float UINT8_InnerProductSIMD16_NEON_DOTPROD(const void *pVect1v, const void *pVect2v,
                                            size_t dimension) {
    const auto ip =
        static_cast<int64_t>(UINT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension));
    return static_cast<float>(1 - ip);
}

template <unsigned char residual> // 0..63
float UINT8_CosineSIMD_NEON_DOTPROD(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float ip = static_cast<float>(UINT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension));
    const float norm_v1 = load_unaligned<float>(static_cast<const uint8_t *>(pVect1v) + dimension);
    const float norm_v2 = load_unaligned<float>(static_cast<const uint8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}

__attribute__((noinline)) static uint32_t
UINT8_InnerProductFullChunk_NEON_DOTPROD(const uint8_t *pVect1, const uint8_t *pVect2,
                                         size_t dimension) {
    return UINT8_InnerProductImp<0>(pVect1, pVect2, dimension);
}

template <unsigned char residual> // 0..63
struct UINT8_IPChunkKernel_NEON_DOTPROD {
    static constexpr size_t granule() { return 64; }
    __attribute__((always_inline)) static inline uint32_t
    first(const uint8_t *pVect1, const uint8_t *pVect2, size_t dimension) {
        return UINT8_InnerProductImp<residual>(pVect1, pVect2, dimension);
    }
    static uint32_t rest(const uint8_t *pVect1, const uint8_t *pVect2, size_t dimension) {
        return UINT8_InnerProductFullChunk_NEON_DOTPROD(pVect1, pVect2, dimension);
    }
};

template <unsigned char residual> // 0..63
float UINT8_InnerProductSIMD16_NEON_DOTPROD_Chunked(const void *pVect1v, const void *pVect2v,
                                                    size_t dimension) {
    const auto ip = static_cast<int64_t>(
        spaces::uint8_chunked_total<UINT8_IPChunkKernel_NEON_DOTPROD<residual>>(pVect1v, pVect2v,
                                                                                dimension));
    return static_cast<float>(1 - ip);
}

template <unsigned char residual> // 0..63
float UINT8_CosineSIMD_NEON_DOTPROD_Chunked(const void *pVect1v, const void *pVect2v,
                                            size_t dimension) {
    float ip =
        static_cast<float>(spaces::uint8_chunked_total<UINT8_IPChunkKernel_NEON_DOTPROD<residual>>(
            pVect1v, pVect2v, dimension));
    const float norm_v1 = load_unaligned<float>(static_cast<const uint8_t *>(pVect1v) + dimension);
    const float norm_v2 = load_unaligned<float>(static_cast<const uint8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}
