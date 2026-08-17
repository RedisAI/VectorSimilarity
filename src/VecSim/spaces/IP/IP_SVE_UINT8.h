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
#include <arm_sve.h>

inline void InnerProductStep(const uint8_t *&pVect1, const uint8_t *&pVect2, size_t &offset,
                             svuint32_t &sum, const size_t chunk) {
    svbool_t pg = svptrue_b8();

    // Load uint8 vectors
    svuint8_t v1_ui8 = svld1_u8(pg, pVect1 + offset);
    svuint8_t v2_ui8 = svld1_u8(pg, pVect2 + offset);

    sum = svdot_u32(sum, v1_ui8, v2_ui8);

    offset += chunk; // Move to the next set of uint8 elements
}

// Split so the chunked wrapper below can fold each chunk's total in 64 bits; summing the float
// results per chunk would round each one. always_inline because the chunked wrapper calls this
// twice, and GCC outlines a template once it has several callers, which also costs the plain
// wrapper its inlining. static keeps each translation unit's copy to itself: SVE.cpp and SVE2.cpp
// both include this header, and other headers define the same name with different bodies.
template <bool partial_chunk, unsigned char additional_steps>
__attribute__((always_inline)) static inline uint32_t
UINT8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const uint8_t *pVect1 = reinterpret_cast<const uint8_t *>(pVect1v);
    const uint8_t *pVect2 = reinterpret_cast<const uint8_t *>(pVect2v);

    size_t offset = 0;
    const size_t vl = svcntb();
    const size_t chunk_size = 4 * vl;

    // Each innerProductStep adds maximum 2^8 & 2^8 = 2^16
    // Therefore, on a single accumulator, we can perform 2^16 steps before overflowing
    // That scenario will happen only is the dimension of the vector is larger than 16*4*2^16 = 2^22
    // (16 uint8 in 1 SVE register) * (4 accumulators) * (2^16 steps)
    // We can safely assume that the dimension is smaller than that
    // So using int32_t is safe

    svuint32_t sum0 = svdup_u32(0);
    svuint32_t sum1 = svdup_u32(0);
    svuint32_t sum2 = svdup_u32(0);
    svuint32_t sum3 = svdup_u32(0);

    size_t num_chunks = dimension / chunk_size;

    for (size_t i = 0; i < num_chunks; ++i) {
        InnerProductStep(pVect1, pVect2, offset, sum0, vl);
        InnerProductStep(pVect1, pVect2, offset, sum1, vl);
        InnerProductStep(pVect1, pVect2, offset, sum2, vl);
        InnerProductStep(pVect1, pVect2, offset, sum3, vl);
    }

    // Process remaining complete SVE vectors that didn't fit into the main loop
    // These are full vector operations (0-3 elements)
    if constexpr (additional_steps > 0) {
        if constexpr (additional_steps >= 1) {
            InnerProductStep(pVect1, pVect2, offset, sum0, vl);
        }
        if constexpr (additional_steps >= 2) {
            InnerProductStep(pVect1, pVect2, offset, sum1, vl);
        }
        if constexpr (additional_steps >= 3) {
            InnerProductStep(pVect1, pVect2, offset, sum2, vl);
        }
    }

    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b8_u64(offset, dimension);

        svuint8_t v1_ui8 = svld1_u8(pg, pVect1 + offset); // Load uint8 vectors
        svuint8_t v2_ui8 = svld1_u8(pg, pVect2 + offset); // Load uint8 vectors

        sum3 = svdot_u32(sum3, v1_ui8, v2_ui8);

        pVect1 += vl;
        pVect2 += vl;
    }

    sum0 = svadd_u32_x(svptrue_b32(), sum0, sum1);
    sum2 = svadd_u32_x(svptrue_b32(), sum2, sum3);

    // svaddv_u32 reduces into a 64-bit scalar; the previous int32_t truncated it, which wrapped
    // negative from dimension 33,027. Narrowed to uint32_t, which is exact for up to
    // spaces::UINT8_CHUNK_ELEMENTS elements, and that is what the caller guarantees.
    return static_cast<uint32_t>(svaddv_u32(svptrue_b32(), svadd_u32_x(svptrue_b32(), sum0, sum2)));
}

template <bool partial_chunk, unsigned char additional_steps>
float UINT8_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - static_cast<float>(UINT8_InnerProductImp<partial_chunk, additional_steps>(
                      pVect1v, pVect2v, dimension));
}

template <bool partial_chunk, unsigned char additional_steps>
float UINT8_CosineSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float ip = static_cast<float>(
        UINT8_InnerProductImp<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension));
    const float norm_v1 = load_unaligned<float>(static_cast<const uint8_t *>(pVect1v) + dimension);
    const float norm_v2 = load_unaligned<float>(static_cast<const uint8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}

// Chunked variant, selected by the chooser past spaces::UINT8_CHUNK_ELEMENTS. Each chunk's 32-bit
// total is exact because 65025 * 65536 = 4,261,478,400 <= UINT32_MAX, and every contribution is
// non-negative, so no accumulator lane can exceed the chunk total either. That is the whole
// correctness argument; no reasoning about how work spreads across lanes is needed.
template <bool partial_chunk, unsigned char additional_steps>
static inline uint64_t UINT8_InnerProductChunkedImp(const void *pVect1v, const void *pVect2v,
                                                    size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // The SVE vector length is a runtime value, so unlike the fixed-width kernels the split is
    // computed here rather than at compile time. chunk_size matches the kernel's 4-accumulator main
    // loop, and tail is the part the template parameters describe.
    const size_t chunk_size = 4 * svcntb();
    const size_t tail = dimension % chunk_size;
    const size_t max_step = spaces::UINT8_CHUNK_ELEMENTS / chunk_size * chunk_size;
    const size_t first = tail + (spaces::UINT8_CHUNK_ELEMENTS - tail) / chunk_size * chunk_size;

    // first keeps this instantiation's own residual shape: it is congruent to dimension modulo
    // chunk_size, so partial_chunk and additional_steps still describe its tail.
    uint64_t total = UINT8_InnerProductImp<partial_chunk, additional_steps>(pVect1, pVect2, first);
    pVect1 += first;
    pVect2 += first;
    size_t remaining = dimension - first;

    // remaining is a whole multiple of chunk_size, and so is every step, which is the <false, 0>
    // shape: no partial vector and no leftover single steps.
    while (remaining) {
        const size_t step = remaining < max_step ? remaining : max_step;
        total += UINT8_InnerProductImp<false, 0>(pVect1, pVect2, step);
        pVect1 += step;
        pVect2 += step;
        remaining -= step;
    }
    return total;
}

template <bool partial_chunk, unsigned char additional_steps>
float UINT8_InnerProductSIMD_SVE_Chunked(const void *pVect1v, const void *pVect2v,
                                         size_t dimension) {
    return 1.0f - static_cast<float>(UINT8_InnerProductChunkedImp<partial_chunk, additional_steps>(
                      pVect1v, pVect2v, dimension));
}

template <bool partial_chunk, unsigned char additional_steps>
float UINT8_CosineSIMD_SVE_Chunked(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float ip = static_cast<float>(
        UINT8_InnerProductChunkedImp<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension));
    const float norm_v1 = load_unaligned<float>(static_cast<const uint8_t *>(pVect1v) + dimension);
    const float norm_v2 = load_unaligned<float>(static_cast<const uint8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}
