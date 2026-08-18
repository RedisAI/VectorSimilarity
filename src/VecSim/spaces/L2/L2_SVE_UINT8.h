/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/spaces.h" // spaces::UINT8_CHUNK_ELEMENTS
#include <arm_sve.h>

// Aligned step using svptrue_b8()
inline void L2SquareStep(const uint8_t *&pVect1, const uint8_t *&pVect2, size_t &offset,
                         svuint32_t &sum, const size_t chunk) {
    svbool_t pg = svptrue_b8();
    // Note: Because all the bits are 1, the extention to 16 and 32 bits does not make a difference
    // Otherwise, pg should be recalculated for 16 and 32 operations

    svuint8_t v1_ui8 = svld1_u8(pg, pVect1 + offset); // Load uint8 vectors from pVect1
    svuint8_t v2_ui8 = svld1_u8(pg, pVect2 + offset); // Load uint8 vectors from pVect2

    svuint8_t abs_diff = svabd_u8_x(pg, v1_ui8, v2_ui8);

    sum = svdot_u32(sum, abs_diff, abs_diff);

    offset += chunk; // Move to the next set of uint8 elements
}

// Split so the chunked wrapper below can fold each chunk's total in 64 bits; summing the float
// results per chunk would round each one. always_inline because the chunked wrapper calls this
// twice, and GCC outlines a template once it has several callers, which also costs the plain
// wrapper its inlining. static keeps each translation unit's copy to itself: SVE.cpp and SVE2.cpp
// both include this header, and other headers define the same name with different bodies.
template <bool partial_chunk, unsigned char additional_steps>
__attribute__((always_inline)) static inline uint32_t
UINT8_L2SqrImp_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const uint8_t *pVect1 = reinterpret_cast<const uint8_t *>(pVect1v);
    const uint8_t *pVect2 = reinterpret_cast<const uint8_t *>(pVect2v);

    // number of uint8 per SVE register
    const size_t vl = svcntb();
    const size_t chunk_size = 4 * vl;
    svbool_t all = svptrue_b8();

    // Each L2SquareStep adds maximum (2^8)^2 = 2^16
    // Therefor, on a single accumulator, we can perform 2^16 steps before overflowing
    // That scenario will happen only is the dimension of the vector is larger than 16*4*2^16 = 2^22
    // (16 uint8 in 1 SVE register) * (4 accumulators) * (2^16 steps)
    // We can safely assume that the dimension is smaller than that
    // So using uint32_t is safe

    svuint32_t sum0 = svdup_u32(0);
    svuint32_t sum1 = svdup_u32(0);
    svuint32_t sum2 = svdup_u32(0);
    svuint32_t sum3 = svdup_u32(0);

    size_t offset = 0;
    size_t num_main_blocks = dimension / chunk_size;

    for (size_t i = 0; i < num_main_blocks; ++i) {
        L2SquareStep(pVect1, pVect2, offset, sum0, vl);
        L2SquareStep(pVect1, pVect2, offset, sum1, vl);
        L2SquareStep(pVect1, pVect2, offset, sum2, vl);
        L2SquareStep(pVect1, pVect2, offset, sum3, vl);
    }

    if constexpr (additional_steps > 0) {
        if constexpr (additional_steps >= 1) {
            L2SquareStep(pVect1, pVect2, offset, sum0, vl);
        }
        if constexpr (additional_steps >= 2) {
            L2SquareStep(pVect1, pVect2, offset, sum1, vl);
        }
        if constexpr (additional_steps >= 3) {
            L2SquareStep(pVect1, pVect2, offset, sum2, vl);
        }
    }

    if constexpr (partial_chunk) {

        svbool_t pg = svwhilelt_b8_u64(offset, dimension);
        svuint8_t v1_ui8 = svld1_u8(pg, pVect1 + offset); // Load uint8 vectors from pVect1
        svuint8_t v2_ui8 = svld1_u8(pg, pVect2 + offset); // Load uint8 vectors from pVect2

        svuint8_t abs_diff = svabd_u8_x(all, v1_ui8, v2_ui8);

        // Can sum with taking into account pg because svld1 will set inactive lanes to 0
        sum3 = svdot_u32(sum3, abs_diff, abs_diff);
    }

    sum0 = svadd_u32_x(all, sum0, sum1);
    sum2 = svadd_u32_x(all, sum2, sum3);
    svuint32_t sum_all = svadd_u32_x(all, sum0, sum2);
    // svaddv_u32 reduces into a 64-bit scalar. The total is a sum of squared byte differences,
    // reaching 255*255*dim, and is exact for up to spaces::UINT8_CHUNK_ELEMENTS elements, which is
    // what the caller guarantees.
    return static_cast<uint32_t>(svaddv_u32(svptrue_b32(), sum_all));
}

template <bool partial_chunk, unsigned char additional_steps>
float UINT8_L2SqrSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return static_cast<float>(
        UINT8_L2SqrImp_SVE<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension));
}

// One out-of-line copy of the residual-0 kernel, called once per whole chunk. Inlining it into
// every chunked wrapper cost text for nothing: one call per 65,536 elements is unmeasurable, and
// keeping it out of line leaves the first chunk's register allocation alone.
__attribute__((noinline)) static uint32_t
UINT8_L2SqrFullChunk_SVE(const uint8_t *pVect1, const uint8_t *pVect2, size_t dimension) {
    return UINT8_L2SqrImp_SVE<false, 0>(pVect1, pVect2, dimension);
}

// Chunked variant, selected by the chooser past spaces::UINT8_CHUNK_ELEMENTS. Each chunk's 32-bit
// total is exact because 65025 * 65536 = 4,261,478,400 <= UINT32_MAX, and every contribution is
// non-negative, so no accumulator lane can exceed the chunk total either. That is the whole
// correctness argument; no reasoning about how work spreads across lanes is needed.
template <bool partial_chunk, unsigned char additional_steps>
float UINT8_L2SqrSIMD_SVE_Chunked(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // The SVE vector length is a runtime value, so unlike the fixed-width kernels the split is
    // computed here rather than at compile time. chunk_size matches the kernel's 4-accumulator main
    // loop, and tail is the part the template parameters describe.
    const size_t chunk_size = 4 * svcntb();
    const size_t tail = dimension % chunk_size;
    const size_t max_step = spaces::UINT8_CHUNK_ELEMENTS / chunk_size * chunk_size;
    const size_t first_chunk =
        tail + (spaces::UINT8_CHUNK_ELEMENTS - tail) / chunk_size * chunk_size;
    // Clamped so this wrapper is correct at any dimension, not only past the chunk size.
    const size_t first = dimension < first_chunk ? dimension : first_chunk;

    // first keeps this instantiation's own residual shape: it is congruent to dimension modulo
    // chunk_size, so partial_chunk and additional_steps still describe its tail.
    uint64_t total = UINT8_L2SqrImp_SVE<partial_chunk, additional_steps>(pVect1, pVect2, first);
    pVect1 += first;
    pVect2 += first;
    size_t remaining = dimension - first;

    // remaining is a whole multiple of chunk_size, and so is every step, which is the <false, 0>
    // shape: no partial vector and no leftover single steps.
    while (remaining) {
        const size_t step = remaining < max_step ? remaining : max_step;
        total += UINT8_L2SqrFullChunk_SVE(pVect1, pVect2, step);
        pVect1 += step;
        pVect2 += step;
        remaining -= step;
    }
    return static_cast<float>(total);
}
