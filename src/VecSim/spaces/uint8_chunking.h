/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

// Shared chunked-accumulation driver for the uint8 SIMD kernels. It splits a distance
// computation into chunks of at most UINT8_CHUNK_ELEMENTS elements so each chunk's 32-bit
// SIMD total stays exact, and folds the per-chunk totals into a 64-bit scalar. The same
// formula serves both fixed-width kernels (granule 64) and SVE (granule 4 * svcntb()); only
// the granule differs. The caller supplies a Kernel adapter with:
//   static size_t granule()                                            - the kernel's block size
//   static uint32_t first(const uint8_t *, const uint8_t *, size_t)    - the residual-bearing
//                                                                         kernel, shape already
//                                                                         bound
//   static uint32_t rest(const uint8_t *, const uint8_t *, size_t)     - the out-of-line
//                                                                         residual-0 kernel
// Invariants below hold for any Kernel whose granule() is in (0, UINT8_CHUNK_ELEMENTS]; all
// current adapters return 64 (fixed-width) or 4 * svcntb() (SVE, 64 to 1024 for a 16 to 256 byte
// vector length), so both stay within that range. Given that precondition: first <= dimension
// always, so this is correct at any dimension, including ones below the chunk size (the loop
// then does not execute). first is congruent to dimension modulo granule, so Kernel::first's
// residual shape still describes it. remaining is therefore a whole multiple of granule, and so
// is every step, which is Kernel::rest's precondition. No single call ever gets more than
// UINT8_CHUNK_ELEMENTS elements, which is what keeps each chunk's 32-bit total exact.

#include "VecSim/spaces/spaces.h" // spaces::UINT8_CHUNK_ELEMENTS

#include <cassert>
#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace spaces {

template <typename Kernel>
static inline uint64_t uint8_chunked_total(const void *pVect1v, const void *pVect2v,
                                           size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);

    constexpr size_t chunk = UINT8_CHUNK_ELEMENTS;
    // Enforce the granule precondition at compile time when the adapter can express it, which is
    // every fixed-width kernel. A plain assert would vanish under NDEBUG, so it is the fallback
    // only for SVE, whose granule depends on the runtime vector length and cannot be constant.
    if constexpr (requires { std::integral_constant<size_t, Kernel::granule()>{}; }) {
        static_assert(Kernel::granule() > 0 && Kernel::granule() <= UINT8_CHUNK_ELEMENTS,
                      "Kernel::granule() must be in (0, UINT8_CHUNK_ELEMENTS]");
    }
    const size_t granule = Kernel::granule();
    assert(granule > 0 && granule <= chunk);
    const size_t tail = dimension % granule;
    const size_t first_chunk = tail + ((chunk - tail) / granule) * granule;
    const size_t first = dimension < first_chunk ? dimension : first_chunk;
    const size_t max_step = (chunk / granule) * granule;

    uint64_t total = Kernel::first(pVect1, pVect2, first);
    pVect1 += first;
    pVect2 += first;
    size_t remaining = dimension - first;

    while (remaining) {
        const size_t step = remaining < max_step ? remaining : max_step;
        total += Kernel::rest(pVect1, pVect2, step);
        pVect1 += step;
        pVect2 += step;
        remaining -= step;
    }
    return total;
}

} // namespace spaces
