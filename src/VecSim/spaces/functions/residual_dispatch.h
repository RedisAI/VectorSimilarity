/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <array>
#include <cstddef>
#include <utility>

/*
 * Picks the compile-time specialization of a residual-templated SIMD kernel — the one matching
 * `dim % Chunk` — via a static lookup table built once at compile time, instead of a hand-written
 * switch. Every kernel in spaces/IP and spaces/L2 is templated on a "residual" (the leftover
 * elements after the main SIMD loop, e.g. `template <unsigned char residual> float
 * BF16_InnerProductSIMD32_AVX2(...)`), so there is one compiled specialization per possible
 * residual value 0..Chunk-1; this picks the right one at runtime with one array index.
 *
 * Template arguments, using `Choose_BF16_IP_implementation_AVX2` as a worked example:
 *   Dist     the distance-function pointer type being selected, e.g. `dist_func_t<float>`
 *            (matches the `Choose_*` function's own return type).
 *   Chunk    the kernel's residual period — how many specializations exist (0..Chunk-1) — e.g.
 *            `32` for `BF16_InnerProductSIMD32_AVX2` (the "32" in its own name). This must match
 *            the kernel's own `template <unsigned char residual>` range; get it from the kernel
 *            header, not by guessing.
 *   Selector a templated lambda `[]<size_t N>() { return some_kernel<N>; }` — `some_kernel<N>` is
 *            the actual kernel being dispatched, one call site's worth of "which function is
 *            this" living entirely in this one line, not spread across a macro.
 *
 * Chunk is stated exactly once, as this function's own template argument — the table's size and
 * the modulus used to index it are the same Chunk inside the same function, so they cannot drift
 * apart the way two independently-written literals at a call site could. The table is `static
 * constexpr`, built once at compile time into read-only storage; the only runtime work is
 * `dim % Chunk` plus one array load.
 */
namespace spaces {

template <typename Dist, size_t Chunk, typename Selector>
Dist dispatch_by_residual(size_t dim, Selector selector) {
    static constexpr std::array<Dist, Chunk> table = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array<Dist, Chunk> { selector.template operator()<Is>()... };
    }(std::make_index_sequence<Chunk>{});
    return table[dim % Chunk];
}

/*
 * SVE analog of dispatch_by_residual. SVE kernels are templated on <bool partial_chunk,
 * unsigned char additional_steps> instead of a single residual, because the actual chunk size is
 * a runtime property of the hardware's vector length (queried via svcntw()/svcnth()/svcntb()/
 * svcntd(), depending on the kernel's element width), not known at compile time — so the kernel
 * can't be templated on "the residual" directly the way the fixed-width x86/NEON kernels are.
 * `additional_steps` instead counts how many extra passes of the kernel's own 4-step main loop
 * are needed (assumed to be in 0..3), and `partial_chunk` says whether there's a leftover partial
 * vector at all.
 *
 * Template argument:
 *   Dist     same meaning as in dispatch_by_residual — the distance-function pointer type.
 *
 * Runtime arguments:
 *   dim      the vector dimension, same as elsewhere.
 *   chunk    the hardware's actual vector length for this kernel's element type — the caller
 *            must pass the right one (e.g. `svcntw()` for 32-bit elements), the same as the old
 *            macro's chunk_getter argument; this can't be hidden in here since it depends on the
 *            kernel's element type, not on anything this function knows.
 *   selector a templated lambda `[]<bool P, size_t S>() { return some_kernel<P, S>; }`.
 *
 * The "4 steps" assumption is not a separately-written literal here — the modulus for
 * additional_steps is derived from the table's own row length, so it can't drift from the shape
 * the table was actually built with.
 */
template <typename Dist, typename Selector>
Dist dispatch_by_sve_residual(size_t dim, size_t chunk, Selector selector) {
    static constexpr std::array<Dist, 4> full_chunk_row =
        [&]<size_t... Steps>(std::index_sequence<Steps...>) {
            return std::array<Dist, 4> { selector.template operator()<false, Steps>()... };
        }(std::make_index_sequence<4>{});
    static constexpr std::array<Dist, 4> partial_chunk_row =
        [&]<size_t... Steps>(std::index_sequence<Steps...>) {
            return std::array<Dist, 4> { selector.template operator()<true, Steps>()... };
        }(std::make_index_sequence<4>{});
    bool partial_chunk = dim % chunk;
    size_t additional_steps = (dim / chunk) % full_chunk_row.size();
    return partial_chunk ? partial_chunk_row[additional_steps] : full_chunk_row[additional_steps];
}

} // namespace spaces
