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
 * Replaces the CHOOSE_IMPLEMENTATION / CHOOSE_SVE_IMPLEMENTATION macro pyramid
 * (implementation_chooser.h) for picking a residual-templated kernel by dim % chunk.
 *
 * Chunk is stated exactly once, as dispatch_by_residual's own template argument — the table's
 * size and the modulus used to index it are the same Chunk inside the same function, so they
 * cannot drift apart the way two independently-written literals at a call site could. The table
 * itself is `static constexpr`, built once at compile time into read-only storage; the only
 * runtime work is `dim % Chunk` plus one array load, matching the switch statement it replaces.
 *
 * `selector` is a templated lambda: []<size_t N>() { return some_kernel<N>; }
 */
namespace spaces {

template <typename Dist, size_t Chunk, typename Selector>
Dist dispatch_by_residual(size_t dim, Selector selector) {
    static constexpr std::array<Dist, Chunk> table =
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::array<Dist, Chunk>{selector.template operator()<Is>()...};
        }(std::make_index_sequence<Chunk>{});
    return table[dim % Chunk];
}

/*
 * SVE analog, for CHOOSE_SVE_IMPLEMENTATION's replacement. SVE kernels are templated on
 * <bool partial_chunk, unsigned char additional_steps> (additional_steps assumed to be in 0..3,
 * i.e. a 4-step main loop) because the actual chunk size is a runtime property of the hardware's
 * vector length, not known at compile time. `chunk` (e.g. svcntw()/svcntb()/svcntd(), matching the
 * kernel's element width) is still a runtime value supplied by the caller, same as the old
 * macro's chunk_getter argument — that part cannot be hidden here since it depends on the kernel's
 * element type. What IS hidden here is the 4-steps assumption: the modulus for additional_steps is
 * derived from the table's own row length, not a separately-written literal.
 *
 * `selector` is a templated lambda: []<bool P, size_t S>() { return some_kernel<P, S>; }
 */
template <typename Dist, typename Selector>
Dist dispatch_by_sve_residual(size_t dim, size_t chunk, Selector selector) {
    static constexpr std::array<Dist, 4> full_chunk_row =
        [&]<size_t... Steps>(std::index_sequence<Steps...>) {
            return std::array<Dist, 4>{selector.template operator()<false, Steps>()...};
        }(std::make_index_sequence<4>{});
    static constexpr std::array<Dist, 4> partial_chunk_row =
        [&]<size_t... Steps>(std::index_sequence<Steps...>) {
            return std::array<Dist, 4>{selector.template operator()<true, Steps>()...};
        }(std::make_index_sequence<4>{});
    bool partial_chunk = dim % chunk;
    size_t additional_steps = (dim / chunk) % full_chunk_row.size();
    return partial_chunk ? partial_chunk_row[additional_steps] : full_chunk_row[additional_steps];
}

} // namespace spaces
