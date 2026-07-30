/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/spaces/spaces.h"
#include <array>
#include <cstddef>

/*
 * One row of a per-(dtype,metric) dispatch table, replacing a single `#ifdef OPT_<TIER> if
 * (features.<flag>) { ... return Choose_*(dim); }` block from the old cascade with one struct
 * literal. Rows are listed in priority order (best tier first) inside a
 * `inline constexpr std::array<DispatchTier<DistType>, N>` defined in a dispatch-tables header
 * (e.g. IP_dispatch_tables.h) so both the real GetDistFunc and test_spaces.cpp read the exact
 * same table.
 */
namespace spaces {

template <typename DistType>
struct DispatchTier {
    // Pure CPU-feature check only - no dimension logic here, see min_dim below. Must be a plain
    // function pointer (a captureless lambda decays to one), never std::function - this sits on
    // a cold path but there's no reason to pay for type erasure/possible allocation anyway.
    bool (*predicate)(const FeaturesType &) noexcept;
    // Dimension floor below which this tier is skipped even if predicate matches. 0 = no floor.
    size_t min_dim;
    // Alignment hint, in elements of the tier's storage type (not bytes - the caller multiplies
    // by sizeof(StorageElemType), which this struct doesn't know). 0 = this tier never sets an
    // alignment hint (the two documented cosine skip-alignment special cases use this).
    size_t alignment_chunk_elems;
    // The existing Choose_* function for this tier - unchanged by this refactor.
    dist_func_t<DistType> (*chooser)(size_t dim);
};

/*
 * Returns the index of the first row in `rows` whose predicate matches `features` and whose
 * min_dim is satisfied by `dim`, or `rows.size()` if no row matches (caller falls back to the
 * naive/scalar implementation, exactly as the old cascade's final `return ret_dist_func;` did).
 *
 * Pure comparisons only - this never calls any row's `chooser`, and therefore never touches any
 * arch-specific translation unit. That makes it safe to call with an arbitrary or fabricated
 * `FeaturesType` value regardless of what the host CPU actually supports - unlike the full
 * GetDistFunc path, which calls into a TU compiled for a specific ISA target and so must only
 * ever be exercised with real, host-detected features (see SPACES-REFACTOR-PLAN.md's
 * execution-safety rule).
 */
template <typename DistType, size_t N>
size_t select_tier_index(const FeaturesType &features, size_t dim,
                         const std::array<DispatchTier<DistType>, N> &rows) {
    for (size_t i = 0; i < N; i++) {
        if (rows[i].predicate(features) && dim >= rows[i].min_dim) {
            return i;
        }
    }
    return N;
}

} // namespace spaces
