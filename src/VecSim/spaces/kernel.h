/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

// What a tier provides for one (metric, data type) pair, and the per-type rules that hold whatever
// tier is chosen. Like tier_info.h this header names no intrinsics: the specializations live in the
// per-tier declaration headers and their make() bodies live in the tier's own translation unit, the
// only one compiled with that tier's flags.

#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/tier_info.h"

#include <cstddef>
#include <limits>

namespace spaces {

enum class Metric { L2, IP, Cosine };

// The stored/query type pair a kernel works on. SQ8_FP32 means SQ8 storage against an FP32 query,
// and so on for the other asymmetric combinations.
enum class DataType { FP32, FP64, BF16, FP16, INT8, UINT8, SQ8_FP32, SQ8_FP16, SQ8_SQ8 };

// How a kernel's tail is handled. FixedModulus selects a compile-time specialization on
// dim % residual_mod, which is what CHOOSE_IMPLEMENTATION does. SveRuntime defers to the runtime
// vector length, which is what CHOOSE_SVE_IMPLEMENTATION does, so residual_mod is not meaningful.
enum class ResidualPolicy { FixedModulus, SveRuntime };

// Unspecialized primary: this tier does not provide this (metric, type). dispatch() skips it, and
// because the check is `if constexpr (Kernel<...>::exists)`, make() is never referenced for a
// combination that does not exist and never linked for a tier that did not compile.
template <Metric M, DataType D, Tier T>
struct Kernel {
    static constexpr bool exists = false;
};

// Per-type rules that hold regardless of tier. Two kinds, both real in the current code:
// a predicate gate (BF16 needs a little-endian host) and a dimension ceiling (the uint8 families
// stop being exact past UINT8_MAX_EXACT_SIMD_DIM because every SIMD kernel accumulates into a
// signed 32-bit total, while the scalar kernel accumulates into 64 bits).
template <DataType D>
struct TypePolicy {
    static bool eligible() { return true; }
    static constexpr size_t simd_max_dim = std::numeric_limits<size_t>::max();
};

// Declares what a tier provides for one (metric, type). Kept as a macro so a tier header reads as a
// short table of rows and a contributor adding a kernel copies a neighbouring line rather than
// writing a template specialization.
//
//   residual_step  the modulus CHOOSE_IMPLEMENTATION specializes on, or 0 under SveRuntime
//   policy         FixedModulus or SveRuntime
//   align_modulus  alignment is only worth requesting when dim % align_modulus == 0, since a
//                  residual offsets every subsequent load anyway. 0 means this kernel never asks
//   align_byte_cnt the hint to publish when the modulus divides, in bytes
//   minimum_dim    below this the scalar kernel is at least as fast, so the tier declines
#define VECSIM_KERNEL(metric, type, tier, residual_step, policy, align_modulus, align_byte_cnt,    \
                      minimum_dim)                                                                 \
    template <>                                                                                    \
    struct Kernel<Metric::metric, DataType::type, Tier::tier> {                                    \
        static constexpr bool exists = true;                                                       \
        static constexpr size_t residual_mod = residual_step;                                      \
        static constexpr ResidualPolicy residual_policy = ResidualPolicy::policy;                  \
        static constexpr size_t align_mod = align_modulus;                                         \
        static constexpr size_t align_bytes = align_byte_cnt;                                      \
        static constexpr size_t min_dim = minimum_dim;                                             \
        static constexpr size_t max_dim = std::numeric_limits<size_t>::max();                      \
        static dist_func_t<float> make(size_t dim);                                                \
    };

} // namespace spaces
