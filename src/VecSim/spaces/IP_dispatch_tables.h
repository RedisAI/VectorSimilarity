/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/spaces/dispatch_tier.h"
#include "VecSim/spaces/functions/SVE2.h"
#include "VecSim/spaces/functions/SVE.h"
#include "VecSim/spaces/functions/NEON.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"

/*
 * `inline constexpr` dispatch tables for IP_space.cpp's GetDistFunc-per-combo functions, one per
 * migrated (dtype, metric) combo. Defined here (not in IP_space.cpp) because both IP_space.cpp
 * and test_spaces.cpp need to see the same table object with its full initializer visible -
 * `inline constexpr` variables need their definition present in every translation unit that uses
 * them, so a header is the only place this can live. Row order matches the priority order of the
 * `#ifdef` cascade this replaces exactly (best tier first); each row is still individually
 * `#ifdef OPT_<TIER>`-guarded, so the table only ever contains rows for tiers this toolchain can
 * actually emit - the compile-time file-gating this refactor deliberately keeps.
 */
namespace spaces {

// Deliberately size-deduced (std::array CTAD), not a fixed <..., 6> - the actual row count varies
// with which OPT_* tiers this toolchain/target compiles in (e.g. this table has only 3 rows on
// an x86 build with no AVX512F, and a different 3 on ARM), so a fixed size would either fail to
// compile or (worse) silently zero-initialize missing rows into null-predicate garbage.
inline constexpr auto IP_FP32_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_FP32_IP_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_FP32_IP_implementation_SVE},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 0, 0,
                        Choose_FP32_IP_implementation_NEON},
#endif
#ifdef OPT_AVX512F
    // Optimizations assume at least 8 floats (see the residual handling in the kernels); below
    // that, the scalar implementation is at least as fast anyway - hence min_dim=8 on every x86
    // row here, matching the shared `if (dim < 8) return ret_dist_func;` gate it replaces.
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx512f; }, 8, 16,
                        Choose_FP32_IP_implementation_AVX512F},
#endif
#ifdef OPT_AVX
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx; }, 8, 8,
                        Choose_FP32_IP_implementation_AVX},
#endif
#ifdef OPT_SSE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sse; }, 8, 4,
                        Choose_FP32_IP_implementation_SSE},
#endif
};

} // namespace spaces
