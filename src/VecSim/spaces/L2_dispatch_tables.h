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
 * `inline constexpr` dispatch tables for L2_space.cpp's GetDistFunc-per-combo functions - see
 * IP_dispatch_tables.h for the full rationale (same reasons apply here, mirrored for L2).
 */
namespace spaces {

// Deliberately size-deduced (std::array CTAD), not a fixed size - see IP_dispatch_tables.h.
inline constexpr auto L2_FP32_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_FP32_L2_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_FP32_L2_implementation_SVE},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 0, 0,
                        Choose_FP32_L2_implementation_NEON},
#endif
#ifdef OPT_AVX512F
    // Optimizations assume at least 8 floats (see the residual handling in the kernels); below
    // that, the scalar implementation is at least as fast anyway - hence min_dim=8 on every x86
    // row here, matching the shared `if (dim < 8) return ret_dist_func;` gate it replaces.
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx512f; }, 8, 16,
                        Choose_FP32_L2_implementation_AVX512F},
#endif
#ifdef OPT_AVX
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx; }, 8, 8,
                        Choose_FP32_L2_implementation_AVX},
#endif
#ifdef OPT_SSE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sse; }, 8, 4,
                        Choose_FP32_L2_implementation_SSE},
#endif
};

} // namespace spaces
