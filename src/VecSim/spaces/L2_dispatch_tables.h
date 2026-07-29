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
#include "VecSim/spaces/functions/SVE_BF16.h"
#include "VecSim/spaces/functions/NEON.h"
#include "VecSim/spaces/functions/NEON_DOTPROD.h"
#include "VecSim/spaces/functions/NEON_HP.h"
#include "VecSim/spaces/functions/NEON_BF16.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512FP16_VL.h"
#include "VecSim/spaces/functions/AVX512F_BW_VL_VNNI.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/AVX2_F16C.h"
#include "VecSim/spaces/functions/AVX2_FMA.h"
#include "VecSim/spaces/functions/AVX2_FMA_F16C.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/SSE4.h"
#include "VecSim/spaces/functions/SSE4_F16C.h"

/*
 * `inline constexpr` dispatch tables for L2_space.cpp's GetDistFunc-per-combo functions - see
 * IP_dispatch_tables.h for the full rationale (same reasons apply here, mirrored for L2). Note
 * L2 has no Cosine tables at all: for int8/uint8, Cosine is dispatched through IP_space.cpp's
 * Cosine_* functions, never through L2_space.cpp - so L2's AVX-512 int8/uint8 tiers set alignment
 * normally, with none of IP's documented cosine skip-alignment special case.
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

// Alignment hints below refer to the SQ8 (first) operand per the GetDistFunc contract.
inline constexpr auto L2_SQ8_FP32_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_SQ8_FP32_L2_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_SQ8_FP32_L2_implementation_SVE},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 0, 0,
                        Choose_SQ8_FP32_L2_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vnni);
                        },
                        8, 16, Choose_SQ8_FP32_L2_implementation_AVX512F_BW_VL_VNNI},
#endif
#ifdef OPT_AVX2_FMA
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)(f.avx2 && f.fma3); }, 8,
                        8, Choose_SQ8_FP32_L2_implementation_AVX2_FMA},
#endif
#ifdef OPT_AVX2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx2; }, 8, 8,
                        Choose_SQ8_FP32_L2_implementation_AVX2},
#endif
#ifdef OPT_SSE4
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sse4_1; }, 8, 4,
                        Choose_SQ8_FP32_L2_implementation_SSE4},
#endif
};

// min_dim=16 on every row - both the x86 and the ARM block in the original cascade shared one
// `if (dim < 16) return ret_dist_func;` gate over all their rows.
inline constexpr auto L2_SQ8_FP16_DispatchTable = std::array{
#ifdef OPT_AVX512F
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx512f; }, 16, 16,
                        Choose_SQ8_FP16_L2_implementation_AVX512F},
#endif
#ifdef OPT_F16C
#ifdef OPT_AVX2_FMA
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.avx2 && f.fma3 && f.f16c); }, 16, 8,
        Choose_SQ8_FP16_L2_implementation_AVX2_FMA},
#endif
#ifdef OPT_AVX2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)(f.avx2 && f.f16c); }, 16,
                        8, Choose_SQ8_FP16_L2_implementation_AVX2},
#endif
#ifdef OPT_SSE4
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.sse4_1 && f.f16c && f.avx); }, 16, 4,
        Choose_SQ8_FP16_L2_implementation_SSE4},
#endif
#endif // OPT_F16C
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 16, 0,
                        Choose_SQ8_FP16_L2_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 16, 0,
                        Choose_SQ8_FP16_L2_implementation_SVE},
#endif
#ifdef OPT_NEON_HP
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimdfhm; }, 16, 0,
                        Choose_SQ8_FP16_L2_implementation_NEON_FHM},
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimdhp; }, 16, 0,
                        Choose_SQ8_FP16_L2_implementation_NEON_HP},
#endif
};

inline constexpr auto L2_FP64_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                         Choose_FP64_L2_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                         Choose_FP64_L2_implementation_SVE},
#endif
#ifdef OPT_NEON
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 0, 0,
                         Choose_FP64_L2_implementation_NEON},
#endif
#ifdef OPT_AVX512F
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.avx512f; }, 4, 8,
                         Choose_FP64_L2_implementation_AVX512F},
#endif
#ifdef OPT_AVX
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.avx; }, 4, 4,
                         Choose_FP64_L2_implementation_AVX},
#endif
#ifdef OPT_SSE
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.sse; }, 4, 2,
                         Choose_FP64_L2_implementation_SSE},
#endif
};

// The BF16 big/little-endian split stays as a special-cased branch in L2_BF16_GetDistFunc itself,
// before this table is ever consulted - see L2_space.cpp (mirrors IP_BF16_DispatchTable).
inline constexpr auto L2_BF16_DispatchTable = std::array{
#ifdef OPT_SVE_BF16
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.svebf16; }, 0, 0,
                        Choose_BF16_L2_implementation_SVE_BF16},
#endif
#ifdef OPT_NEON_BF16
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.bf16; }, 8, 0,
                        Choose_BF16_L2_implementation_NEON_BF16},
#endif
#ifdef OPT_AVX512_BW_VBMI2
    // Note: unlike IP_BF16_DispatchTable, there is no AVX512BF16_VL row here - the original
    // L2_BF16_GetDistFunc cascade never had one, only AVX512BW_VBMI2.
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.avx512bw && f.avx512vbmi2); }, 32, 32,
        Choose_BF16_L2_implementation_AVX512BW_VBMI2},
#endif
#ifdef OPT_AVX2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx2; }, 32, 16,
                        Choose_BF16_L2_implementation_AVX2},
#endif
#ifdef OPT_SSE3
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sse3; }, 32, 8,
                        Choose_BF16_L2_implementation_SSE3},
#endif
};

inline constexpr auto L2_FP16_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_FP16_L2_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_FP16_L2_implementation_SVE},
#endif
#ifdef OPT_NEON_HP
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimdhp; }, 8, 0,
                        Choose_FP16_L2_implementation_NEON_HP},
#endif
#ifdef OPT_AVX512_FP16_VL
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.avx512_fp16 && f.avx512vl); }, 32, 32,
        Choose_FP16_L2_implementation_AVX512FP16_VL},
#endif
#ifdef OPT_AVX512F
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx512f; }, 16, 32,
                        Choose_FP16_L2_implementation_AVX512F},
#endif
#ifdef OPT_F16C
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.f16c && f.fma3 && f.avx); }, 8, 16,
        Choose_FP16_L2_implementation_F16C},
#endif
};

// Unlike IP_space.cpp's Cosine_INT8/Cosine_UINT8, there is no skip-alignment special case here -
// L2's AVX-512 tier sets alignment normally.
inline constexpr auto L2_INT8_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_INT8_L2_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_INT8_L2_implementation_SVE},
#endif
#ifdef OPT_NEON_DOTPROD
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimddp; }, 16, 0,
                        Choose_INT8_L2_implementation_NEON_DOTPROD},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 16, 0,
                        Choose_INT8_L2_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vl && f.avx512vnni);
                        },
                        32, 32, Choose_INT8_L2_implementation_AVX512F_BW_VL_VNNI},
#endif
};

inline constexpr auto L2_UINT8_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_UINT8_L2_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_UINT8_L2_implementation_SVE},
#endif
#ifdef OPT_NEON_DOTPROD
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimddp; }, 16, 0,
                        Choose_UINT8_L2_implementation_NEON_DOTPROD},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 16, 0,
                        Choose_UINT8_L2_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vl && f.avx512vnni);
                        },
                        32, 32, Choose_UINT8_L2_implementation_AVX512F_BW_VL_VNNI},
#endif
};

// Both operands are SQ8 with a precomputed sum; the ARM rows carry their own per-row dim>=16
// floor, and the single x86 tier uses 64-element chunks with its own dim>=64 floor - mirrors
// IP_SQ8_SQ8_DispatchTable.
inline constexpr auto L2_SQ8_SQ8_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_SQ8_SQ8_L2_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_SQ8_SQ8_L2_implementation_SVE},
#endif
#ifdef OPT_NEON_DOTPROD
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimddp; }, 16, 0,
                        Choose_SQ8_SQ8_L2_implementation_NEON_DOTPROD},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 16, 0,
                        Choose_SQ8_SQ8_L2_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vnni);
                        },
                        64, 32, Choose_SQ8_SQ8_L2_implementation_AVX512F_BW_VL_VNNI},
#endif
};

} // namespace spaces
