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
#include "VecSim/spaces/functions/AVX512BF16_VL.h"
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
 * `inline constexpr` dispatch tables for IP_space.cpp's GetDistFunc-per-combo functions, one per
 * migrated (dtype, metric) combo. Defined here (not in IP_space.cpp) because both IP_space.cpp
 * and test_spaces.cpp need to see the same table object with its full initializer visible -
 * `inline constexpr` variables need their definition present in every translation unit that uses
 * them, so a header is the only place this can live. Row order matches the priority order of the
 * `#ifdef` cascade this replaces exactly (best tier first); each row is still individually
 * `#ifdef OPT_<TIER>`-guarded, so the table only ever contains rows for tiers this toolchain can
 * actually emit - the compile-time file-gating this refactor deliberately keeps.
 *
 * Every table here is deliberately size-deduced (`std::array` CTAD), never a fixed `<..., N>` -
 * see IP_FP32_DispatchTable's comment for why a fixed size would be actively dangerous, not just
 * inconvenient.
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

// Alignment hints below refer to the SQ8 (first) operand per the GetDistFunc contract.
inline constexpr auto IP_SQ8_FP32_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_SQ8_FP32_IP_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_SQ8_FP32_IP_implementation_SVE},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 0, 0,
                        Choose_SQ8_FP32_IP_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vnni);
                        },
                        8, 16, Choose_SQ8_FP32_IP_implementation_AVX512F_BW_VL_VNNI},
#endif
#ifdef OPT_AVX2_FMA
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)(f.avx2 && f.fma3); }, 8,
                        8, Choose_SQ8_FP32_IP_implementation_AVX2_FMA},
#endif
#ifdef OPT_AVX2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx2; }, 8, 8,
                        Choose_SQ8_FP32_IP_implementation_AVX2},
#endif
#ifdef OPT_SSE4
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sse4_1; }, 8, 4,
                        Choose_SQ8_FP32_IP_implementation_SSE4},
#endif
};

inline constexpr auto Cosine_SQ8_FP32_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_SQ8_FP32_Cosine_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_SQ8_FP32_Cosine_implementation_SVE},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 0, 0,
                        Choose_SQ8_FP32_Cosine_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vnni);
                        },
                        8, 16, Choose_SQ8_FP32_Cosine_implementation_AVX512F_BW_VL_VNNI},
#endif
#ifdef OPT_AVX2_FMA
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)(f.avx2 && f.fma3); }, 8,
                        8, Choose_SQ8_FP32_Cosine_implementation_AVX2_FMA},
#endif
#ifdef OPT_AVX2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx2; }, 8, 8,
                        Choose_SQ8_FP32_Cosine_implementation_AVX2},
#endif
#ifdef OPT_SSE4
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sse4_1; }, 8, 4,
                        Choose_SQ8_FP32_Cosine_implementation_SSE4},
#endif
};

// SQ8<->FP16 tiers all need F16C (vcvtph2ps) except AVX-512 (cvtph_ps is part of AVX-512F
// itself). min_dim=16 on every row - both the x86 and the ARM `#ifdef` blocks in the original
// cascade shared one `if (dim < 16) return ret_dist_func;` gate over all their rows.
inline constexpr auto IP_SQ8_FP16_DispatchTable = std::array{
#ifdef OPT_AVX512F
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx512f; }, 16, 16,
                        Choose_SQ8_FP16_IP_implementation_AVX512F},
#endif
#ifdef OPT_F16C
#ifdef OPT_AVX2_FMA
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.avx2 && f.fma3 && f.f16c); }, 16, 8,
        Choose_SQ8_FP16_IP_implementation_AVX2_FMA},
#endif
#ifdef OPT_AVX2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)(f.avx2 && f.f16c); }, 16,
                        8, Choose_SQ8_FP16_IP_implementation_AVX2},
#endif
#ifdef OPT_SSE4
    // F16C is VEX-encoded - require AVX as well, matching the existing F16C/FP16 dispatcher.
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.sse4_1 && f.f16c && f.avx); }, 16, 4,
        Choose_SQ8_FP16_IP_implementation_SSE4},
#endif
#endif // OPT_F16C
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 16, 0,
                        Choose_SQ8_FP16_IP_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 16, 0,
                        Choose_SQ8_FP16_IP_implementation_SVE},
#endif
#ifdef OPT_NEON_HP
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimdfhm; }, 16, 0,
                        Choose_SQ8_FP16_IP_implementation_NEON_FHM},
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimdhp; }, 16, 0,
                        Choose_SQ8_FP16_IP_implementation_NEON_HP},
#endif
};

inline constexpr auto Cosine_SQ8_FP16_DispatchTable = std::array{
#ifdef OPT_AVX512F
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx512f; }, 16, 16,
                        Choose_SQ8_FP16_Cosine_implementation_AVX512F},
#endif
#ifdef OPT_F16C
#ifdef OPT_AVX2_FMA
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.avx2 && f.fma3 && f.f16c); }, 16, 8,
        Choose_SQ8_FP16_Cosine_implementation_AVX2_FMA},
#endif
#ifdef OPT_AVX2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)(f.avx2 && f.f16c); }, 16,
                        8, Choose_SQ8_FP16_Cosine_implementation_AVX2},
#endif
#ifdef OPT_SSE4
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.sse4_1 && f.f16c && f.avx); }, 16, 4,
        Choose_SQ8_FP16_Cosine_implementation_SSE4},
#endif
#endif // OPT_F16C
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 16, 0,
                        Choose_SQ8_FP16_Cosine_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 16, 0,
                        Choose_SQ8_FP16_Cosine_implementation_SVE},
#endif
#ifdef OPT_NEON_HP
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimdfhm; }, 16, 0,
                        Choose_SQ8_FP16_Cosine_implementation_NEON_FHM},
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimdhp; }, 16, 0,
                        Choose_SQ8_FP16_Cosine_implementation_NEON_HP},
#endif
};

// Both operands are SQ8 with a precomputed sum; the ARM rows carry their own per-row dim>=16
// floor (no shared gate existed for them in the original cascade), and the single x86 tier uses
// 64-element chunks (residual handling is in 32-byte sub-chunks) with its own dim>=64 floor.
inline constexpr auto IP_SQ8_SQ8_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_SQ8_SQ8_IP_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_SQ8_SQ8_IP_implementation_SVE},
#endif
#ifdef OPT_NEON_DOTPROD
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimddp; }, 16, 0,
                        Choose_SQ8_SQ8_IP_implementation_NEON_DOTPROD},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 16, 0,
                        Choose_SQ8_SQ8_IP_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vnni);
                        },
                        64, 32, Choose_SQ8_SQ8_IP_implementation_AVX512F_BW_VL_VNNI},
#endif
};

inline constexpr auto Cosine_SQ8_SQ8_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_SQ8_SQ8_Cosine_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_SQ8_SQ8_Cosine_implementation_SVE},
#endif
#ifdef OPT_NEON_DOTPROD
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimddp; }, 16, 0,
                        Choose_SQ8_SQ8_Cosine_implementation_NEON_DOTPROD},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 16, 0,
                        Choose_SQ8_SQ8_Cosine_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vnni);
                        },
                        64, 32, Choose_SQ8_SQ8_Cosine_implementation_AVX512F_BW_VL_VNNI},
#endif
};

inline constexpr auto IP_FP64_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                         Choose_FP64_IP_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                         Choose_FP64_IP_implementation_SVE},
#endif
#ifdef OPT_NEON
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 0, 0,
                         Choose_FP64_IP_implementation_NEON},
#endif
#ifdef OPT_AVX512F
    // Optimizations assume at least 4 doubles; below that, scalar is at least as fast anyway.
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.avx512f; }, 4, 8,
                         Choose_FP64_IP_implementation_AVX512F},
#endif
#ifdef OPT_AVX
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.avx; }, 4, 4,
                         Choose_FP64_IP_implementation_AVX},
#endif
#ifdef OPT_SSE
    DispatchTier<double>{[](const FeaturesType &f) noexcept { return (bool)f.sse; }, 4, 2,
                         Choose_FP64_IP_implementation_SSE},
#endif
};

// The BF16 big/little-endian split is not a tier-selection concern at all (it's not a CPU-feature
// or dimension decision) and stays as a special-cased branch in IP_BF16_GetDistFunc itself,
// before this table is ever consulted - see IP_space.cpp.
inline constexpr auto IP_BF16_DispatchTable = std::array{
#ifdef OPT_SVE_BF16
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.svebf16; }, 0, 0,
                        Choose_BF16_IP_implementation_SVE_BF16},
#endif
#ifdef OPT_NEON_BF16
    // Optimization assumes at least 8 BF16s (a full chunk).
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.bf16; }, 8, 0,
                        Choose_BF16_IP_implementation_NEON_BF16},
#endif
#ifdef OPT_AVX512_BF16_VL
    // Optimizations assume at least 32 bfloats; below that, scalar is at least as fast anyway.
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.avx512_bf16 && f.avx512vl); }, 32, 32,
        Choose_BF16_IP_implementation_AVX512BF16_VL},
#endif
#ifdef OPT_AVX512_BW_VBMI2
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.avx512bw && f.avx512vbmi2); }, 32, 32,
        Choose_BF16_IP_implementation_AVX512BW_VBMI2},
#endif
#ifdef OPT_AVX2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx2; }, 32, 16,
                        Choose_BF16_IP_implementation_AVX2},
#endif
#ifdef OPT_SSE3
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sse3; }, 32, 8,
                        Choose_BF16_IP_implementation_SSE3},
#endif
};

// Each x86/ARM tier here carries its own per-row dimension floor, implied by its residual
// handling (see the original cascade's comment in IP_space.cpp for the exact rationale per tier)
// - there was no single shared gate to absorb, unlike IP_FP32/IP_SQ8_FP16/etc.
inline constexpr auto IP_FP16_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_FP16_IP_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_FP16_IP_implementation_SVE},
#endif
#ifdef OPT_NEON_HP
    // Optimization assumes at least 8 16FPs (a full chunk).
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimdhp; }, 8, 0,
                        Choose_FP16_IP_implementation_NEON_HP},
#endif
#ifdef OPT_AVX512_FP16_VL
    // The AVX512FP16_VL kernel loads full 512-bit blocks (32 elements).
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.avx512_fp16 && f.avx512vl); }, 32, 32,
        Choose_FP16_IP_implementation_AVX512FP16_VL},
#endif
#ifdef OPT_AVX512F
    // The AVX512F kernel loads full 256-bit blocks (16 elements).
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.avx512f; }, 16, 32,
                        Choose_FP16_IP_implementation_AVX512F},
#endif
#ifdef OPT_F16C
    // The F16C kernel loads full 128-bit blocks (8 elements).
    DispatchTier<float>{
        [](const FeaturesType &f) noexcept { return (bool)(f.f16c && f.fma3 && f.avx); }, 8, 16,
        Choose_FP16_IP_implementation_F16C},
#endif
};

inline constexpr auto IP_INT8_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_INT8_IP_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_INT8_IP_implementation_SVE},
#endif
#ifdef OPT_NEON_DOTPROD
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimddp; }, 16, 0,
                        Choose_INT8_IP_implementation_NEON_DOTPROD},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 16, 0,
                        Choose_INT8_IP_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    // Optimizations assume at least 32 int8; below that, scalar is at least as fast anyway.
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vl && f.avx512vnni);
                        },
                        32, 32, Choose_INT8_IP_implementation_AVX512F_BW_VL_VNNI},
#endif
};

// Cosine's AVX-512 tier deliberately never sets an alignment hint (alignment_chunk_elems=0): for
// int8 vectors with cosine distance, the extra float for the norm shifts the effective alignment
// to `(dim + sizeof(float)) % 32`, and vectors satisfying THAT have a residual, causing offset
// loads during calculation. The original cascade skips computing this to avoid the complexity,
// assuming the performance impact is negligible - carried over unchanged. There is no L2 mirror
// of this special case: L2_INT8_GetDistFunc's AVX-512 tier sets alignment normally.
inline constexpr auto Cosine_INT8_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_INT8_Cosine_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_INT8_Cosine_implementation_SVE},
#endif
#ifdef OPT_NEON_DOTPROD
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimddp; }, 16, 0,
                        Choose_INT8_Cosine_implementation_NEON_DOTPROD},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 16, 0,
                        Choose_INT8_Cosine_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vl && f.avx512vnni);
                        },
                        32, 0, Choose_INT8_Cosine_implementation_AVX512F_BW_VL_VNNI},
#endif
};

inline constexpr auto IP_UINT8_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_UINT8_IP_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_UINT8_IP_implementation_SVE},
#endif
#ifdef OPT_NEON_DOTPROD
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimddp; }, 16, 0,
                        Choose_UINT8_IP_implementation_NEON_DOTPROD},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 16, 0,
                        Choose_UINT8_IP_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vl && f.avx512vnni);
                        },
                        32, 32, Choose_UINT8_IP_implementation_AVX512F_BW_VL_VNNI},
#endif
};

// See Cosine_INT8_DispatchTable's comment - the same documented skip-alignment special case
// applies here (uint8 cosine's extra norm float shifts effective alignment the same way).
inline constexpr auto Cosine_UINT8_DispatchTable = std::array{
#ifdef OPT_SVE2
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve2; }, 0, 0,
                        Choose_UINT8_Cosine_implementation_SVE2},
#endif
#ifdef OPT_SVE
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.sve; }, 0, 0,
                        Choose_UINT8_Cosine_implementation_SVE},
#endif
#ifdef OPT_NEON_DOTPROD
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimddp; }, 16, 0,
                        Choose_UINT8_Cosine_implementation_NEON_DOTPROD},
#endif
#ifdef OPT_NEON
    DispatchTier<float>{[](const FeaturesType &f) noexcept { return (bool)f.asimd; }, 16, 0,
                        Choose_UINT8_Cosine_implementation_NEON},
#endif
#ifdef OPT_AVX512_F_BW_VL_VNNI
    DispatchTier<float>{[](const FeaturesType &f) noexcept {
                            return (bool)(f.avx512f && f.avx512bw && f.avx512vl && f.avx512vnni);
                        },
                        32, 0, Choose_UINT8_Cosine_implementation_AVX512F_BW_VL_VNNI},
#endif
};

} // namespace spaces
