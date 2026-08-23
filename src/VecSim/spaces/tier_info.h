/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

// The C++ view of isa_tiers.def and isa_features.def. Deliberately free of intrinsics: this header
// is compiled into every translation unit that dispatches, including the baseline ones, so it may
// only name cpu_features fields and the availability macros CMake defines.
//
// Two facts per tier come from two different columns and must not be conflated:
//   compiled     from VECSIM_BUILT_<TIER>, which CMake sets from the whole-tier compile probe. It
//                answers "did this build produce this tier's object".
//   supported()  folded exclusively from the tier's GUARANTEES column. It answers "may the CPU in
//                front of us execute that object".
// supported() never consults FLAGS_FROM. A flag fragment expands to a compiler-defined bundle that
// differs between compilers for the same -march string, so a predicate derived from flags would be
// unsound in the one direction that matters: it would claim support the hardware does not have.

#include "VecSim/spaces/spaces.h"

namespace spaces {

enum class Arch { X86, ARM };

#if defined(CPU_FEATURES_ARCH_AARCH64)
inline constexpr Arch kBuildArch = Arch::ARM;
#else
inline constexpr Arch kBuildArch = Arch::X86;
#endif

// ---- the tier enumeration, one enumerator per manifest row -----------------------------------
#define ARCH_IMPLIES(a, b)
#define FLAG_ENABLES(a, b)
#define LEVEL_REQUIRES(a, b)
#define FEATURE(token, arch, fragment, level, field)
#define TIER(name, arch, stem, priority, flags_from, guarantees) name,
enum class Tier {
#include "VecSim/spaces/isa_tiers.def"
};
#undef TIER
#undef FEATURE

// ---- one accessor per feature token, for the current architecture only -----------------------
// A tier may only name tokens of its own architecture (CMake enforces that), so emitting accessors
// for the other architecture would only produce references to fields that do not exist on this
// build's features struct.
namespace detail {

#if defined(CPU_FEATURES_ARCH_AARCH64)
#define VECSIM_FEATURE_ARM(token, field)                                                           \
    inline bool feature_##token(const FeaturesType &f) { return f.field; }
#define VECSIM_FEATURE_X86(token, field)
#else
#define VECSIM_FEATURE_X86(token, field)                                                           \
    inline bool feature_##token(const FeaturesType &f) { return f.field; }
#define VECSIM_FEATURE_ARM(token, field)
#endif

#define FEATURE(token, arch, fragment, level, field) VECSIM_FEATURE_##arch(token, field)
#include "VecSim/spaces/isa_features.def"
#undef FEATURE
#undef VECSIM_FEATURE_X86
#undef VECSIM_FEATURE_ARM

} // namespace detail

#undef ARCH_IMPLIES
#undef FLAG_ENABLES
#undef LEVEL_REQUIRES

// ---- folding a GUARANTEES tuple into a conjunction -------------------------------------------
// The manifest writes GUARANTEES as a parenthesised, comma-separated tuple, so the fold is by
// arity. Six is deliberately more than the widest row in use (four, for AVX512F+BW+VL+VNNI); a
// seventh token fails to compile here rather than silently dropping the extra guarantees.
#define VECSIM_AND_1(a)                ::spaces::detail::feature_##a(f)
#define VECSIM_AND_2(a, b)             VECSIM_AND_1(a) && VECSIM_AND_1(b)
#define VECSIM_AND_3(a, b, c)          VECSIM_AND_2(a, b) && VECSIM_AND_1(c)
#define VECSIM_AND_4(a, b, c, d)       VECSIM_AND_3(a, b, c) && VECSIM_AND_1(d)
#define VECSIM_AND_5(a, b, c, d, e)    VECSIM_AND_4(a, b, c, d) && VECSIM_AND_1(e)
#define VECSIM_AND_6(a, b, c, d, e, g) VECSIM_AND_5(a, b, c, d, e) && VECSIM_AND_1(g)

#define VECSIM_AND_PICK(_1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define VECSIM_ALL_OF(...)                                                                         \
    VECSIM_AND_PICK(__VA_ARGS__, VECSIM_AND_6, VECSIM_AND_5, VECSIM_AND_4, VECSIM_AND_3,           \
                    VECSIM_AND_2, VECSIM_AND_1)                                                    \
    (__VA_ARGS__)

// ---- TierInfo -------------------------------------------------------------------------------
// Every tier gets a specialization, including tiers of the other architecture, so metadata stays
// visible for tests and tooling on any build. Only the predicate body is architecture-selected.
//
// The selection is done by the preprocessor rather than by if constexpr, because the discarded
// branch of an if constexpr in a non-template function still has to name things that exist, and
// this build only emits feature accessors for its own architecture. An ARM tier's GUARANTEES
// mention feature_SVE2 and friends, which simply are not declared in an x86 build.
template <Tier T>
struct TierInfo;

#define VECSIM_TIER_ARCH_X86 ::spaces::Arch::X86
#define VECSIM_TIER_ARCH_ARM ::spaces::Arch::ARM

#if defined(CPU_FEATURES_ARCH_AARCH64)
#define VECSIM_SUPPORTED_ARM(guarantees) return VECSIM_ALL_OF guarantees;
#define VECSIM_SUPPORTED_X86(guarantees) return false;
#else
#define VECSIM_SUPPORTED_X86(guarantees) return VECSIM_ALL_OF guarantees;
#define VECSIM_SUPPORTED_ARM(guarantees) return false;
#endif

#define TIER(name, arch, stem, priority, flags_from, guarantees)                                   \
    template <>                                                                                    \
    struct TierInfo<Tier::name> {                                                                  \
        static constexpr Arch arch_of = VECSIM_TIER_ARCH_##arch;                                   \
        static constexpr int priority_of = priority;                                               \
        static constexpr bool compiled = VECSIM_BUILT_##name;                                      \
        static constexpr const char *name_of = #name;                                              \
        static bool supported([[maybe_unused]] const FeaturesType &f) {                            \
            VECSIM_SUPPORTED_##arch(guarantees)                                                    \
        }                                                                                          \
    };
#include "VecSim/spaces/isa_tiers.def"
#undef TIER
#undef VECSIM_SUPPORTED_X86
#undef VECSIM_SUPPORTED_ARM
#undef VECSIM_TIER_ARCH_X86
#undef VECSIM_TIER_ARCH_ARM

} // namespace spaces
