/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/vec_sim_common.h" // enum VecSimMetric
#include "space_includes.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace spaces {

template <typename RET_TYPE>
using dist_func_t = RET_TYPE (*)(const void *, const void *, size_t);

// Get the distance function for comparing vectors of type VecType1 and VecType2, for a given metric
// and dimension. The returned function has the signature: dist(VecType1*, VecType2*, size_t) ->
// DistType. VecType2 defaults to VecType1 when both vectors are of the same type. The alignment
// hint is set based on the chosen implementation and available optimizations.
//
// Asymmetric-types contract (e.g. VecType1 = SQ8 storage, VecType2 = FP32 query):
//   The returned alignment hint refers to the FIRST operand only (the storage operand).
//   The query operand alignment is governed by the symmetric query-type dispatcher
//   (e.g. GetDistFunc<float, float>). Callers that need both operand alignments must
//   query both dispatchers and combine the results with combineAlignments().
template <typename VecType1, typename DistType, typename VecType2 = VecType1>
dist_func_t<DistType> GetDistFunc(VecSimMetric metric, size_t dim, unsigned char *alignment);

// Combine two alignment hints into the strictest requirement that satisfies both.
// Each input must be a power of two or zero (zero means "no alignment requirement").
// The result is the maximum of the two, which for power-of-two values is also the LCM
// and therefore the smallest alignment that simultaneously satisfies both consumers.
static inline unsigned char combineAlignments(unsigned char a, unsigned char b) {
    assert((a == 0 || (a & (a - 1)) == 0) && "alignment must be a power of two or zero");
    assert((b == 0 || (b & (b - 1)) == 0) && "alignment must be a power of two or zero");
    return a > b ? a : b;
}

template <typename DataType>
using normalizeVector_f = void (*)(void *input_vector, const size_t dim);

template <typename DataType>
normalizeVector_f<DataType> GetNormalizeFunc();

static int inline is_little_endian() {
    unsigned int x = 1;
    return *(char *)&x;
}

// The uint8 kernels accumulate products or squared byte differences, so the worst-case total is
// 255 * 255 * dim, and every kernel holds that total in a signed 32-bit int. That makes 33,025 the
// largest dimension they are exact at: 65025 * 33,025 is 2,147,450,625 and fits INT32_MAX, and one
// dimension more does not. Above it the choosers hand back the scalar kernel, which accumulates
// into a 64-bit ret_t and is exact at any dimension. Decided once per index, so no distance
// computation pays for the check. The pull request covers why the bound is here and not at the
// unsigned limit.
static constexpr size_t UINT8_MAX_EXACT_SIMD_DIM =
    std::numeric_limits<int32_t>::max() /
    (std::numeric_limits<uint8_t>::max() * std::numeric_limits<uint8_t>::max());

// Intersect an overridden feature mask with the features actually detected on this machine, so
// that the override can only clear a capability, never claim one the hardware does not have.
//
// `arch_opt` (below) is a test-only affordance: no production caller passes a non-null value
// (every chooser in spaces.cpp calls getCpuOptimizationFeatures() with the `= nullptr` default),
// tests use it to force a specific SIMD tier onto whatever machine runs the suite. Without this
// intersection, a test that claims a feature the real CPU lacks would hand back a function
// pointer into a tier that executes an illegal instruction.
//
// This is sound only because cpu_features::X86Features and cpu_features::Aarch64Features are
// composed entirely of `int <name> : 1;` bitfields with no other members, so a byte-wise AND of
// the two object representations performs a per-feature logical AND. The
// CpuFeatureIntersection.* tests in test_spaces.cpp walk every feature via cpu_features' own
// GetX86FeaturesEnumValue / GetAarch64FeaturesEnumValue readers and enforce that layout
// assumption, so this breaks loudly if cpu_features ever adds a non-bitfield member.
template <typename FeaturesType>
static inline FeaturesType intersectWithDetectedFeatures(const FeaturesType &overridden,
                                                         const FeaturesType &detected) {
    static_assert(std::is_trivially_copyable_v<FeaturesType>,
                  "byte-wise AND requires a trivially copyable features type");
    unsigned char overridden_bytes[sizeof(FeaturesType)];
    unsigned char detected_bytes[sizeof(FeaturesType)];
    std::memcpy(overridden_bytes, &overridden, sizeof(FeaturesType));
    std::memcpy(detected_bytes, &detected, sizeof(FeaturesType));
    for (size_t i = 0; i < sizeof(FeaturesType); i++) {
        overridden_bytes[i] &= detected_bytes[i];
    }
    FeaturesType result;
    std::memcpy(&result, overridden_bytes, sizeof(FeaturesType));
    return result;
}

static inline auto getCpuOptimizationFeatures(const void *arch_opt = nullptr) {

#if defined(CPU_FEATURES_ARCH_AARCH64)
    using FeaturesType = cpu_features::Aarch64Features;
    constexpr auto getFeatures = cpu_features::GetAarch64Info;
#else
    using FeaturesType = cpu_features::X86Features; // Fallback
    constexpr auto getFeatures = cpu_features::GetX86Info;
#endif
    const FeaturesType detected = getFeatures().features;
    return arch_opt ? intersectWithDetectedFeatures(*static_cast<const FeaturesType *>(arch_opt),
                                                    detected)
                    : detected;
}

} // namespace spaces
