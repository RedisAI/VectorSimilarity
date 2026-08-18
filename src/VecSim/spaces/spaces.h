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
#include <limits>

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
// 255 * 255 * dim. A 32-bit accumulator holds that exactly up to this dimension and no further, so
// the choosers hand back the scalar kernel above it, which accumulates into a 64-bit ret_t and is
// exact at any dimension. Derived from the types rather than written as a literal: the bound is a
// property of uint8 in a uint32 accumulator, and at the limit there are only 1,020 to spare.
//
// This is the UNSIGNED bound. The total passes INT32_MAX from dimension 33,026, well inside the
// range kept on SIMD, which is why the kernels return uint32_t.
static constexpr size_t UINT8_MAX_EXACT_SIMD_DIM =
    std::numeric_limits<uint32_t>::max() /
    (std::numeric_limits<uint8_t>::max() * std::numeric_limits<uint8_t>::max());

static inline auto getCpuOptimizationFeatures(const void *arch_opt = nullptr) {

#if defined(CPU_FEATURES_ARCH_AARCH64)
    using FeaturesType = cpu_features::Aarch64Features;
    constexpr auto getFeatures = cpu_features::GetAarch64Info;
#else
    using FeaturesType = cpu_features::X86Features; // Fallback
    constexpr auto getFeatures = cpu_features::GetX86Info;
#endif
    return arch_opt ? *static_cast<const FeaturesType *>(arch_opt) : getFeatures().features;
}

} // namespace spaces
