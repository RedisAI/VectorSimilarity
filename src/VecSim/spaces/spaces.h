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

// Largest dimension for which a uint8 SIMD kernel may be selected.
//
// The uint8 kernels accumulate products or squared differences of bytes into 32-bit SIMD lanes and
// finish with a 32-bit horizontal reduce. Per element that caps at 255 * 255 = 65025, so the total
// is 65025 * dim, exactly representable in uint32 through dim 66,051 and wrapping at 66,052. Note
// this is twice the old signed limit of 33,025: nothing about the accumulation changed there, the
// top bit was simply being read as a sign, which is why the unsigned reduce costs nothing.
//
// Above this dimension the choosers hand back the scalar kernel, which accumulates into a 64-bit
// ret_t and is exact to roughly dim 2.8e14. One comparison at index creation, no branch in any
// kernel, and no second set of instantiations. Two alternatives were considered and rejected:
//
//   * Widening the horizontal reduce. Measured on an Ice Lake-SP Xeon it costs 4 extra uops in the
//     epilogue: +20% at dim 32, +8-11% across dim 55-200, +4-5% at dim 900-1024, on byte-identical
//     loop code. That is a dependency-chain cost rather than a throughput one, which is why an
//     instruction count understates it. It also only moves the limit rather than removing it, and
//     to a different place per ISA: roughly dim 1,056,816 on AVX-512, but only 264,204 on NEON,
//     where four accumulators are combined with vaddq_u32 in 32 bits before the widening reduce
//     ever sees them.
//
//   * Chunking the accumulation and flushing into a 64-bit total. Exact at any dimension and cheap
//     if the chunk loop lives in the wrapper rather than the kernel (+2 instructions on the fast
//     path, versus +12 to +21 when placed inside the kernel). Deferred rather than dismissed: it
//     is only worth the restructuring if dimensions above 66,051 become a real workload.
//
// Nothing comparable supports that range today. Lucene caps its scalar-quantized format at 1,024
// dimensions and Elasticsearch caps dense vectors at 4,096, both of which keep a 32-bit
// accumulator safe by contract. Faiss's QT_8bit_direct path accumulates full-range bytes into
// int/32-bit lanes with no widening, so it carries the same theoretical limit. Qdrant quantizes to
// 0..127 instead of 0..255, which lowers the per-element cap to 16,129 and pushes the signed limit
// out to dim 133,144, and its raw uint8 metric still sums into i32. So the scalar fallback here is
// already stricter than the alternatives, and optimizing the SIMD path beyond 66,051 would be
// optimizing a range none of them accept.
static constexpr size_t MAX_EXACT_UINT8_SIMD_DIM = 66051;

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
