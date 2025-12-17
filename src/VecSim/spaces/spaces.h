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

namespace spaces {

template <typename RET_TYPE>
using dist_func_t = RET_TYPE (*)(const void *, const void *, size_t);

// Set the distance function for a given data type, metric and dimension. The alignment hint is
// determined according to the chosen implementation and available optimizations.

template <typename DataType, typename DistType>
dist_func_t<DistType> GetDistFunc(VecSimMetric metric, size_t dim, unsigned char *alignment);

// Get distance function for 4-bit (INT4) quantized vectors
template <typename DistType>
dist_func_t<DistType> GetDistFunc_INT4(VecSimMetric metric, size_t dim, unsigned char *alignment);

template <typename DataType>
using normalizeVector_f = void (*)(void *input_vector, const size_t dim);

template <typename DataType>
normalizeVector_f<DataType> GetNormalizeFunc();

static int inline is_little_endian() {
    unsigned int x = 1;
    return *(char *)&x;
}

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
