/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/L2_dispatch_tables.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512FP16_VL.h"
#include "VecSim/spaces/functions/AVX512F_BW_VL_VNNI.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/AVX2_F16C.h"
#include "VecSim/spaces/functions/AVX2_FMA.h"
#include "VecSim/spaces/functions/AVX2_FMA_F16C.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/SSE4.h"
#include "VecSim/spaces/functions/SSE4_F16C.h"
#include "VecSim/spaces/functions/NEON.h"
#include "VecSim/spaces/functions/NEON_DOTPROD.h"
#include "VecSim/spaces/functions/NEON_HP.h"
#include "VecSim/spaces/functions/NEON_BF16.h"
#include "VecSim/spaces/functions/SVE.h"
#include "VecSim/spaces/functions/SVE_BF16.h"
#include "VecSim/spaces/functions/SVE2.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

namespace spaces {

// SQ8-FP32: asymmetric L2 distance between SQ8 storage and FP32 query
dist_func_t<float> L2_SQ8_FP32_GetDistFunc(size_t dim, unsigned char *alignment,
                                           const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, L2_SQ8_FP32_DispatchTable);
    if (idx == L2_SQ8_FP32_DispatchTable.size()) {
        return SQ8_FP32_L2Sqr;
    }
    const auto &tier = L2_SQ8_FP32_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

// SQ8-FP16: asymmetric L2 distance between SQ8 storage and FP16 query.
dist_func_t<float> L2_SQ8_FP16_GetDistFunc(size_t dim, unsigned char *alignment,
                                           const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, L2_SQ8_FP16_DispatchTable);
    if (idx == L2_SQ8_FP16_DispatchTable.size()) {
        return SQ8_FP16_L2Sqr;
    }
    const auto &tier = L2_SQ8_FP16_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

dist_func_t<float> L2_FP32_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, L2_FP32_DispatchTable);
    if (idx == L2_FP32_DispatchTable.size()) {
        return FP32_L2Sqr;
    }
    const auto &tier = L2_FP32_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(float);
    }
    return tier.chooser(dim);
}

dist_func_t<double> L2_FP64_GetDistFunc(size_t dim, unsigned char *alignment,
                                        const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, L2_FP64_DispatchTable);
    if (idx == L2_FP64_DispatchTable.size()) {
        return FP64_L2Sqr;
    }
    const auto &tier = L2_FP64_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(double);
    }
    return tier.chooser(dim);
}

dist_func_t<float> L2_BF16_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    // Big/little-endian is not a tier-selection concern - handled before the table is consulted.
    if (!is_little_endian()) {
        return BF16_L2Sqr_BigEndian;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, L2_BF16_DispatchTable);
    if (idx == L2_BF16_DispatchTable.size()) {
        return BF16_L2Sqr_LittleEndian;
    }
    const auto &tier = L2_BF16_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(bfloat16);
    }
    return tier.chooser(dim);
}

dist_func_t<float> L2_FP16_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, L2_FP16_DispatchTable);
    if (idx == L2_FP16_DispatchTable.size()) {
        return FP16_L2Sqr;
    }
    const auto &tier = L2_FP16_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(float16);
    }
    return tier.chooser(dim);
}

dist_func_t<float> L2_INT8_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, L2_INT8_DispatchTable);
    if (idx == L2_INT8_DispatchTable.size()) {
        return INT8_L2Sqr;
    }
    const auto &tier = L2_INT8_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(int8_t);
    }
    return tier.chooser(dim);
}

dist_func_t<float> L2_UINT8_GetDistFunc(size_t dim, unsigned char *alignment,
                                        const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, L2_UINT8_DispatchTable);
    if (idx == L2_UINT8_DispatchTable.size()) {
        return UINT8_L2Sqr;
    }
    const auto &tier = L2_UINT8_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

// SQ8-to-SQ8 L2 squared distance function (both vectors are uint8 quantized)
dist_func_t<float> L2_SQ8_SQ8_GetDistFunc(size_t dim, unsigned char *alignment,
                                          const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, L2_SQ8_SQ8_DispatchTable);
    if (idx == L2_SQ8_SQ8_DispatchTable.size()) {
        return SQ8_SQ8_L2Sqr;
    }
    const auto &tier = L2_SQ8_SQ8_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

} // namespace spaces
