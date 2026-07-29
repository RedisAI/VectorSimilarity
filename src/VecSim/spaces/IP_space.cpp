/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/IP_dispatch_tables.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512FP16_VL.h"
#include "VecSim/spaces/functions/AVX512BF16_VL.h"
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
// SQ8-FP32: asymmetric distance between SQ8 storage and FP32 query
dist_func_t<float> IP_SQ8_FP32_GetDistFunc(size_t dim, unsigned char *alignment,
                                           const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, IP_SQ8_FP32_DispatchTable);
    if (idx == IP_SQ8_FP32_DispatchTable.size()) {
        return SQ8_FP32_InnerProduct;
    }
    const auto &tier = IP_SQ8_FP32_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

// SQ8-FP32: asymmetric cosine distance between SQ8 storage and FP32 query
dist_func_t<float> Cosine_SQ8_FP32_GetDistFunc(size_t dim, unsigned char *alignment,
                                               const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, Cosine_SQ8_FP32_DispatchTable);
    if (idx == Cosine_SQ8_FP32_DispatchTable.size()) {
        return SQ8_FP32_Cosine;
    }
    const auto &tier = Cosine_SQ8_FP32_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

// SQ8-FP16: asymmetric inner product distance between SQ8 storage and FP16 query.
dist_func_t<float> IP_SQ8_FP16_GetDistFunc(size_t dim, unsigned char *alignment,
                                           const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    // Alignment hints below refer to the SQ8 (first) operand per the GetDistFunc contract.
    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, IP_SQ8_FP16_DispatchTable);
    if (idx == IP_SQ8_FP16_DispatchTable.size()) {
        return SQ8_FP16_InnerProduct;
    }
    const auto &tier = IP_SQ8_FP16_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

// SQ8-FP16: asymmetric cosine distance between SQ8 storage and FP16 query.
dist_func_t<float> Cosine_SQ8_FP16_GetDistFunc(size_t dim, unsigned char *alignment,
                                               const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, Cosine_SQ8_FP16_DispatchTable);
    if (idx == Cosine_SQ8_FP16_DispatchTable.size()) {
        return SQ8_FP16_Cosine;
    }
    const auto &tier = Cosine_SQ8_FP16_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

// SQ8-to-SQ8 Inner Product distance function (both vectors are uint8 quantized with precomputed
// sum)
dist_func_t<float> IP_SQ8_SQ8_GetDistFunc(size_t dim, unsigned char *alignment,
                                          const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, IP_SQ8_SQ8_DispatchTable);
    if (idx == IP_SQ8_SQ8_DispatchTable.size()) {
        return SQ8_SQ8_InnerProduct;
    }
    const auto &tier = IP_SQ8_SQ8_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

// SQ8-to-SQ8 Cosine distance function (both vectors are uint8 quantized with precomputed sum)
dist_func_t<float> Cosine_SQ8_SQ8_GetDistFunc(size_t dim, unsigned char *alignment,
                                              const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, Cosine_SQ8_SQ8_DispatchTable);
    if (idx == Cosine_SQ8_SQ8_DispatchTable.size()) {
        return SQ8_SQ8_Cosine;
    }
    const auto &tier = Cosine_SQ8_SQ8_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

dist_func_t<float> IP_FP32_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, IP_FP32_DispatchTable);
    if (idx == IP_FP32_DispatchTable.size()) {
        return FP32_InnerProduct;
    }
    const auto &tier = IP_FP32_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(float);
    }
    return tier.chooser(dim);
}

dist_func_t<double> IP_FP64_GetDistFunc(size_t dim, unsigned char *alignment,
                                        const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, IP_FP64_DispatchTable);
    if (idx == IP_FP64_DispatchTable.size()) {
        return FP64_InnerProduct;
    }
    const auto &tier = IP_FP64_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(double);
    }
    return tier.chooser(dim);
}

dist_func_t<float> IP_BF16_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    // Big/little-endian is not a tier-selection concern - handled before the table is consulted.
    if (!is_little_endian()) {
        return BF16_InnerProduct_BigEndian;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, IP_BF16_DispatchTable);
    if (idx == IP_BF16_DispatchTable.size()) {
        return BF16_InnerProduct_LittleEndian;
    }
    const auto &tier = IP_BF16_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(bfloat16);
    }
    return tier.chooser(dim);
}

dist_func_t<float> IP_FP16_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, IP_FP16_DispatchTable);
    if (idx == IP_FP16_DispatchTable.size()) {
        return FP16_InnerProduct;
    }
    const auto &tier = IP_FP16_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(float16);
    }
    return tier.chooser(dim);
}

dist_func_t<float> IP_INT8_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, IP_INT8_DispatchTable);
    if (idx == IP_INT8_DispatchTable.size()) {
        return INT8_InnerProduct;
    }
    const auto &tier = IP_INT8_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(int8_t);
    }
    return tier.chooser(dim);
}

dist_func_t<float> Cosine_INT8_GetDistFunc(size_t dim, unsigned char *alignment,
                                           const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, Cosine_INT8_DispatchTable);
    if (idx == Cosine_INT8_DispatchTable.size()) {
        return INT8_Cosine;
    }
    const auto &tier = Cosine_INT8_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(int8_t);
    }
    return tier.chooser(dim);
}

dist_func_t<float> IP_UINT8_GetDistFunc(size_t dim, unsigned char *alignment,
                                        const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, IP_UINT8_DispatchTable);
    if (idx == IP_UINT8_DispatchTable.size()) {
        return UINT8_InnerProduct;
    }
    const auto &tier = IP_UINT8_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

dist_func_t<float> Cosine_UINT8_GetDistFunc(size_t dim, unsigned char *alignment,
                                            const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    auto features = getCpuOptimizationFeatures(arch_opt);
    size_t idx = select_tier_index(features, dim, Cosine_UINT8_DispatchTable);
    if (idx == Cosine_UINT8_DispatchTable.size()) {
        return UINT8_Cosine;
    }
    const auto &tier = Cosine_UINT8_DispatchTable[idx];
    if (tier.alignment_chunk_elems != 0 && dim % tier.alignment_chunk_elems == 0) {
        *alignment = tier.alignment_chunk_elems * sizeof(uint8_t);
    }
    return tier.chooser(dim);
}

} // namespace spaces
