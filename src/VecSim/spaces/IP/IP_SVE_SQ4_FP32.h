/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/types/sq4.h"
#include "VecSim/utils/alignment.h"
#include <arm_sve.h>
#include <cmath>

// SVE.cpp and SVE2.cpp both compile this header, under different -march flags. The
// anonymous namespace keeps each tier's bodies to itself; without it they are weak
// symbols that both objects define and link order picks the -march. Only the Choose_*
// entry points stay external. Dependencies above must stay outside the namespace.
namespace {

/*
 * Asymmetric SQ4 inner product with the same identity as SQ8:
 *
 *   IP(x, y) ≈ Σ((min + delta * q_i) * y_i) = min * y_sum + delta * Σ(q_i * y_i)
 *
 * Storage packs coordinate i into byte i / 2, the low nibble for even i and the high nibble for
 * odd i, followed by unaligned FP32 metadata. The query blob carries y_sum after its dim floats.
 */

// Widen packed bytes into the u32 lanes selected by packed_pg. Interleaving their low and high
// nibbles gives one FP32 vector in coordinate order. The byte predicate keeps every load inside
// the code bytes, including a final odd coordinate.
static inline svfloat32_t SQ4_SVE_Unpack(const uint8_t *data, svbool_t packed_pg) {
    const svbool_t pg = svptrue_b32();
    const svuint32_t packed = svld1ub_u32(packed_pg, data);
    const svuint32_t low = svand_n_u32_x(pg, packed, 15);
    const svuint32_t high = svlsr_n_u32_x(pg, packed, 4);
    return svcvt_f32_u32_x(pg, svzip1_u32(low, high));
}

// Helper: accumulate Σ(q_i * y_i) for one SVE vector of coordinates (no dequantization).
static inline void InnerProductStepSQ4_FP32(const uint8_t *x, const float *y, size_t &offset,
                                            svfloat32_t &sum, const size_t chunk,
                                            svbool_t packed_pg) {
    const svbool_t pg = svptrue_b32();
    const svfloat32_t codes = SQ4_SVE_Unpack(x + offset / 2, packed_pg);
    sum = svmla_f32_x(pg, sum, codes, svld1_f32(pg, y + offset));
    offset += chunk;
}

// Four accumulators and one final affine correction, as in SQ8. Full chunks precede the tail, so
// every packed load starts at an even coordinate (a whole byte).
template <bool partial_chunk, unsigned char additional_steps>
float SQ4_FP32_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    using vecsim_types::sq4;
    const auto *x = static_cast<const uint8_t *>(pVect1v);
    const auto *y = static_cast<const float *>(pVect2v);
    const size_t chunk = svcntw();
    const svbool_t pg = svptrue_b32();
    const svbool_t packed_pg = svwhilelt_b32(uint64_t(0), uint64_t(chunk / 2));
    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);
    size_t offset = 0;
    const size_t blocks = dimension / (4 * chunk);
    for (size_t i = 0; i < blocks; ++i) {
        InnerProductStepSQ4_FP32(x, y, offset, sum0, chunk, packed_pg);
        InnerProductStepSQ4_FP32(x, y, offset, sum1, chunk, packed_pg);
        InnerProductStepSQ4_FP32(x, y, offset, sum2, chunk, packed_pg);
        InnerProductStepSQ4_FP32(x, y, offset, sum3, chunk, packed_pg);
    }
    if constexpr (additional_steps > 0) {
        InnerProductStepSQ4_FP32(x, y, offset, sum0, chunk, packed_pg);
    }
    if constexpr (additional_steps > 1) {
        InnerProductStepSQ4_FP32(x, y, offset, sum1, chunk, packed_pg);
    }
    if constexpr (additional_steps > 2) {
        InnerProductStepSQ4_FP32(x, y, offset, sum2, chunk, packed_pg);
    }
    if constexpr (partial_chunk) {
        const size_t remaining = dimension - offset;
        const svbool_t tail_pg = svwhilelt_b32(uint64_t(0), uint64_t(remaining));
        const svbool_t tail_packed_pg = svwhilelt_b32(uint64_t(0), uint64_t((remaining + 1) / 2));
        const svfloat32_t codes = SQ4_SVE_Unpack(x + offset / 2, tail_packed_pg);
        // tail_pg drops the unused high nibble of an odd dim.
        sum0 = svmla_f32_m(tail_pg, sum0, codes, svld1_f32(tail_pg, y + offset));
    }
    svfloat32_t sum = svadd_f32_x(pg, sum0, sum1);
    sum = svadd_f32_x(pg, sum, sum2);
    sum = svadd_f32_x(pg, sum, sum3);
    const float dot = svaddv_f32(pg, sum);
    const auto *params = x + sq4::data_bytes_count(dimension);
    const float min_val = load_unaligned<float>(params + sq4::MIN_VAL * sizeof(float));
    const float delta = load_unaligned<float>(params + sq4::DELTA * sizeof(float));
    const float distance = 1.0f - (min_val * y[dimension + sq4::SUM_QUERY] + delta * dot);
    // The unscaled code dot can overflow even when the reconstructed products stay finite.
    if (!std::isfinite(distance)) {
        return SQ4_FP32_InnerProductReconstructed(pVect1v, pVect2v, dimension);
    }
    return distance;
}

} // namespace
