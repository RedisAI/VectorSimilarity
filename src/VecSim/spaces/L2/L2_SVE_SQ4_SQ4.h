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
#include "VecSim/spaces/IP/IP_SVE_SQ4_SQ4.h" // SQ4_SQ4_InnerProductSIMD_SVE_IMP
#include "VecSim/types/sq4.h"
#include "VecSim/utils/alignment.h"
#include <arm_sve.h>

// SVE.cpp and SVE2.cpp both compile this header, under different -march flags. The
// anonymous namespace keeps each tier's bodies to itself; without it they are weak
// symbols that both objects define and link order picks the -march. Only the Choose_*
// entry points stay external. Dependencies above must stay outside the namespace.
namespace {

/*
 * Symmetric SQ4-SQ4 L2 squared distance, reconstructed per element as the scalar SQ4_SQ4_L2Sqr:
 *
 *   ||x - y||^2 = Σ((min_x - min_y) + (delta_x * qx_i - delta_y * qy_i))^2
 *
 * Subtracting the offsets first keeps rounding at the scale of the residual, not of the norms,
 * when the data has a large common offset (the ||x||^2 + ||y||^2 - 2 * IP identity loses HNSW
 * recall there). Storage packs coordinate i into byte i / 2, the low nibble for even i and the
 * high nibble for odd i, followed by unaligned FP32 metadata. Both operands share the packing and
 * L2 is a plain sum, so the low nibbles (even coordinates) and the high nibbles (odd coordinates)
 * are processed in separate vectors without reordering. Each difference is evaluated with two
 * fused multiply-adds as ((min_x - min_y) + delta_x * qx_i) - delta_y * qy_i, so the offsets are
 * still subtracted first.
 */

// Coordinates per step: one u32 lane per packed byte, two coordinates per lane.
static inline size_t SQ4_SQ4_L2_SVE_CoordinatesPerStep() { return 2 * svcntw(); }

static inline svfloat32_t SQ4_SQ4_L2Diff_SVE(svbool_t pg, svuint32_t qx, svuint32_t qy,
                                             svfloat32_t min_diff, svfloat32_t delta_x,
                                             svfloat32_t delta_y) {
    const svfloat32_t fx = svcvt_f32_u32_x(pg, qx);
    const svfloat32_t fy = svcvt_f32_u32_x(pg, qy);
    return svmls_f32_x(pg, svmla_f32_x(pg, min_diff, delta_x, fx), delta_y, fy);
}

// Accumulate diff^2 for the 2 * svcntw() coordinates held in svcntw() packed bytes: the even
// coordinates into sum_even, the odd ones into sum_odd.
static inline void L2StepSQ4_SQ4_SVE(const uint8_t *x, const uint8_t *y, svfloat32_t &sum_even,
                                     svfloat32_t &sum_odd, svfloat32_t min_diff,
                                     svfloat32_t delta_x, svfloat32_t delta_y) {
    const svbool_t pg = svptrue_b32();
    const svuint32_t bx = svld1ub_u32(pg, x), by = svld1ub_u32(pg, y);
    const svfloat32_t d_even = SQ4_SQ4_L2Diff_SVE(
        pg, svand_n_u32_x(pg, bx, 15), svand_n_u32_x(pg, by, 15), min_diff, delta_x, delta_y);
    const svfloat32_t d_odd = SQ4_SQ4_L2Diff_SVE(
        pg, svlsr_n_u32_x(pg, bx, 4), svlsr_n_u32_x(pg, by, 4), min_diff, delta_x, delta_y);
    sum_even = svmla_f32_x(pg, sum_even, d_even, d_even);
    sum_odd = svmla_f32_x(pg, sum_odd, d_odd, d_odd);
}

// Four accumulators, full steps first and a predicated tail last, as SQ4-FP32. Every full step is
// an even number of coordinates, so every packed load starts at a whole byte.
template <bool partial_chunk, unsigned char additional_steps>
float SQ4_SQ4_L2SqrSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    using vecsim_types::sq4;
    const auto *x = static_cast<const uint8_t *>(pVect1v);
    const auto *y = static_cast<const uint8_t *>(pVect2v);
    const size_t vl = svcntw();  // packed bytes per step
    const size_t chunk = 2 * vl; // coordinates per step
    const svbool_t pg = svptrue_b32();
    const auto *params_x = x + sq4::data_bytes_count(dimension);
    const auto *params_y = y + sq4::data_bytes_count(dimension);
    const float min_x = load_unaligned<float>(params_x + sq4::MIN_VAL * sizeof(float));
    const float delta_x = load_unaligned<float>(params_x + sq4::DELTA * sizeof(float));
    const float min_y = load_unaligned<float>(params_y + sq4::MIN_VAL * sizeof(float));
    const float delta_y = load_unaligned<float>(params_y + sq4::DELTA * sizeof(float));
    const svfloat32_t min_diff = svdup_f32(min_x - min_y);
    const svfloat32_t delta_x_vec = svdup_f32(delta_x);
    const svfloat32_t delta_y_vec = svdup_f32(delta_y);
    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);
    size_t offset = 0; // in packed bytes
    const size_t main_bytes = (dimension / (4 * chunk)) * 4 * vl;
    for (; offset < main_bytes; offset += 4 * vl) {
        L2StepSQ4_SQ4_SVE(x + offset, y + offset, sum0, sum1, min_diff, delta_x_vec, delta_y_vec);
        L2StepSQ4_SQ4_SVE(x + offset + vl, y + offset + vl, sum2, sum3, min_diff, delta_x_vec,
                          delta_y_vec);
        L2StepSQ4_SQ4_SVE(x + offset + 2 * vl, y + offset + 2 * vl, sum0, sum1, min_diff,
                          delta_x_vec, delta_y_vec);
        L2StepSQ4_SQ4_SVE(x + offset + 3 * vl, y + offset + 3 * vl, sum2, sum3, min_diff,
                          delta_x_vec, delta_y_vec);
    }
    if constexpr (additional_steps > 0) {
        L2StepSQ4_SQ4_SVE(x + offset, y + offset, sum0, sum1, min_diff, delta_x_vec, delta_y_vec);
        offset += vl;
    }
    if constexpr (additional_steps > 1) {
        L2StepSQ4_SQ4_SVE(x + offset, y + offset, sum2, sum3, min_diff, delta_x_vec, delta_y_vec);
        offset += vl;
    }
    if constexpr (additional_steps > 2) {
        L2StepSQ4_SQ4_SVE(x + offset, y + offset, sum0, sum1, min_diff, delta_x_vec, delta_y_vec);
        offset += vl;
    }
    if constexpr (partial_chunk) {
        // remaining = dimension - 2 * offset coordinates: (remaining + 1) / 2 bytes hold them, and
        // the even coordinate of each of those bytes is in the tail; the odd one is for the first
        // remaining / 2 bytes (an odd dim's last high nibble is padding).
        const size_t remaining = dimension - 2 * offset;
        const svbool_t even_pg = svwhilelt_b32(uint64_t(0), uint64_t((remaining + 1) / 2));
        const svbool_t odd_pg = svwhilelt_b32(uint64_t(0), uint64_t(remaining / 2));
        const svuint32_t bx = svld1ub_u32(even_pg, x + offset);
        const svuint32_t by = svld1ub_u32(even_pg, y + offset);
        const svfloat32_t d_even =
            SQ4_SQ4_L2Diff_SVE(pg, svand_n_u32_x(pg, bx, 15), svand_n_u32_x(pg, by, 15), min_diff,
                               delta_x_vec, delta_y_vec);
        const svfloat32_t d_odd =
            SQ4_SQ4_L2Diff_SVE(pg, svlsr_n_u32_x(pg, bx, 4), svlsr_n_u32_x(pg, by, 4), min_diff,
                               delta_x_vec, delta_y_vec);
        // Merging (_m): lanes outside the tail keep the running sums.
        sum3 = svmla_f32_m(even_pg, sum3, d_even, d_even);
        sum2 = svmla_f32_m(odd_pg, sum2, d_odd, d_odd);
    }
    svfloat32_t sum = svadd_f32_x(pg, sum0, sum1);
    sum = svadd_f32_x(pg, sum, sum2);
    sum = svadd_f32_x(pg, sum, sum3);
    return svaddv_f32(pg, sum);
}

/*
 * Symmetric SQ4-SQ4 L2 squared distance with the same identity as SQ8-SQ8, for centered indexes:
 *
 *   ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * IP(x, y)
 *
 * where ||x||^2 and ||y||^2 are the stored SUM_SQUARES metadata and IP(x, y) comes from the
 * integer code dot of the inner product kernel. It rounds at the scale of the norms, so the index
 * factory uses it only when the stored vectors are centered (a mean is set); plain indexes keep the
 * per-element reconstruction above. Chunk (SQ4_SQ4_SVE_CoordinatesPerVector),
 * exact-dim bound and dispatch as the inner product kernel.
 */
template <bool partial_chunk, unsigned char additional_steps>
float SQ4_SQ4_L2SqrIdentitySIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    using vecsim_types::sq4;
    const float ip = SQ4_SQ4_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(
        pVect1v, pVect2v, dimension);
    const auto *x = static_cast<const uint8_t *>(pVect1v);
    const auto *y = static_cast<const uint8_t *>(pVect2v);
    const size_t data_bytes = sq4::data_bytes_count(dimension);
    const float sum_sq_x = load_unaligned<float>(x + data_bytes + sq4::SUM_SQUARES * sizeof(float));
    const float sum_sq_y = load_unaligned<float>(y + data_bytes + sq4::SUM_SQUARES * sizeof(float));
    return sum_sq_x + sum_sq_y - 2.0f * ip;
}

} // namespace
