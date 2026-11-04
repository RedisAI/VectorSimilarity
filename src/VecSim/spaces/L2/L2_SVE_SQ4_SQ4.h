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
#include "VecSim/spaces/IP/IP_SVE_SQ4_FP32.h" // SQ4_SVE_Unpack
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
 * high nibble for odd i, followed by unaligned FP32 metadata.
 */

// Helper: accumulate Σ(diff_i²) for one SVE vector of coordinates.
static inline void L2StepSQ4_SQ4_SVE(const uint8_t *x, const uint8_t *y, size_t offset,
                                     svfloat32_t &sum, svbool_t packed_pg, svfloat32_t min_diff,
                                     svfloat32_t delta_x, svfloat32_t delta_y) {
    const svbool_t pg = svptrue_b32();
    const svfloat32_t qx = SQ4_SVE_Unpack(x + offset / 2, packed_pg);
    const svfloat32_t qy = SQ4_SVE_Unpack(y + offset / 2, packed_pg);
    const svfloat32_t scaled = svmls_f32_x(pg, svmul_f32_x(pg, delta_x, qx), delta_y, qy);
    const svfloat32_t diff = svadd_f32_x(pg, min_diff, scaled);
    sum = svmla_f32_x(pg, sum, diff, diff);
}

// Four accumulators, full vectors first and a predicated tail last, as SQ4-FP32. Every full chunk
// is an even number of coordinates, so every packed load starts at a whole byte.
template <bool partial_chunk, unsigned char additional_steps>
float SQ4_SQ4_L2SqrSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    using vecsim_types::sq4;
    const auto *x = static_cast<const uint8_t *>(pVect1v);
    const auto *y = static_cast<const uint8_t *>(pVect2v);
    const size_t chunk = svcntw();
    const svbool_t pg = svptrue_b32();
    const svbool_t packed_pg = svwhilelt_b32(uint64_t(0), uint64_t(chunk / 2));
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
    size_t offset = 0;
    // Bounds compared rather than divided, as in the SQ8 kernel: chunk is a runtime value.
    const size_t chunk_size = 4 * chunk;
    for (; offset + chunk_size <= dimension; offset += chunk_size) {
        L2StepSQ4_SQ4_SVE(x, y, offset, sum0, packed_pg, min_diff, delta_x_vec, delta_y_vec);
        L2StepSQ4_SQ4_SVE(x, y, offset + chunk, sum1, packed_pg, min_diff, delta_x_vec,
                          delta_y_vec);
        L2StepSQ4_SQ4_SVE(x, y, offset + 2 * chunk, sum2, packed_pg, min_diff, delta_x_vec,
                          delta_y_vec);
        L2StepSQ4_SQ4_SVE(x, y, offset + 3 * chunk, sum3, packed_pg, min_diff, delta_x_vec,
                          delta_y_vec);
    }
    if constexpr (additional_steps > 0) {
        L2StepSQ4_SQ4_SVE(x, y, offset, sum0, packed_pg, min_diff, delta_x_vec, delta_y_vec);
        offset += chunk;
    }
    if constexpr (additional_steps > 1) {
        L2StepSQ4_SQ4_SVE(x, y, offset, sum1, packed_pg, min_diff, delta_x_vec, delta_y_vec);
        offset += chunk;
    }
    if constexpr (additional_steps > 2) {
        L2StepSQ4_SQ4_SVE(x, y, offset, sum2, packed_pg, min_diff, delta_x_vec, delta_y_vec);
        offset += chunk;
    }
    if constexpr (partial_chunk) {
        const size_t remaining = dimension - offset;
        const svbool_t tail_pg = svwhilelt_b32(uint64_t(0), uint64_t(remaining));
        const svbool_t tail_packed_pg = svwhilelt_b32(uint64_t(0), uint64_t((remaining + 1) / 2));
        const svfloat32_t qx = SQ4_SVE_Unpack(x + offset / 2, tail_packed_pg);
        const svfloat32_t qy = SQ4_SVE_Unpack(y + offset / 2, tail_packed_pg);
        const svfloat32_t scaled =
            svmls_f32_z(tail_pg, svmul_f32_z(tail_pg, delta_x_vec, qx), delta_y_vec, qy);
        const svfloat32_t diff = svadd_f32_z(tail_pg, min_diff, scaled);
        // Merging (_m): sum3 keeps its main-loop lanes. tail_pg also drops the unused high nibble
        // of an odd dim.
        sum3 = svmla_f32_m(tail_pg, sum3, diff, diff);
    }
    svfloat32_t sum = svadd_f32_x(pg, sum0, sum1);
    sum = svadd_f32_x(pg, sum, sum2);
    sum = svadd_f32_x(pg, sum, sum3);
    return svaddv_f32(pg, sum);
}

} // namespace
