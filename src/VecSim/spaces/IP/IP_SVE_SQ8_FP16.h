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
#include "VecSim/types/sq8.h"
#include "VecSim/types/float16.h"
#include <arm_sve.h>
#include <cassert>

using sq8 = vecsim_types::sq8;
using float16 = vecsim_types::float16;

/*
 * Optimised asymmetric SQ8<->FP16 inner product using the algebraic identity:
 *
 *   IP(x, y) ~= min * y_sum + delta * sum(q_i * y_i)
 *
 * Hot loop accumulates sum(q_i * y_i) only; FP16 query lanes are widened to FP32
 * inside each step via svcvt_f32_f16_x. Metadata loads use load_unaligned<float>.
 */

// Helper: one SVE-vector-width-of-FP32 step.
//   chunk = svcntw() - number of FP32 lanes per step.
//   pg    = svptrue_b32() - predicate for FP32 lanes.
static inline void
SQ8_FP16_InnerProductStep_SVE(const uint8_t *pVect1, const float16 *pVect2, size_t &offset,
                              svfloat32_t &sum, svbool_t pg, size_t chunk) {
    // SQ8 -> uint32 (widen on load), then to FP32.
    svuint32_t v1_u32 = svld1ub_u32(pg, pVect1 + offset);
    svfloat32_t v1_f = svcvt_f32_u32_x(pg, v1_u32);

    // FP16 query -> FP32. svld1_f16 uses a b16 predicate sized to `chunk` half lanes.
    svbool_t pg16 = svwhilelt_b16(uint32_t(0), uint32_t(chunk));
    svfloat16_t q_h =
        svld1_f16(pg16, reinterpret_cast<const float16_t *>(pVect2) + offset);
    svfloat32_t v2_f = svcvt_f32_f16_x(pg, q_h);

    sum = svmla_f32_x(pg, sum, v1_f, v2_f);
    offset += chunk;
}

// pVect1v = SQ8 storage, pVect2v = FP16 query
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP16_InnerProductSIMD_SVE_IMP(const void *pVect1v, const void *pVect2v,
                                        size_t dimension) {
    assert(dimension >= 16 && "kernel precondition: dispatcher must guard dim >= 16");

    const uint8_t *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const float16 *pVect2 = static_cast<const float16 *>(pVect2v);
    size_t offset = 0;
    svbool_t pg = svptrue_b32();
    const size_t chunk = svcntw();

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    // Partial chunk for dim % chunk lanes. Use _z form so inactive lanes are zero -
    // the final reduction below walks all lanes via svptrue_b32().
    if constexpr (partial_chunk) {
        size_t remaining = dimension % chunk;
        if (remaining > 0) {
            svbool_t pg_partial =
                svwhilelt_b32(uint32_t(0), uint32_t(remaining));
            svbool_t pg16_partial =
                svwhilelt_b16(uint32_t(0), uint32_t(remaining));
            svuint32_t v1_u32 = svld1ub_u32(pg_partial, pVect1 + offset);
            svfloat32_t v1_f = svcvt_f32_u32_z(pg_partial, v1_u32);
            svfloat16_t q_h = svld1_f16(
                pg16_partial, reinterpret_cast<const float16_t *>(pVect2) + offset);
            svfloat32_t v2_f = svcvt_f32_f16_z(pg_partial, q_h);
            sum0 = svmla_f32_z(pg_partial, sum0, v1_f, v2_f);
            offset += remaining;
        }
    }

    // Main loop: 4 chunks per iteration via 4 accumulators.
    const size_t chunk_size = 4 * chunk;
    const size_t number_of_chunks =
        (dimension - (partial_chunk ? dimension % chunk : 0)) / chunk_size;
    for (size_t i = 0; i < number_of_chunks; i++) {
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum0, pg, chunk);
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum1, pg, chunk);
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum2, pg, chunk);
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum3, pg, chunk);
    }

    // Additional steps 0..3.
    if constexpr (additional_steps > 0)
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum0, pg, chunk);
    if constexpr (additional_steps > 1)
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum1, pg, chunk);
    if constexpr (additional_steps > 2)
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum2, pg, chunk);

    svfloat32_t sum = svadd_f32_x(pg, sum0, sum1);
    sum = svadd_f32_x(pg, sum, sum2);
    sum = svadd_f32_x(pg, sum, sum3);
    float quantized_dot = svaddv_f32(pg, sum);

    // Metadata loads - unaligned because odd dim leaves trailers unaligned.
    const uint8_t *params_bytes = static_cast<const uint8_t *>(pVect1v) + dimension;
    const float min_val =
        load_unaligned<float>(params_bytes + sq8::MIN_VAL * sizeof(float));
    const float delta =
        load_unaligned<float>(params_bytes + sq8::DELTA * sizeof(float));
    const uint8_t *query_meta_bytes = reinterpret_cast<const uint8_t *>(
        static_cast<const float16 *>(pVect2v) + dimension);
    const float y_sum =
        load_unaligned<float>(query_meta_bytes + sq8::SUM_QUERY * sizeof(float));

    return min_val * y_sum + delta * quantized_dot;
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP16_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v,
                                    size_t dimension) {
    return 1.0f - SQ8_FP16_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(
                      pVect1v, pVect2v, dimension);
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP16_CosineSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return SQ8_FP16_InnerProductSIMD_SVE<partial_chunk, additional_steps>(
        pVect1v, pVect2v, dimension);
}
