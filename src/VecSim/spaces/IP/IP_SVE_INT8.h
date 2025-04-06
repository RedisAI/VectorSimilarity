/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

static void InnerProductStep(const int8_t *&pVect1, const int8_t *&pVect2, size_t &offset,
                             svint32_t &sum) {
    svbool_t pg = svptrue_b8();

    // Load int8 vectors
    svint8_t v1_i8 = svld1_s8(pg, pVect1 + offset);
    svint8_t v2_i8 = svld1_s8(pg, pVect2 + offset);

    sum = svdot_s32(sum, v1_i8, v2_i8);

    offset += svcntb(); // Move to the next set of int8 elements
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const int8_t *pVect1 = reinterpret_cast<const int8_t *>(pVect1v);
    const int8_t *pVect2 = reinterpret_cast<const int8_t *>(pVect2v);

    size_t offset = 0;
    const size_t vl = svcntb();
    const size_t chunk_size = 4 * vl;

    // Each innerProductStep adds maximum 2^8 & 2^8 = 2^16
    // Therefore, on a single accumulator, we can perform 2^15 steps before overflowing
    // That scenario will happen only is the dimension of the vector is larger than 16*4*2^15 = 2^21
    // (16 int8 in 1 SVE register) * (4 accumulators) * (2^15 steps)
    // We can safely assume that the dimension is smaller than that
    // So using int32_t is safe

    svint32_t sum0 = svdup_s32(0);
    svint32_t sum1 = svdup_s32(0);
    svint32_t sum2 = svdup_s32(0);
    svint32_t sum3 = svdup_s32(0);

    size_t num_chunks = dimension / chunk_size;

    for (size_t i = 0; i < num_chunks; ++i) {
        InnerProductStep(pVect1, pVect2, offset, sum0);
        InnerProductStep(pVect1, pVect2, offset, sum1);
        InnerProductStep(pVect1, pVect2, offset, sum2);
        InnerProductStep(pVect1, pVect2, offset, sum3);
    }

    // Process remaining complete SVE vectors that didn't fit into the main loop
    // These are full vector operations (0-3 elements)
    if constexpr (additional_steps > 0) {
        if constexpr (additional_steps >= 1) {
            InnerProductStep(pVect1, pVect2, offset, sum0);
        }
        if constexpr (additional_steps >= 2) {
            InnerProductStep(pVect1, pVect2, offset, sum1);
        }
        if constexpr (additional_steps >= 3) {
            InnerProductStep(pVect1, pVect2, offset, sum2);
        }
    }

    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b8(offset, dimension);

        svint8_t v1_i8 = svld1_s8(pg, pVect1 + offset); // Load int8 vectors
        svint8_t v2_i8 = svld1_s8(pg, pVect2 + offset); // Load int8 vectors

        sum3 = svdot_s32(sum3, v1_i8, v2_i8);

        pVect1 += svcntb();
        pVect2 += svcntb();
    }

    sum0 = svadd_s32_x(svptrue_b32(), sum0, sum1);
    sum2 = svadd_s32_x(svptrue_b32(), sum2, sum3);

    // Perform vector addition in parallel and Horizontal sum
    int32_t sum_all = svaddv_s32(svptrue_b32(), svadd_s32_x(svptrue_b32(), sum0, sum2));

    return sum_all;
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f -
           INT8_InnerProductImp<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_CosineSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float ip = INT8_InnerProductImp<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);
    float norm_v1 =
        *reinterpret_cast<const float *>(static_cast<const int8_t *>(pVect1v) + dimension);
    float norm_v2 =
        *reinterpret_cast<const float *>(static_cast<const int8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}
