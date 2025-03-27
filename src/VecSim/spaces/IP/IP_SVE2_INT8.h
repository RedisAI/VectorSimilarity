/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

static void InnerProductStep(const int8_t *&pVect1, const int8_t *&pVect2, svfloat32_t &sum) {
    svbool_t pg = svptrue_b8();

    // Load int8 vectors
    svint8_t v1_i8 = svld1_s8(pg, pVect1);
    svint8_t v2_i8 = svld1_s8(pg, pVect2);

    svfloat32_t ipf32 = svcvt_f32_s32_z(pg, svdot_s32(svdup_s32(0), v1_i8, v2_i8));

    sum = svadd_f32_m(pg, sum, ipf32);

    pVect1 += svcntb();
    pVect2 += svcntb();
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const int8_t *pVect1 = reinterpret_cast<const int8_t *>(pVect1v);
    const int8_t *pVect2 = reinterpret_cast<const int8_t *>(pVect2v);

    size_t offset = 0;
    const size_t vl = svcntb();
    const size_t chunk_size = 4 * vl;

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    size_t num_chunks = dimension / chunk_size;

    for (size_t i = 0; i < num_chunks; ++i) {
        InnerProductStep(pVect1, pVect2, sum0);
        InnerProductStep(pVect1, pVect2, sum1);
        InnerProductStep(pVect1, pVect2, sum2);
        InnerProductStep(pVect1, pVect2, sum3);
        offset += chunk_size;
    }

    if constexpr (additional_steps > 0) {
        for (unsigned char c = 0; c < additional_steps; ++c) {
            InnerProductStep(pVect1, pVect2, sum0);
            offset += vl;
        }
    }

    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b8(offset, dimension);
        svbool_t pg32 = svwhilelt_b32(offset, dimension);

        svint8_t v1_i8 = svld1_s8(pg, pVect1);
        svint8_t v2_i8 = svld1_s8(pg, pVect2);

        svfloat32_t ipf32 = svcvt_f32_s32_z(pg32, svdot_s32(svdup_s32(0), v1_i8, v2_i8));

        sum0 = svadd_f32_m(pg32, sum0, ipf32);

        pVect1 += svcntb();
        pVect2 += svcntb();
    }

    // Combine and reduce
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum1);
    sum2 = svadd_f32_z(svptrue_b32(), sum2, sum3);
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum2);

    // Horizontal sum
    float result = svaddv_f32(svptrue_b32(), sum0);
    return result;
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_InnerProductSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f -
           INT8_InnerProductImp<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_CosineSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float ip = INT8_InnerProductImp<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);
    float norm_v1 =
        *reinterpret_cast<const float *>(static_cast<const int8_t *>(pVect1v) + dimension);
    float norm_v2 =
        *reinterpret_cast<const float *>(static_cast<const int8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}
