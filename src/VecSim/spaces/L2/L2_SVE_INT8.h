/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

// Aligned step using svptrue_b8()
static inline void L2SquareStep(const int8_t *&pVect1, const int8_t *&pVect2, svfloat32_t &sum) {
    svbool_t pg = svptrue_b8();
    // Note: Because all the bits are 1, the extention to 16 and 32 bits does not make a difference
    // Otherwise, pg should be recalculated for 16 and 32 operations

    svint8_t v1_i8 = svld1_s8(pg, pVect1);
    svint8_t v2_i8 = svld1_s8(pg, pVect2);

    svint16_t v1_16_l = svunpklo_s16(v1_i8);
    svint16_t v1_16_h = svunpkhi_s16(v1_i8);
    svint16_t v2_16_l = svunpklo_s16(v2_i8);
    svint16_t v2_16_h = svunpkhi_s16(v2_i8);

    // Calculate difference and square for low part
    svint16_t diff_l = svsub_s16_x(pg, v1_16_l, v2_16_l);

    svint32_t diff32_l_l = svunpklo_s32(diff_l);
    svint32_t diff32_l_h = svunpkhi_s32(diff_l);

    svint32_t sq_l_l = svmul_s32_z(pg, diff32_l_l, diff32_l_l);
    svint32_t sq_l_h = svmul_s32_z(pg, diff32_l_h, diff32_l_h);

    svint32_t sq_l = svadd_s32_z(pg, sq_l_l, sq_l_h);

    svint16_t diff_h = svsub_s16_x(pg, v1_16_h, v2_16_h);

    svint32_t diff32_h_l = svunpklo_s32(diff_h);
    svint32_t diff32_h_h = svunpkhi_s32(diff_h);

    svint32_t sq_h_l = svmul_s32_z(pg, diff32_h_l, diff32_h_l);
    svint32_t sq_h_h = svmul_s32_z(pg, diff32_h_h, diff32_h_h);

    svint32_t sq_h = svadd_s32_z(pg, sq_h_l, sq_h_h);

    // Convert to float and accumulate
    svfloat32_t sqf_l = svcvt_f32_s32_z(pg, sq_l);
    svfloat32_t sqf_h = svcvt_f32_s32_z(pg, sq_h);

    sum = svadd_f32_z(pg, sum, sqf_l);
    sum = svadd_f32_z(pg, sum, sqf_h);

    pVect1 += svcntb();
    pVect2 += svcntb();
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_L2SqrSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const int8_t *pVect1 = reinterpret_cast<const int8_t *>(pVect1v);
    const int8_t *pVect2 = reinterpret_cast<const int8_t *>(pVect2v);

    // number of int8 per SVE register
    const size_t vl = svcntb();
    const size_t chunk_size = 4 * vl;

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    size_t offset = 0;
    size_t num_main_blocks = dimension / chunk_size;

    for (size_t i = 0; i < num_main_blocks; ++i) {
        L2SquareStep(pVect1, pVect2, sum0);
        L2SquareStep(pVect1, pVect2, sum1);
        L2SquareStep(pVect1, pVect2, sum2);
        L2SquareStep(pVect1, pVect2, sum3);
        offset += chunk_size;
    }

    if constexpr (additional_steps > 0) {
        for (unsigned char c = 0; c < additional_steps; ++c) {
            L2SquareStep(pVect1, pVect2, sum0);
            offset += vl;
        }
    }

    if constexpr (partial_chunk) {

        svbool_t pg = svwhilelt_b8(offset, dimension);
        svint8_t v1_i8 = svld1_s8(pg, pVect1);
        svint8_t v2_i8 = svld1_s8(pg, pVect2);

        svbool_t pg32 = svwhilelt_b32(offset, dimension);

        svint16_t v1_16_l = svunpklo_s16(v1_i8);
        svint16_t v1_16_h = svunpkhi_s16(v1_i8);
        svint16_t v2_16_l = svunpklo_s16(v2_i8);
        svint16_t v2_16_h = svunpkhi_s16(v2_i8);

        // Calculate difference and square for low part
        svint16_t diff_l = svsub_s16_x(svwhilelt_b16(offset, dimension), v1_16_l, v2_16_l);

        svint32_t diff32_l_l = svunpklo_s32(diff_l);
        svint32_t diff32_l_h = svunpkhi_s32(diff_l);

        svint32_t sq_l_l = svmul_s32_z(pg32, diff32_l_l, diff32_l_l);
        svint32_t sq_l_h = svmul_s32_z(pg32, diff32_l_h, diff32_l_h);

        svint32_t sq_l = svadd_s32_z(pg32, sq_l_l, sq_l_h);

        svint16_t diff_h = svsub_s16_x(pg32, v1_16_h, v2_16_h);

        svint32_t diff32_h_l = svunpklo_s32(diff_h);
        svint32_t diff32_h_h = svunpkhi_s32(diff_h);

        svint32_t sq_h_l = svmul_s32_z(pg32, diff32_h_l, diff32_h_l);
        svint32_t sq_h_h = svmul_s32_z(pg32, diff32_h_h, diff32_h_h);

        svint32_t sq_h = svadd_s32_z(pg32, sq_h_l, sq_h_h);

        // Convert to float and accumulate
        svfloat32_t sqf_l = svcvt_f32_s32_z(pg32, sq_l);
        svfloat32_t sqf_h = svcvt_f32_s32_z(pg32, sq_h);

        sum0 = svadd_f32_m(pg32, sum0, sqf_l);
        sum0 = svadd_f32_m(pg32, sum0, sqf_h);
    }

    // Combine the partial sums
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum1);
    sum2 = svadd_f32_z(svptrue_b32(), sum2, sum3);
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum2);

    // Horizontal sum
    return svaddv_f32(svptrue_b32(), sum0);
}
