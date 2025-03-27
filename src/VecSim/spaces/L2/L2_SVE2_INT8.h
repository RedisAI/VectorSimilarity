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

    // Subtract v2 from v1 and widen the results to int16 for the even indexes
    svint16_t diff_e = svsublb_s16(v1_i8, v2_i8);

    // Subtract v2 from v1 and widen the results to int16 for the odd indexes
    svint16_t diff_o = svsublt_s16(v1_i8, v2_i8);

    svint32_t sum_int = svdup_s32(0);

    // sum_int can't overflow because max value is  4*(2^16)

    sum_int = svmlalb_s32(sum_int, diff_e, diff_e);
    sum_int = svmlalt_s32(sum_int, diff_e, diff_e);

    sum_int = svmlalb_s32(sum_int, diff_o, diff_o);
    sum_int = svmlalt_s32(sum_int, diff_o, diff_o);

    sum = svadd_f32_z(pg, sum, svcvt_f32_s32_z(pg, sum_int));

    pVect1 += svcntb();
    pVect2 += svcntb();
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_L2SqrSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const int8_t *pVect1 = reinterpret_cast<const int8_t *>(pVect1v);
    const int8_t *pVect2 = reinterpret_cast<const int8_t *>(pVect2v);

    // number of int8 per SVE2 register
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
        svbool_t pg32 = svwhilelt_b32(offset, dimension);

        svint8_t v1_i8 = svld1_s8(pg, pVect1);
        svint8_t v2_i8 = svld1_s8(pg, pVect2);

        // Subtract v2 from v1 and widen the results to int16 for the even indexes
        svint16_t diff_e = svsublb_s16(v1_i8, v2_i8);

        // Subtract v2 from v1 and widen the results to int16 for the odd indexes
        svint16_t diff_o = svsublt_s16(v1_i8, v2_i8);

        svint32_t sum_int = svdup_s32(0);

        sum_int = svmlalb_s32(sum_int, diff_e, diff_e);
        sum_int = svmlalt_s32(sum_int, diff_e, diff_e);

        sum_int = svmlalb_s32(sum_int, diff_o, diff_o);
        sum_int = svmlalt_s32(sum_int, diff_o, diff_o);

        sum0 = svadd_f32_z(svptrue_b32(), sum0, svcvt_f32_s32_z(pg32, sum_int));
    }

    // Combine the partial sums
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum1);
    sum2 = svadd_f32_z(svptrue_b32(), sum2, sum3);
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum2);

    // Horizontal sum
    return svaddv_f32(svptrue_b32(), sum0);
}
