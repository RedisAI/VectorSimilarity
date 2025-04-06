/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

// Aligned step using svptrue_b8()
static inline void L2SquareStep(const int8_t *&pVect1, const int8_t *&pVect2, size_t &offset,
                                svint32_t &sum, const size_t chunk) {
    svbool_t pg = svptrue_b8();
    // Note: Because all the bits are 1, the extention to 16 and 32 bits does not make a difference
    // Otherwise, pg should be recalculated for 16 and 32 operations

    svint8_t v1_i8 = svld1_s8(pg, pVect1 + offset); // Load int8 vectors from pVect1
    svint8_t v2_i8 = svld1_s8(pg, pVect2 + offset); // Load int8 vectors from pVect2

    svint16_t v1_16_l = svunpklo_s16(v1_i8);
    svint16_t v1_16_h = svunpkhi_s16(v1_i8);
    svint16_t v2_16_l = svunpklo_s16(v2_i8);
    svint16_t v2_16_h = svunpkhi_s16(v2_i8);

    // Calculate difference and square for low part
    svint16_t diff_l = svsub_s16_x(pg, v1_16_l, v2_16_l);

    // Unpacking to 32 bits is necessary for the multiplication
    // The multiplication of two 16 bits numbers can overflow
    // Maximal value of int8 - int8 is 255 (127 - (-128))
    // 255^2 = 65025 while int16 can hold upto 32767

    svint32_t diff32_l_l = svunpklo_s32(diff_l);
    svint32_t diff32_l_h = svunpkhi_s32(diff_l);

    // Result register is the same as the accumulator for better performance
    svint32_t sq_l = svmul_s32_x(pg, diff32_l_l, diff32_l_l);
    sq_l = svmla_s32_x(pg, sq_l, diff32_l_h, diff32_l_h);

    svint16_t diff_h = svsub_s16_x(pg, v1_16_h, v2_16_h);

    svint32_t diff32_h_l = svunpklo_s32(diff_h);
    svint32_t diff32_h_h = svunpkhi_s32(diff_h);

    // Result register is the same as the accumulator for better performance
    svint32_t sq_h = svmul_s32_x(pg, diff32_h_l, diff32_h_l);
    sq_h = svmla_s32_x(pg, sq_h, diff32_h_h, diff32_h_h);

    // Accumulate
    sum = svadd_s32_x(pg, sum, sq_l);
    sum = svadd_s32_x(pg, sum, sq_h);

    offset += chunk; // Move to the next set of int8 elements
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_L2SqrSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const int8_t *pVect1 = reinterpret_cast<const int8_t *>(pVect1v);
    const int8_t *pVect2 = reinterpret_cast<const int8_t *>(pVect2v);

    // number of int8 per SVE register
    const size_t vl = svcntb();
    const size_t chunk_size = 4 * vl;
    svbool_t all = svptrue_b8();

    // Each L2SquareStep adds maximum (2*2^8)^2 = 2^18
    // Therefor, on a single accumulator, we can perform 2^13 steps before overflowing
    // That scenario will happen only is the dimension of the vector is larger than 16*4*2^13 = 2^19
    // (16 int8 in 1 SVE register) * (4 accumulators) * (2^13 steps)
    // We can safely assume that the dimension is smaller than that
    // So using int32_t is safe

    svint32_t sum0 = svdup_s32(0);
    svint32_t sum1 = svdup_s32(0);
    svint32_t sum2 = svdup_s32(0);
    svint32_t sum3 = svdup_s32(0);

    size_t offset = 0;
    size_t num_main_blocks = dimension / chunk_size;

    for (size_t i = 0; i < num_main_blocks; ++i) {
        L2SquareStep(pVect1, pVect2, offset, sum0, vl);
        L2SquareStep(pVect1, pVect2, offset, sum1, vl);
        L2SquareStep(pVect1, pVect2, offset, sum2, vl);
        L2SquareStep(pVect1, pVect2, offset, sum3, vl);
    }

    if constexpr (additional_steps > 0) {
        if constexpr (additional_steps >= 1) {
            L2SquareStep(pVect1, pVect2, offset, sum0, vl);
        }
        if constexpr (additional_steps >= 2) {
            L2SquareStep(pVect1, pVect2, offset, sum1, vl);
        }
        if constexpr (additional_steps >= 3) {
            L2SquareStep(pVect1, pVect2, offset, sum2, vl);
        }
    }

    if constexpr (partial_chunk) {

        /* TODO: Test using svptrue_b8() instead of svwhilelt_b8/b16/b32) in vector arithmetics
        Because Inactive lanes are set to 0 in load */

        svbool_t pg = svwhilelt_b8(offset, dimension);
        svbool_t pg16 = svwhilelt_b16(offset, dimension);
        svbool_t pg32 = svwhilelt_b32(offset, dimension);

        svint8_t v1_i8 = svld1_s8(pg, pVect1 + offset); // Load int8 vectors from pVect1
        svint8_t v2_i8 = svld1_s8(pg, pVect2 + offset); // Load int8 vectors from pVect2

        svint16_t v1_16_l = svunpklo_s16(v1_i8);
        svint16_t v1_16_h = svunpkhi_s16(v1_i8);
        svint16_t v2_16_l = svunpklo_s16(v2_i8);
        svint16_t v2_16_h = svunpkhi_s16(v2_i8);

        // Calculate difference and square for low part
        svint16_t diff_l = svsub_s16_x(pg16, v1_16_l, v2_16_l);

        svint32_t diff32_l_l = svunpklo_s32(diff_l);
        svint32_t diff32_l_h = svunpkhi_s32(diff_l);

        // Result register is the same as the accumulator for better performance
        svint32_t sq_l = svmul_s32_x(pg32, diff32_l_l, diff32_l_l);
        sq_l = svmla_s32_x(pg32, sq_l, diff32_l_h, diff32_l_h);

        svint16_t diff_h = svsub_s16_x(pg16, v1_16_h, v2_16_h);

        svint32_t diff32_h_l = svunpklo_s32(diff_h);
        svint32_t diff32_h_h = svunpkhi_s32(diff_h);

        // Result register is the same as the accumulator for better performance
        svint32_t sq_h = svmul_s32_x(pg32, diff32_h_l, diff32_h_l);
        sq_h = svmla_s32_x(pg32, sq_h, diff32_h_h, diff32_h_h);

        // Accumulate
        sum3 = svadd_s32_x(pg32, sum3, sq_l);
        sum3 = svadd_s32_x(pg32, sum3, sq_h);
    }

    sum0 = svadd_s32_x(all, sum0, sum1);
    sum2 = svadd_s32_x(all, sum2, sum3);
    svint32_t sum_all = svadd_s32_x(all, sum0, sum2);
    return svaddv_s32(svptrue_b32(), sum_all);
}
