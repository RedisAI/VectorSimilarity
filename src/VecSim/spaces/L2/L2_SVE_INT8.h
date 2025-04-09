/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

// Aligned step using svptrue_b8()
inline void L2SquareStep(const int8_t *&pVect1, const int8_t *&pVect2, size_t &offset,
                         svuint32_t &sum, const size_t chunk) {
    svbool_t pg = svptrue_b8();
    // Note: Because all the bits are 1, the extention to 16 and 32 bits does not make a difference
    // Otherwise, pg should be recalculated for 16 and 32 operations

    svint8_t v1_i8 = svld1_s8(pg, pVect1 + offset); // Load int8 vectors from pVect1
    svint8_t v2_i8 = svld1_s8(pg, pVect2 + offset); // Load int8 vectors from pVect2

    svuint8_t abs_diff = svreinterpret_u8_s8(svabd_s8_x(pg, v1_i8, v2_i8));

    sum = svdot_u32(sum, abs_diff, abs_diff);
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

    svuint32_t sum0 = svdup_u32(0);
    svuint32_t sum1 = svdup_u32(0);
    svuint32_t sum2 = svdup_u32(0);
    svuint32_t sum3 = svdup_u32(0);

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

        svbool_t pg = svwhilelt_b8_u64(offset, dimension);

        svint8_t v1_i8 = svld1_s8(pg, pVect1 + offset); // Load int8 vectors from pVect1
        svint8_t v2_i8 = svld1_s8(pg, pVect2 + offset); // Load int8 vectors from pVect2

        svuint8_t abs_diff = svreinterpret_u8_s8(svabd_s8_x(pg, v1_i8, v2_i8));

        // Can sum with taking into account pg because svld1 will set inactive lanes to 0
        sum3 = svdot_u32(sum3, abs_diff, abs_diff);
    }

    sum0 = svadd_u32_x(all, sum0, sum1);
    sum2 = svadd_u32_x(all, sum2, sum3);
    svuint32_t sum_all = svadd_u32_x(all, sum0, sum2);
    return svaddv_u32(svptrue_b32(), sum_all);
}
