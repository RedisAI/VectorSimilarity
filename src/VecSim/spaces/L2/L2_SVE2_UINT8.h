/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

 #include "VecSim/spaces/space_includes.h"
 #include <arm_sve.h>

 // Aligned step using svptrue_b8()
 static inline void L2SquareStep(const uint8_t *&pVect1, const uint8_t *&pVect2, size_t &offset,
     svint32_t &sum) {
     svbool_t pg = svptrue_b8();
     // Note: Because all the bits are 1, the extention to 16 and 32 bits does not make a difference
     // Otherwise, pg should be recalculated for 16 and 32 operations

     svuint8_t v1_i8 = svld1_u8(pg, pVect1 + offset); // Load int8 vectors from pVect1
     svuint8_t v2_i8 = svld1_u8(pg, pVect2 + offset); // Load int8 vectors from pVect2

     // Subtract v2 from v1 and widen the results to int16 for the even indexes
     svint16_t diff_e = svreinterpret_s16(svsublb_u16(v1_i8, v2_i8));

     // Subtract v2 from v1 and widen the results to int16 for the odd indexes
     svint16_t diff_o = svreinterpret_s16(svsublt_u16(v1_i8, v2_i8));

     sum = svmlalb_s32(sum, diff_e, diff_e);
     sum = svmlalt_s32(sum, diff_e, diff_e);

     sum = svmlalb_s32(sum, diff_o, diff_o);
     sum = svmlalt_s32(sum, diff_o, diff_o);

     offset += svcntb(); // Move to the next set of int8 elements
 }

 template <bool partial_chunk, unsigned char additional_steps>
 float UINT8_L2SqrSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
     const uint8_t *pVect1 = reinterpret_cast<const uint8_t *>(pVect1v);
     const uint8_t *pVect2 = reinterpret_cast<const uint8_t *>(pVect2v);

     // number of int8 per SVE2 register
     const size_t vl = svcntb();
     const size_t chunk_size = 4 * vl;
     svbool_t all = svptrue_b8();

     // Each L2SquareStep adds maximum (2*2^8)^2 = 2^18
     // Therefor, on a single accumulator, we can perform 2^13 steps before overflowing
     // That scenario will happen only is the dimension of the vector is larger than 16*4*2^13 = 2^19
     // (at least 16 int8 in 1 SVE2 register) * (4 accumulators) * (2^13 steps)
     // We can safely assume that the dimension is smaller than that
     // So using int32_t is safe

     svint32_t sum0 = svdup_s32(0);
     svint32_t sum1 = svdup_s32(0);
     svint32_t sum2 = svdup_s32(0);
     svint32_t sum3 = svdup_s32(0);

     size_t offset = 0;
     size_t num_main_blocks = dimension / chunk_size;

     for (size_t i = 0; i < num_main_blocks; ++i) {
         L2SquareStep(pVect1, pVect2, offset, sum0);
         L2SquareStep(pVect1, pVect2, offset, sum1);
         L2SquareStep(pVect1, pVect2, offset, sum2);
         L2SquareStep(pVect1, pVect2, offset, sum3);
     }

     if constexpr (additional_steps > 0) {
         if constexpr (additional_steps >= 1) {
             L2SquareStep(pVect1, pVect2, offset, sum0);
         }
         if constexpr (additional_steps >= 2) {
             L2SquareStep(pVect1, pVect2, offset, sum1);
         }
         if constexpr (additional_steps >= 3) {
             L2SquareStep(pVect1, pVect2, offset, sum2);
         }
     }

     if constexpr (partial_chunk) {

         svbool_t pg = svwhilelt_b8(offset, dimension);

         svuint8_t v1_i8 = svld1_u8(pg, pVect1 + offset); // Load int8 vectors from pVect1
         svuint8_t v2_i8 = svld1_u8(pg, pVect2 + offset); // Load int8 vectors from pVect2

         // Subtract v2 from v1 and widen the results to int16 for the even indexes
         svint16_t diff_e = svreinterpret_s16(svsublb_u16(v1_i8, v2_i8));

         // Subtract v2 from v1 and widen the results to int16 for the odd indexes
         svint16_t diff_o = svreinterpret_s16(svsublt_u16(v1_i8, v2_i8));

         // Can sum without lanes because diffs are zero where it's inactive

         sum2 = svmlalb_s32(sum2, diff_e, diff_e);
         sum2 = svmlalt_s32(sum2, diff_e, diff_e);

         sum2 = svmlalb_s32(sum2, diff_o, diff_o);
         sum2 = svmlalt_s32(sum2, diff_o, diff_o);

     }

     sum0 = svadd_s32_x(all, sum0, sum1);
     sum2 = svadd_s32_x(all, sum2, sum3);
     svint32_t sum_all = svadd_s32_x(all, sum0, sum2);
     return svaddv_s32(all, sum_all);
 }
