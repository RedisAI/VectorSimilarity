/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

 #include "VecSim/spaces/space_includes.h"
 #include <arm_sve.h>

 // Aligned step using svptrue_b8()
 static inline void L2SquareStep_SVE1(const int8_t *&pVect1, const int8_t *&pVect2, svfloat32_t &sum) {
     svbool_t pg = svptrue_b8();

     svint8_t v1_i8 = svld1_s8(pg, pVect1);
     svint8_t v2_i8 = svld1_s8(pg, pVect2);

     // Manually widen to int16 (since svunpklo_s8 is SVE2, using reinterpretation)
     svint16_t v1_16 = svreinterpret_s16(svadd_s8_x(pg, v1_i8, svdup_s8(0)));
     svint16_t v2_16 = svreinterpret_s16(svadd_s8_x(pg, v2_i8, svdup_s8(0)));

     // Calculate difference and square for low part
     svint16_t diff = svsub_s16_x(pg, v1_16, v2_16);
     svint32_t diff32 = svreinterpret_s32(diff);
     svint32_t sq = svmul_s32_z(pg, diff32, diff32);

     // Convert to float and accumulate
     svfloat32_t sqf = svcvt_f32_s32_z(pg, sq);
     sum = svadd_f32_z(pg, sum, sqf);

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
         L2SquareStep_SVE1(pVect1, pVect2, sum0);
         L2SquareStep_SVE1(pVect1, pVect2, sum1);
         L2SquareStep_SVE1(pVect1, pVect2, sum2);
         L2SquareStep_SVE1(pVect1, pVect2, sum3);
         offset += chunk_size;
     }

     if constexpr (additional_steps > 0) {
         for (unsigned char c = 0; c < additional_steps; ++c) {
             L2SquareStep_SVE1(pVect1, pVect2, sum0);
             offset += vl;
         }
     }

     if constexpr (partial_chunk) {
         svbool_t pg = svwhilelt_b8(offset, dimension);

         svint8_t v1_i8 = svld1_s8(pg, reinterpret_cast<const int8_t *>(pVect1v) + offset);
         svint8_t v2_i8 = svld1_s8(pg, reinterpret_cast<const int8_t *>(pVect2v) + offset);

         // Manually widen to int16
         svint16_t v1_16 = svreinterpret_s16(svadd_s8_x(pg, v1_i8, svdup_s8(0)));
         svint16_t v2_16 = svreinterpret_s16(svadd_s8_x(pg, v2_i8, svdup_s8(0)));

         // Calculate difference and square for low part
         svint16_t diff = svsub_s16_m(pg, v1_16, v2_16);
         svint32_t diff32 = svreinterpret_s32(diff);
         svint32_t sq = svmul_s32_z(pg, diff32, diff32);

         // Convert to float and accumulate
         svfloat32_t sqf = svcvt_f32_s32_z(pg, sq);
         sum0 = svadd_f32_z(pg, sum0, sqf);
     }

     // Combine the partial sums
     sum0 = svadd_f32_z(svptrue_b32(), sum0, sum1);
     sum2 = svadd_f32_z(svptrue_b32(), sum2, sum3);
     sum0 = svadd_f32_z(svptrue_b32(), sum0, sum2);

     // Horizontal sum
     return svaddv_f32(svptrue_b32(), sum0);
 }
