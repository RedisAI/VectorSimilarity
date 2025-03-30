/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

 #include "VecSim/spaces/space_includes.h"
 #include <arm_neon.h>
 
 static inline void InnerProductStepInt8(int8_t *&pVect1, int8_t *&pVect2, int32x4_t &sum) {
     // Load 16 int8 elements (16 bytes) into NEON registers
     int8x16_t v1 = vld1q_s8(pVect1);
     int8x16_t v2 = vld1q_s8(pVect2);
 
     // Multiply and accumulate low 8 elements (first half)
     int16x8_t prod_low = vmull_s8(vget_low_s8(v1), vget_low_s8(v2));
 
     // Multiply and accumulate high 8 elements (second half)
     int16x8_t prod_high = vmull_s8(vget_high_s8(v1), vget_high_s8(v2));
 
     // Pairwise add adjacent elements to 32-bit accumulators
     sum = vpadalq_s16(sum, prod_low);
     sum = vpadalq_s16(sum, prod_high);
 
     pVect1 += 16;
     pVect2 += 16;
 }
 
 static inline void InnerProductStepInt8_x4(int8_t *&pVect1, int8_t *&pVect2, 
     int32x4_t &sum0, int32x4_t &sum1, 
     int32x4_t &sum2, int32x4_t &sum3) {
     // Load 4 consecutive 16-element vectors (64 bytes) at once
     int8x16x4_t v1 = vld1q_s8_x4(pVect1);
     int8x16x4_t v2 = vld1q_s8_x4(pVect2);
 
     // Process first chunk
     int16x8_t prod_low0 = vmull_s8(vget_low_s8(v1.val[0]), vget_low_s8(v2.val[0]));
     int16x8_t prod_high0 = vmull_s8(vget_high_s8(v1.val[0]), vget_high_s8(v2.val[0]));
     sum0 = vpadalq_s16(sum0, prod_low0);
     sum0 = vpadalq_s16(sum0, prod_high0);
 
     // Process second chunk
     int16x8_t prod_low1 = vmull_s8(vget_low_s8(v1.val[1]), vget_low_s8(v2.val[1]));
     int16x8_t prod_high1 = vmull_s8(vget_high_s8(v1.val[1]), vget_high_s8(v2.val[1]));
     sum1 = vpadalq_s16(sum1, prod_low1);
     sum1 = vpadalq_s16(sum1, prod_high1);
 
     // Process third chunk
     int16x8_t prod_low2 = vmull_s8(vget_low_s8(v1.val[2]), vget_low_s8(v2.val[2]));
     int16x8_t prod_high2 = vmull_s8(vget_high_s8(v1.val[2]), vget_high_s8(v2.val[2]));
     sum2 = vpadalq_s16(sum2, prod_low2);
     sum2 = vpadalq_s16(sum2, prod_high2);
 
     // Process fourth chunk
     int16x8_t prod_low3 = vmull_s8(vget_low_s8(v1.val[3]), vget_low_s8(v2.val[3]));
     int16x8_t prod_high3 = vmull_s8(vget_high_s8(v1.val[3]), vget_high_s8(v2.val[3]));
     sum3 = vpadalq_s16(sum3, prod_low3);
     sum3 = vpadalq_s16(sum3, prod_high3);
 
     pVect1 += 64;
     pVect2 += 64;
 }
 
 template <unsigned char residual> // 0..63
 float INT8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
     int8_t *pVect1 = (int8_t *)pVect1v;
     int8_t *pVect2 = (int8_t *)pVect2v;
 
     // Initialize multiple sum accumulators for better parallelism
     int32x4_t sum0 = vdupq_n_s32(0);
     int32x4_t sum1 = vdupq_n_s32(0);
     int32x4_t sum2 = vdupq_n_s32(0);
     int32x4_t sum3 = vdupq_n_s32(0);
 
     // Process 64 elements at a time in the main loop
     const size_t num_of_chunks = dimension / 64;
 
     for (size_t i = 0; i < num_of_chunks; i++) {
         InnerProductStepInt8(pVect1, pVect2, sum0);
         InnerProductStepInt8(pVect1, pVect2, sum1);
         InnerProductStepInt8(pVect1, pVect2, sum2);
         InnerProductStepInt8(pVect1, pVect2, sum3);
     }
 
     constexpr size_t residual_chunks = residual / 16;
 
     if constexpr (residual_chunks > 0) {
         if constexpr (residual_chunks >= 1) {
             InnerProductStepInt8(pVect1, pVect2, sum0);
         }
         if constexpr (residual_chunks >= 2) {
             InnerProductStepInt8(pVect1, pVect2, sum1);
         }
         if constexpr (residual_chunks >= 3) {
             InnerProductStepInt8(pVect1, pVect2, sum2);
         }
     }
 
     constexpr size_t final_residual = residual % 16;
     if constexpr (final_residual > 0) {
         // Create an index vector: 0, 1, 2, ..., 15
         uint8x16_t indices = vcombine_u8(
             vcreate_u8(0x0706050403020100ULL), 
             vcreate_u8(0x0F0E0D0C0B0A0908ULL)
         );
         
         // Create threshold vector with all elements = final_residual
         uint8x16_t threshold = vdupq_n_u8(final_residual);
         
         // Create mask: indices < final_residual ? 0xFF : 0x00
         uint8x16_t mask = vcltq_u8(indices, threshold);
         
         // Load data directly from input vectors
         int8x16_t v1 = vld1q_s8(pVect1);
         int8x16_t v2 = vld1q_s8(pVect2);
         
         // Apply mask to zero out irrelevant elements
         v1 = vandq_s8(v1, vreinterpretq_s8_u8(mask));
         v2 = vandq_s8(v2, vreinterpretq_s8_u8(mask));
         
         // Split vectors into low and high parts
         int8x8_t v1_low = vget_low_s8(v1);
         int8x8_t v1_high = vget_high_s8(v1);
         int8x8_t v2_low = vget_low_s8(v2);
         int8x8_t v2_high = vget_high_s8(v2);
         
         // Multiply and accumulate
         int16x8_t prod_low = vmull_s8(v1_low, v2_low);
         int16x8_t prod_high = vmull_s8(v1_high, v2_high);
         
         // Accumulate products
         sum3 = vpadalq_s16(sum3, prod_low);
         if constexpr (final_residual > 8) {
             sum3 = vpadalq_s16(sum3, prod_high);
         }
     }
 
     // Combine all four sum registers
     int32x4_t total_sum = vaddq_s32(sum0, sum1);
     total_sum = vaddq_s32(total_sum, sum2);
     total_sum = vaddq_s32(total_sum, sum3);
 
     // Horizontal sum of the 4 elements in the combined sum register
     int32_t result = vaddvq_s32(total_sum);
 
     return static_cast<float>(result);
 }
 
 template <unsigned char residual> // 0..15
 float INT8_InnerProductSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
     return 1.0f - INT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension);
 }
 
 template <unsigned char residual> // 0..63
 float INT8_CosineSIMD_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
     float ip = INT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension);
     float norm_v1 =
         *reinterpret_cast<const float *>(static_cast<const uint8_t *>(pVect1v) + dimension);
     float norm_v2 =
         *reinterpret_cast<const float *>(static_cast<const uint8_t *>(pVect2v) + dimension);
     return 1.0f - ip / (norm_v1 * norm_v2);
 }
 