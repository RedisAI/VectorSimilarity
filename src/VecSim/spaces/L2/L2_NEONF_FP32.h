/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

 #include "VecSim/spaces/space_includes.h"

 static inline void L2SqrStep(float*& pVect1, float*& pVect2, float32x4_t& sum_vec0, 
     float32x4_t& sum_vec1, float32x4_t& sum_vec2, 
     float32x4_t& sum_vec3) {
     // Load 16 floats (4x4) from each vector
     float32x4_t v1_0 = vld1q_f32(pVect1);
     float32x4_t v1_1 = vld1q_f32(pVect1 + 4);
     float32x4_t v1_2 = vld1q_f32(pVect1 + 8);
     float32x4_t v1_3 = vld1q_f32(pVect1 + 12);
     pVect1 += 16;
 
     float32x4_t v2_0 = vld1q_f32(pVect2);
     float32x4_t v2_1 = vld1q_f32(pVect2 + 4);
     float32x4_t v2_2 = vld1q_f32(pVect2 + 8);
     float32x4_t v2_3 = vld1q_f32(pVect2 + 12);
     pVect2 += 16;
 
     // Calculate differences
     float32x4_t diff0 = vsubq_f32(v1_0, v2_0);
     float32x4_t diff1 = vsubq_f32(v1_1, v2_1);
     float32x4_t diff2 = vsubq_f32(v1_2, v2_2);
     float32x4_t diff3 = vsubq_f32(v1_3, v2_3);
 
     // Square and accumulate using FMA
     sum_vec0 = vfmaq_f32(sum_vec0, diff0, diff0);
     sum_vec1 = vfmaq_f32(sum_vec1, diff1, diff1);
     sum_vec2 = vfmaq_f32(sum_vec2, diff2, diff2);
     sum_vec3 = vfmaq_f32(sum_vec3, diff3, diff3);
     }
 
     // Helper function for horizontal sum
     static inline float horizontal_sum(float32x4_t v) {
     float32x2_t sum = vadd_f32(vget_high_f32(v), vget_low_f32(v));
     sum = vpadd_f32(sum, sum);
     return vget_lane_f32(sum, 0);
 }
 
 
 template <unsigned char residual> // 0..15
 float FP32_L2SqrSIMD16_NEONF(const void *pVect1v, const void *pVect2v, size_t dimension) {
     float* pVect1 = (float*)pVect1v;
     float* pVect2 = (float*)pVect2v;
     const float* pEnd1 = pVect1 + dimension;
 
     // Initialize four sum vectors to handle 16 elements at a time
     float32x4_t sum_vec0 = vdupq_n_f32(0);
     float32x4_t sum_vec1 = vdupq_n_f32(0);
     float32x4_t sum_vec2 = vdupq_n_f32(0);
     float32x4_t sum_vec3 = vdupq_n_f32(0);
 
     // Handle residual elements first
     if constexpr (residual) {
         // Create partial masks for loading
         alignas(16) float mask_arr[16] = {0};
         for (unsigned char i = 0; i < residual; i++) {
             mask_arr[i] = 1.0f;
         }
         
         // Load mask vectors
         float32x4_t mask0 = vld1q_f32(mask_arr);
         float32x4_t mask1 = vld1q_f32(mask_arr + 4);
         float32x4_t mask2 = vld1q_f32(mask_arr + 8);
         float32x4_t mask3 = vld1q_f32(mask_arr + 12);
 
         // Load and mask vectors
         float32x4_t v1_0 = vmulq_f32(vld1q_f32(pVect1), mask0);
         float32x4_t v2_0 = vmulq_f32(vld1q_f32(pVect2), mask0);
         
         float32x4_t v1_1 = residual > 4 ? vmulq_f32(vld1q_f32(pVect1 + 4), mask1) : vdupq_n_f32(0);
         float32x4_t v2_1 = residual > 4 ? vmulq_f32(vld1q_f32(pVect2 + 4), mask1) : vdupq_n_f32(0);
         
         float32x4_t v1_2 = residual > 8 ? vmulq_f32(vld1q_f32(pVect1 + 8), mask2) : vdupq_n_f32(0);
         float32x4_t v2_2 = residual > 8 ? vmulq_f32(vld1q_f32(pVect2 + 8), mask2) : vdupq_n_f32(0);
         
         float32x4_t v1_3 = residual > 12 ? vmulq_f32(vld1q_f32(pVect1 + 12), mask3) : vdupq_n_f32(0);
         float32x4_t v2_3 = residual > 12 ? vmulq_f32(vld1q_f32(pVect2 + 12), mask3) : vdupq_n_f32(0);
 
         // Calculate differences
         float32x4_t diff0 = vsubq_f32(v1_0, v2_0);
         float32x4_t diff1 = vsubq_f32(v1_1, v2_1);
         float32x4_t diff2 = vsubq_f32(v1_2, v2_2);
         float32x4_t diff3 = vsubq_f32(v1_3, v2_3);
 
         // Square and accumulate
         sum_vec0 = vmulq_f32(diff0, diff0);
         sum_vec1 = vmulq_f32(diff1, diff1);
         sum_vec2 = vmulq_f32(diff2, diff2);
         sum_vec3 = vmulq_f32(diff3, diff3);
 
         pVect1 += residual;
         pVect2 += residual;
     }
 
     // Process remaining elements in blocks of 16
     while (pVect1 < pEnd1) {
         L2SqrStep(pVect1, pVect2, sum_vec0, sum_vec1, sum_vec2, sum_vec3);
     }
 
     // Combine all sum vectors
     sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
     sum_vec2 = vaddq_f32(sum_vec2, sum_vec3);
     sum_vec0 = vaddq_f32(sum_vec0, sum_vec2);
 
     // Use helper function for horizontal sum
     return horizontal_sum(sum_vec0);
 }
 