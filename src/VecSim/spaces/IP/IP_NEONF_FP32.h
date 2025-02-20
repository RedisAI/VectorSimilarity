/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

 #include "VecSim/spaces/space_includes.h"

 static inline void InnerProductStep(float*& pVect1, float*& pVect2, float32x4_t& sum_vec0, 
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
 
 // Multiply and accumulate using FMA
 sum_vec0 = vfmaq_f32(sum_vec0, v1_0, v2_0);
 sum_vec1 = vfmaq_f32(sum_vec1, v1_1, v2_1);
 sum_vec2 = vfmaq_f32(sum_vec2, v1_2, v2_2);
 sum_vec3 = vfmaq_f32(sum_vec3, v1_3, v2_3);
 }
 
 template <unsigned char residual> // 0..15
 float FP32_InnerProductSIMD16_NEONF(const void *pVect1v, const void *pVect2v, size_t dimension) {
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
         // Load residual elements (up to 15)
         float32x4_t v1[4] = {vdupq_n_f32(0)};
         float32x4_t v2[4] = {vdupq_n_f32(0)};
         
         // Load available elements
         for (int i = 0; i < residual; i++) {
             int vec_idx = i >> 2;        // Divide by 4 to get vector index
             int elem_idx = i & 3;        // Modulo 4 to get element index
             
             float* ptr1 = (float*)&v1[vec_idx];
             float* ptr2 = (float*)&v2[vec_idx];
             
             ptr1[elem_idx] = pVect1[i];
             ptr2[elem_idx] = pVect2[i];
         }
         
         // Multiply and accumulate residual elements
         sum_vec0 = vfmaq_f32(sum_vec0, v1[0], v2[0]);
         if (residual > 4)  sum_vec1 = vfmaq_f32(sum_vec1, v1[1], v2[1]);
         if (residual > 8)  sum_vec2 = vfmaq_f32(sum_vec2, v1[2], v2[2]);
         if (residual > 12) sum_vec3 = vfmaq_f32(sum_vec3, v1[3], v2[3]);
         
         pVect1 += residual;
         pVect2 += residual;
     }
 
     // Process remaining elements in blocks of 16
     while (pVect1 < pEnd1) {
         InnerProductStep(pVect1, pVect2, sum_vec0, sum_vec1, sum_vec2, sum_vec3);
     }
 
     // Combine all sum vectors
     sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
     sum_vec2 = vaddq_f32(sum_vec2, sum_vec3);
     sum_vec0 = vaddq_f32(sum_vec0, sum_vec2);
 
     // Horizontal sum of final vector
     float32x2_t sum_2 = vadd_f32(vget_low_f32(sum_vec0), vget_high_f32(sum_vec0));
     float32x2_t sum_1 = vpadd_f32(sum_2, sum_2);
     
     return 1.0f - vget_lane_f32(sum_1, 0);
 }
 