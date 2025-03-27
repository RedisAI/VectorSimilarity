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

template <unsigned char residual> // 0..15
float INT8_InnerProductSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;

    // Initialize sum accumulators to zero (4 lanes of 32-bit integers)
    int32x4_t sum = vdupq_n_s32(0);
    
    // Process 16 elements at a time
    const size_t num_of_chunks = dimension / 16;
    
    for (size_t i = 0; i < num_of_chunks; i++) {
        InnerProductStepInt8(pVect1, pVect2, sum);
    }

    // Handle remaining elements (0-15)
    if constexpr (residual > 0) {
        // For residual elements, we need to handle them carefully
        int8_t temp1[16] = {0};
        int8_t temp2[16] = {0};
        
        // Copy residual elements to temporary buffers
        for (size_t i = 0; i < residual; i++) {
            temp1[i] = pVect1[i];
            temp2[i] = pVect2[i];
        }
        
        int8_t *pTemp1 = temp1;
        int8_t *pTemp2 = temp2;
        
        // Process the residual elements
        InnerProductStepInt8(pTemp1, pTemp2, sum);
    }

    // Horizontal sum of the 4 elements in the sum register to get final result
    int32_t result = vaddvq_s32(sum);
    
    // Normalize and invert the result similar to the floating-point version
    // The scaling factor might need adjustment based on your specific use case
    float normalized_result = static_cast<float>(result) / (127.0f * 127.0f * dimension);
    
    return 1.0f - normalized_result;
}