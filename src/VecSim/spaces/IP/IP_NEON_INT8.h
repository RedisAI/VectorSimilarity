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

template <unsigned char residual> // 0..63
float INT8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;

    // Initialize sum accumulators to zero (4 lanes of 32-bit integers)
    int32x4_t sum = vdupq_n_s32(0);

    // Process 16 elements at a time
    const size_t num_of_chunks = dimension / 64;

    for (size_t i = 0; i < num_of_chunks; i++) {
        InnerProductStepInt8(pVect1, pVect2, sum);
        InnerProductStepInt8(pVect1, pVect2, sum);
        InnerProductStepInt8(pVect1, pVect2, sum);
        InnerProductStepInt8(pVect1, pVect2, sum);
    }

    constexpr size_t remaining_chunks = residual / 16;
    if constexpr (remaining_chunks > 0)
    {
        // Process remaining full chunks of 16 elements
        for (size_t i = 0; i < remaining_chunks; i++) {
            L2SquareStep(pVect1, pVect2, sum);
        }
    }


    // Handle remaining elements (0-15)
    constexpr size_t final_residual = residual % 16;
    if constexpr (final_residual > 0) {
        // For residual elements, we need to handle them carefully
        int8_t temp1[16] = {0};
        int8_t temp2[16] = {0};

        // Copy residual elements to temporary buffers
        for (size_t i = 0; i < final_residual; i++) {
            temp1[i] = pVect1[i];
            temp2[i] = pVect2[i];
        }

        int8_t *pTemp1 = temp1;
        int8_t *pTemp2 = temp2;

        // Process the residual elements
        InnerProductStepInt8(pTemp1, pTemp2, sum);
    }

    int32_t result = vaddvq_s32(sum);
    return static_cast<float>(result);
}

template <unsigned char residual> // 0..15
float INT8_InnerProductSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - INT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension);
}

template <unsigned char residual> // 0..63
float UINT8_CosineSIMD64_AVX512F_BW_VL_VNNI(const void *pVect1v, const void *pVect2v,
                                            size_t dimension) {
    float ip = UINT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension);
    float norm_v1 =
        *reinterpret_cast<const float *>(static_cast<const uint8_t *>(pVect1v) + dimension);
    float norm_v2 =
        *reinterpret_cast<const float *>(static_cast<const uint8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}