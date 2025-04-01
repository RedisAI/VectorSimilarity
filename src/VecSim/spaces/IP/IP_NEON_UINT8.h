/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

__attribute__((always_inline)) static inline void InnerProductStep(uint8x16_t &v1, uint8x16_t &v2,
                                                                   uint32x4_t &sum) {
    // Multiply and accumulate low 8 elements (first half)
    uint16x8_t prod_low = vmull_u8(vget_low_u8(v1), vget_low_u8(v2));

    // Multiply and accumulate high 8 elements (second half)
    uint16x8_t prod_high = vmull_u8(vget_high_u8(v1), vget_high_u8(v2));

    // Pairwise add adjacent elements to 32-bit accumulators
    sum = vpadalq_u16(sum, prod_low);
    sum = vpadalq_u16(sum, prod_high);
}

__attribute__((always_inline)) static inline void
InnerProductStep16(uint8_t *&pVect1, uint8_t *&pVect2, uint32x4_t &sum) {
    // Load 16 uint8 elements (16 bytes) into NEON registers
    uint8x16_t v1 = vld1q_u8(pVect1);
    uint8x16_t v2 = vld1q_u8(pVect2);
    InnerProductStep(v1, v2, sum);

    pVect1 += 16;
    pVect2 += 16;
}

template <unsigned char residual> // 0..63
float UINT8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
    uint8_t *pVect1 = (uint8_t *)pVect1v;
    uint8_t *pVect2 = (uint8_t *)pVect2v;

    // Initialize multiple sum accumulators for better parallelism
    uint32x4_t sum0 = vdupq_n_u32(0);
    uint32x4_t sum1 = vdupq_n_u32(0);

    // Process 64 elements at a time in the main loop
    const size_t num_of_chunks = dimension / 64;

    for (size_t i = 0; i < num_of_chunks; i++) {
        InnerProductStep16(pVect1, pVect2, sum0);
        InnerProductStep16(pVect1, pVect2, sum1);
        InnerProductStep16(pVect1, pVect2, sum0);
        InnerProductStep16(pVect1, pVect2, sum1);
    }

    constexpr size_t residual_chunks = residual / 16;

    if constexpr (residual_chunks > 0) {
        if constexpr (residual_chunks >= 1) {
            InnerProductStep16(pVect1, pVect2, sum0);
        }
        if constexpr (residual_chunks >= 2) {
            InnerProductStep16(pVect1, pVect2, sum1);
        }
        if constexpr (residual_chunks >= 3) {
            InnerProductStep16(pVect1, pVect2, sum0);
        }
    }

    constexpr size_t final_residual = residual % 16;
    if constexpr (final_residual > 0) {
        // Create an index vector: 0, 1, 2, ..., 15
        uint8x16_t indices =
            vcombine_u8(vcreate_u8(0x0706050403020100ULL), vcreate_u8(0x0F0E0D0C0B0A0908ULL));

        // Create threshold vector with all elements = final_residual
        uint8x16_t threshold = vdupq_n_u8(final_residual);

        // Create mask: indices < final_residual ? 0xFF : 0x00
        uint8x16_t mask = vcltq_u8(indices, threshold);

        // Load data directly from input vectors
        uint8x16_t v1 = vld1q_u8(pVect1);
        uint8x16_t v2 = vld1q_u8(pVect2);

        // Apply mask to zero out irrelevant elements
        v1 = vandq_u8(v1, mask);
        v2 = vandq_u8(v2, mask);
        InnerProductStep(v1, v2, sum1);
    }

    uint32x4_t total_sum = vaddq_u32(sum0, sum1);

    // Horizontal sum of the 4 elements in the combined sum register
    int32_t result = vaddvq_u32(total_sum);

    return static_cast<float>(result);
}

template <unsigned char residual> // 0..15
float UINT8_InnerProductSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - UINT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension);
}

template <unsigned char residual> // 0..63
float UINT8_CosineSIMD_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float ip = UINT8_InnerProductImp<residual>(pVect1v, pVect2v, dimension);
    float norm_v1 =
        *reinterpret_cast<const float *>(static_cast<const uint8_t *>(pVect1v) + dimension);
    float norm_v2 =
        *reinterpret_cast<const float *>(static_cast<const uint8_t *>(pVect2v) + dimension);
    return 1.0f - ip / (norm_v1 * norm_v2);
}
