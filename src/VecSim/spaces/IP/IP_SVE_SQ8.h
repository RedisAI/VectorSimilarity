/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>
#include <iostream>
#include <string.h>

static inline void InnerProductStep(float *&pVect1, uint8_t *&pVect2, size_t &offset,
                                    svfloat32_t &sum, const svfloat32_t &min_val_vec, 
                                    const svfloat32_t &delta_vec) {
    svbool_t pg = svptrue_b32();
    
    // Load float elements from pVect1
    svfloat32_t v1 = svld1_f32(pg, pVect1 + offset);
    
    // Convert uint8 to uint32
    svuint32_t v2_u32 = svld1ub_u32(pg, pVect2 + offset); // LD1UB: loa
    
    // Convert uint32 to float32
    svfloat32_t v2_f = svcvt_f32_u32_z(pg, v2_u32);
    
    // Dequantize: (val * delta) + min_val
    svfloat32_t v2_dequant = svadd_f32_z(pg, svmul_f32_z(pg, v2_f, delta_vec), min_val_vec);
    
    // Compute dot product and add to sum
    sum = svmla_f32_z(pg, sum, v1, v2_dequant);
    
    // Move to the next set of elements
    offset += svcntw();
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_InnerProductSIMD_SVE_IMP(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    uint8_t *pVect2 = (uint8_t *)pVect2v;
    size_t offset = 0;

    // Get dequantization parameters from the end of quantized vector
    float min = *(float *)(pVect2 + dimension);
    float delta = *(float *)(pVect2 + dimension + sizeof(float));
    
    // Create broadcast vectors for SIMD operations
    svbool_t pg = svptrue_b32();
    svfloat32_t min_val_vec = svdup_f32(min);
    svfloat32_t delta_vec = svdup_f32(delta);

    // Get the number of 32-bit elements per vector at runtime
    uint64_t sve_word_count = svcntw();
    
    // Multiple accumulators to increase instruction-level parallelism
    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    // Handle partial chunk if needed
    if constexpr (partial_chunk) {
        size_t remaining = dimension % sve_word_count;
        if (remaining > 0) {
            // Create predicate for the remaining elements
            svbool_t pg_partial = svwhilelt_b32(static_cast<uint32_t>(0), static_cast<uint32_t>(remaining));

            // Load float elements from pVect1 with predicate
            svfloat32_t v1 = svld1_f32(pg_partial, pVect1);
            

            // load 8-bit bytes from pVect2+offset and zero-extend each into a 32-bit lane
            svuint32_t v2_u32 = svld1ub_u32(pg_partial, pVect2 + offset);  // LD1UB: load 8-bit, zero-extend to 32-bit :contentReference[oaicite:0]{index=0}

            
            // Convert uint32 to float32
            svfloat32_t v2_f = svcvt_f32_u32_z(pg_partial, v2_u32);
            
            // Dequantize: (val * delta) + min_val
            svfloat32_t v2_dequant = svadd_f32_z(pg_partial, svmul_f32_z(pg_partial, v2_f, delta_vec), min_val_vec);
            
            // Compute dot product and add to sum
            sum0 = svmla_f32_z(pg_partial, sum0, v1, v2_dequant);
            
            // Move pointers past the partial chunk
            offset += remaining;
        }
    }

    // Process 4 chunks at a time in the main loop
    auto chunk_size = 4 * sve_word_count;
    const size_t number_of_chunks = (dimension - (partial_chunk ? dimension % sve_word_count : 0)) / chunk_size;
    
    for (size_t i = 0; i < number_of_chunks; i++) {
        InnerProductStep(pVect1, pVect2, offset, sum0, min_val_vec, delta_vec);
        InnerProductStep(pVect1, pVect2, offset, sum1, min_val_vec, delta_vec);
        InnerProductStep(pVect1, pVect2, offset, sum2, min_val_vec, delta_vec);
        InnerProductStep(pVect1, pVect2, offset, sum3, min_val_vec, delta_vec);
    }
    
    // Handle remaining steps (0-3)
    if constexpr (additional_steps > 0) {
        InnerProductStep(pVect1, pVect2, offset, sum0, min_val_vec, delta_vec);
    }
    if constexpr (additional_steps > 1) {
        InnerProductStep(pVect1, pVect2, offset, sum1, min_val_vec, delta_vec);
    }
    if constexpr (additional_steps > 2) {
        InnerProductStep(pVect1, pVect2, offset, sum2, min_val_vec, delta_vec);
    }
    
    // Combine the accumulators
    svfloat32_t sum = svadd_f32_z(pg, sum0, sum1);
    sum = svadd_f32_z(pg, sum, sum2);
    sum = svadd_f32_z(pg, sum, sum3);
    
    // Horizontal sum of all elements in the vector
    float result = svaddv_f32(pg, sum);
    
    return result;
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - SQ8_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_CosineSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);
    
    // Get quantization parameters
    const float inv_norm = *reinterpret_cast<const float *>(pVect2 + dimension + 2 * sizeof(float));
    
    // Compute inner product with dequantization using the common function
    const float res = SQ8_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);
    
    // For cosine, we need to account for the vector norms
    // The inv_norm parameter is stored after min_val and delta in the quantized vector
    return 1.0f - res * inv_norm;
}
