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

// Helper function to perform L2 squared distance calculation for a chunk of elements
static inline void L2SqrStep(const float *&pVect1, const uint8_t *&pVect2, size_t &offset,
                            svfloat32_t &sum, const svfloat32_t &min_val_vec, 
                            const svfloat32_t &delta_vec) {
    svbool_t pg = svptrue_b32();
    
    // Load float elements from pVect1
    svfloat32_t v1 = svld1_f32(pg, pVect1 + offset);
    
    // Load uint8 elements from pVect2, convert to int32, then to float
    svbool_t pg_b8 = svptrue_b8();
    svuint8_t v2_u8 = svld1_u8(pg_b8, pVect2 + offset);
    
    // Convert uint8 to uint32
    svuint32_t v2_u32 = svzext_u32(svreinterpret_u32_u8(v2_u8));
    
    // Convert uint32 to float32
    svfloat32_t v2_f = svcvt_f32_u32_z(pg, v2_u32);
    
    // Dequantize: (val * delta) + min_val
    svfloat32_t v2_dequant = svadd_f32_z(pg, svmul_f32_z(pg, v2_f, delta_vec), min_val_vec);
    
    // Compute difference
    svfloat32_t diff = svsub_f32_z(pg, v1, v2_dequant);
    
    // Square difference and add to sum
    sum = svmla_f32_z(pg, sum, diff, diff);
    
    // Move to the next set of elements
    offset += svcntw();
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_L2SqrSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    uint8_t *pVect2 = (uint8_t *)pVect2v;
    size_t offset = 0;

    // Get dequantization parameters from the end of quantized vector
    const float min_val = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));
    
    // Create broadcast vectors for SIMD operations
    svbool_t pg = svptrue_b32();
    svfloat32_t min_val_vec = svdup_f32(min_val);
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
            svbool_t pg_partial = svwhilelt_b32(0, remaining);
            
            // Load float elements from pVect1 with predicate
            svfloat32_t v1 = svld1_f32(pg_partial, pVect1);
            
            // Load uint8 elements from pVect2 with predicate, convert to int32, then to float
            svbool_t pg_b8_partial = svwhilelt_b8(0, remaining);
            svuint8_t v2_u8 = svld1_u8(pg_b8_partial, pVect2);
            
            // Convert uint8 to uint32
            svuint32_t v2_u32 = svzext_u32(svreinterpret_u32_u8(v2_u8));
            
            // Convert uint32 to float32
            svfloat32_t v2_f = svcvt_f32_u32_z(pg_partial, v2_u32);
            
            // Dequantize: (val * delta) + min_val
            svfloat32_t v2_dequant = svadd_f32_z(pg_partial, svmul_f32_z(pg_partial, v2_f, delta_vec), min_val_vec);
            
            // Compute difference
            svfloat32_t diff = svsub_f32_z(pg_partial, v1, v2_dequant);
            
            // Square difference and add to sum
            sum0 = svmla_f32_z(pg_partial, sum0, diff, diff);
            
            // Move pointers past the partial chunk
            offset += remaining;
        }
    }
    // Handle remaining steps (0-3)
    if constexpr (additional_steps > 0) {
        L2SqrStep(pVect1, pVect2, offset, sum0, min_val_vec, delta_vec);
    }
    if constexpr (additional_steps > 1) {
        L2SqrStep(pVect1, pVect2, offset, sum1, min_val_vec, delta_vec);
    }
    if constexpr (additional_steps > 2) {
        L2SqrStep(pVect1, pVect2, offset, sum2, min_val_vec, delta_vec);
    }
    

    // Process 4 chunks at a time in the main loop
    auto chunk_size = 4 * sve_word_count;
    size_t number_of_chunks = dimension / chunk_size;
    
    for (size_t i = 0; i < number_of_chunks; i++) {
        L2SqrStep(pVect1, pVect2, offset, sum0, min_val_vec, delta_vec);
        L2SqrStep(pVect1, pVect2, offset, sum1, min_val_vec, delta_vec);
        L2SqrStep(pVect1, pVect2, offset, sum2, min_val_vec, delta_vec);
        L2SqrStep(pVect1, pVect2, offset, sum3, min_val_vec, delta_vec);
    }
    

    // Combine the accumulators
    svfloat32_t sum = svadd_f32_z(pg, sum0, sum1);
    sum = svadd_f32_z(pg, sum, sum2);
    sum = svadd_f32_z(pg, sum, sum3);
    
    // Horizontal sum of all elements in the vector
    float result = svaddv_f32(pg, sum);
    
    return result;
}
