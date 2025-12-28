/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

/**
 * SQ8-to-SQ8 distance functions for SVE.
 * These functions compute distance between two SQ8 (scalar quantized 8-bit) vectors,
 * where BOTH vectors are uint8 quantized and dequantization is applied to both
 * during computation.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [inv_norm (float)]
 * Dequantization formula: dequantized_value = quantized_value * delta + min_val
 */

// Helper function to perform inner product step for one chunk with dual dequantization
static inline void SQ8_SQ8_InnerProductStep_SVE(const uint8_t *&pVec1, const uint8_t *&pVec2,
                                                size_t &offset, svfloat32_t &sum,
                                                const svfloat32_t &min_val_vec1,
                                                const svfloat32_t &delta_vec1,
                                                const svfloat32_t &min_val_vec2,
                                                const svfloat32_t &delta_vec2, const size_t chunk) {
    svbool_t pg = svptrue_b32();

    // Load uint8 elements from pVec1 and convert to float
    svuint32_t v1_u32 = svld1ub_u32(pg, pVec1 + offset);
    svfloat32_t v1_f = svcvt_f32_u32_x(pg, v1_u32);

    // Dequantize v1: (val * delta1) + min_val1
    svfloat32_t v1_dequant = svmla_f32_x(pg, min_val_vec1, v1_f, delta_vec1);

    // Load uint8 elements from pVec2 and convert to float
    svuint32_t v2_u32 = svld1ub_u32(pg, pVec2 + offset);
    svfloat32_t v2_f = svcvt_f32_u32_x(pg, v2_u32);

    // Dequantize v2: (val * delta2) + min_val2
    svfloat32_t v2_dequant = svmla_f32_x(pg, min_val_vec2, v2_f, delta_vec2);

    // Compute dot product and add to sum: sum += v1_dequant * v2_dequant
    sum = svmla_f32_x(pg, sum, v1_dequant, v2_dequant);

    // Move to the next set of elements
    offset += chunk;
}

// Common implementation for inner product between two SQ8 vectors
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_SQ8_InnerProductSIMD_SVE_IMP(const void *pVec1v, const void *pVec2v, size_t dimension) {
    const uint8_t *pVec1 = static_cast<const uint8_t *>(pVec1v);
    const uint8_t *pVec2 = static_cast<const uint8_t *>(pVec2v);
    size_t offset = 0;

    // Get dequantization parameters from the end of pVec1
    const float min1 = *reinterpret_cast<const float *>(pVec1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVec1 + dimension + sizeof(float));

    // Get dequantization parameters from the end of pVec2
    const float min2 = *reinterpret_cast<const float *>(pVec2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVec2 + dimension + sizeof(float));

    // Create broadcast vectors for SIMD operations
    svbool_t pg = svptrue_b32();
    svfloat32_t min_val_vec1 = svdup_f32(min1);
    svfloat32_t delta_vec1 = svdup_f32(delta1);
    svfloat32_t min_val_vec2 = svdup_f32(min2);
    svfloat32_t delta_vec2 = svdup_f32(delta2);

    // Get the number of 32-bit elements per vector at runtime
    uint64_t chunk = svcntw();

    // Multiple accumulators to increase instruction-level parallelism
    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    // Handle partial chunk if needed
    if constexpr (partial_chunk) {
        size_t remaining = dimension % chunk;
        if (remaining > 0) {
            // Create predicate for the remaining elements
            svbool_t pg_partial =
                svwhilelt_b32(static_cast<uint32_t>(0), static_cast<uint32_t>(remaining));

            // Load and convert v1 elements
            svuint32_t v1_u32 = svld1ub_u32(pg_partial, pVec1 + offset);
            svfloat32_t v1_f = svcvt_f32_u32_z(pg_partial, v1_u32);

            // Dequantize v1
            svfloat32_t v1_dequant = svmla_f32_z(pg_partial, min_val_vec1, v1_f, delta_vec1);

            // Load and convert v2 elements
            svuint32_t v2_u32 = svld1ub_u32(pg_partial, pVec2 + offset);
            svfloat32_t v2_f = svcvt_f32_u32_z(pg_partial, v2_u32);

            // Dequantize v2
            svfloat32_t v2_dequant = svmla_f32_z(pg_partial, min_val_vec2, v2_f, delta_vec2);

            // Compute dot product and add to sum
            sum0 = svmla_f32_z(pg_partial, sum0, v1_dequant, v2_dequant);

            // Move past the partial chunk
            offset += remaining;
        }
    }

    // Process 4 chunks at a time in the main loop
    auto chunk_size = 4 * chunk;
    const size_t number_of_chunks =
        (dimension - (partial_chunk ? dimension % chunk : 0)) / chunk_size;

    for (size_t i = 0; i < number_of_chunks; i++) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, sum0, min_val_vec1, delta_vec1,
                                     min_val_vec2, delta_vec2, chunk);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, sum1, min_val_vec1, delta_vec1,
                                     min_val_vec2, delta_vec2, chunk);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, sum2, min_val_vec1, delta_vec1,
                                     min_val_vec2, delta_vec2, chunk);
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, sum3, min_val_vec1, delta_vec1,
                                     min_val_vec2, delta_vec2, chunk);
    }

    // Handle remaining steps (0-3)
    if constexpr (additional_steps > 0) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, sum0, min_val_vec1, delta_vec1,
                                     min_val_vec2, delta_vec2, chunk);
    }
    if constexpr (additional_steps > 1) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, sum1, min_val_vec1, delta_vec1,
                                     min_val_vec2, delta_vec2, chunk);
    }
    if constexpr (additional_steps > 2) {
        SQ8_SQ8_InnerProductStep_SVE(pVec1, pVec2, offset, sum2, min_val_vec1, delta_vec1,
                                     min_val_vec2, delta_vec2, chunk);
    }

    // Combine the accumulators
    svfloat32_t sum = svadd_f32_z(pg, sum0, sum1);
    sum = svadd_f32_z(pg, sum, sum2);
    sum = svadd_f32_z(pg, sum, sum3);

    // Horizontal sum of all elements in the vector
    return svaddv_f32(pg, sum);
}

// SQ8-to-SQ8 Inner Product distance function
// Returns 1 - inner_product (distance form)
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_SQ8_InnerProductSIMD_SVE(const void *pVec1v, const void *pVec2v, size_t dimension) {
    return 1.0f - SQ8_SQ8_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(pVec1v, pVec2v,
                                                                                    dimension);
}

// SQ8-to-SQ8 Cosine distance function
// Returns 1 - inner_product (assumes vectors are pre-normalized)
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_SQ8_CosineSIMD_SVE(const void *pVec1v, const void *pVec2v, size_t dimension) {
    float ip = SQ8_SQ8_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(pVec1v, pVec2v,
                                                                                 dimension);
    return 1.0f - ip;
}
