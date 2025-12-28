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

/**
 * SQ8 distance functions (float32 query vs uint8 stored) for SVE.
 *
 * Uses algebraic optimization to reduce operations per element:
 *
 * IP = Σ query[i] * (val[i] * δ + min)
 *    = δ * Σ(query[i] * val[i]) + min * Σ(query[i])
 *
 * This saves 1 FMA per chunk by deferring dequantization to scalar math at the end.
 *
 * Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [inv_norm (float)]
 */

// Helper function to perform inner product step with algebraic optimization
static inline void InnerProductStep(const float *pVect1, const uint8_t *pVect2, size_t offset,
                                    svfloat32_t &dot_sum, svfloat32_t &query_sum,
                                    const size_t chunk) {
    svbool_t pg = svptrue_b32();

    // Load float elements from query
    svfloat32_t v1 = svld1_f32(pg, pVect1 + offset);

    // Load uint8 elements and convert to float
    svuint32_t v2_u32 = svld1ub_u32(pg, pVect2 + offset);
    svfloat32_t v2_f = svcvt_f32_u32_x(pg, v2_u32);

    // Accumulate query * val (without dequantization)
    dot_sum = svmla_f32_x(pg, dot_sum, v1, v2_f);

    // Accumulate query sum
    query_sum = svadd_f32_x(pg, query_sum, v1);
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_InnerProductSIMD_SVE_IMP(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *pVect1 = static_cast<const float *>(pVect1v);
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);
    size_t offset = 0;

    // Get dequantization parameters from the end of quantized vector
    const float min_val = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));

    svbool_t pg = svptrue_b32();

    // Get the number of 32-bit elements per vector at runtime
    uint64_t chunk = svcntw();

    // Multiple accumulators for instruction-level parallelism
    // dot_sum: accumulates query[i] * val[i]
    // query_sum: accumulates query[i]
    svfloat32_t dot_sum0 = svdup_f32(0.0f);
    svfloat32_t dot_sum1 = svdup_f32(0.0f);
    svfloat32_t dot_sum2 = svdup_f32(0.0f);
    svfloat32_t dot_sum3 = svdup_f32(0.0f);
    svfloat32_t query_sum0 = svdup_f32(0.0f);
    svfloat32_t query_sum1 = svdup_f32(0.0f);
    svfloat32_t query_sum2 = svdup_f32(0.0f);
    svfloat32_t query_sum3 = svdup_f32(0.0f);

    // Handle partial chunk if needed
    if constexpr (partial_chunk) {
        size_t remaining = dimension % chunk;
        if (remaining > 0) {
            // Create predicate for the remaining elements
            svbool_t pg_partial =
                svwhilelt_b32(static_cast<uint32_t>(0), static_cast<uint32_t>(remaining));

            // Load query float elements with predicate
            svfloat32_t v1 = svld1_f32(pg_partial, pVect1);

            // Load uint8 elements and convert to float
            svuint32_t v2_u32 = svld1ub_u32(pg_partial, pVect2 + offset);
            svfloat32_t v2_f = svcvt_f32_u32_z(pg_partial, v2_u32);

            // Accumulate dot product (no dequantization)
            dot_sum0 = svmla_f32_z(pg_partial, dot_sum0, v1, v2_f);

            // Accumulate query sum
            query_sum0 = svadd_f32_z(pg_partial, query_sum0, v1);

            // Move past the partial chunk
            offset += remaining;
        }
    }

    // Process 4 chunks at a time in the main loop
    auto chunk_size = 4 * chunk;
    const size_t number_of_chunks =
        (dimension - (partial_chunk ? dimension % chunk : 0)) / chunk_size;

    for (size_t i = 0; i < number_of_chunks; i++) {
        InnerProductStep(pVect1, pVect2, offset, dot_sum0, query_sum0, chunk);
        InnerProductStep(pVect1, pVect2, offset + chunk, dot_sum1, query_sum1, chunk);
        InnerProductStep(pVect1, pVect2, offset + 2 * chunk, dot_sum2, query_sum2, chunk);
        InnerProductStep(pVect1, pVect2, offset + 3 * chunk, dot_sum3, query_sum3, chunk);
        offset += chunk_size;
    }

    // Handle remaining steps (0-3)
    if constexpr (additional_steps > 0) {
        InnerProductStep(pVect1, pVect2, offset, dot_sum0, query_sum0, chunk);
        offset += chunk;
    }
    if constexpr (additional_steps > 1) {
        InnerProductStep(pVect1, pVect2, offset, dot_sum1, query_sum1, chunk);
        offset += chunk;
    }
    if constexpr (additional_steps > 2) {
        InnerProductStep(pVect1, pVect2, offset, dot_sum2, query_sum2, chunk);
    }

    // Combine the accumulators
    svfloat32_t dot_total = svadd_f32_x(pg, svadd_f32_x(pg, dot_sum0, dot_sum1),
                                        svadd_f32_x(pg, dot_sum2, dot_sum3));
    svfloat32_t query_total = svadd_f32_x(pg, svadd_f32_x(pg, query_sum0, query_sum1),
                                          svadd_f32_x(pg, query_sum2, query_sum3));

    // Horizontal sum of all elements
    float dot_product = svaddv_f32(pg, dot_total);
    float query_sum = svaddv_f32(pg, query_total);

    // Apply algebraic formula: IP = δ * Σ(query*val) + min * Σ(query)
    return delta * dot_product + min_val * query_sum;
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f - SQ8_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(pVect1v, pVect2v,
                                                                                dimension);
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_CosineSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // Get quantization parameters
    const float inv_norm = *reinterpret_cast<const float *>(pVect2 + dimension + 2 * sizeof(float));

    // Compute inner product with dequantization using the common function
    const float res =
        SQ8_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);

    // For cosine, we need to account for the vector norms
    // The inv_norm parameter is stored after min_val and delta in the quantized vector
    return 1.0f - res * inv_norm;
}
