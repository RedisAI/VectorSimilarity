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
#include "VecSim/spaces/IP/IP_NEON_SQ8.h"
#include "VecSim/types/sq8.h"
#include <arm_neon.h>

using sq8 = vecsim_types::sq8;

/*
 * Optimized asymmetric SQ8 L2 squared distance using algebraic identity:
 *
 *   ||x - y||² = Σx_i² - 2*IP(x, y) + Σy_i²
 *              = x_sum_squares - 2 * IP(x, y) + y_sum_squares
 *
 * where:
 *   - IP(x, y) = min * y_sum + delta * Σ(q_i * y_i)  (computed via SQ8_InnerProductSIMD16_NEON_IMP)
 *   - x_sum_squares and y_sum_squares are precomputed
 *
 * This avoids dequantization in the hot loop.
 */

template <unsigned char residual> // 0..15
float SQ8_L2SqrSIMD16_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // Get the raw inner product using the common SIMD implementation
    const float ip = SQ8_InnerProductSIMD16_NEON_IMP<residual>(pVect1v, pVect2v, dimension);

    // Get precomputed sum of squares from storage blob
    const uint8_t *pVect2 = static_cast<const uint8_t *>(pVect2v);
    const float *params = reinterpret_cast<const float *>(pVect2 + dimension);
    const float x_sum_sq = params[sq8::SUM_SQUARES];

    // Get precomputed sum of squares from query blob
    const float y_sum_sq = static_cast<const float *>(pVect1v)[dimension + sq8::SUM_SQUARES_QUERY];

    // L2² = ||x||² + ||y||² - 2*IP(x, y)
    return x_sum_sq + y_sum_sq - 2.0f * ip;
}
