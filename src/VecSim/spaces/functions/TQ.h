/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/spaces/spaces.h"

#include <cstddef>
#include <cstdint>

namespace spaces {

using tq_inner_product_func_t = float (*)(const void *, const void *, size_t);
using tq_sum_squares_func_t = float (*)(const void *, size_t);
using tq_packed_sign_dot_func_t = int (*)(const uint8_t *, const uint8_t *, size_t);
using tq_symmetric_polar_estimate_func_t = float (*)(const float *, const void *, const float *,
                                                     const void *, size_t, size_t, const float *,
                                                     bool, bool);
using tq_packed_residual_sign_dot_func_t = int (*)(const uint8_t *, const uint8_t *, size_t);
using tq_symmetric_polar_func_t = float (*)(const float *, const uint8_t *, const float *,
                                            const uint8_t *, const float *, uint8_t, size_t);

float TQ_FP32_InnerProduct(const void *lhs, const void *rhs, size_t dim);
float TQ_FP32_SumSquares(const void *values, size_t dim);

tq_inner_product_func_t Choose_FP32_InnerProduct_implementation_TQ(size_t dim,
                                                                   const void *arch_opt = nullptr);
tq_sum_squares_func_t Choose_FP32_SumSquares_implementation_TQ(size_t dim,
                                                               const void *arch_opt = nullptr);
tq_symmetric_polar_estimate_func_t
Choose_TQ_SymmetricPolarEstimate_implementation(const void *arch_opt = nullptr);
tq_packed_sign_dot_func_t Choose_TQ_PackedSignDot_implementation(const void *arch_opt = nullptr);

tq_packed_residual_sign_dot_func_t
Choose_TQ_PackedResidualSignDot_implementation(size_t projections, const void *arch_opt = nullptr);

tq_symmetric_polar_func_t Choose_TQ_SymmetricPolar_implementation(size_t pairs,
                                                                  const void *arch_opt = nullptr);

float TQ_SymmetricPolarEstimate(const float *lhs_radii, const void *lhs_angles,
                                const float *rhs_radii, const void *rhs_angles, size_t pairs,
                                size_t angle_delta_mask, const float *delta_cos_lut,
                                bool nibble_angle_codes, bool compact_angle_codes);
int TQ_PackedSignDot(const uint8_t *lhs, const uint8_t *rhs, size_t projections);

} // namespace spaces
