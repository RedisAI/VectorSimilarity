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
#include "VecSim/spaces/kernel.h"

namespace spaces {

dist_func_t<float> Choose_SQ8_FP32_IP_implementation_AVX2(size_t dim);
dist_func_t<float> Choose_SQ8_FP32_Cosine_implementation_AVX2(size_t dim);
dist_func_t<float> Choose_SQ8_FP32_L2_implementation_AVX2(size_t dim);

dist_func_t<float> Choose_BF16_IP_implementation_AVX2(size_t dim);
dist_func_t<float> Choose_BF16_L2_implementation_AVX2(size_t dim);

// What this tier provides, one row per (metric, type). The values come from what the kernels
// actually do: CHOOSE_IMPLEMENTATION specializes on dim % 32 here, and the alignment hint is only
// requested when dim % 8 == 0 because the SQ8 operand is read in 8-byte chunks. min_dim is 8
// because below that the residual handling has nothing to work with and the scalar kernel is at
// least as fast.
//
//            metric  type      tier  step  policy        align_mod  align_bytes  min_dim
VECSIM_KERNEL(L2, SQ8_FP32, AVX2, 32, FixedModulus, 8, 8, 8)
VECSIM_KERNEL(IP, SQ8_FP32, AVX2, 32, FixedModulus, 8, 8, 8)
VECSIM_KERNEL(Cosine, SQ8_FP32, AVX2, 32, FixedModulus, 8, 8, 8)

} // namespace spaces
