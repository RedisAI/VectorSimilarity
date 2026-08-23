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

dist_func_t<float> Choose_SQ8_FP32_IP_implementation_SSE4(size_t dim);
dist_func_t<float> Choose_SQ8_FP32_Cosine_implementation_SSE4(size_t dim);
dist_func_t<float> Choose_SQ8_FP32_L2_implementation_SSE4(size_t dim);

// What this tier provides, one row per (metric, type). Values taken from the legacy
//            metric  type      tier  step  policy  align_mod  align_bytes  min_dim
VECSIM_KERNEL(L2, SQ8_FP32, SSE4, 16, FixedModulus, 4, 4, 8)
VECSIM_KERNEL(IP, SQ8_FP32, SSE4, 16, FixedModulus, 4, 4, 8)
VECSIM_KERNEL(Cosine, SQ8_FP32, SSE4, 16, FixedModulus, 4, 4, 8)

} // namespace spaces
