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

dist_func_t<float> Choose_INT8_L2_implementation_AVX512F_BW_VL_VNNI(size_t dim);
dist_func_t<float> Choose_INT8_IP_implementation_AVX512F_BW_VL_VNNI(size_t dim);
dist_func_t<float> Choose_INT8_Cosine_implementation_AVX512F_BW_VL_VNNI(size_t dim);

dist_func_t<float> Choose_UINT8_L2_implementation_AVX512F_BW_VL_VNNI(size_t dim);
dist_func_t<float> Choose_UINT8_IP_implementation_AVX512F_BW_VL_VNNI(size_t dim);
dist_func_t<float> Choose_UINT8_Cosine_implementation_AVX512F_BW_VL_VNNI(size_t dim);

dist_func_t<float> Choose_SQ8_FP32_IP_implementation_AVX512F_BW_VL_VNNI(size_t dim);
dist_func_t<float> Choose_SQ8_FP32_Cosine_implementation_AVX512F_BW_VL_VNNI(size_t dim);
dist_func_t<float> Choose_SQ8_FP32_L2_implementation_AVX512F_BW_VL_VNNI(size_t dim);

// SQ8-to-SQ8 distance functions (both vectors are uint8 quantized with precomputed sum)
dist_func_t<float> Choose_SQ8_SQ8_IP_implementation_AVX512F_BW_VL_VNNI(size_t dim);
dist_func_t<float> Choose_SQ8_SQ8_Cosine_implementation_AVX512F_BW_VL_VNNI(size_t dim);
dist_func_t<float> Choose_SQ8_SQ8_L2_implementation_AVX512F_BW_VL_VNNI(size_t dim);

// What this tier provides, one row per (metric, type). Values taken from the legacy chooser.
// The tier name and the file stem differ here (AVX512_F_BW_VL_VNNI against
// AVX512F_BW_VL_VNNI), which is why the manifest carries the stem as its own column.
VECSIM_KERNEL(L2, SQ8_FP32, AVX512_F_BW_VL_VNNI, 32, FixedModulus, 16, 16, 8)
VECSIM_KERNEL(IP, SQ8_FP32, AVX512_F_BW_VL_VNNI, 32, FixedModulus, 16, 16, 8)
VECSIM_KERNEL(Cosine, SQ8_FP32, AVX512_F_BW_VL_VNNI, 32, FixedModulus, 16, 16, 8)

} // namespace spaces
