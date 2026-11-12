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
#include "VecSim/types/float16.h"

namespace spaces {

dist_func_t<float> Choose_FP16_IP_implementation_AVX512F(size_t dim);
dist_func_t<float> Choose_FP32_IP_implementation_AVX512F(size_t dim);
dist_func_t<double> Choose_FP64_IP_implementation_AVX512F(size_t dim);

dist_func_t<float> Choose_FP16_L2_implementation_AVX512F(size_t dim);
dist_func_t<float> Choose_FP32_L2_implementation_AVX512F(size_t dim);
dist_func_t<double> Choose_FP64_L2_implementation_AVX512F(size_t dim);

// SQ8↔FP16 kernels — only need AVX-512F, not VNNI/BW/VL.
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_AVX512F(size_t dim);
dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_AVX512F(size_t dim);
dist_func_t<float> Choose_SQ8_FP16_L2_implementation_AVX512F(size_t dim);

// A8 query encoders (vecsim_types::sq8_a8_query) for float and float16 input.
sq8_a8_encode_func_t<float> Choose_SQ8_A8_FP32_QueryEncoder_AVX512F();
sq8_a8_encode_func_t<vecsim_types::float16> Choose_SQ8_A8_FP16_QueryEncoder_AVX512F();

} // namespace spaces
