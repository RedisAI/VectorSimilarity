/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/vec_sim_common.h" // enum VecSimMetric

namespace spaces {

template <typename RET_TYPE>
using dist_func_t = RET_TYPE (*)(const void *, const void *, size_t);

using bf16_converter_t = void (*)(const void *, const void *, size_t);

// Set the distance function for a given metric and dimension, and the alignment hint according to
// the chosen implementation and available optimizations.
void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *index_dist_func,
                 unsigned char *alignment);
void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<double> *index_dist_func,
                 unsigned char *alignment);

void SetBF16DistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *out_func,
                 unsigned char *alignment);
void SetBF16DistFunc(VecSimMetric metric, size_t dim, dist_func_t<double> *out_func,
                 unsigned char *alignment);

//fp32_to_bf16_encoder_t GetFP32ToBFloat16Encoder(size_t dim);

} // namespace spaces
