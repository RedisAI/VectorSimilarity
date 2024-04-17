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

// Set the distance function for a given data type, metric and dimension. The alignment hint is
// determined according to the chosen implementation and available optimizations.

template <typename DataType, typename DistType>
dist_func_t<DistType> GetDistFunc(VecSimMetric metric, size_t dim, unsigned char *alignment);

template <typename DataType>
using normalizeVector_f = void (*)(void *input_vector, size_t dim);

template <typename DataType>
normalizeVector_f<DataType> GetNormalizeFunc();
} // namespace spaces
