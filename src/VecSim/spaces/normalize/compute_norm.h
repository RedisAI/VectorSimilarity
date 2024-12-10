/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include <cmath>
#include <vector>

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

namespace spaces {

template <typename DataType>
static inline float IntegralType_ComputeNorm(const DataType *vec, const size_t dim) {
    int sum = 0;

    for (size_t i = 0; i < dim; i++) {
        int val = static_cast<int>(vec[i]);
        sum += val * val;
    }
    float norm = sqrt(sum);
}



} // namespace spaces
