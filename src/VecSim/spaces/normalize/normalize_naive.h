/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/types/bfloat16.h"
#include <cmath>

using bfloat16 = vecsim_types::bfloat16;

namespace spaces {

template <typename DataType>
static inline void normalizeVector_imp(void *vec, const size_t dim) {
    DataType *input_vector = (DataType *)vec;
    // Cast to double to avoid float overflow.
    double sum = 0;

    for (size_t i = 0; i < dim; i++) {
        sum += (double)input_vector[i] * (double)input_vector[i];
    }
    DataType norm = sqrt(sum);

    for (size_t i = 0; i < dim; i++) {
        input_vector[i] = input_vector[i] / norm;
    }
}

template <bool is_little>
static inline void bfloat16_normalizeVector(void *vec, const size_t dim) {
    bfloat16 *input_vector = (bfloat16 *)vec;

    float f32_tmp[dim];

    float sum = 0;

    if constexpr (is_little) {
        for (size_t i = 0; i < dim; i++) {
            float val = vecsim_types::bfloat16_to_float32(input_vector[i]);
            f32_tmp[i] = val;
            sum += val * val;
        }
    } else {
        for (size_t i = 0; i < dim; i++) {
            float val = vecsim_types::bfloat16_to_float32_bigEndian(input_vector[i]);
            f32_tmp[i] = val;
            sum += val * val;
        }
    }

    float norm = sqrt(sum);

    for (size_t i = 0; i < dim; i++) {
        input_vector[i] = vecsim_types::float_to_bf16(f32_tmp[i] / norm);
    }
}

} // namespace spaces
