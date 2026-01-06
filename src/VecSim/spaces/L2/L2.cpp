/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "L2.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include <cstring>
#include <iostream>

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

float SQ8_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const float *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);
    // pvect2 is a vector of uint8_t, so we need to dequantize it, normalize it and then multiply
    // it. it structred as [quantized values (uint8_t * dim)][min_val (float)][delta
    // (float)][inv_norm (float)] The last two values are used to dequantize the vector.
    const float min_val = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        auto dequantized_V2 = (pVect2[i] * delta + min_val);
        float t = pVect1[i] - dequantized_V2;
        res += t * t;
    }
    return res;
}

float FP32_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *vec1 = (float *)pVect1v;
    float *vec2 = (float *)pVect2v;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float t = vec1[i] - vec2[i];
        res += t * t;
    }
    return res;
}

double FP64_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension) {
    double *vec1 = (double *)pVect1v;
    double *vec2 = (double *)pVect2v;

    double res = 0;
    for (size_t i = 0; i < dimension; i++) {
        double t = vec1[i] - vec2[i];
        res += t * t;
    }
    return res;
}

template <bool is_little>
float BF16_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension) {
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float a = vecsim_types::bfloat16_to_float32<is_little>(pVect1[i]);
        float b = vecsim_types::bfloat16_to_float32<is_little>(pVect2[i]);
        float diff = a - b;
        res += diff * diff;
    }
    return res;
}

float BF16_L2Sqr_LittleEndian(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return BF16_L2Sqr<true>(pVect1v, pVect2v, dimension);
}

float BF16_L2Sqr_BigEndian(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return BF16_L2Sqr<false>(pVect1v, pVect2v, dimension);
}

float FP16_L2Sqr(const void *pVect1, const void *pVect2, size_t dimension) {
    auto *vec1 = (float16 *)pVect1;
    auto *vec2 = (float16 *)pVect2;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float t = vecsim_types::FP16_to_FP32(vec1[i]) - vecsim_types::FP16_to_FP32(vec2[i]);
        res += t * t;
    }
    return res;
}

// Return type for the L2 functions.
// The type should be able to hold `dimension * MAX_VAL(int_elem_t) * MAX_VAL(int_elem_t)`.
// To support dimension up to 2^16, we need the difference between the type and int_elem_t to be at
// least 2 bytes. We assert that in the implementation.
template <typename int_elem_t>
using ret_t = std::conditional_t<sizeof(int_elem_t) == 1, int, long long>;

// Difference type for the L2 functions.
// The type should be able to hold `MIN_VAL(int_elem_t)-MAX_VAL(int_elem_t)`, and should be signed
// to avoid unsigned arithmetic. This means that the difference type should be bigger than the
// size of the element type. We assert that in the implementation.
template <typename int_elem_t>
using diff_t = std::conditional_t<sizeof(int_elem_t) == 1, int16_t, int>;

template <typename int_elem_t>
static inline ret_t<int_elem_t> INTEGER_L2Sqr(const int_elem_t *pVect1, const int_elem_t *pVect2,
                                              size_t dimension) {
    static_assert(sizeof(ret_t<int_elem_t>) - sizeof(int_elem_t) * 2 >= sizeof(uint16_t));
    static_assert(std::is_signed_v<diff_t<int_elem_t>>);
    static_assert(sizeof(diff_t<int_elem_t>) >= 2 * sizeof(int_elem_t));

    ret_t<int_elem_t> res = 0;
    for (size_t i = 0; i < dimension; i++) {
        diff_t<int_elem_t> diff = pVect1[i] - pVect2[i];
        res += diff * diff;
    }
    return res;
}

float INT8_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const int8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const int8_t *>(pVect2v);
    return float(INTEGER_L2Sqr(pVect1, pVect2, dimension));
}

float UINT8_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);
    return float(INTEGER_L2Sqr(pVect1, pVect2, dimension));
}

// SQ8-to-SQ8 L2 squared distance (both vectors are uint8 quantized)
// Vector layout: [uint8_t values (dim)] [min_val (float)] [delta (float)] [sum (float)]
// [sum_of_squares (float)]
//  ||x - y||² = ||x||² + ||y||² - 2*IP(x, y)
//   where:
//     - ||x||² = sum_squares_x is precomputed and stored
//     - ||y||² = sum_squares_y is precomputed and stored
//     - IP(x, y) is computed using SQ8_SQ8_InnerProduct_Impl

float SQ8_SQ8_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // Get precomputed sum of squares from both vectors
    // Layout: [uint8_t values (dim)] [min_val] [delta] [sum] [sum_of_squares]
    const float sum_sq_1 = *reinterpret_cast<const float *>(pVect1 + dimension + 3 * sizeof(float));
    const float sum_sq_2 = *reinterpret_cast<const float *>(pVect2 + dimension + 3 * sizeof(float));

    // Use the common inner product implementation
    const float ip = SQ8_SQ8_InnerProduct_Impl(pVect1v, pVect2v, dimension);

    // L2² = ||x||² + ||y||² - 2*IP(x, y)
    return sum_sq_1 + sum_sq_2 - 2.0f * ip;
}
