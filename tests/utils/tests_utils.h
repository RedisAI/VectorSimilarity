/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <random>
#include <vector>
#include "VecSim/spaces/normalize/compute_norm.h"
#include "VecSim/types/float16.h"

namespace test_utils {

// Assuming v is a memory allocation of size dim * sizeof(float)
static void populate_int8_vec(int8_t *v, size_t dim, int seed = 1234) {

    std::mt19937 gen(seed); // Mersenne Twister engine initialized with the fixed seed

    // uniform_int_distribution doesn't support int8,
    // Define a distribution range for int8_t
    std::uniform_int_distribution<int16_t> dis(INT8_MIN, INT8_MAX);

    for (size_t i = 0; i < dim; i++) {
        v[i] = static_cast<int8_t>(dis(gen));
    }
}
static void populate_uint8_vec(uint8_t *v, size_t dim, int seed = 1234) {

    std::mt19937 gen(seed); // Mersenne Twister engine initialized with the fixed seed

    // uniform_int_distribution doesn't support uint8,
    // Define a distribution range for uint8_t
    std::uniform_int_distribution<uint16_t> dis(0, UINT8_MAX);

    for (size_t i = 0; i < dim; i++) {
        v[i] = static_cast<uint8_t>(dis(gen));
    }
}

// Assuming v is a memory allocation of size dim * sizeof(float)
static void populate_float_vec(float *v, size_t dim, int seed = 1234, float min = -1.0f,
                               float max = 1.0f) {

    std::mt19937 gen(seed); // Mersenne Twister engine initialized with the fixed seed

    // Define a distribution range for float values
    std::uniform_real_distribution<float> dis(min, max);

    for (size_t i = 0; i < dim; i++) {
        v[i] = dis(gen);
    }
}

// Assuming v is a memory allocation of size dim * sizeof(float)
static void populate_float16_vec(vecsim_types::float16 *v, const size_t dim, int seed = 1234,
                                 float min = -1.0f, float max = 1.0f) {
    float v_f[dim];
    populate_float_vec(v_f, dim, seed, min, max);

    for (size_t i = 0; i < dim; i++) {
        v[i] = vecsim_types::FP32_to_FP16(v_f[i]);
    }
}

static void quantize_float_vec_to_uint8(float *v, size_t dim, uint8_t *qv, int seed = 1234) {

    float min_val = v[0];
    float max_val = v[0];
    for (size_t i = 1; i < dim; i++) {
        min_val = std::min(min_val, v[i]);
        max_val = std::max(max_val, v[i]);
    }
    // Calculate delta
    float delta = (max_val - min_val) / 255.0f;
    if (delta == 0)
        delta = 1.0f; // Avoid division by zero
    // Quantize each value
    for (size_t i = 0; i < dim; i++) {
        float normalized = (v[i] - min_val) / delta;
        normalized = std::max(0.0f, std::min(255.0f, normalized));
        qv[i] = static_cast<uint8_t>(std::round(normalized));
    }
    // Store parameters
    float *params = reinterpret_cast<float *>(qv + dim);
    params[0] = min_val;
    params[1] = delta;
}

static void populate_float_vec_to_sq8(uint8_t *v, size_t dim, int seed = 1234) {

    std::mt19937 gen(seed); // Mersenne Twister engine initialized with the fixed seed
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; i++) {
        vec[i] = dis(gen);
    }
    quantize_float_vec_to_uint8(vec.data(), dim, v, seed);
}

/**
 * Quantize float vector to SQ8 with precomputed sum and norm.
 * Vector layout: [uint8_t values (dim)] [min (float)] [delta (float)] [sum (float)] [norm (float)]
 * where sum = Σv[i] and norm = Σv[i]² (sum of squares of uint8 elements)
 */
static void quantize_float_vec_to_uint8_with_sum_norm(float *v, size_t dim, uint8_t *qv,
                                                      int seed = 1234) {
    float min_val = v[0];
    float max_val = v[0];
    for (size_t i = 1; i < dim; i++) {
        min_val = std::min(min_val, v[i]);
        max_val = std::max(max_val, v[i]);
    }
    // Calculate delta
    float delta = (max_val - min_val) / 255.0f;
    if (delta == 0)
        delta = 1.0f; // Avoid division by zero

    // Quantize each value and compute sum and norm
    float sum = 0.0f;
    float norm = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float normalized = (v[i] - min_val) / delta;
        normalized = std::max(0.0f, std::min(255.0f, normalized));
        qv[i] = static_cast<uint8_t>(std::round(normalized));
        sum += static_cast<float>(qv[i]);
        norm += static_cast<float>(qv[i]) * static_cast<float>(qv[i]);
    }

    // Store parameters: [min, delta, sum, norm]
    float *params = reinterpret_cast<float *>(qv + dim);
    params[0] = min_val;
    params[1] = delta;
    params[2] = sum;
    params[3] = norm;
}

/**
 * Populate a float vector and quantize to SQ8 with precomputed sum and norm.
 * Vector layout: [uint8_t values (dim)] [min (float)] [delta (float)] [sum (float)] [norm (float)]
 */
static void populate_float_vec_to_sq8_with_sum_norm(uint8_t *v, size_t dim, int seed = 1234) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; i++) {
        vec[i] = dis(gen);
    }
    quantize_float_vec_to_uint8_with_sum_norm(vec.data(), dim, v, seed);
}

template <typename datatype>
float integral_compute_norm(const datatype *vec, size_t dim) {
    return spaces::IntegralType_ComputeNorm<datatype>(vec, dim);
}

} // namespace test_utils
