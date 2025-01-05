#pragma once

#include <random>
#include <vector>
#include "VecSim/spaces/normalize/compute_norm.h"

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

template <typename datatype>
float integral_compute_norm(const datatype *vec, size_t dim) {
    return spaces::IntegralType_ComputeNorm<datatype>(vec, dim);
}

} // namespace test_utils
