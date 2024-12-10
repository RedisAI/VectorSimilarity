#pragma once

#include <random>
#include <vector>

namespace test_utils {

// Assuming v is a memory allocation of size dim * sizeof(float)
static void populate_int8_vec(int8_t *v, size_t dim, int seed = 1234) {

    std::mt19937 gen(seed); // Mersenne Twister engine initialized with the fixed seed

    // uniform_int_distribution doesn't support int8,
    // Define a distribution range for int8_t
    std::uniform_int_distribution<int16_t> dis(-128, 127);

    for (size_t i = 0; i < dim; i++) {
        v[i] = static_cast<int8_t>(dis(gen));
    }
}

// TODO: replace with normalize function from VecSim
float compute_norm(const int8_t *vec, size_t dim) {
    int norm = 0;
    for (size_t i = 0; i < dim; i++) {
        int val = static_cast<int>(vec[i]);
        norm += val * val;
    }
    return sqrt(norm);
}

} // namespace test_utils
