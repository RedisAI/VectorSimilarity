#pragma once

#include <random>
#include <vector>

namespace test_utils {

static std::vector<int8_t> create_int8_vec(size_t dim, int seed = 1234) {

    std::mt19937 gen(seed); // Mersenne Twister engine initialized with the fixed seed

    // uniform_int_distribution doesn't support int8,
    // Define a distribution range for int8_t
    std::uniform_int_distribution<int16_t> dis(-128, 127);

    std::vector<int8_t> vec(dim);
    for (auto &num : vec) {
        num = static_cast<int8_t>(dis(gen));
    }

    return vec;
}

} // namespace test_utils
