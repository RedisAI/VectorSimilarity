#pragma once

#include <random>
#include <vector>

namespace test_utils {

std::vector<int8_t> create_int8_vec(size_t dim) {

    std::mt19937 gen(1234); // Mersenne Twister engine initialized with the fixed seed

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
