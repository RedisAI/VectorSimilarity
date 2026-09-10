/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/computer/preprocessors.h"
#include <array>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <limits>

int main(int argc, char **argv) {
    static_assert(std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24);
    if (argc != 4 || std::fegetround() != FE_TONEAREST) {
        std::fprintf(stderr, "Expected three coordinates and round-to-nearest mode.\n");
        return 1;
    }

    // Runtime inputs keep the production preprocessing path in the executed reproduction.
    constexpr size_t dim = 3;
    std::array<float, dim> input;
    for (size_t i = 0; i < dim; ++i) {
        input[i] = std::strtof(argv[i + 1], nullptr);
    }
    auto allocator = VecSimAllocator::newVecsimAllocator();
    QuantPreprocessor<float, VecSimMetric_L2> preprocessor(allocator, dim);
    void *storage = nullptr;
    size_t storage_bytes = sizeof(input);
    preprocessor.preprocessForStorage(input.data(), storage, storage_bytes, 0);
    const auto *bytes = static_cast<const uint8_t *>(storage);
    using sq8 = vecsim_types::sq8;
    const float min_val = load_unaligned<float>(bytes + dim + sq8::MIN_VAL * sizeof(float));
    const float delta = load_unaligned<float>(bytes + dim + sq8::DELTA * sizeof(float));
    std::printf("Production metadata: min=%.9g delta=%.9g bytes=[%u,%u,%u]\n", min_val, delta,
                unsigned(bytes[0]), unsigned(bytes[1]), unsigned(bytes[2]));
    bool reproduced = min_val == 1.0f && delta == 262144.0f && bytes[0] == 0 && bytes[1] == 128 &&
                      bytes[2] == 255;

    auto check = [&](const char *label, const std::array<float, dim> &query_input,
                     double expected_reference, float expected_actual) {
        void *query = nullptr;
        size_t query_bytes = sizeof(query_input);
        preprocessor.preprocessQuery(query_input.data(), query, query_bytes, 0);
        const auto *values = static_cast<const float *>(query);
        double reference = 0.0;
        std::printf("%s:\n", label);
        for (size_t i = 0; i < dim; ++i) {
            // Match the PR's independent double reference, using the actual stored metadata.
            const double reconstructed = double(min_val) + double(delta) * bytes[i];
            const double diff = reconstructed - double(values[i]);
            reference += diff * diff;
            const float min_minus_y = min_val - values[i];
            std::printf("  i=%zu q=%u y=%.9g reconstructed=%.17g exact_diff=%.17g "
                        "fp32_min_minus_y=%.9g fma_diff=%.9g\n",
                        i, unsigned(bytes[i]), values[i], reconstructed, diff, min_minus_y,
                        std::fma(delta, float(bytes[i]), min_minus_y));
        }
        const float actual = SQ8_FP32_L2Sqr(storage, query, dim);
        std::printf("  Production scalar L2^2=%.9g; double-reference L2^2=%.17g\n", actual,
                    reference);
        allocator->free_allocation(query);
        return reference == expected_reference && actual == expected_actual;
    };

    reproduced &= check("Original query", input, 2.0, 0.0f);
    auto shifted_query = input;
    shifted_query[1] += 4.0f;
    reproduced &= check("Middle query coordinate increased by 4", shifted_query, 10.0, 16.0f);
    allocator->free_allocation(storage);
    std::puts(reproduced ? "REPRODUCED both predicted distance errors using production code."
                         : "NOT REPRODUCED: at least one prediction did not match.");
    return reproduced ? 0 : 1;
}
