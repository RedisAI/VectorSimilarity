/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/computer/preprocessors.h"
#include <bit>
#include <cfenv>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

void expectDistance(const void *storage, const float *query, size_t dim, float expected) {
    EXPECT_EQ(std::bit_cast<uint32_t>(SQ8_FP32_L2Sqr(storage, query, dim)),
              std::bit_cast<uint32_t>(expected));
    EXPECT_EQ(std::bit_cast<uint32_t>(spaces::L2_SQ8_FP32_GetDistFunc(dim)(storage, query, dim)),
              std::bit_cast<uint32_t>(expected));
}

std::vector<uint8_t> rawStorage(size_t dim, float min = 0.0f, float delta = 1.0f) {
    std::vector<uint8_t> storage(vecsim_types::sq8::storage_bytes_count<VecSimMetric_L2>(dim));
    const float metadata[] = {min, delta, 0.0f, 0.0f};
    memcpy(storage.data() + dim, metadata, sizeof(metadata));
    return storage;
}

} // namespace

TEST(SQ8ExactL2, ProductionQuantizationCancellation) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    for (size_t dim : {3UL, 8UL, 9UL, 16UL, 128UL, 1536UL}) {
        std::vector<float> input(dim, 1.0f);
        input[1] = 33554432.0f;
        input.back() = 66846720.0f;
        QuantPreprocessor<float, VecSimMetric_L2> preprocessor(allocator, dim);
        void *storage = nullptr;
        size_t bytes = input.size() * sizeof(float);
        preprocessor.preprocessForStorage(input.data(), storage, bytes, 0);
        expectDistance(storage, input.data(), dim, 2.0f);
        input[1] += 4.0f;
        expectDistance(storage, input.data(), dim, 10.0f);
        allocator->free_allocation(storage);
    }
}

TEST(SQ8ExactL2, CancellationBeyondDoublePrecision) {
    const std::vector<float> input = {1.0f, 0x1p60f, 255.0f * 0x1p53f};
    auto allocator = VecSimAllocator::newVecsimAllocator();
    QuantPreprocessor<float, VecSimMetric_L2> preprocessor(allocator, input.size());
    void *storage = nullptr;
    size_t bytes = input.size() * sizeof(float);
    preprocessor.preprocessForStorage(input.data(), storage, bytes, 0);
    expectDistance(storage, input.data(), input.size(), 2.0f);
    allocator->free_allocation(storage);
}

TEST(SQ8ExactL2, RoundingMidpointsAndStickyBits) {
    const std::vector<std::vector<float>> queries = {
        {1.0f, 0x1p-12f}, {1.0f, 0x1p-12f, 0x1p-24f}, {1.0f, 0x1p-12f, 0x1p-12f, 0x1p-12f}};
    const uint32_t expected[] = {0x3f800000, 0x3f800001, 0x3f800002};
    for (size_t i = 0; i < queries.size(); ++i) {
        auto storage = rawStorage(queries[i].size());
        expectDistance(storage.data(), queries[i].data(), queries[i].size(),
                       std::bit_cast<float>(expected[i]));
    }
}

TEST(SQ8ExactL2, OutputUnderflowAndOverflow) {
    auto storage = rawStorage(1);
    float query = 0x1p-75f;
    expectDistance(storage.data(), &query, 1, 0.0f);
    query = std::nextafter(query, 1.0f);
    expectDistance(storage.data(), &query, 1, std::numeric_limits<float>::denorm_min());
    query = 0x1p-74f;
    expectDistance(storage.data(), &query, 1, std::bit_cast<float>(uint32_t{2}));
    query = 0x1p-63f;
    expectDistance(storage.data(), &query, 1, std::numeric_limits<float>::min());
    query = std::numeric_limits<float>::max();
    expectDistance(storage.data(), &query, 1, std::numeric_limits<float>::infinity());
}

TEST(SQ8ExactL2, NonfiniteAndEmptyInputs) {
    EXPECT_EQ(SQ8_FP32_L2Sqr(nullptr, nullptr, 0), 0.0f);
    auto storage = rawStorage(2);
    float query[] = {std::numeric_limits<float>::infinity(), 0.0f};
    expectDistance(storage.data(), query, 2, std::numeric_limits<float>::infinity());
    query[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(std::isnan(SQ8_FP32_L2Sqr(storage.data(), query, 2)));
    storage = rawStorage(2, std::numeric_limits<float>::infinity());
    query[0] = query[1] = 0.0f;
    EXPECT_TRUE(std::isnan(SQ8_FP32_L2Sqr(storage.data(), query, 2)));
}

TEST(SQ8ExactL2, RoundingModeDoesNotChangeContract) {
    const int original = std::fegetround();
    auto storage = rawStorage(2);
    const float query[] = {1.0f, 0x1p-12f};
    for (const int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        expectDistance(storage.data(), query, 2, 1.0f);
    }
    EXPECT_EQ(std::fesetround(original), 0);
}
