/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "VecSim/index_factories/components/components_factory.h"

namespace {
using vecsim_types::sq8_fp64;

double readMetadata(const void *blob, size_t offset, size_t index) {
    double result;
    std::memcpy(&result, static_cast<const uint8_t *>(blob) + offset + index * sizeof(double),
                sizeof(result));
    return result;
}

template <VecSimMetric Metric>
void checkLayout(size_t dim, double constant) {
    const auto allocator = VecSimAllocator::newVecsimAllocator();
    SQ8FP64Preprocessor<Metric> preprocessor(allocator, dim);
    const std::vector<double> input(dim, constant);
    const auto original = input;
    void *storage = nullptr;
    void *query = nullptr;
    size_t storage_size = input.size() * sizeof(double);
    size_t query_size = storage_size;
    preprocessor.preprocess(input.data(), storage, query, storage_size, query_size, 32, 64);
    EXPECT_EQ(storage_size, sq8_fp64::storage_bytes_count<Metric>(dim));
    EXPECT_EQ(query_size, sq8_fp64::query_bytes_count<Metric>(dim));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(storage) % 32, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(query) % 64, 0);
    EXPECT_EQ(std::memcmp(input.data(), query, dim * sizeof(double)), 0);
    EXPECT_EQ(input, original);
    EXPECT_DOUBLE_EQ(readMetadata(storage, dim, sq8_fp64::MIN_VAL), constant);
    EXPECT_DOUBLE_EQ(readMetadata(storage, dim, sq8_fp64::DELTA), 1.0);
    EXPECT_DOUBLE_EQ(readMetadata(storage, dim, sq8_fp64::SUM), dim * constant);
    EXPECT_DOUBLE_EQ(readMetadata(query, dim * sizeof(double), sq8_fp64::SUM_QUERY),
                     dim * constant);
    if constexpr (Metric == VecSimMetric_L2) {
        EXPECT_DOUBLE_EQ(readMetadata(storage, dim, sq8_fp64::SUM_SQUARES),
                         dim * constant * constant);
        EXPECT_DOUBLE_EQ(readMetadata(query, dim * sizeof(double), sq8_fp64::SUM_SQUARES_QUERY),
                         dim * constant * constant);
    }
    for (size_t i = 0; i < dim; ++i) {
        EXPECT_EQ(static_cast<const uint8_t *>(storage)[i], 0);
    }
    std::vector<double> inplace((std::max(storage_size, dim * sizeof(double)) + 7) / 8);
    std::memcpy(inplace.data(), input.data(), dim * sizeof(double));
    preprocessor.preprocessStorageInPlace(inplace.data(), inplace.size() * sizeof(double));
    EXPECT_EQ(std::memcmp(inplace.data(), storage, storage_size), 0);
    allocator->free_allocation(storage);
    allocator->free_allocation(query);
}

TEST(SQ8FP64Components, LayoutAlignmentConstantsAndInPlace) {
    for (size_t dim : {1, 3, 4, 5, 17}) {
        checkLayout<VecSimMetric_L2>(dim, 0.0);
        checkLayout<VecSimMetric_L2>(dim, 1.25);
        checkLayout<VecSimMetric_IP>(dim, -2.0);
        checkLayout<VecSimMetric_Cosine>(dim, 1.0 / std::sqrt(double(dim)));
    }
}

TEST(SQ8FP64Components, DoublePrecisionRangeAndMetadata) {
    const auto allocator = VecSimAllocator::newVecsimAllocator();
    SQ8FP64Preprocessor<VecSimMetric_L2> preprocessor(allocator, 3);
    for (double scale : {1.0, 1e100, 1e-300}) {
        const double input[] = {scale, 1.25 * scale, 2.0 * scale};
        void *storage = nullptr;
        size_t size = sizeof(input);
        preprocessor.preprocessForStorage(input, storage, size, 0);
        std::vector<double> inplace((std::max(size, sizeof(input)) + sizeof(double) - 1) /
                                    sizeof(double));
        std::memcpy(inplace.data(), input, sizeof(input));
        preprocessor.preprocessStorageInPlace(inplace.data(), inplace.size() * sizeof(double));
        EXPECT_EQ(std::memcmp(inplace.data(), storage, size), 0);
        const auto *const codes = static_cast<const uint8_t *>(storage);
        EXPECT_EQ(codes[0], 0);
        EXPECT_EQ(codes[1], 64);
        EXPECT_EQ(codes[2], 255);
        const double delta = readMetadata(storage, 3, sq8_fp64::DELTA);
        EXPECT_GT(delta, 0.0);
        EXPECT_TRUE(std::isfinite(delta));
        const double middle = scale + delta * 64;
        EXPECT_DOUBLE_EQ(readMetadata(storage, 3, sq8_fp64::SUM), scale + middle + 2 * scale);
        EXPECT_DOUBLE_EQ(readMetadata(storage, 3, sq8_fp64::SUM_SQUARES),
                         scale * scale + middle * middle + 4 * scale * scale);
        allocator->free_allocation(storage);
    }
}

TEST(SQ8FP64Components, DefinedConversionOutsideRepresentableDeltaDomain) {
    const auto allocator = VecSimAllocator::newVecsimAllocator();
    SQ8FP64Preprocessor<VecSimMetric_IP> preprocessor(allocator, 2);
    const double tiny = std::numeric_limits<double>::denorm_min();
    for (const auto &input : {std::vector<double>{-1e308, 1e308}, std::vector<double>{0.0, tiny}}) {
        void *storage = nullptr;
        size_t size = input.size() * sizeof(double);
        preprocessor.preprocessForStorage(input.data(), storage, size, 0);
        const auto *const codes = static_cast<const uint8_t *>(storage);
        EXPECT_EQ(codes[0], 0);
        EXPECT_EQ(codes[1], input[0] == 0.0 ? 255 : 0);
        allocator->free_allocation(storage);
    }
}

template <VecSimMetric Metric>
void checkZeroDistance(double expected) {
    const auto allocator = VecSimAllocator::newVecsimAllocator();
    const auto components = CreateSQ8FP64IndexComponents<Metric>(allocator, 3);
    const std::unique_ptr<IndexCalculatorInterface<double>> calculator(components.indexCalculator);
    const std::unique_ptr<PreprocessorsContainerAbstract> preprocessors(components.preprocessors);
    const double input[] = {0.0, 0.0, 0.0};
    const auto blobs = preprocessors->preprocess(input, sizeof(input));
    EXPECT_DOUBLE_EQ(calculator->calcDistance(blobs.getStorageBlob(), blobs.getStorageBlob(), 3),
                     expected);
    EXPECT_DOUBLE_EQ(
        calculator->calcDistanceForQuery(blobs.getStorageBlob(), blobs.getQueryBlob(), 3),
        expected);
}

TEST(SQ8FP64Components, ZeroDistances) {
    checkZeroDistance<VecSimMetric_L2>(0.0);
    checkZeroDistance<VecSimMetric_IP>(1.0);
}

TEST(SQ8FP64Components, NearbyLargeValuesKeepDoublePrecision) {
    const auto allocator = VecSimAllocator::newVecsimAllocator();
    const auto components = CreateSQ8FP64IndexComponents<VecSimMetric_L2>(allocator, 2);
    const std::unique_ptr<IndexCalculatorInterface<double>> calculator(components.indexCalculator);
    const std::unique_ptr<PreprocessorsContainerAbstract> preprocessors(components.preprocessors);
    const double input[] = {1e8, 1e8 + 1.0};
    const double query[] = {1e8, 1e8};
    const auto stored = preprocessors->preprocessForStorage(input, sizeof(input));
    const auto processed = preprocessors->preprocessQuery(query, sizeof(query));
    EXPECT_EQ(static_cast<const uint8_t *>(stored.get())[0], 0);
    EXPECT_EQ(static_cast<const uint8_t *>(stored.get())[1], 255);
    EXPECT_NEAR(calculator->calcDistanceForQuery(stored.get(), processed.get(), 2), 1.0, 1e-14);
    const double close[] = {1.0, 1.0 + std::ldexp(1.0, -40)};
    const auto precise = preprocessors->preprocessForStorage(close, sizeof(close));
    EXPECT_DOUBLE_EQ(readMetadata(precise.get(), 2, sq8_fp64::DELTA), std::ldexp(1.0, -40) / 255);
    EXPECT_NEAR(calculator->calcDistance(precise.get(), precise.get(), 2), 0.0, 1e-40);
}
} // namespace
