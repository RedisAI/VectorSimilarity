/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 * Licensed under your choice of RSALv2, SSPLv1, or AGPLv3.
 */
#include "gtest/gtest.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/index_factories/components/components_factory.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "VecSim/types/float16.h"
#include "VecSim/types/sq8.h"
#include "VecSim/utils/alignment.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#ifndef MOD19169_SQ8_QUERY
#error "SQ8 query mode tests require an explicit experimental mode"
#endif

namespace {
using FP16 = vecsim_types::float16;
using SQ8 = vecsim_types::sq8;
constexpr bool quantized_ip = MOD19169_SQ8_QUERY == 1;

// Component lifetimes must exceed those of blobs whose deleters use their allocator.
template <typename T, VecSimMetric Metric>
struct Components {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    std::unique_ptr<IndexCalculatorInterface<float>> calculator;
    std::unique_ptr<PreprocessorsContainerAbstract> preprocessors;

    Components(size_t dim, const float *mean = nullptr) {
        auto components = CreateSQ8IndexComponents<T, Metric>(allocator, dim, mean);
        calculator.reset(components.indexCalculator);
        preprocessors.reset(components.preprocessors);
    }

    PreprocessorInterface *preprocessor() {
        return static_cast<MultiPreprocessorsContainer<T, 1> *>(preprocessors.get())
            ->getPreprocessors()[0];
    }

    MemoryUtils::unique_blob own(void *blob) {
        return MemoryUtils::unique_blob(
            blob, [allocator = allocator](void *ptr) { allocator->free_allocation(ptr); });
    }
};

float reconstruct(const void *blob, size_t dim, size_t i) {
    const auto *bytes = static_cast<const uint8_t *>(blob);
    const float min = load_unaligned<float>(bytes + dim + SQ8::MIN_VAL * sizeof(float));
    const float delta = load_unaligned<float>(bytes + dim + SQ8::DELTA * sizeof(float));
    return min + delta * bytes[i];
}

void expect_alignment(const void *blob, unsigned char alignment) {
    if (alignment)
        EXPECT_EQ(reinterpret_cast<uintptr_t>(blob) % alignment, 0);
}

TEST(SQ8QueryModes, FP16ValuesSizesAndInsertionOwnership) {
    for (size_t dim : {1UL, 2UL, 3UL, 17UL, 32UL, 33UL, 129UL, 512UL, 768UL}) {
        for (bool with_mean : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "dim=" << dim << " mean=" << with_mean);
            std::vector<float> mean(dim, 0.125f);
            Components<FP16, VecSimMetric_IP> components(dim, with_mean ? mean.data() : nullptr);
            std::vector<FP16> input(dim);
            for (size_t i = 0; i < dim; ++i)
                input[i] = vecsim_types::FP32_to_FP16((int(i % 9) - 4) * 0.0625f);
            const auto original = input;

            void *raw = nullptr;
            size_t size = input.size() * sizeof(FP16);
            const unsigned char alignment = components.preprocessors->getQueryAlignment();
            components.preprocessor()->preprocessQuery(input.data(), raw, size, alignment);
            auto query = components.own(raw);
            ASSERT_NE(query.get(), nullptr);
            expect_alignment(query.get(), alignment);
            const size_t body_size =
                quantized_ip ? dim : dim * (MOD19169_SQ8_QUERY == 2 ? sizeof(float) : sizeof(FP16));
            const size_t metadata_count =
                quantized_ip ? SQ8::storage_metadata_count<VecSimMetric_IP>() + size_t(with_mean)
                             : SQ8::query_metadata_count<VecSimMetric_IP>() + size_t(with_mean);
            ASSERT_EQ(size, body_size + metadata_count * sizeof(float));
            if constexpr (!quantized_ip) {
                for (size_t i = 0; i < dim; ++i) {
                    const float actual =
                        MOD19169_SQ8_QUERY == 2
                            ? static_cast<const float *>(query.get())[i]
                            : vecsim_types::FP16_to_FP32(static_cast<const FP16 *>(query.get())[i]);
                    EXPECT_FLOAT_EQ(actual, vecsim_types::FP16_to_FP32(input[i]));
                }
            }
            auto insertion =
                components.preprocessors->preprocess(input.data(), input.size() * sizeof(FP16));
            auto storage = components.preprocessors->preprocessForStorage(
                input.data(), input.size() * sizeof(FP16));
            ASSERT_NE(insertion.getStorageBlob(), nullptr);
            ASSERT_NE(insertion.getQueryBlob(), nullptr);
            EXPECT_EQ(std::memcmp(insertion.getQueryBlob(), query.get(), size), 0);
            EXPECT_EQ(std::memcmp(insertion.getStorageBlob(), storage.get(),
                                  SQ8::storage_bytes_count<VecSimMetric_IP>(dim) +
                                      size_t(with_mean) * sizeof(float)),
                      0);
            if constexpr (quantized_ip)
                EXPECT_EQ(insertion.getStorageBlob(), insertion.getQueryBlob());
            else
                EXPECT_NE(insertion.getStorageBlob(), insertion.getQueryBlob());
            expect_alignment(insertion.getStorageBlob(),
                             components.preprocessors->getStorageAlignment());
            expect_alignment(insertion.getQueryBlob(), alignment);
            EXPECT_EQ(std::memcmp(input.data(), original.data(), input.size() * sizeof(FP16)), 0);

            unsigned char expected_alignment = 0;
            if constexpr (quantized_ip)
                spaces::GetDistFunc<SQ8, float>(VecSimMetric_IP, dim, &expected_alignment);
            else if constexpr (MOD19169_SQ8_QUERY == 2)
                spaces::GetDistFunc<float, float>(VecSimMetric_IP, dim, &expected_alignment);
            else
                spaces::GetDistFunc<FP16, float>(VecSimMetric_IP, dim, &expected_alignment);
            EXPECT_EQ(alignment, expected_alignment);
        }
    }
}

TEST(SQ8QueryModes, FP16ResidualAndSIMDDistances) {
    for (size_t dim :
         {1UL, 2UL, 3UL, 7UL, 16UL, 17UL, 32UL, 33UL, 65UL, 128UL, 129UL, 512UL, 768UL}) {
        SCOPED_TRACE(::testing::Message() << "dim=" << dim);
        Components<FP16, VecSimMetric_IP> components(dim);
        std::vector<FP16> candidate(dim), query(dim);
        for (size_t i = 0; i < dim; ++i) {
            candidate[i] = vecsim_types::FP32_to_FP16((int(i % 13) - 6) * 0.03125f);
            query[i] = vecsim_types::FP32_to_FP16((int(i % 11) - 5) * 0.0625f);
        }
        auto stored =
            components.preprocessors->preprocessForStorage(candidate.data(), dim * sizeof(FP16));
        auto prepared = components.preprocessors->preprocessQuery(query.data(), dim * sizeof(FP16));
        double expected = 1.0;
        for (size_t i = 0; i < dim; ++i) {
            const float q = quantized_ip ? reconstruct(prepared.get(), dim, i)
                                         : vecsim_types::FP16_to_FP32(query[i]);
            expected -= double(reconstruct(stored.get(), dim, i)) * q;
        }
        const float actual =
            components.calculator->calcDistanceForQuery(stored.get(), prepared.get(), dim);
        const auto dispatch =
            components.calculator->getDistanceDispatch(DistanceMode::StoredToQuery);
        ASSERT_TRUE(dispatch.isValid());
        EXPECT_TRUE(std::isfinite(actual));
        EXPECT_NEAR(actual, expected, 1e-4);
        EXPECT_FLOAT_EQ(dispatch(stored.get(), prepared.get(), dim), actual);
        if constexpr (MOD19169_SQ8_QUERY == 2) {
            unsigned char query_alignment = 0, storage_alignment = 0;
            spaces::GetDistFunc<FP16, float>(VecSimMetric_IP, dim, &query_alignment);
            const auto mixed =
                spaces::GetDistFunc<SQ8, float, FP16>(VecSimMetric_IP, dim, &storage_alignment);
            auto original_query = components.own(components.allocator->allocate_aligned(
                dim * sizeof(FP16) + sizeof(float), query_alignment));
            std::memcpy(original_query.get(), query.data(), dim * sizeof(FP16));
            float sum = 0;
            for (FP16 value : query)
                sum += vecsim_types::FP16_to_FP32(value);
            std::memcpy(static_cast<uint8_t *>(original_query.get()) + dim * sizeof(FP16), &sum,
                        sizeof(sum));
            auto original_storage = components.own(components.allocator->allocate_aligned(
                SQ8::storage_bytes_count<VecSimMetric_IP>(dim), storage_alignment));
            std::memcpy(original_storage.get(), stored.get(),
                        SQ8::storage_bytes_count<VecSimMetric_IP>(dim));
            EXPECT_NEAR(actual, mixed(original_storage.get(), original_query.get(), dim), 1e-4f);
        }
    }
}

TEST(SQ8QueryModes, CachedMeanCorrectionSeesInstalledMean) {
    for (size_t dim : {3UL, 17UL, 32UL, 129UL}) {
        SCOPED_TRACE(::testing::Message() << "dim=" << dim);
        std::vector<float> initial_mean(dim, 0.0f), mean(dim, 0.125f);
        Components<FP16, VecSimMetric_IP> components(dim, initial_mean.data());
        auto *preprocessor = dynamic_cast<QuantPreprocessor<FP16, VecSimMetric_IP, true> *>(
            components.preprocessor());
        auto *calculator = dynamic_cast<DistanceCalculatorWithNorm<FP16, float, VecSimMetric_IP> *>(
            components.calculator.get());
        ASSERT_NE(preprocessor, nullptr);
        ASSERT_NE(calculator, nullptr);
        const auto cached_query = calculator->getDistanceDispatch(DistanceMode::StoredToQuery);
        const auto cached_stored = calculator->getDistanceDispatch(DistanceMode::StoredToStored);
        preprocessor->setMean(mean);
        calculator->setMeanSumSquares(mean);
        std::vector<FP16> x(dim, vecsim_types::FP32_to_FP16(0.25f));
        std::vector<FP16> y(dim, vecsim_types::FP32_to_FP16(-0.5f));
        auto stored_x =
            components.preprocessors->preprocessForStorage(x.data(), dim * sizeof(FP16));
        auto stored_y =
            components.preprocessors->preprocessForStorage(y.data(), dim * sizeof(FP16));
        auto query = components.preprocessors->preprocessQuery(y.data(), dim * sizeof(FP16));
        const float expected = 1.0f + dim * 0.125f;
        EXPECT_FLOAT_EQ(calculator->calcDistanceForQuery(stored_x.get(), query.get(), dim),
                        expected);
        EXPECT_FLOAT_EQ(cached_query(stored_x.get(), query.get(), dim), expected);
        EXPECT_FLOAT_EQ(calculator->calcDistance(stored_x.get(), stored_y.get(), dim), expected);
        EXPECT_FLOAT_EQ(cached_stored(stored_x.get(), stored_y.get(), dim), expected);
    }
}

TEST(SQ8QueryModes, HNSWMeanInstallationAndPublicInsertion) {
    constexpr size_t dim = 17;
    std::vector<float> initial_mean(dim, 0.0f), mean(dim, 0.125f);
    HNSWParams params{};
    params.type = VecSimType_FLOAT16;
    params.dim = dim;
    params.metric = VecSimMetric_IP;
    params.M = 4;
    params.efConstruction = 10;
    params.efRuntime = 10;
    params.quantType = VecSimQuant_SQ8;
    params.quantParams = initial_mean.data();
    std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)> index(HNSWFactory::NewIndex(&params),
                                                                    VecSimIndex_Free);
    auto *hnsw = dynamic_cast<HNSWIndex<FP16, float> *>(index.get());
    ASSERT_NE(hnsw, nullptr);
    hnsw->setQuantizationMean(mean);
    std::vector<FP16> candidate(dim, vecsim_types::FP32_to_FP16(0.25f));
    std::vector<FP16> query(dim, vecsim_types::FP32_to_FP16(-0.5f));
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), candidate.data(), 42), 1);
    EXPECT_FLOAT_EQ(VecSimIndex_GetDistanceFrom_Unsafe(index.get(), 42, query.data()),
                    1.0f + dim * 0.125f);
    std::unique_ptr<VecSimQueryReply, decltype(&VecSimQueryReply_Free)> reply(
        VecSimIndex_TopKQuery(index.get(), query.data(), 1, nullptr, BY_SCORE),
        VecSimQueryReply_Free);
    ASSERT_NE(reply, nullptr);
    ASSERT_EQ(VecSimQueryReply_GetCode(reply.get()), VecSim_QueryReply_OK);
    ASSERT_EQ(VecSimQueryReply_Len(reply.get()), 1);
}

TEST(SQ8QueryModes, FiniteFP16BoundaryInputs) {
    for (const auto &values :
         {std::array<float, 2>{65504.0f, -65504.0f}, std::array<float, 2>{0x1p-24f, -0x1p-24f}}) {
        Components<FP16, VecSimMetric_IP> components(2);
        std::array<FP16, 2> candidate{vecsim_types::FP32_to_FP16(0.00006103515625f),
                                      vecsim_types::FP32_to_FP16(0.00006103515625f)};
        std::array<FP16, 2> query{vecsim_types::FP32_to_FP16(values[0]),
                                  vecsim_types::FP32_to_FP16(values[1])};
        auto stored =
            components.preprocessors->preprocessForStorage(candidate.data(), sizeof(candidate));
        auto prepared = components.preprocessors->preprocessQuery(query.data(), sizeof(query));
        const float score =
            components.calculator->calcDistanceForQuery(stored.get(), prepared.get(), 2);
        const auto dispatch =
            components.calculator->getDistanceDispatch(DistanceMode::StoredToQuery);
        EXPECT_TRUE(std::isfinite(score));
        EXPECT_NEAR(score, 1.0f, 1e-5f);
        EXPECT_FLOAT_EQ(dispatch(stored.get(), prepared.get(), 2), score);
    }
}

template <typename T, VecSimMetric Metric>
void verify_unaffected_queries() {
    constexpr size_t dim = 17;
    for (bool with_mean : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "mean=" << with_mean);
        std::vector<float> mean(dim, 0.125f);
        Components<T, Metric> components(dim, with_mean ? mean.data() : nullptr);
        std::vector<T> input(dim);
        for (size_t i = 0; i < dim; ++i) {
            if constexpr (std::is_same_v<T, FP16>)
                input[i] = vecsim_types::FP32_to_FP16(0.5f);
            else
                input[i] = 0.5f;
        }
        auto prepared = components.preprocessors->preprocessQuery(input.data(), dim * sizeof(T));
        auto candidate =
            components.preprocessors->preprocessForStorage(input.data(), dim * sizeof(T));
        const float score =
            components.calculator->calcDistanceForQuery(candidate.get(), prepared.get(), dim);
        EXPECT_TRUE(std::isfinite(score));
        EXPECT_NEAR(score, Metric == VecSimMetric_L2 ? 0.0f : 1.0f - dim * 0.25f, 1e-5f);
        EXPECT_FLOAT_EQ(components.calculator->getDistanceDispatch(DistanceMode::StoredToQuery)(
                            candidate.get(), prepared.get(), dim),
                        score);
        constexpr bool quantized = quantized_ip && Metric == VecSimMetric_IP;
        if constexpr (quantized) {
            auto stored =
                components.preprocessors->preprocessForStorage(input.data(), dim * sizeof(T));
            EXPECT_EQ(std::memcmp(prepared.get(), stored.get(),
                                  SQ8::storage_bytes_count<Metric>(dim) +
                                      size_t(with_mean) * sizeof(float)),
                      0);
        } else {
            for (size_t i = 0; i < dim; ++i) {
                const bool fp32_body =
                    std::is_same_v<T, float> || (with_mean && Metric == VecSimMetric_L2);
                const float actual =
                    fp32_body
                        ? static_cast<const float *>(prepared.get())[i]
                        : vecsim_types::FP16_to_FP32(static_cast<const FP16 *>(prepared.get())[i]);
                const float expected = with_mean && Metric == VecSimMetric_L2 ? 0.375f : 0.5f;
                EXPECT_FLOAT_EQ(actual, expected);
            }
        }
    }
}

TEST(SQ8QueryModes, L2AndFP32QueryRepresentations) {
    verify_unaffected_queries<FP16, VecSimMetric_L2>();
    verify_unaffected_queries<float, VecSimMetric_L2>();
    verify_unaffected_queries<float, VecSimMetric_IP>();
}
} // namespace
