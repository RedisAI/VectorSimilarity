/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include <benchmark/benchmark.h>

#include "VecSim/algorithms/tq/tq_flat.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace {

using TQState = TQFlatDetails::TQModelState;
using TQPreprocessor = TQFlatDetails::TQPreprocessor<VecSimMetric_Cosine>;

constexpr size_t kDim = 1024;
constexpr size_t kProjections = 512;
constexpr size_t kSeed = 7;
constexpr size_t kVectorPairs = 1024;

std::vector<float> GenerateUnitVector(std::mt19937_64 &rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> values(kDim);
    float norm_sq = 0.0f;
    for (float &value : values) {
        value = normal(rng);
        norm_sq += value * value;
    }
    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (float &value : values) {
        value *= inv_norm;
    }
    return values;
}

class BM_TQSymmetricKernel : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State &state) override {
        const size_t bits = static_cast<size_t>(state.range(0));
        allocator = VecSimAllocator::newVecsimAllocator();
        tq_state =
            std::make_shared<TQState>(kDim, bits, kProjections, kSeed, true);
        storage_blobs.clear();
        storage_blobs.reserve(kVectorPairs * 2);
        TQPreprocessor preprocessor(allocator, tq_state);

        std::mt19937_64 rng(kSeed + bits);
        for (size_t i = 0; i < kVectorPairs * 2; ++i) {
            auto input = GenerateUnitVector(rng);
            void *storage_blob = nullptr;
            size_t storage_blob_size = input.size() * sizeof(float);
            preprocessor.preprocessForStorage(input.data(), storage_blob, storage_blob_size);

            auto *encoded = static_cast<uint8_t *>(storage_blob);
            storage_blobs.emplace_back(encoded, encoded + storage_blob_size);
            allocator->free_allocation(storage_blob);
        }
    }

    void TearDown(const benchmark::State &) override {
        storage_blobs.clear();
        tq_state.reset();
        allocator.reset();
    }

protected:
    std::shared_ptr<VecSimAllocator> allocator;
    std::shared_ptr<TQState> tq_state;
    std::vector<std::vector<uint8_t>> storage_blobs;
};

BENCHMARK_DEFINE_F(BM_TQSymmetricKernel, EstimateInnerProductSymmetric)
(benchmark::State &state) {
    size_t idx = 0;
    for (auto _ : state) {
        const auto &lhs_blob = storage_blobs[idx];
        const auto &rhs_blob = storage_blobs[idx + 1];
        const auto lhs = tq_state->storageView(lhs_blob.data());
        const auto rhs = tq_state->storageView(rhs_blob.data());
        const float estimate = tq_state->estimateInnerProductSymmetric(lhs, rhs);
        benchmark::DoNotOptimize(estimate);
        idx += 2;
        if (idx >= storage_blobs.size()) {
            idx = 0;
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(BM_TQSymmetricKernel, EstimateInnerProductSymmetric)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->ArgName("bits")
    ->Unit(benchmark::kMicrosecond);

} // namespace

BENCHMARK_MAIN();
