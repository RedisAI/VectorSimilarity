/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/AVX512FP16_BW_VL.h"
#include "VecSim/types/float16.h"
#include "VecSim/types/sq8.h"

#if defined(CPU_FEATURES_ARCH_X86_64) && defined(OPT_AVX512_FP16_BW_VL) && defined(OPT_AVX512F) && \
    defined(MOD19169_NATIVE_FP16) && MOD19169_NATIVE_FP16 && defined(MOD19169_FP16_VARIANT)
namespace {
using vecsim_types::float16;
using vecsim_types::sq8;
constexpr uint32_t seed = 19169;
constexpr size_t large_pool_bytes = 240 * 1024 * 1024;
constexpr int variant = MOD19169_FP16_VARIANT;
static_assert(variant >= 0 && variant <= 2);
enum class Mode { Hot, Dependent, LargePool };

struct Inputs {
    size_t dim, stride, count;
    std::vector<uint8_t> storage;
    std::vector<float16> query;
    std::vector<uint32_t> order, next;
    uint64_t hash = 14695981039346656037ULL;
    double public_checksum = 0, mixed_checksum = 0;
    spaces::dist_func_t<float> public_func, mixed_func;

    Inputs(size_t dimension, bool large)
        : dim(dimension),
          stride((sq8::storage_bytes_count<VecSimMetric_IP>(dim) + 63) & ~size_t(63)),
          count(large ? large_pool_bytes / stride : 8), storage(count * stride),
          query(dim + sizeof(float) / sizeof(float16)), order(count), next(count) {
        public_func = spaces::IP_SQ8_FP16_GetDistFunc(dim);
        mixed_func = spaces::Choose_SQ8_FP16_IP_implementation_AVX512F(dim);
        const bool native_expected = dim >= (variant == 0 ? 32 : 128);
        const auto expected = native_expected
                                  ? spaces::Choose_SQ8_FP16_IP_implementation_AVX512FP16_BW_VL(dim)
                                  : mixed_func;
        if (public_func != expected)
            throw std::runtime_error("Public dispatcher did not select the expected variant tier");

        std::mt19937 rng(seed + static_cast<uint32_t>(dim));
        std::vector<float> values(dim);
        double norm2 = 0;
        for (float &value : values) {
            value = static_cast<int32_t>(rng() % 65536) - 32768;
            norm2 += double(value) * value;
        }
        float query_sum = 0;
        for (size_t j = 0; j < dim; ++j) {
            query[j] = vecsim_types::FP32_to_FP16(values[j] / std::sqrt(norm2));
            query_sum += vecsim_types::FP16_to_FP32(query[j]);
        }
        std::memcpy(query.data() + dim, &query_sum, sizeof(query_sum));

        for (size_t i = 0; i < count; ++i) {
            auto *blob = storage.data() + i * stride;
            double code_norm2 = 0;
            float code_sum = 0;
            for (size_t j = 0; j < dim; ++j) {
                blob[j] = rng() & 255;
                const double centered = double(blob[j]) - 127.5;
                code_norm2 += centered * centered;
                code_sum += blob[j];
            }
            // Affine decoding produces a unit-length storage vector without a mean correction.
            const float delta = 1.0 / std::sqrt(code_norm2);
            const float metadata[] = {-127.5f * delta, delta, code_sum};
            std::memcpy(blob + dim, metadata, sizeof(metadata));
            const float a = public_func(blob, query.data(), dim);
            const float b = mixed_func(blob, query.data(), dim);
            if (!std::isfinite(a) || !std::isfinite(b) || a <= 0 || b <= 0)
                throw std::runtime_error("Expected finite positive scores for dependency cycle");
            public_checksum += a;
            mixed_checksum += b;
        }
        std::iota(order.begin(), order.end(), 0);
        for (size_t n = count; n > 1; --n)
            std::swap(order[n - 1], order[rng() % n]);
        for (size_t i = 0; i < count; ++i)
            next[order[i]] = order[(i + 1) % count];
        const auto hash_bytes = [this](const void *data, size_t bytes) {
            const auto *p = static_cast<const uint8_t *>(data);
            for (size_t i = 0; i < bytes; ++i) {
                hash ^= p[i];
                hash *= 1099511628211ULL;
            }
        };
        hash_bytes(storage.data(), storage.size());
        hash_bytes(query.data(), query.size() * sizeof(float16));
        hash_bytes(order.data(), order.size() * sizeof(uint32_t));
    }
};

Inputs &inputs(size_t dim, bool large) {
    static std::map<std::pair<size_t, bool>, std::unique_ptr<Inputs>> pools;
    auto &pool = pools[{dim, large}];
    if (!pool)
        pool = std::make_unique<Inputs>(dim, large);
    return *pool;
}

void sweep(benchmark::State &state, size_t dim, bool mixed, Mode mode) {
    auto &data = inputs(dim, mode == Mode::LargePool);
    const auto function = mixed ? data.mixed_func : data.public_func;
    const auto *query = data.query.data();
    uint32_t index = data.order[0];
    size_t cursor = 0;
    float score = 0;
    // First-touch initialization and hot warming happen on the single benchmark thread.
    if (mode != Mode::LargePool) {
        for (const auto node : data.order)
            benchmark::DoNotOptimize(
                function(data.storage.data() + node * data.stride, query, dim));
    }
    if (mode == Mode::LargePool) {
        // A full pool per batch prevents short calibration runs from repeatedly timing a prefix
        // that fits in cache. KeepRunningBatch counts each distance as an iteration.
        while (state.KeepRunningBatch(data.count)) {
            for (const auto node : data.order) {
                score = function(data.storage.data() + node * data.stride, query, dim);
                benchmark::DoNotOptimize(score);
            }
        }
    } else {
        for (auto _ : state) {
            score = function(data.storage.data() + index * data.stride, query, dim);
            benchmark::DoNotOptimize(score);
            if (mode == Mode::Dependent) {
                // Positive scores make the sign bit zero: identical shuffled addresses, but the
                // next address cannot be resolved until this call's result is available.
                index = data.next[index] ^ (std::bit_cast<uint32_t>(score) >> 31);
            } else {
                if (++cursor == data.count)
                    cursor = 0;
                index = data.order[cursor];
            }
        }
    }
    benchmark::DoNotOptimize(index);
    state.SetItemsProcessed(state.iterations());
    state.counters["dimension"] = dim;
    state.counters["kernel_variant"] = variant;
    state.counters["mixed"] = mixed;
    state.counters["mode"] = static_cast<int>(mode);
    state.counters["native_dispatch"] = !mixed && dim >= (variant == 0 ? 32 : 128);
    state.counters["seed"] = seed;
    state.counters["pool_bytes"] = data.storage.size();
    state.counters["nodes"] = data.count;
    state.counters["input_hash_hi"] = static_cast<uint32_t>(data.hash >> 32);
    state.counters["input_hash_lo"] = static_cast<uint32_t>(data.hash);
    state.counters["checksum"] = mixed ? data.mixed_checksum : data.public_checksum;
    state.counters["last_score"] = score;
}

void register_sweep() {
    const size_t dimensions[] = {32,  33,  48,  64,  80,  96,  100, 127, 128,  129,  160,
                                 192, 224, 255, 256, 257, 300, 512, 768, 1024, 1536, 2048};
    for (const auto dim : dimensions) {
        for (const auto mode : {Mode::Hot, Mode::Dependent, Mode::LargePool}) {
            if (mode == Mode::LargePool && dim != 768)
                continue;
            for (const bool mixed : {false, true}) {
                const std::string mode_name = mode == Mode::Hot         ? "hot"
                                              : mode == Mode::Dependent ? "dependent"
                                                                        : "large_pool";
                const std::string name = "SQ8FP16/" + mode_name + "/" +
                                         (mixed ? "mixed" : "public") +
                                         "/dim:" + std::to_string(dim);
                benchmark::RegisterBenchmark(
                    name.c_str(), [=](benchmark::State &state) { sweep(state, dim, mixed, mode); })
                    ->Unit(benchmark::kNanosecond)
                    ->MinTime(0.02)
                    ->Repetitions(3);
            }
        }
    }
}
} // namespace
#endif

int main(int argc, char **argv) {
#if defined(CPU_FEATURES_ARCH_X86_64) && defined(OPT_AVX512_FP16_BW_VL) && defined(OPT_AVX512F) && \
    defined(MOD19169_NATIVE_FP16) && MOD19169_NATIVE_FP16 && defined(MOD19169_FP16_VARIANT)
    const auto features = spaces::getCpuOptimizationFeatures();
    if (!(features.avx512f && features.avx512bw && features.avx512vl && features.avx512_fp16)) {
        std::cerr << "Kernel sweep requires runtime AVX512F/BW/VL/FP16; refusing fallback\n";
        return 2;
    }
    if ((_mm_getcsr() & _MM_ROUND_MASK) != _MM_ROUND_NEAREST) {
        std::cerr << "Kernel sweep requires nearest MXCSR rounding\n";
        return 2;
    }
    try {
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv))
            return 2;
        benchmark::AddCustomContext("kernel_variant", std::to_string(variant));
        benchmark::AddCustomContext("seed", std::to_string(seed));
        benchmark::AddCustomContext("large_pool",
                                    "240 MiB allocated; shuffled access, not guaranteed DRAM");
        register_sweep();
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
    } catch (const std::exception &error) {
        std::cerr << "Kernel sweep failed: " << error.what() << '\n';
        return 2;
    }
    return 0;
#else
    std::cerr << "Kernel sweep requires native FP16 build support, MOD19169_NATIVE_FP16=1, "
                 "and MOD19169_FP16_VARIANT=0/1/2\n";
    return 2;
#endif
}
