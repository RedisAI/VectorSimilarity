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
#ifdef SQ8_EXACT_CANDIDATE
#include "VecSim/spaces/L2/L2_SQ8_exact.h"
#endif
#include <bit>
#include <cfenv>
#include <cstdio>
#include <random>
#include <time.h>
#include <vector>
#ifdef __x86_64__
#include <xmmintrin.h>
#endif

using Distance = float (*)(const void *, const void *, size_t);

static float exactDistance(const void *storage, const void *query, size_t dim) {
#ifdef SQ8_EXACT_CANDIDATE
    const auto *bytes = static_cast<const uint8_t *>(storage);
    const float min = load_unaligned<float>(bytes + dim);
    const float delta = load_unaligned<float>(bytes + dim + sizeof(float));
    return sq8_l2_detail::exact(bytes, static_cast<const float *>(query), dim, min, delta);
#else
    return SQ8_FP32_L2Sqr(storage, query, dim);
#endif
}

static Distance selectDistance(size_t dim, int mode) {
    if (mode == 2)
        return exactDistance;
#ifndef SQ8_STANDALONE
    if (mode == 1)
        return spaces::L2_SQ8_FP32_GetDistFunc(dim);
#endif
    return SQ8_FP32_L2Sqr;
}

extern "C" uint32_t sq8_distance(const void *storage, const void *query, size_t dim, int mode) {
    return std::bit_cast<uint32_t>(selectDistance(dim, mode)(storage, query, dim));
}

extern "C" int sq8_certified(const void *storage, const void *query, size_t dim) {
#ifdef SQ8_EXACT_CANDIDATE
    const auto *bytes = static_cast<const uint8_t *>(storage);
    float result;
    return sq8_l2_detail::certified(bytes, static_cast<const float *>(query), dim,
                                    load_unaligned<float>(bytes + dim),
                                    load_unaligned<float>(bytes + dim + sizeof(float)), result);
#else
    return -1;
#endif
}

extern "C" void sq8_quantize(const float *input, size_t dim, void *output) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    QuantPreprocessor<float, VecSimMetric_L2> preprocessor(allocator, dim);
    void *storage = nullptr;
    size_t bytes = dim * sizeof(float);
    preprocessor.preprocessForStorage(input, storage, bytes, 0);
    memcpy(output, storage, bytes);
    allocator->free_allocation(storage);
}

extern "C" void sq8_query(const float *input, size_t dim, void *output) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    QuantPreprocessor<float, VecSimMetric_L2> preprocessor(allocator, dim);
    void *query = nullptr;
    size_t bytes = dim * sizeof(float);
    preprocessor.preprocessQuery(input, query, bytes, 0);
    memcpy(output, query, bytes);
    allocator->free_allocation(query);
}

extern "C" double sq8_benchmark(const void *storage, const void *query, size_t dim, int mode,
                                size_t iterations) {
    const auto distance = selectDistance(dim, mode);
    timespec begin{}, end{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &begin);
    for (size_t i = 0; i < iterations; ++i) {
        const float result = distance(storage, query, dim);
        asm volatile("" : : "g"(result) : "memory");
    }
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    return (double(end.tv_sec - begin.tv_sec) * 1e9 + double(end.tv_nsec - begin.tv_nsec)) /
           double(iterations);
}

extern "C" void sq8_environment(int mode, int flush) {
    const int modes[] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
    std::fesetround(modes[mode]);
#ifdef __x86_64__
    unsigned control = _mm_getcsr();
    control &= ~unsigned(0x8040);
    if (flush)
        control |= 0x8040;
    _mm_setcsr(control);
#elif defined(__aarch64__)
    uint64_t control;
    asm volatile("mrs %0, fpcr" : "=r"(control));
    control &= ~(uint64_t{1} << 24);
    if (flush)
        control |= uint64_t{1} << 24;
    asm volatile("msr fpcr, %0" : : "r"(control));
#endif
}

#ifdef SQ8_SANITIZER_MAIN
int main() {
    std::mt19937 generator(1028);
    for (size_t sample = 0; sample < 10000; ++sample) {
        const size_t dim = 1 + generator() % 160;
        std::vector<uint8_t> storage(dim + 4 * sizeof(float));
        std::vector<uint32_t> query(dim + 2);
        for (auto &code : storage)
            code = uint8_t(generator());
        for (auto &value : query)
            value = generator() & 0xfeffffffU;
        const uint32_t metadata[] = {uint32_t(generator()) & 0xfeffffffU,
                                     uint32_t(generator()) & 0xfeffffffU, 0, 0};
        memcpy(storage.data() + dim, metadata, sizeof(metadata));
        const auto actual = sq8_distance(storage.data(), query.data(), dim, 0);
        const auto expected = sq8_distance(storage.data(), query.data(), dim, 2);
        if (actual != expected) {
            std::fprintf(stderr, "Certificate mismatch at sample %zu\n", sample);
            return 1;
        }
    }
    std::puts("ASan/UBSan: 10000 randomized exact/certified comparisons passed.");
}
#endif
