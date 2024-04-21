#pragma once
/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>

#pragma once
#include "cpu_features_macros.h"
#ifdef CPU_FEATURES_ARCH_X86_64
#include "cpuinfo_x86.h"
#endif

class BM_VecSimSpaces : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    size_t dim;
    DATA_TYPE *v1, *v2;

public:
    BM_VecSimSpaces();
    ~BM_VecSimSpaces() = default;

    void SetUp(const ::benchmark::State &state);
    void TearDown(const ::benchmark::State &state);

    // Specific architecture optimization flags that are supported on this machine,
    // to be initialized in every executable that is running this benchmarks.
#ifdef CPU_FEATURES_ARCH_X86_64
    static cpu_features::X86Features opt;
#endif
};

BM_VecSimSpaces::BM_VecSimSpaces() { rng.seed(47); }

void BM_VecSimSpaces::SetUp(const ::benchmark::State &state) {
    dim = state.range(0);
    v1 = new DATA_TYPE[dim];
    v2 = new DATA_TYPE[dim];
    std::uniform_real_distribution<double> distrib(-1.0, 1.0);
    for (size_t i = 0; i < dim; i++) {
        v1[i] = (DATA_TYPE)distrib(rng);
        v2[i] = (DATA_TYPE)distrib(rng);
    }
}

void BM_VecSimSpaces::TearDown(const ::benchmark::State &state) {
    delete v1;
    delete v2;
}
