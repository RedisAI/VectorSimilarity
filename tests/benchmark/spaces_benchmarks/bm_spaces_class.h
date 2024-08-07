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

template <typename DATA_TYPE>
class BM_VecSimSpaces : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    size_t dim;
    DATA_TYPE *v1, *v2;

    virtual DATA_TYPE DoubleToType(double val) { return static_cast<DATA_TYPE>(val); }

public:
    BM_VecSimSpaces() { rng.seed(47); }
    ~BM_VecSimSpaces() = default;

    void SetUp(const ::benchmark::State &state) {
        dim = state.range(0);
        v1 = new DATA_TYPE[dim];
        v2 = new DATA_TYPE[dim];
        std::uniform_real_distribution<double> distrib(-1.0, 1.0);
        for (size_t i = 0; i < dim; i++) {
            v1[i] = DoubleToType(distrib(rng));
            v2[i] = DoubleToType(distrib(rng));
        }
    }
    void TearDown(const ::benchmark::State &state) {
        delete v1;
        delete v2;
    }
};
