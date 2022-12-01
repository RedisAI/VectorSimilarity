/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/space_aux.h"

template <typename data_type>
class BM_VecSimSpaces : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    size_t dim;
    data_type *v1, *v2;
    Arch_Optimization opt;

public:
    BM_VecSimSpaces();
    ~BM_VecSimSpaces() {}

    void SetUp(const ::benchmark::State &state);
    void TearDown(const ::benchmark::State &state);
};

template <typename data_type>
BM_VecSimSpaces<data_type>::BM_VecSimSpaces() {
    rng.seed(47);
    opt = getArchitectureOptimization();
}

template <typename data_type>
void BM_VecSimSpaces<data_type>::SetUp(const ::benchmark::State &state) {
    dim = state.range(0);
    v1 = new data_type[dim];
    v2 = new data_type[dim];
    std::uniform_real_distribution<double> distrib(-1.0, 1.0);
    for (size_t i = 0; i < dim; i++) {
        v1[i] = (data_type)distrib(rng);
        v2[i] = (data_type)distrib(rng);
    }
}

template <typename data_type>
void BM_VecSimSpaces<data_type>::TearDown(const ::benchmark::State &state) {
    delete v1;
    delete v2;
}
