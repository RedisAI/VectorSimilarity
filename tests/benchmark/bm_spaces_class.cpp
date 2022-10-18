#include "bm_spaces_class.h"

BM_VecSimSpaces::BM_VecSimSpaces() {
    rng.seed(47);
    opt = getArchitectureOptimization();
}

void BM_VecSimSpaces::SetUp(const ::benchmark::State &state) {
    dim = state.range(0);
    v1 = new float[dim];
    v2 = new float[dim];
    std::uniform_real_distribution<double> distrib(-1.0, 1.0);
    for (size_t i = 0; i < dim; i++) {
        v1[i] = (float)distrib(rng);
        v2[i] = (float)distrib(rng);
    }
}

void BM_VecSimSpaces::TearDown(const ::benchmark::State &state) {
    delete v1;
    delete v2;
}

BM_VecSimSpaces_FP64::BM_VecSimSpaces_FP64() {
    rng.seed(47);
    opt = getArchitectureOptimization();
}

void BM_VecSimSpaces_FP64::SetUp(const ::benchmark::State &state) {
    dim = state.range(0);
    v1 = new double[dim];
    v2 = new double[dim];
    std::uniform_real_distribution<double> distrib(-1.0, 1.0);
    for (size_t i = 0; i < dim; i++) {
        v1[i] = distrib(rng);
        v2[i] = distrib(rng);
    }
}

void BM_VecSimSpaces_FP64::TearDown(const ::benchmark::State &state) {
    delete v1;
    delete v2;
}
