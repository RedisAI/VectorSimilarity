#pragma once
/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "bm_spaces_class.h"
BM_VecSimSpaces::BM_VecSimSpaces() {
    rng.seed(47);
    opt = getArchitectureOptimization();
}

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
