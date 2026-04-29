/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "bm_spaces.h"
#include "VecSim/types/float16.h"
#include "utils/tests_utils.h"

using sq8 = vecsim_types::sq8;
using float16 = vecsim_types::float16;

/**
 * SQ8-to-FP16 benchmarks: SQ8 quantized storage with FP16 query.
 * Only naive (scalar) benchmarks are registered for now; SIMD chooser symbols are added
 * by P1b (MOD-15152, x86) and P1c (MOD-15153, ARM).
 */
class BM_VecSimSpaces_SQ8_FP16 : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    size_t dim;
    float16 *v1;
    uint8_t *v2;

public:
    BM_VecSimSpaces_SQ8_FP16() { rng.seed(47); }
    ~BM_VecSimSpaces_SQ8_FP16() = default;

    void SetUp(const ::benchmark::State &state) {
        dim = state.range(0);
        size_t query_bytes =
            dim * sizeof(float16) + sq8::query_metadata_count<VecSimMetric_L2>() * sizeof(float);
        v1 = reinterpret_cast<float16 *>(new uint8_t[query_bytes]);
        test_utils::populate_sq8_fp16_query(v1, dim, true, 123);
        size_t quantized_size =
            dim * sizeof(uint8_t) + sq8::storage_metadata_count<VecSimMetric_L2>() * sizeof(float);
        v2 = new uint8_t[quantized_size];
        test_utils::populate_float_vec_to_sq8_with_metadata(v2, dim, true, 1234);
    }
    void TearDown(const ::benchmark::State &state) {
        delete[] reinterpret_cast<uint8_t *>(v1);
        delete[] v2;
    }
};

// Naive (scalar) algorithms. SIMD chooser slots will be added by P1b (MOD-15152) and
// P1c (MOD-15153), following the SQ8_FP32 layout in bm_spaces_sq8_fp32.cpp.

INITIALIZE_NAIVE_BM(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, InnerProduct, 16);
INITIALIZE_NAIVE_BM(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, Cosine, 16);
INITIALIZE_NAIVE_BM(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, L2Sqr, 16);

BENCHMARK_MAIN();
