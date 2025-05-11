/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/
#include "bm_spaces.h"
#include "utils/tests_utils.h"

class BM_VecSimSpaces_SQ8 : public benchmark::Fixture {
    protected:
        std::mt19937 rng;
        size_t dim;
        float *v1;
        uint8_t *v2;
    
    public:
    BM_VecSimSpaces_SQ8() { rng.seed(47); }
        ~BM_VecSimSpaces_SQ8() = default;
    
        void SetUp(const ::benchmark::State &state) {
            dim = state.range(0);
            v1 = new float[dim];
            test_utils::populate_float_vec(v1, dim, 123);
            // Allocate vector with extra space for min, delta and cosine calculations
            v2 = new uint8_t[dim + sizeof(float) * 3];
            test_utils::populate_float_vec_to_sq8(v2, dim, 1234);
        }
        void TearDown(const ::benchmark::State &state) {
            delete v1;
            delete v2;
        }
    };

#ifdef CPU_FEATURES_ARCH_X86_64
cpu_features::X86Features opt = cpu_features::GetX86Info().features;

// AVX512_F_BW_VL_VNNI functions
#ifdef OPT_AVX512_F_BW_VL_VNNI
bool avx512_f_bw_vl_vnni_supported = opt.avx512f && opt.avx512bw &&
                                   opt.avx512vl && opt.avx512vnni;
INITIALIZE_BENCHMARKS_SET_IP(BM_VecSimSpaces_SQ8, SQ8, AVX512F_BW_VL_VNNI, 32,
                                avx512_f_bw_vl_vnni_supported);
// INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_Integers_INT8, INT8, AVX512F_BW_VL_VNNI, 32,
//                                  avx512_f_bw_vl_vnni_supported);
#endif // AVX512_F_BW_VL_VNNI

#ifdef AVX2
// AVX2 functions
bool avx2_supported = opt.avx2;
INITIALIZE_BENCHMARKS_SET_IP(BM_VecSimSpaces_SQ8, SQ8, AVX2, 32, avx2_supported);
// INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_Integers_INT8, INT8, AVX2, 32,
//                                  avx2_supported);
#endif // AVX2

// AVX functions
#ifdef OPT_AVX
bool avx_supported = opt.avx;
INITIALIZE_BENCHMARKS_SET_IP(BM_VecSimSpaces_SQ8, SQ8, AVX, 32, avx_supported);
// INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_Integers_INT8, INT8, AVX, 32,
//                                  avx_supported);
#endif // AVX
// SSE functions
#ifdef OPT_SSE
bool sse_supported = opt.sse;
INITIALIZE_BENCHMARKS_SET_IP(BM_VecSimSpaces_SQ8, SQ8, SSE, 32, sse_supported);
// INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8, SQ8, SSE, 32,
//                                   sse_supported);
#endif // SSE
#endif // x86_64

// Naive algorithms

INITIALIZE_NAIVE_BM(BM_VecSimSpaces_SQ8, SQ8, InnerProduct, 16);
INITIALIZE_NAIVE_BM(BM_VecSimSpaces_SQ8, SQ8, Cosine, 16);
INITIALIZE_NAIVE_BM(BM_VecSimSpaces_SQ8, SQ8, L2Sqr, 16);

// Naive

BENCHMARK_MAIN();
