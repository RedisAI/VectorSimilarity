/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#include <benchmark/benchmark.h>
#include <random>
#include "../../utils/tests_utils.h"
#include "bm_spaces.h"

class BM_VecSimSpaces_Integers_INT8 : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    size_t dim;
    int8_t *v1, *v2;

public:
    BM_VecSimSpaces_Integers_INT8() { rng.seed(47); }
    ~BM_VecSimSpaces_Integers_INT8() = default;

    void SetUp(const ::benchmark::State &state) {
        dim = state.range(0);
        v1 = new int8_t[dim];
        v2 = new int8_t[dim];

        // random for int8_t and uint8_t is not provided by the standard library
        memcpy(v1, test_utils::create_int8_vec(dim).data(), dim);
        memcpy(v2, test_utils::create_int8_vec(dim).data(), dim);
    }
    void TearDown(const ::benchmark::State &state) {
        delete v1;
        delete v2;
    }
};


#ifdef CPU_FEATURES_ARCH_X86_64
cpu_features::X86Features opt = cpu_features::GetX86Info().features;

// AVX512_BF16 functions
#ifdef OPT_AVX512_F_BW_VL_VNNI
bool avx512_f_bw_vl_vnni_supported = opt.avx512f && opt.avx512bw && opt.avx512vl && opt.avx512vnni;
INITIALIZE_BENCHMARKS_SET_L2(BM_VecSimSpaces_Integers_INT8, INT8, AVX512F_BW_VL_VNNI, 32,
                             avx512_f_bw_vl_vnni_supported);
// INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_Integers_INT8, INT8, AVX512_F_BW_VL_VNNI, 32,
//                              avx512_f_bw_vl_vnni_supported);
// INITIALIZE_BENCHMARKS_SET_COSINE(BM_VecSimSpaces_Integers_INT8, INT8, AVX512_F_BW_VL_VNNI, 32,
//                              avx512_f_bw_vl_vnni_supported)
#endif // AVX512_BF16


#endif // x86_64

INITIALIZE_NAIVE_BM(BM_VecSimSpaces_Integers_INT8, INT8, InnerProduct, 32);
INITIALIZE_NAIVE_BM(BM_VecSimSpaces_Integers_INT8, INT8, L2Sqr, 32);
BENCHMARK_MAIN();
