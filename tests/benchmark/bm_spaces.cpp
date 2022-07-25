#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/space_interface.h"

#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/IP/IP_SSE.h"
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/L2/L2_SSE.h"

class BM_VecSimSpaces : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    size_t dim;
    // SpaceInterface<float> *space;
    DISTFUNC<float> df;
    float *v1, *v2;

    BM_VecSimSpaces() {
        dim = 768;
        rng.seed(47);
    }

public:
    void SetUp(const ::benchmark::State &state) {}

    void TearDown(const ::benchmark::State &state) {
        free(v1);
        free(v2);
    }

    ~BM_VecSimSpaces() {}
};

BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX512_16)(benchmark::State &st) {
    size_t dim = st.range(0) * 16;
    v1 = (float *)malloc(dim * sizeof(float));
    v2 = (float *)malloc(dim * sizeof(float));
    std::uniform_real_distribution<double> distrib(-1.0, 1.0);
    for (size_t i = 0; i < dim; i++) {
        v1[i] = (float)distrib(rng);
        v2[i] = (float)distrib(rng);
    }

    if (st.range(1))
        df = L2SqrSIMD16Ext_AVX512;
    else
        df = InnerProductSIMD16Ext_AVX512;

    for (auto _ : st) {
        df(v1, v2, &dim);
    }
}

// Register the function as a benchmark
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_16)
    // The actual radius will the given arg divided by 100, since arg must me and integer.
    ->Args({5, 0})
    ->Args({200, 0})
    ->Args({500, 0})
    ->Args({5, 1})
    ->Args({200, 1})
    ->Args({500, 1})
    ->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
