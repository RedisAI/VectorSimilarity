#include <benchmark/benchmark.h>
#include <random>
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/IP/IP_AVX.h"

class BM_Microbench : public benchmark::Fixture {


public:
    void SetUp(const ::benchmark::State &state) {
    }

    void TearDown(const ::benchmark::State &state) {}

    ~BM_Microbench() { }
};



BENCHMARK_DEFINE_F(BM_Microbench, base_ip)(benchmark::State &st) {
    size_t dim = 512;
    std::mt19937 rng;
    rng.seed(47);
    float vector1[dim];
    float vector2[dim];

    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < dim; ++i) {
            vector1[i] = (float)distrib(rng);
            vector2[i] = (float)distrib(rng);
    }
    for (auto _ : st) {
        InnerProduct(vector1, vector2, &dim);
    }
}

BENCHMARK_DEFINE_F(BM_Microbench, avx_ip)(benchmark::State &st) {
    size_t dim = 512;
    std::mt19937 rng;
    rng.seed(47);
    float vector1[dim];
    float vector2[dim];

    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < dim; ++i) {
            vector1[i] = (float)distrib(rng);
            vector2[i] = (float)distrib(rng);
    }
    for (auto _ : st) {
        InnerProductSIMD16Ext_AVX(vector1, vector2, &dim);
    }
}

BENCHMARK_REGISTER_F(BM_Microbench, base_ip);
BENCHMARK_REGISTER_F(BM_Microbench, avx_ip);

BENCHMARK_MAIN();
