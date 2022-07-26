#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/space_interface.h"
#include "VecSim/spaces/space_aux.h"

#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/L2/L2.h"

class BM_VecSimSpaces : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    size_t dim;
    float *v1, *v2;

    BM_VecSimSpaces() { rng.seed(47); }

public:
    void SetUp(const ::benchmark::State &state) {
        dim = state.range(0);
        v1 = (float *)malloc(dim * sizeof(float));
        v2 = (float *)malloc(dim * sizeof(float));
        std::uniform_real_distribution<double> distrib(-1.0, 1.0);
        for (size_t i = 0; i < dim; i++) {
            v1[i] = (float)distrib(rng);
            v2[i] = (float)distrib(rng);
        }
    }

    void TearDown(const ::benchmark::State &state) {
        free(v1);
        free(v2);
    }

    ~BM_VecSimSpaces() {}
};

#ifdef __AVX512F__
#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/L2/L2_AVX512.h"

BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX512_L2_16)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD16Ext_AVX512(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX512_L2_4)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD4Ext_AVX512(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX512_L2_16_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD16ExtResiduals_AVX512(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX512_L2_4_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD4ExtResiduals_AVX512(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX512_IP_16)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD16Ext_AVX512(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX512_IP_4)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD4Ext_AVX512(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX512_IP_16_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD16ExtResiduals_AVX512(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX512_IP_4_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD4ExtResiduals_AVX512(v1, v2, &dim);
    }
}
#endif

#ifdef __AVX__
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/L2/L2_AVX.h"

BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX_L2_16)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD16Ext_AVX(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX_L2_4)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD4Ext_AVX(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX_L2_16_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD16ExtResiduals_AVX(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX_L2_4_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD4ExtResiduals_AVX(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX_IP_16)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD16Ext_AVX(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX_IP_4)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD4Ext_AVX(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX_IP_16_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD16ExtResiduals_AVX(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, AVX_IP_4_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD4ExtResiduals_AVX(v1, v2, &dim);
    }
}
#endif

#ifdef __SSE__
#include "VecSim/spaces/IP/IP_SSE.h"
#include "VecSim/spaces/L2/L2_SSE.h"

BENCHMARK_DEFINE_F(BM_VecSimSpaces, SSE_L2_16)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD16Ext_SSE(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, SSE_L2_4)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD4Ext_SSE(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, SSE_L2_16_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD16ExtResiduals_SSE(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, SSE_L2_4_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        L2SqrSIMD4ExtResiduals_SSE(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, SSE_IP_16)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD16Ext_SSE(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, SSE_IP_4)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD4Ext_SSE(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, SSE_IP_16_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD16ExtResiduals_SSE(v1, v2, &dim);
    }
}
BENCHMARK_DEFINE_F(BM_VecSimSpaces, SSE_IP_4_Residuals)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProductSIMD4ExtResiduals_SSE(v1, v2, &dim);
    }
}
#endif

BENCHMARK_DEFINE_F(BM_VecSimSpaces, NAIVE_IP)(benchmark::State &st) {
    for (auto _ : st) {
        InnerProduct(v1, v2, &dim);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimSpaces, NAIVE_L2)(benchmark::State &st) {
    for (auto _ : st) {
        L2Sqr(v1, v2, &dim);
    }
}

// Register the function as a benchmark
#define EXACT_PARAMS                                                                               \
    ->Arg(16)                                                                                      \
        ->Arg(128)                                                                                 \
        ->Arg(400)                                                                                 \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)
        // ->Iterations(1000000)

#define RESIDUAL_PARAMS                                                                            \
    ->Arg(16 - 1)                                                                                  \
        ->Arg(128 - 1)                                                                             \
        ->Arg(400 - 1)                                                                             \
        ->ArgName("Dimension")                                                                     \
        ->Unit(benchmark::kNanosecond)
        // ->Iterations(1000000)

#ifdef __AVX512F__
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_L2_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_L2_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_L2_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_L2_4_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_IP_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_IP_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_IP_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX512_IP_4_Residuals) RESIDUAL_PARAMS;
#endif

#ifdef __AVX__
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_L2_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_L2_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_L2_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_L2_4_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_IP_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_IP_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_IP_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, AVX_IP_4_Residuals) RESIDUAL_PARAMS;
#endif

#ifdef __SSE__
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_L2_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_L2_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_L2_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_L2_4_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_IP_16) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_IP_4) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_IP_16_Residuals) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, SSE_IP_4_Residuals) RESIDUAL_PARAMS;
#endif

BENCHMARK_REGISTER_F(BM_VecSimSpaces, NAIVE_L2) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, NAIVE_L2) RESIDUAL_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, NAIVE_IP) EXACT_PARAMS;
BENCHMARK_REGISTER_F(BM_VecSimSpaces, NAIVE_IP) RESIDUAL_PARAMS;

BENCHMARK_MAIN();
