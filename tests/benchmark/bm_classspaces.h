#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/space_interface.h"
#include "VecSim/spaces/space_aux.h"

class BM_VecSimSpaces : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    size_t dim;
    float *v1, *v2;
    Arch_Optimization opt;

    BM_VecSimSpaces() {
        rng.seed(47);
        opt = getArchitectureOptimization();
    }

public:
    void SetUp(const ::benchmark::State &state);

    void TearDown(const ::benchmark::State &state) {
        delete v1;
        delete v2;
    }

    ~BM_VecSimSpaces() {}
};