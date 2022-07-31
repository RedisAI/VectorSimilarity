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

public:
    BM_VecSimSpaces();
    ~BM_VecSimSpaces() {}

    void SetUp(const ::benchmark::State &state);
    void TearDown(const ::benchmark::State &state);
};
