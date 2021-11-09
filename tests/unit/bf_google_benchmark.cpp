#include <benchmark/benchmark.h>
#include <random>
#include <iostream>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"

class BM_BatchIteratorBF : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    VecSimIndex *bf_index;
    size_t dim;

public:
    void SetUp(const ::benchmark::State& state) {
        // Initialize BF index
        dim = 4;
        size_t n_vectors = 10;
        VecSimParams params = {.bfParams = {.initialCapacity = n_vectors},
                .type = VecSimType_FLOAT32,
                .size = dim,
                .metric = VecSimMetric_L2,
                .algo = VecSimAlgo_BF};
        bf_index = VecSimIndex_New(&params);

        // Add 1M random vectors with dim=100
        std::vector<float> data(n_vectors * dim);
        rng.seed(47);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < n_vectors * dim; ++i) {
            data[i] = (float)distrib(rng);
        }
        for (size_t i = 0; i < n_vectors; ++i) {
            VecSimIndex_AddVector(bf_index, data.data() + dim*i, i);
        }

    }

    void TearDown(const ::benchmark::State& state) {
        VecSimIndex_Free(bf_index);
    }
};

// Register the function as a benchmark
BENCHMARK_DEFINE_F(BM_BatchIteratorBF, get_10000_total_results)(benchmark::State &st) {
    size_t n_res = st.range(0);
    std::vector<float> query(dim);
    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < dim; ++i) {
        query[i] = (float)distrib(rng);
    }
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(bf_index, query.data());
    size_t res_num = 0;
    while (VecSimBatchIterator_HasNext(batchIterator)) {
        VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, n_res, BY_SCORE);
        res_num += VecSimQueryResult_Len(res);
        if (res_num == 10) {
            break;
        }
    }
}

BENCHMARK_REGISTER_F(BM_BatchIteratorBF, get_10000_total_results)->Arg(2);

BENCHMARK_MAIN();
