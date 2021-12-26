#include <benchmark/benchmark.h>
#include <random>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"

class BM_BatchIterator : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    VecSimIndex *bf_index;
    VecSimIndex *hnsw_index;
    size_t dim;
    std::vector<float> query;

    BM_BatchIterator() {
        // Initialize BF index with dim=100
        dim = 100;
        size_t n_vectors = 1000000;
        VecSimParams params = {.algo = VecSimAlgo_BF,
                               .bfParams = {.type = VecSimType_FLOAT32,
                                            .dim = dim,
                                            .metric = VecSimMetric_L2,
                                            .initialCapacity = n_vectors}};
        bf_index = VecSimIndex_New(&params);

        size_t M = 32;
        size_t ef = 200;
        params = {.algo = VecSimAlgo_HNSWLIB,
                .hnswParams = {.type = VecSimType_FLOAT32,
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .initialCapacity = n_vectors,
                        .M = M,
                        .efConstruction = ef,
                        .efRuntime = ef}};
        hnsw_index = VecSimIndex_New(&params);

        // Add 1M random vectors
        std::vector<float> data(n_vectors * dim);
        rng.seed(47);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < n_vectors * dim; ++i) {
            data[i] = (float)distrib(rng);
        }
        for (size_t i = 0; i < n_vectors; ++i) {
            VecSimIndex_AddVector(bf_index, data.data() + dim * i, i);
            VecSimIndex_AddVector(hnsw_index, data.data() + dim * i, i);
        }
        query.reserve(dim);
    }

public:
    void SetUp(const ::benchmark::State &state) {
        // Generate random query vector before every iteration
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < dim; ++i) {
            query[i] = (float)distrib(rng);
        }
    }

    void TearDown(const ::benchmark::State &state) {}

    ~BM_BatchIterator() {
        VecSimIndex_Free(bf_index);
        VecSimIndex_Free(hnsw_index);
    }
};

BENCHMARK_DEFINE_F(BM_BatchIterator, get_10000_results_BF)(benchmark::State &st) {

    size_t n_res = st.range(0);
    for (auto _ : st) {
        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(bf_index, query.data());
        size_t res_num = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, n_res, BY_SCORE);
            res_num += VecSimQueryResult_Len(res);
            if (res_num == 10000) {
                break;
            }
        }
        VecSimBatchIterator_Free(batchIterator);
    }
}

// Register the function as a benchmark
BENCHMARK_REGISTER_F(BM_BatchIterator, get_10000_results_BF)
    ->Arg(100)
    ->Arg(1000)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);


BENCHMARK_DEFINE_F(BM_BatchIterator, get_10000_results_HNSW)(benchmark::State &st) {

    size_t n_res = st.range(0);
    for (auto _ : st) {
        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(hnsw_index, query.data());
        size_t res_num = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, n_res, BY_SCORE);
            res_num += VecSimQueryResult_Len(res);
            if (res_num == 10000) {
                break;
            }
        }
        VecSimBatchIterator_Free(batchIterator);
    }
}

BENCHMARK_REGISTER_F(BM_BatchIterator, get_10000_results_HNSW)
        ->Arg(100)
        ->Arg(1000)
        ->Iterations(100)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
