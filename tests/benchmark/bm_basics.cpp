#include <benchmark/benchmark.h>
#include <random>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"

class BM_VecSimBasics : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    VecSimIndex *bf_index;
    VecSimIndex *hnsw_index;
    size_t dim;
    size_t n_vectors;
    std::vector<float> query;


    BM_VecSimBasics() {
        // Initialize BF and HNSW indices.
        dim = 128;
        n_vectors = 100000;
        VecSimParams params = {.algo = VecSimAlgo_BF,
                .bfParams = {.type = VecSimType_FLOAT32,
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .initialCapacity = n_vectors}};
        bf_index = VecSimIndex_New(&params);

        size_t M = 36;
        size_t ef = 200;
        params = {.algo = VecSimAlgo_HNSWLIB,
                .hnswParams = {.type = VecSimType_FLOAT32,
                        .dim = dim,
                        .metric = VecSimMetric_L2,
                        .initialCapacity = n_vectors+1,
                        .M = M,
                        .efConstruction = ef,
                        .efRuntime = ef}};
        hnsw_index = VecSimIndex_New(&params);

        // Add random vectors.
        std::vector<float> data(n_vectors * dim);
        rng.seed(47);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < n_vectors * dim; ++i) {
            data[i] = (float) distrib(rng);
        }
        for (size_t i = 0; i < n_vectors; ++i) {
            VecSimIndex_AddVector(bf_index, data.data() + dim * i, i);
            VecSimIndex_AddVector(hnsw_index, data.data() + dim * i, i);
        }
        query.reserve(dim);
    }

public:
    void SetUp(const ::benchmark::State &state) {
        // Generate random query vector before test.
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < dim; ++i) {
            query[i] = (float)distrib(rng);
        }
    }

    void TearDown(const ::benchmark::State &state) {}

    ~BM_VecSimBasics() {
        VecSimIndex_Free(bf_index);
        VecSimIndex_Free(hnsw_index);
    }
};

BENCHMARK_DEFINE_F(BM_VecSimBasics, AddVectorHNSW)(benchmark::State &st) {

    // Add a single vector to the index
    for (auto _: st) {
        VecSimIndex_AddVector(hnsw_index, query.data(), n_vectors);
        st.PauseTiming();
        VecSimIndex_DeleteVector(hnsw_index, n_vectors);
        st.ResumeTiming();
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, DeleteVectorHNSW)(benchmark::State &st) {

    for (auto _: st) {
        st.PauseTiming();
        VecSimIndex_AddVector(hnsw_index, query.data(), n_vectors);
        st.ResumeTiming();
        VecSimIndex_DeleteVector(hnsw_index, n_vectors);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, TopK_BF)(benchmark::State &st) {
    size_t k = st.range(0);
    for (auto _: st) {
        VecSimIndex_TopKQuery(bf_index, query.data(), k, nullptr, BY_SCORE);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, TopK_HNSW)(benchmark::State &st) {
    size_t k = st.range(0);
    auto bf_results = VecSimIndex_TopKQuery(bf_index, query.data(), k, nullptr, BY_SCORE);
    auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, query.data(), k, nullptr, BY_SCORE);

    // measure recall:
    auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
    size_t correct = 0;
    while (VecSimQueryResult_IteratorHasNext(hnsw_it)) {
        auto hnsw_res_item = VecSimQueryResult_IteratorNext(hnsw_it);
        auto bf_it = VecSimQueryResult_List_GetIterator(bf_results);
        while (VecSimQueryResult_IteratorHasNext(bf_it)) {
            auto bf_res_item = VecSimQueryResult_IteratorNext(bf_it);
            if (VecSimQueryResult_GetId(hnsw_res_item) == VecSimQueryResult_GetId(bf_res_item)) {
                correct++;
                break;
            }
        }
        VecSimQueryResult_IteratorFree(bf_it);
    }
    VecSimQueryResult_IteratorFree(hnsw_it);
    st.counters["Recall"] = (float)correct / k;

    VecSimQueryResult_Free(bf_results);
    VecSimQueryResult_Free(hnsw_results);


    for (auto _: st) {
        VecSimIndex_TopKQuery(hnsw_index, query.data(), k, nullptr, BY_SCORE);
    }
}

// Register the function as a benchmark
BENCHMARK_REGISTER_F(BM_VecSimBasics, AddVectorHNSW)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, DeleteVectorHNSW)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, TopK_BF)
        ->Arg(10)
        ->Arg(100)
        ->Arg(500)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, TopK_HNSW)
        ->Arg(10)
        ->Arg(100)
        ->Arg(500)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
