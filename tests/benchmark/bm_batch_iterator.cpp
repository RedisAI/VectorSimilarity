#include <benchmark/benchmark.h>
#include <random>
#include <iostream>
#include <unistd.h>
#include <VecSim/algorithms/hnsw/serialization.h>
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

        size_t M = 50;
        size_t ef = 350;
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
            //VecSimIndex_AddVector(hnsw_index, data.data() + dim * i, i);
        }
        char *location = get_current_dir_name();
        auto serializer = hnswlib::HNSWIndexSerializer(reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex());
        auto file_name = std::string(location) + "/tests/benchmark/data/random-1M-100-l2.hnsw";
        //serializer.saveIndex(file_name);
        serializer.loadIndex(file_name, reinterpret_cast<HNSWIndex *>(hnsw_index)->getSpace().get());
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
    ->Unit(benchmark::kMillisecond);


BENCHMARK_DEFINE_F(BM_BatchIterator, get_10000_results_HNSW)(benchmark::State &st) {

    size_t n_res = st.range(0);
    size_t total_res_num = 10000;
    auto bf_results = VecSimIndex_TopKQuery(bf_index, query.data(), total_res_num, nullptr, BY_SCORE);
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(hnsw_index, query.data());
    VecSimQueryResult_List accumulated_results[total_res_num];
    size_t batch_num = 0, res_num = 0;
    while (VecSimBatchIterator_HasNext(batchIterator)) {
        VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, n_res, BY_SCORE);
        res_num += VecSimQueryResult_Len(res);
        accumulated_results[batch_num++] = res;
        if (res_num == total_res_num) {
            break;
        }
    }

    // Measure recall
    size_t correct = 0;
    for (size_t i = 0; i < batch_num; i++) {
        auto hnsw_results = accumulated_results[i];
        auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
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
        VecSimQueryResult_Free(hnsw_results);
    }
    st.counters["Recall"] = (float)correct / total_res_num;
    VecSimQueryResult_Free(bf_results);

    for (auto _ : st) {
        batchIterator = VecSimBatchIterator_New(hnsw_index, query.data());
        res_num = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, n_res, BY_SCORE);
            res_num += VecSimQueryResult_Len(res);
            if (res_num == total_res_num) {
                break;
            }
        }
        VecSimBatchIterator_Free(batchIterator);
    }
}

BENCHMARK_REGISTER_F(BM_BatchIterator, get_10000_results_HNSW)
    ->Arg(100)
    ->Arg(1000)
    ->Unit(benchmark::kMillisecond);


BENCHMARK_DEFINE_F(BM_BatchIterator, TopK_BF)(benchmark::State &st) {
    size_t k = st.range(0);
    for (auto _: st) {
        VecSimIndex_TopKQuery(bf_index, query.data(), k, nullptr, BY_SCORE);
    }
}

BENCHMARK_DEFINE_F(BM_BatchIterator, TopK_HNSW)(benchmark::State &st) {
    size_t k = st.range(0);
    auto bf_results = VecSimIndex_TopKQuery(bf_index, query.data(), k, nullptr, BY_SCORE);
    auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, query.data(), k, nullptr, BY_SCORE);

    // Measure recall
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
BENCHMARK_REGISTER_F(BM_BatchIterator, TopK_BF)
        ->Arg(10000)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_BatchIterator, TopK_HNSW)
        ->Arg(10000)
        ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
