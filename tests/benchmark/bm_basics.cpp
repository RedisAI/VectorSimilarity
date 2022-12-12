/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "bm_utils.h"

// Global benchmark data
size_t BM_VecSimBasics::n_vectors = 1000000;
size_t BM_VecSimBasics::n_queries = 10000;
size_t BM_VecSimBasics::dim = 768;
VecSimIndex *BM_VecSimBasics::bf_index;
VecSimIndex *BM_VecSimBasics::hnsw_index;
std::vector<std::vector<float>> BM_VecSimBasics::queries;
size_t BM_VecSimBasics::M = 64;
size_t BM_VecSimBasics::EF_C = 512;
size_t BM_VecSimBasics::block_size = 1024;
const char *BM_VecSimBasics::hnsw_index_file =
    "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_blocks";
const char *BM_VecSimBasics::test_vectors_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

size_t BM_VecSimBasics::ref_count = 0;

BENCHMARK_DEFINE_F(BM_VecSimBasics, AddVectorHNSW)(benchmark::State &st) {
    // Add a new vector from the test vectors in every iteration.
    size_t iter = 0;
    size_t new_id = VecSimIndex_IndexSize(hnsw_index);
    size_t memory_delta = 0;
    for (auto _ : st) {
        memory_delta +=
            VecSimIndex_AddVector(hnsw_index, queries[(iter % n_queries)].data(), new_id++);
        iter++;
    }
    st.counters["memory"] = (double)memory_delta / (double)iter;

    // Clean-up.
    size_t new_index_size = VecSimIndex_IndexSize(hnsw_index);
    for (size_t id = n_vectors; id < new_index_size; id++) {
        VecSimIndex_DeleteVector(hnsw_index, id);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, AddVectorBF)(benchmark::State &st) {
    // Add a new vector from the test vectors in every iteration.
    size_t iter = 0;
    size_t new_id = VecSimIndex_IndexSize(bf_index);
    size_t memory_delta = 0;
    for (auto _ : st) {
        memory_delta +=
            VecSimIndex_AddVector(bf_index, queries[(iter % n_queries)].data(), new_id++);
        iter++;
    }
    st.counters["memory"] = (double)memory_delta / (double)iter;

    // Clean-up.
    size_t new_index_size = VecSimIndex_IndexSize(bf_index);
    for (size_t id = n_vectors; id < new_index_size; id++) {
        VecSimIndex_DeleteVector(bf_index, id);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, DeleteVectorHNSW)(benchmark::State &st) {
    // Remove a different vector in every execution.
    std::vector<std::vector<float>> blobs;
    size_t id_to_remove = 0;
    double memory_delta = 0;
    size_t iter = 0;
    auto hnsw_index_casted = reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index);
    for (auto _ : st) {
        st.PauseTiming();
        auto removed_vec = std::vector<float>(dim);
        memcpy(removed_vec.data(), hnsw_index_casted->getDataByInternalId(id_to_remove),
               dim * sizeof(float));
        blobs.push_back(removed_vec);
        st.ResumeTiming();

        iter++;
        auto delta = (double)VecSimIndex_DeleteVector(hnsw_index, id_to_remove++);
        memory_delta += delta;
    }
    st.counters["memory"] = memory_delta / (double)iter;

    // Restore index state.
    for (size_t i = 0; i < blobs.size(); i++) {
        VecSimIndex_AddVector(hnsw_index, blobs[i].data(), i);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, DeleteVectorBF)(benchmark::State &st) {
    // Remove a different vector in every execution.
    std::vector<std::vector<float>> blobs;
    size_t id_to_remove = 0;
    double memory_delta = 0;
    size_t iter = 0;
    for (auto _ : st) {
        st.PauseTiming();
        auto removed_vec = std::vector<float>(dim);
        float *destination =
            reinterpret_cast<BruteForceIndex_Single<float, float> *>(bf_index)->getDataByInternalId(
                id_to_remove);
        memcpy(removed_vec.data(), destination, dim * sizeof(float));
        blobs.push_back(removed_vec);
        iter++;
        st.ResumeTiming();

        memory_delta += (double)VecSimIndex_DeleteVector(bf_index, id_to_remove++);
    }
    st.counters["memory"] = memory_delta / (double)iter;

    // Restore index state.
    for (size_t i = 0; i < blobs.size(); i++) {
        VecSimIndex_AddVector(bf_index, blobs[i].data(), i);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, TopK_BF)(benchmark::State &st) {
    size_t k = st.range(0);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(bf_index, queries[iter % n_queries].data(), k, nullptr, BY_SCORE);
        iter++;
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, TopK_HNSW)(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    size_t correct = 0;
    size_t iter = 0;
    for (auto _ : st) {
        RunTopK_HNSW(st, ef, iter, k, correct, hnsw_index, bf_index);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, Range_BF)(benchmark::State &st) {
    float radius = (1.0f / 100.0f) * (float)st.range(0);
    size_t iter = 0;
    size_t total_res = 0;

    for (auto _ : st) {
        auto res = VecSimIndex_RangeQuery(bf_index, queries[iter % n_queries].data(), radius,
                                          nullptr, BY_ID);
        total_res += VecSimQueryResult_Len(res);
        iter++;
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, Range_HNSW)(benchmark::State &st) {
    double radius = (1.0f / 100.0f) * (float)st.range(0);
    double epsilon = (1.0f / 1000.0f) * (float)st.range(1);
    size_t iter = 0;
    size_t total_res = 0;
    size_t total_res_bf = 0;
    auto query_params =
        VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.epsilon = epsilon}};

    for (auto _ : st) {
        auto hnsw_results = VecSimIndex_RangeQuery(hnsw_index, queries[iter % n_queries].data(),
                                                   radius, &query_params, BY_ID);
        st.PauseTiming();
        total_res += VecSimQueryResult_Len(hnsw_results);

        // Measure recall:
        auto bf_results = VecSimIndex_RangeQuery(bf_index, queries[iter % n_queries].data(), radius,
                                                 nullptr, BY_ID);
        total_res_bf += VecSimQueryResult_Len(bf_results);

        VecSimQueryResult_Free(bf_results);
        VecSimQueryResult_Free(hnsw_results);
        iter++;
        st.ResumeTiming();
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
    st.counters["Recall"] = (float)total_res / total_res_bf;
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, Memory_FLAT)(benchmark::State &st) {
    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_Info(bf_index).bfInfo.memory;
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, Memory_HNSW)(benchmark::State &st) {
    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_Info(hnsw_index).hnswInfo.memory;
}

BENCHMARK_REGISTER_F(BM_VecSimBasics, Memory_FLAT)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimBasics, Memory_HNSW)->Iterations(1);

BENCHMARK_REGISTER_F(BM_VecSimBasics, AddVectorHNSW)
    ->Unit(benchmark::kMillisecond)
    ->Iterations((long)BM_VecSimBasics::block_size);

BENCHMARK_REGISTER_F(BM_VecSimBasics, AddVectorBF)
    ->Unit(benchmark::kMillisecond)
    ->Iterations((long)BM_VecSimBasics::block_size);

BENCHMARK_REGISTER_F(BM_VecSimBasics, DeleteVectorHNSW)
    ->Unit(benchmark::kMillisecond)
    ->Iterations((long)BM_VecSimBasics::block_size);

BENCHMARK_REGISTER_F(BM_VecSimBasics, DeleteVectorBF)
    ->Unit(benchmark::kMillisecond)
    ->Iterations((long)BM_VecSimBasics::block_size);

BENCHMARK_REGISTER_F(BM_VecSimBasics, TopK_BF)
    ->Arg(10)
    ->ArgName("k")
    ->Arg(100)
    ->ArgName("k")
    ->Arg(500)
    ->ArgName("k")
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, TopK_HNSW)
// {ef_runtime, k} (recall that always ef_runtime >= k)
HNSW_TOP_K_ARGS(10, 10)
HNSW_TOP_K_ARGS(200, 10)
HNSW_TOP_K_ARGS(100, 100)
HNSW_TOP_K_ARGS(200, 100)
HNSW_TOP_K_ARGS(500, 500)->Iterations(100)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, Range_BF)
    // The actual radius will be the given arg divided by 100, since arg must be an integer.
    ->Arg(20)
    ->ArgName("radiusX100")
    ->Arg(35)
    ->ArgName("radiusX100")
    ->Arg(50)
    ->ArgName("radiusX100")
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, Range_HNSW)
// {radius*100, epsilon*1000}
// The actual radius will be the given arg divided by 100, and the actual epsilon values
// will be the given arg divided by 1000.
#define HNSW_RANGE_ARGS(radius, epsilon)                                                           \
    ->Args({radius, epsilon})->ArgNames({"radiusX100", "epsilonX1000"})

HNSW_RANGE_ARGS(20, 1)
HNSW_RANGE_ARGS(20, 10)
HNSW_RANGE_ARGS(20, 100)
HNSW_RANGE_ARGS(35, 1)
HNSW_RANGE_ARGS(35, 10)
HNSW_RANGE_ARGS(35, 100)
HNSW_RANGE_ARGS(50, 1)
HNSW_RANGE_ARGS(50, 10)
HNSW_RANGE_ARGS(50, 100)->Iterations(100)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
