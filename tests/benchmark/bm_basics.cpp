#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/hnsw/serialization.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "bm_utils.h"

// Global benchmark data
size_t BM_VecSimBasics::n_vectors = 1000000;
size_t BM_VecSimBasics::n_queries = 10000;
size_t BM_VecSimBasics::dim = 768;
VecSimIndex *BM_VecSimBasics::bf_index;
VecSimIndex *BM_VecSimBasics::hnsw_index;
std::vector<std::vector<float>> *BM_VecSimBasics::queries;
size_t HNSW_M = 64;
size_t HNSW_EF_C = 512;
const char *hnsw_index_file = "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1";
const char *test_vectors_file = "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

size_t BM_VecSimBasics::ref_count = 0;

BENCHMARK_DEFINE_F(BM_VecSimBasics, AddVectorHNSW)(benchmark::State &st) {
    // Add a new vector from the test vectors in every iteration.
    size_t iter = 0;
    size_t new_id = VecSimIndex_IndexSize(hnsw_index);
    for (auto _ : st) {
        VecSimIndex_AddVector(hnsw_index, (*queries)[(iter % n_queries)].data(), new_id++);
        iter++;
    }
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
    for (auto _ : st) {
        VecSimIndex_AddVector(bf_index, (*queries)[(iter % n_queries)].data(), new_id++);
        iter++;
    }
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

    for (auto _ : st) {
        st.PauseTiming();
        auto removed_vec = std::vector<float>(dim);
        memcpy(removed_vec.data(),
               reinterpret_cast<HNSWIndex *>(hnsw_index)
                   ->getHNSWIndex()
                   ->getDataByInternalId(id_to_remove),
               dim * sizeof(float));
        blobs.push_back(removed_vec);
        st.ResumeTiming();
        VecSimIndex_DeleteVector(hnsw_index, id_to_remove++);
    }

    // Restore index state.
    for (size_t i = 0; i < blobs.size(); i++) {
        VecSimIndex_AddVector(hnsw_index, blobs[i].data(), i);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, DeleteVectorBF)(benchmark::State &st) {
    // Remove a different vector in every execution.
    std::vector<std::vector<float>> blobs;
    size_t id_to_remove = 0;

    for (auto _ : st) {
        st.PauseTiming();
        auto removed_vec = std::vector<float>(dim);
        auto *vector_block_member = reinterpret_cast<BruteForceIndex *>(bf_index)
                                        ->idToVectorBlockMemberMapping[id_to_remove];
        size_t index = vector_block_member->index;
        float *destination = vector_block_member->block->getVector(index);
        memcpy(removed_vec.data(), destination, dim * sizeof(float));
        blobs.push_back(removed_vec);
        st.ResumeTiming();

        VecSimIndex_DeleteVector(bf_index, id_to_remove++);
    }

    // Restore index state.
    for (size_t i = 0; i < blobs.size(); i++) {
        VecSimIndex_AddVector(bf_index, blobs[i].data(), i);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, TopK_BF)(benchmark::State &st) {
    size_t k = st.range(0);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(bf_index, (*queries)[iter % n_queries].data(), k, nullptr, BY_SCORE);
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
        auto res = VecSimIndex_RangeQuery(bf_index, (*queries)[iter % n_queries].data(), radius,
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
        auto hnsw_results = VecSimIndex_RangeQuery(hnsw_index, (*queries)[iter % n_queries].data(),
                                                   radius, &query_params, BY_ID);
        st.PauseTiming();
        total_res += VecSimQueryResult_Len(hnsw_results);

        // Measure recall:
        auto bf_results = VecSimIndex_RangeQuery(bf_index, (*queries)[iter % n_queries].data(),
                                                 radius, nullptr, BY_ID);
        total_res_bf += VecSimQueryResult_Len(bf_results);

        VecSimQueryResult_Free(bf_results);
        VecSimQueryResult_Free(hnsw_results);
        iter++;
        st.ResumeTiming();
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
    st.counters["Recall"] = (float)total_res / total_res_bf;
}

BENCHMARK_REGISTER_F(BM_VecSimBasics, AddVectorHNSW)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BM_VecSimBasics, AddVectorBF)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, DeleteVectorHNSW)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BM_VecSimBasics, DeleteVectorBF)->Unit(benchmark::kMillisecond);

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
    ->Args({10, 10})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({200, 10})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({100, 100})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({200, 100})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({500, 500})
    ->ArgNames({"ef_runtime", "k"})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

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
    ->Args({20, 1})
    ->ArgNames({"radiusX100", "epsilonX1000"})
    ->Args({20, 10})
    ->ArgNames({"radiusX100", "epsilonX1000"})
    ->Args({20, 100})
    ->ArgNames({"radiusX100", "epsilonX1000"})
    ->Args({35, 1})
    ->ArgNames({"radiusX100", "epsilonX1000"})
    ->Args({35, 10})
    ->ArgNames({"radiusX100", "epsilonX1000"})
    ->Args({35, 100})
    ->ArgNames({"radiusX100", "epsilonX1000"})
    ->Args({50, 1})
    ->ArgNames({"radiusX100", "epsilonX1000"})
    ->Args({50, 10})
    ->ArgNames({"radiusX100", "epsilonX1000"})
    ->Args({50, 100})
    ->ArgNames({"radiusX100", "epsilonX1000"})
    ->Iterations(100)
    ->ArgNames({"radiusX100", "epsilonX1000"})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
