#pragma once

#include "bm_utils.h"
#include "bm_common.h"

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::block_size = 1024;

const char *BM_VecSimGeneral::test_vectors_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

size_t BM_VecSimGeneral::ref_count = 0;
template <typename index_type_t>
class BM_VecSimBasics : public BM_VecSimIndex<index_type_t> {
public:
    BM_VecSimBasics() : BM_VecSimIndex<index_type_t>(false){};
    ~BM_VecSimBasics() = default;

    static void RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                             size_t &correct, Offset_t index_offset = 0);

    static void TopK_BF(size_t k, benchmark::State &st, Offset_t index_offset = 0);
    static void TopK_HNSW(size_t ef, size_t k, benchmark::State &st, Offset_t index_offset = 0);

    static void Memory_FLAT(benchmark::State &st, Offset_t index_offset = 0);
    static void Memory_HNSW(benchmark::State &st, Offset_t index_offset = 0);
};

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter,
                                                 size_t k, size_t &correct, Offset_t index_offset) {
    auto query_params = VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.efRuntime = ef}};
    auto hnsw_results =
        VecSimIndex_TopKQuery(INDICES[VecSimAlgo_HNSWLIB + index_offset],
                              QUERIES[iter % N_QUERIES].data(), k, &query_params, BY_SCORE);
    st.PauseTiming();

    // Measure recall:
    auto bf_results = VecSimIndex_TopKQuery(INDICES[VecSimAlgo_BF + index_offset],
                                            QUERIES[iter % N_QUERIES].data(), k, nullptr, BY_SCORE);

    BM_VecSimGeneral::MeasureRecall(hnsw_results, bf_results, correct);

    VecSimQueryResult_Free(bf_results);
    VecSimQueryResult_Free(hnsw_results);
    st.ResumeTiming();
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::TopK_BF(size_t k, benchmark::State &st, Offset_t index_offset) {
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(INDICES[VecSimAlgo_BF + index_offset],
                              QUERIES[iter % N_QUERIES].data(), k, nullptr, BY_SCORE);
        iter++;
    }
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::TopK_HNSW(size_t ef, size_t k, benchmark::State &st,
                                              Offset_t index_offset) {
    size_t correct = 0;
    size_t iter = 0;
    for (auto _ : st) {
        RunTopK_HNSW(st, ef, iter, k, correct, index_offset);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::Memory_FLAT(benchmark::State &st, Offset_t index_offset) {

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] =
        (double)VecSimIndex_Info(INDICES[VecSimAlgo_BF + index_offset]).bfInfo.memory;
}
template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::Memory_HNSW(benchmark::State &st, Offset_t index_offset) {

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] =
        (double)VecSimIndex_Info(INDICES[VecSimAlgo_HNSWLIB + index_offset]).hnswInfo.memory;
}

// TopK search BM
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, TopK_BF_fp32, fp32_index_t)
(benchmark::State &st) {
    size_t k = st.range(0);
    TopK_BF(k, st);
}
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, TopK_BF_fp64, fp64_index_t)
(benchmark::State &st) {
    size_t k = st.range(0);
    TopK_BF(k, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, TopK_HNSW_fp32, fp32_index_t)
(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    TopK_HNSW(ef, k, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, TopK_HNSW_fp64, fp64_index_t)
(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    TopK_HNSW(ef, k, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Memory_FLAT_fp32, fp32_index_t)
(benchmark::State &st) { Memory_FLAT(st); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Memory_FLAT_fp64, fp64_index_t)
(benchmark::State &st) { Memory_FLAT(st); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Memory_HNSW_fp32, fp32_index_t)
(benchmark::State &st) { Memory_HNSW(st); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Memory_HNSW_fp64, fp64_index_t)
(benchmark::State &st) { Memory_HNSW(st); }

#define HNSW_TOP_K_ARGS(ef_runtime, k) Args({ef_runtime, k})->ArgNames({"ef_runtime", "k"})

#define REGISTER_TopK_BF(BM_CLASS, BM_FUNC)                                                        \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Arg(10)                                                                                  \
        ->ArgName("k")                                                                             \
        ->Arg(100)                                                                                 \
        ->ArgName("k")                                                                             \
        ->Arg(500)                                                                                 \
        ->ArgName("k")                                                                             \
        ->Unit(benchmark::kMillisecond)

REGISTER_TopK_BF(BM_VecSimBasics, TopK_BF_fp32);
REGISTER_TopK_BF(BM_VecSimBasics, TopK_BF_fp64);

// {ef_runtime, k} (recall that always ef_runtime >= k)
#define REGISTER_TopK_HNSW(BM_CLASS, BM_FUNC)                                                      \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->HNSW_TOP_K_ARGS(10, 10)                                                                  \
        ->HNSW_TOP_K_ARGS(200, 10)                                                                 \
        ->HNSW_TOP_K_ARGS(100, 100)                                                                \
        ->HNSW_TOP_K_ARGS(200, 100)                                                                \
        ->HNSW_TOP_K_ARGS(500, 500)                                                                \
        ->Iterations(100)                                                                          \
        ->Unit(benchmark::kMillisecond)

REGISTER_TopK_HNSW(BM_VecSimBasics, TopK_HNSW_fp32);
REGISTER_TopK_HNSW(BM_VecSimBasics, TopK_HNSW_fp64);

BENCHMARK_REGISTER_F(BM_VecSimBasics, Memory_FLAT_fp32)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimBasics, Memory_FLAT_fp64)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimBasics, Memory_HNSW_fp32)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimBasics, Memory_HNSW_fp64)->Iterations(1);
