/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include "bm_vecsim_index.h"

size_t BM_VecSimGeneral::block_size = 1024;

// Class for common bm for basic index and updated index.
template <typename index_type_t>
class BM_VecSimCommon : public BM_VecSimIndex<index_type_t> {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    BM_VecSimCommon() = default;
    ~BM_VecSimCommon() = default;

    static void RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                             std::atomic_int &correct, unsigned short index_offset = 0,
                             bool is_tiered = false);
    static void RunTopK_SVS(benchmark::State &st, size_t ws, VecSimOptionMode searchHistory,
                            double epsilon, size_t iter, size_t k, std::atomic_int &correct,
                            unsigned short index_offset = 0, bool is_tiered = false);

    // Search for the K closest vectors to the query in the index. K is defined in the
    // test registration (initialization file).
    static void TopK_BF(benchmark::State &st, unsigned short index_offset = 0);
    // Run TopK using both HNSW and flat index and calculate the recall of the HNSW algorithm
    // with respect to the results returned by the flat index.
    static void TopK_HNSW(benchmark::State &st, unsigned short index_offset = 0);
    static void TopK_Tiered(benchmark::State &st, unsigned short index_offset = 0);
    static void TopK_SVS(benchmark::State &st, unsigned short index_offset = 0);

    // Does nothing but returning the index memory.
    static void Memory_FLAT(benchmark::State &st, unsigned short index_offset = 0);
    static void Memory_HNSW(benchmark::State &st, unsigned short index_offset = 0);
    static void Memory_Tiered(benchmark::State &st, unsigned short index_offset = 0);
    static void Memory_SVS(benchmark::State &st, unsigned short index_offset = 0);
    static void Memory_Tiered_SVS(benchmark::State &st, unsigned short index_offset = 0);
};

// for the svs HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
// for svs change to SVSRuntimeParams svsRuntimeParams = {.ws = ef};
// run once with history and once without history
// mix congiguration with and without history
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter,
                                                 size_t k, std::atomic_int &correct,
                                                 unsigned short index_offset, bool is_tiered) {
    HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
    auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
    auto hnsw_results = VecSimIndex_TopKQuery(
        INDICES[is_tiered ? INDEX_VecSimAlgo_TIERED_HNSW : INDEX_VecSimAlgo_HNSWLIB + index_offset],
        QUERIES[iter % N_QUERIES].data(), k, &query_params, BY_SCORE);
    st.PauseTiming();

    // Measure recall:
    auto bf_results = VecSimIndex_TopKQuery(INDICES[INDEX_VecSimAlgo_BF + index_offset],
                                            QUERIES[iter % N_QUERIES].data(), k, nullptr, BY_SCORE);

    BM_VecSimGeneral::MeasureRecall(hnsw_results, bf_results, correct);

    VecSimQueryReply_Free(bf_results);
    VecSimQueryReply_Free(hnsw_results);
    st.ResumeTiming();
}

// runtopk svs
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::RunTopK_SVS(benchmark::State &st, size_t ws,
                                                VecSimOptionMode searchHistory, double epsilon,
                                                size_t iter, size_t k, std::atomic_int &correct,
                                                unsigned short index_offset, bool is_tiered) {
    SVSRuntimeParams svsRuntimeParams = {
        .windowSize = ws, .searchHistory = searchHistory, .epsilon = epsilon};
    auto query_params = BM_VecSimGeneral::CreateQueryParams(svsRuntimeParams);
    auto svs_results =
        VecSimIndex_TopKQuery(INDICES[INDEX_VecSimAlgo_SVS + index_offset],
                              QUERIES[iter % N_QUERIES].data(), k, &query_params, BY_SCORE);
    st.PauseTiming();

    // Measure recall:
    auto bf_results = VecSimIndex_TopKQuery(INDICES[INDEX_VecSimAlgo_BF + index_offset],
                                            QUERIES[iter % N_QUERIES].data(), k, nullptr, BY_SCORE);

    BM_VecSimGeneral::MeasureRecall(svs_results, bf_results, correct);

    VecSimQueryReply_Free(bf_results);
    VecSimQueryReply_Free(svs_results);
    st.ResumeTiming();
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::Memory_FLAT(benchmark::State &st, unsigned short index_offset) {
    auto index = INDICES[INDEX_VecSimAlgo_BF + index_offset];
    index->fitMemory();

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_StatsInfo(index).memory;
}
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::Memory_HNSW(benchmark::State &st, unsigned short index_offset) {
    auto index = INDICES[INDEX_VecSimAlgo_HNSWLIB + index_offset];
    index->fitMemory();

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_StatsInfo(index).memory;
}
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::Memory_Tiered(benchmark::State &st,
                                                  unsigned short index_offset) {
    auto index = INDICES[INDEX_VecSimAlgo_TIERED_HNSW + index_offset];
    index->fitMemory();
    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_StatsInfo(index).memory;
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::Memory_Tiered_SVS(benchmark::State &st,
                                                      unsigned short index_offset) {
    auto index = INDICES[INDEX_VecSimAlgo_TIERED_SVS + index_offset];
    index->fitMemory();
    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_StatsInfo(index).memory;
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::Memory_SVS(benchmark::State &st, unsigned short index_offset) {
    auto index = INDICES[INDEX_VecSimAlgo_SVS + index_offset];
    index->fitMemory();
    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_StatsInfo(index).memory;
}

// TopK search BM

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_BF(benchmark::State &st, unsigned short index_offset) {
    size_t k = st.range(0);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(INDICES[INDEX_VecSimAlgo_BF + index_offset],
                              QUERIES[iter % N_QUERIES].data(), k, nullptr, BY_SCORE);
        iter++;
    }
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW(benchmark::State &st, unsigned short index_offset) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    std::atomic_int correct = 0;
    size_t iter = 0;
    for (auto _ : st) {
        RunTopK_HNSW(st, ef, iter, k, correct, index_offset);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_Tiered(benchmark::State &st, unsigned short index_offset) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    std::atomic_int correct = 0;
    std::atomic_int iter = 0;
    auto *tiered_index =
        dynamic_cast<TieredHNSWIndex<data_t, dist_t> *>(INDICES[INDEX_VecSimAlgo_TIERED_HNSW]);
    size_t total_iters = 50;
    VecSimQueryReply *all_results[total_iters];

    auto parallel_knn_search = [](AsyncJob *job) {
        auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = search_job->ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        size_t cur_iter = search_job->iter;
        auto hnsw_results = VecSimIndex_TopKQuery(INDICES[INDEX_VecSimAlgo_TIERED_HNSW],
                                                  QUERIES[cur_iter % N_QUERIES].data(),
                                                  search_job->k, &query_params, BY_SCORE);
        search_job->all_results[cur_iter] = hnsw_results;
        delete job;
    };

    for (auto _ : st) {
        auto search_job = new (tiered_index->getAllocator())
            tieredIndexMock::SearchJobMock(tiered_index->getAllocator(), parallel_knn_search,
                                           tiered_index, k, ef, iter++, all_results);
        tiered_index->submitSingleJob(search_job);
        if (iter == total_iters) {
            BM_VecSimGeneral::mock_thread_pool.thread_pool_wait();
        }
    }

    // Measure recall
    for (iter = 0; iter < total_iters; iter++) {
        auto bf_results =
            VecSimIndex_TopKQuery(INDICES[INDEX_VecSimAlgo_BF + index_offset],
                                  QUERIES[iter % N_QUERIES].data(), k, nullptr, BY_SCORE);
        BM_VecSimGeneral::MeasureRecall(all_results[iter], bf_results, correct);

        VecSimQueryReply_Free(bf_results);
        VecSimQueryReply_Free(all_results[iter]);
    }

    st.counters["Recall"] = (float)correct / (float)(k * iter);
    st.counters["num_threads"] = (double)BM_VecSimGeneral::mock_thread_pool.thread_pool_size;
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_SVS(benchmark::State &st, unsigned short index_offset) {
    size_t ws = st.range(0);
    size_t k = st.range(1);
    VecSimOptionMode searchHistory = static_cast<VecSimOptionMode>(st.range(2));
    double epsilon =
        static_cast<double>(st.range(3)) / 100.0; // Convert from int percentage to double
    std::atomic_int correct = 0;
    size_t iter = 0;
    for (auto _ : st) {
        RunTopK_SVS(st, ws, searchHistory, epsilon, iter, k, correct, index_offset, false);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
}

#define REGISTER_TopK_BF(BM_CLASS, BM_FUNC)                                                        \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Arg(10)                                                                                  \
        ->Arg(100)                                                                                 \
        ->Arg(500)                                                                                 \
        ->ArgName("k")                                                                             \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

// {ef_runtime, k} (recall that always ef_runtime >= k)
#define REGISTER_TopK_HNSW(BM_CLASS, BM_FUNC)                                                      \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({10, 10})                                                                           \
        ->Args({200, 10})                                                                          \
        ->Args({100, 100})                                                                         \
        ->Args({200, 100})                                                                         \
        ->Args({500, 500})                                                                         \
        ->ArgNames({"ef_runtime", "k"})                                                            \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

// {ef_runtime, k} (recall that always ef_runtime >= k)
#define REGISTER_TopK_Tiered(BM_CLASS, BM_FUNC)                                                    \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({10, 10})                                                                           \
        ->Args({200, 10})                                                                          \
        ->Args({100, 100})                                                                         \
        ->Args({200, 100})                                                                         \
        ->Args({500, 500})                                                                         \
        ->ArgNames({"ef_runtime", "k"})                                                            \
        ->Iterations(50)                                                                           \
        ->Unit(benchmark::kMillisecond)

// {window_size, k, search_history, epsilon_percent}
// search_history: 0=AUTO, 1=ENABLE, 2=DISABLE
// epsilon_percent: epsilon value * 100 (e.g., 1 = 0.01, 5 = 0.05)
#define REGISTER_TopK_SVS(BM_CLASS, BM_FUNC)                                                       \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->ArgsProduct({                                                                            \
            benchmark::CreateRange(10, 500, /*multiplier=*/10), /* window_size: 10,100,500 */      \
            benchmark::CreateRange(10, 500, /*multiplier=*/10), /* k: 10,100,500 */                \
            benchmark::CreateDenseRange(0, 2, /*step=*/1),      /* search_history: 0,1,2 */        \
            {1, 5, 10}                                          /* epsilon_percent: 1,5,10 */      \
        })                                                                                         \
        ->ArgNames({"window_size", "k", "search_history", "epsilon_percent"})                      \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)
