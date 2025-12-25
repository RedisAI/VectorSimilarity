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

    // index_offset: Offset added to base index types to access variants (0=original, 1=updated)

    static void RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                             std::atomic_int &correct, unsigned short index_offset = 0);

    // Search for the K closest vectors to the query in the index. K is defined in the
    // test registration (initialization file).
    static void TopK_BF(benchmark::State &st, unsigned short index_offset = 0);
    // Run TopK using both HNSW and flat index and calculate the recall of the HNSW algorithm
    // with respect to the results returned by the flat index.
    static void TopK_HNSW(benchmark::State &st, unsigned short index_offset = 0);
    static void TopK_Tiered(benchmark::State &st, unsigned short index_offset = 0);

    // Does nothing but returning the index memory.
    static void Memory(benchmark::State &st, IndexTypeIndex index_type);
};

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter,
                                                 size_t k, std::atomic_int &correct,
                                                 unsigned short index_offset) {
    HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
    auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
    auto hnsw_results =
        VecSimIndex_TopKQuery(GET_INDEX(INDEX_HNSW + index_offset),
                              QUERIES[iter % N_QUERIES].data(), k, &query_params, BY_SCORE);
    st.PauseTiming();

    // Measure recall:
    auto bf_results = VecSimIndex_TopKQuery(GET_INDEX(INDEX_BF + index_offset),
                                            QUERIES[iter % N_QUERIES].data(), k, nullptr, BY_SCORE);

    BM_VecSimGeneral::MeasureRecall(hnsw_results, bf_results, correct);

    VecSimQueryReply_Free(bf_results);
    VecSimQueryReply_Free(hnsw_results);
    st.ResumeTiming();
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::Memory(benchmark::State &st, IndexTypeIndex index_type) {
    auto index = GET_INDEX(index_type);
    index->fitMemory();

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] =
        benchmark::Counter((double)VecSimIndex_StatsInfo(index).memory,
                           benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1024);
}

// TopK search BM

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_BF(benchmark::State &st, unsigned short index_offset) {
    size_t k = st.range(0);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(GET_INDEX(INDEX_BF + index_offset), QUERIES[iter % N_QUERIES].data(),
                              k, nullptr, BY_SCORE);
        iter++;
    }
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW(benchmark::State &st, unsigned short index_offset) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    std::atomic_int correct = 0;
    size_t iter = 0;

    // Get initial metrics
    auto index = GET_INDEX(INDEX_HNSW + index_offset);
    VecSimIndexDebugInfo info_before = VecSimIndex_DebugInfo(index);
    size_t num_searches_before = info_before.hnswInfo.num_searches;
    size_t num_visited_nodes_before = info_before.hnswInfo.num_visited_nodes;
    size_t num_visited_nodes_higher_levels_before =
        info_before.hnswInfo.num_visited_nodes_higher_levels;

    for (auto _ : st) {
        RunTopK_HNSW(st, ef, iter, k, correct, index_offset);
        iter++;
    }

    // Get final metrics
    VecSimIndexDebugInfo info_after = VecSimIndex_DebugInfo(index);
    size_t num_searches_after = info_after.hnswInfo.num_searches;
    size_t num_visited_nodes_after = info_after.hnswInfo.num_visited_nodes;
    size_t num_visited_nodes_higher_levels_after =
        info_after.hnswInfo.num_visited_nodes_higher_levels;

    // Calculate deltas
    size_t total_searches = num_searches_after - num_searches_before;
    size_t total_visited_nodes = num_visited_nodes_after - num_visited_nodes_before;
    size_t total_visited_nodes_higher_levels =
        num_visited_nodes_higher_levels_after - num_visited_nodes_higher_levels_before;

    st.counters["Recall"] = (float)correct / (float)(k * iter);
    st.counters["Avg_visited_nodes_level_0"] =
        total_searches > 0 ? (double)total_visited_nodes / (double)total_searches : 0.0;
    st.counters["Avg_visited_nodes_higher_levels"] =
        total_searches > 0 ? (double)total_visited_nodes_higher_levels / (double)total_searches
                           : 0.0;
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_Tiered(benchmark::State &st, unsigned short index_offset) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    std::atomic_int correct = 0;
    std::atomic_int iter = 0;
    auto tiered_index =
        dynamic_cast<TieredHNSWIndex<data_t, dist_t> *>(GET_INDEX(INDEX_TIERED_HNSW));
    size_t total_iters = 50;
    VecSimQueryReply *all_results[total_iters];

    // Get initial metrics from the backend HNSW index
    auto hnsw_index = GET_INDEX(INDEX_HNSW + index_offset);
    VecSimIndexDebugInfo info_before = VecSimIndex_DebugInfo(hnsw_index);
    size_t num_searches_before = info_before.hnswInfo.num_searches;
    size_t num_visited_nodes_before = info_before.hnswInfo.num_visited_nodes;

    auto parallel_knn_search = [](AsyncJob *job) {
        auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = search_job->ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        size_t cur_iter = search_job->iter;
        auto hnsw_results = VecSimIndex_TopKQuery(GET_INDEX(INDEX_TIERED_HNSW),
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
            BM_VecSimGeneral::mock_thread_pool->thread_pool_wait();
        }
    }

    // Get final metrics
    VecSimIndexDebugInfo info_after = VecSimIndex_DebugInfo(hnsw_index);
    size_t num_searches_after = info_after.hnswInfo.num_searches;
    size_t num_visited_nodes_after = info_after.hnswInfo.num_visited_nodes;

    // Calculate deltas
    size_t total_searches = num_searches_after - num_searches_before;
    size_t total_visited_nodes = num_visited_nodes_after - num_visited_nodes_before;

    // Measure recall
    for (iter = 0; iter < total_iters; iter++) {
        auto bf_results =
            VecSimIndex_TopKQuery(GET_INDEX(INDEX_BF + index_offset),
                                  QUERIES[iter % N_QUERIES].data(), k, nullptr, BY_SCORE);
        BM_VecSimGeneral::MeasureRecall(all_results[iter], bf_results, correct);

        VecSimQueryReply_Free(bf_results);
        VecSimQueryReply_Free(all_results[iter]);
    }

    st.counters["Recall"] = (float)correct / (float)(k * iter);
    st.counters["num_threads"] = (double)BM_VecSimGeneral::mock_thread_pool->thread_pool_size;
    st.counters["Avg_visited_nodes_level_0"] =
        total_searches > 0 ? (double)total_visited_nodes / (double)total_searches : 0.0;
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
